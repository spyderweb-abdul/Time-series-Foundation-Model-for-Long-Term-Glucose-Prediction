# Finalised TTM Glucose Pipeline (Full Integration)
# Includes:
# - Safe CustomTrainer with eval_loss logging
# - Auto-skip short users
# - Auto-repair empty validation
# - Multivariate NaN handling (forward+backward fill)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import torch
import psutil
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

print("GPU Available:", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
# CUDA memory optimisation
DEVICE = torch.device("cuda")# if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, default_data_collator, TrainerCallback, EarlyStoppingCallback, set_seed, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
from tsfm_public import TimeSeriesPreprocessor, get_datasets
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions

from pathlib import Path
import inspect
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Dataset Definition
class GlucoseDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        context_length: int,
        prediction_length: int,
        feature_cols=None,
        n_time_features: int = 4,
    ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.feature_cols = feature_cols or []
        # only allocate zero time-features if we truly have none
        self.n_time_features = n_time_features if self.feature_cols else 0

        # Per-user cached arrays and a global index map
        self.user_data = {}           # user_id -> {"target": np.ndarray, "features": Optional[np.ndarray]}
        self.window_indices = []      # list[(user_id, i)], where i is the *right* edge of context

        grouped = df.groupby("USUBJID", sort=False)
        for user_id, user_df in grouped:
            user_df = user_df.sort_values("LBDTC")
            n = len(user_df)
            if n < context_length + prediction_length:
                print(f"[SKIP] User {user_id} has {n} records (< {context_length + prediction_length}). Skipping.")
                continue

            target = user_df["LBORRES"].values.astype(np.float32)
            features = (
                user_df[self.feature_cols].values.astype(np.float32)
                if self.feature_cols else None
            )

            self.user_data[user_id] = {"target": target, "features": features}

            # i is the index where the forecast window begins; context ends at i-1
            # valid: i in [context_length, n - prediction_length]
            for i in range(context_length, n - prediction_length + 1):  # <-- inclusive upper bound
                self.window_indices.append((user_id, i))                # <-- record i directly

        if len(self.window_indices) == 0:
            print("[WARNING] No samples generated in this dataset after skipping small users.")

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, idx: int):
        user_id, i = self.window_indices[idx]
        data = self.user_data[user_id]
        target = data["target"]
        features = data["features"]

        # Context: [i - context_length, i) ; Future: [i, i + prediction_length)
        past_values = target[i - self.context_length : i]
        future_values = target[i : i + self.prediction_length]

        if features is not None:
            past_feats = features[i - self.context_length : i]
            future_feats = features[i : i + self.prediction_length]
        else:
            past_feats = np.zeros((self.context_length, self.n_time_features), dtype=np.float32)
            future_feats = np.zeros((self.prediction_length, self.n_time_features), dtype=np.float32)

        return {
            "past_values": torch.tensor(past_values).unsqueeze(-1),          # [context, 1]
            "future_values": torch.tensor(future_values).unsqueeze(-1),      # [prediction, 1]
            "past_time_features": torch.tensor(past_feats),                  # [context, n_feat] or [context, 0]
            "future_time_features": torch.tensor(future_feats),              # [prediction, n_feat] or [prediction, 0]
            "past_observed_mask": torch.ones(self.context_length, 1, dtype=torch.bool),
            "future_observed_mask": torch.ones(self.prediction_length, 1, dtype=torch.bool),
            "labels": torch.tensor(future_values).unsqueeze(-1),             # Trainer target
            "USUBJID": user_id,
        }


# Loss Function
def compute_custom_loss(preds, targets):
    
    if preds.shape != targets.shape:        
        if preds.shape[1] < targets.shape[1]:
            targets = targets[:, -preds.shape[1]:, :]
        else:
            raise ValueError(f"Shape mismatch: preds={preds.shape}, targets={targets.shape}")

    targets = targets.to(preds.device)
    huber = torch.nn.SmoothL1Loss(beta=27.0)(preds, targets)
    abs_err = torch.abs(preds - targets)
    large_error_threshold = 10.0
    mask = abs_err >= large_error_threshold
    clarke_penalty = torch.mean(abs_err[mask]) if mask.any() else torch.tensor(0.0, device=preds.device)
    return 0.7 * huber + 0.3 * clarke_penalty

# Custom Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if inputs is None:
            raise ValueError("Received None inputs in compute_loss.")
        labels = inputs.get("labels").to(DEVICE)
        if labels is None:
            raise ValueError("Missing 'labels' key in inputs.")
        model_inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k != "labels"}
        preds = model(**model_inputs)
        loss = compute_custom_loss(preds, labels)
        return (loss, {"logits": preds, "labels": labels}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **kwargs):
        if inputs is None:
            return None, None, None

        labels = inputs.get("labels").to(DEVICE)
        if labels is None:
            return None, None, None

        model_inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k != "labels"}
        with torch.no_grad():
            preds = model(**model_inputs)
            loss = compute_custom_loss(preds, labels)

        return (loss, preds, labels)

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        if "eval_loss" not in metrics and "loss" in metrics:
            metrics["eval_loss"] = metrics["loss"]
        elif "eval_loss" not in metrics:
            raise ValueError("Evaluation did not return 'eval_loss'.")
        return metrics


# LoRA Injection

def inject_lora(model, target_modules=None):
    config = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.1,
        target_modules=target_modules or [
            "attention.q_proj", "attention.k_proj", "attention.v_proj",
            "attention.out_proj", "mlp.fc1", "mlp.fc2"
        ],
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        bias="none"
    )
    model.print_trainable_parameters()
    
    return get_peft_model(model, config)

# NaN Repair for Multivariate Data

def repair_nans(df, required_cols):
    df = df.sort_values("LBDTC")
    df[required_cols] = df[required_cols].ffill().bfill()
    return df.dropna(subset=required_cols)

# Training Arguments Helper

def build_training_args(output_dir, lr, batch_size, epochs):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        learning_rate=lr,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        seed=42,
        label_names=[],
        gradient_accumulation_steps=4,
    )

# ==== DUAL HEAD WRAPPER ====
class DualHeadTTM(torch.nn.Module):
    model_input_names = [
        "past_values",
        "past_time_features",
        "future_time_features",
        "past_observed_mask",
        "future_observed_mask",
    ]
    def __init__(self, ttm, d_model, pred_len):
        super().__init__()
        self.ttm = ttm
        self.pred_len = pred_len
        self.uni_head = torch.nn.Sequential(torch.nn.LayerNorm(d_model), torch.nn.Linear(d_model, 1))
        self.multi_head = torch.nn.Sequential(torch.nn.LayerNorm(d_model), torch.nn.Linear(d_model, 1))
        self.use_aux = False

    def forward(self, past_values, past_observed_mask=None,
                past_time_features=None, future_time_features=None,
                future_observed_mask=None, labels=None, **kwargs):

        ttm_out = self.ttm(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            past_time_features=past_time_features,
            future_time_features=future_time_features,
            future_observed_mask=future_observed_mask,
            future_values=labels,
            output_hidden_states=True
        )

        if hasattr(ttm_out, 'backbone_hidden_state') and ttm_out.backbone_hidden_state is not None:
            h = ttm_out.backbone_hidden_state
        elif hasattr(ttm_out, 'hidden_states') and ttm_out.hidden_states is not None:
            h = ttm_out.hidden_states[-1]
        else:
            raise ValueError("No valid hidden state found in TTM output.")
        
        #print(h.shape)

        if h.dim() == 4 and h.shape[1] == 1:
            h = h.squeeze(1)
        
        # Interpolate to prediction_length
        h_interp = F.interpolate(h.transpose(1, 2), size=self.pred_len, mode='linear').transpose(1, 2)

        # Project
        out = self.multi_head(h_interp) if self.use_aux else self.uni_head(h_interp)

        return out
    
    def set_aux(self, value: bool):
        self.use_aux = value

class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            self.train_losses.append({
                'step': state.global_step,
                'epoch': state.epoch if state.epoch is not None else None,
                'loss': logs['loss']
            })

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            self.eval_losses.append({
                'step': state.global_step,
                'epoch': state.epoch,
                'eval_loss': metrics.get('eval_loss')
            })

    def save(self, output_path: str, plot_path: str = None):
        df_train = pd.DataFrame(self.train_losses)
        df_eval = pd.DataFrame(self.eval_losses)

        if df_train.empty and df_eval.empty:
            print("[LossLogger] No logs collected; skipping save.")
            return pd.DataFrame()

        df = pd.merge_asof(
            df_train.sort_values('step'),
            df_eval.sort_values('step'),
            on='step',
            direction='nearest',
            suffixes=('_train', '_eval')
        )

        # coalesce epoch columns to a single one
        epoch_cols = [c for c in ['epoch', 'epoch_train', 'epoch_eval'] if c in df.columns]
        if epoch_cols:
            df['epoch_plot'] = None
            for c in epoch_cols:
                df['epoch_plot'] = df['epoch_plot'].fillna(df[c])
        else:
            df['epoch_plot'] = np.nan

        df.to_csv(output_path, index=False)
        # print(f"[Saved] Loss CSV to {output_path}")

        if plot_path:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))

            # prefer epoch if available, else step
            x_key = 'epoch_plot' if df['epoch_plot'].notnull().any() else 'step'

            if 'loss' in df.columns:
                plt.plot(df[x_key], df['loss'], label='Training Loss')
            if 'eval_loss' in df.columns:
                plt.plot(df[x_key], df['eval_loss'], label='Validation Loss')

            plt.xlabel('Epoch' if x_key == 'epoch_plot' else 'Step')
            plt.ylabel("Loss")
            plt.title("Loss Curves")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            # print(f"[Saved] Loss Plot to {plot_path}")

        return df


class MemoryMonitorCallback(TrainerCallback):
    def __init__(self, log_every_n_steps: int = 1):
        self.log_every_n_steps = log_every_n_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.log_every_n_steps != 0:
            return

        print(f"\n[Memory Monitor] Step {state.global_step}")
        
        # CPU memory usage
        cpu_mem = psutil.virtual_memory()
        used_cpu = (cpu_mem.total - cpu_mem.available) / (1024 ** 3)
        print(f"  CPU Memory Used     : {used_cpu:.2f} GB")

        # GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"  GPU Memory Allocated: {allocated:.2f} GB")
            print(f"  GPU Memory Reserved : {reserved:.2f} GB")
            print(f"  GPU Max Allocated   : {max_allocated:.2f} GB")    

# ==== FULL PIPELINE ====
class TTMGlucosePipeline:
    def __init__(self, train_uni, val_uni, train_multi, val_multi,
                 save_dir, context_length, prediction_length, feature_cols=None):
        self.pred_len = prediction_length
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.train_uni = train_uni
        self.val_uni = val_uni
        self.train_multi = train_multi
        self.val_multi = val_multi

        base_model = get_model(
            "ibm-granite/granite-timeseries-ttm-r2",
            context_length=context_length,
            prediction_length=prediction_length,
        )
        print("TTM config context_length", base_model.config.context_length)
        print("TTM config prediction_length", base_model.config.prediction_length)

        peft_model = inject_lora(base_model)
        base_model_actual = peft_model.base_model.model

        def super_safe_forward(self, *args, **kwargs):
            allowed = set(inspect.signature(self.__original_forward__).parameters.keys())
            safe_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            return self.__original_forward__(*args, **safe_kwargs)

        base_model_actual = peft_model.base_model.model
        if not hasattr(base_model_actual, '__original_forward__'):
            base_model_actual.__original_forward__ = base_model_actual.forward
            base_model_actual.forward = super_safe_forward.__get__(base_model_actual, type(base_model_actual))

        d_model = peft_model.config.d_model
        self.model = DualHeadTTM(peft_model, d_model, prediction_length)
        self.model.to(DEVICE)

        self.model.model_input_names = DualHeadTTM.model_input_names
        self.data_collator = default_data_collator

    def run_pass(self, use_aux, train_set, val_set, lr, epochs, batch_size, outdir):
        self.model.set_aux(use_aux)

        if len(train_set) == 0 or len(val_set) == 0:
            raise ValueError(f"Empty dataset: train_set={len(train_set)}, val_set={len(val_set)}")
       
        if lr is None:
            lr = 5e-4 if not use_aux else 5e-5

        args = build_training_args(
            output_dir=str(self.save_dir / outdir), lr=lr,
            batch_size=batch_size, epochs=epochs)


        optimizer = AdamW(self.model.parameters(), lr=lr)
        # Steps per epoch accounting for accumulation
        steps_per_epoch = math.ceil(len(train_set) / batch_size)
        effective_steps = math.ceil(steps_per_epoch / max(1, args.gradient_accumulation_steps))

        # OneCycleLR over total steps = effective_steps * epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=effective_steps * epochs,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )
        
        loss_logger = LossLoggerCallback()

        trainer = CustomTrainer(
            model=self.model,
            args=args,
            train_dataset=train_set,
            eval_dataset=val_set,
            data_collator=self.data_collator,
            callbacks=[loss_logger, 
                       MemoryMonitorCallback(log_every_n_steps=1), 
                       #EarlyStoppingCallback(early_stopping_patience=4, early_stopping_threshold=1e-5),
                       ],
            #optimizers=(optimizer, scheduler),
        )

        trainer.train()

        # Save metrics
        csv_path = self.save_dir / outdir / "loss_history.csv"
        plot_path = self.save_dir / outdir / "loss_plot.png"
        loss_logger.save(csv_path, plot_path)


    def run_pass1(self, epochs, batch_size, lr):
        self.run_pass(
            use_aux=False,
            train_set=self.train_uni, val_set=self.val_uni,
            lr=lr, epochs=epochs, batch_size=batch_size,
            outdir="pass1_uni"
        )

    def run_pass2(self, epochs, batch_size, lr):
        self.run_pass(
            use_aux=True,
            train_set=self.train_multi, val_set=self.val_multi,
            lr=lr, epochs=epochs, batch_size=batch_size,
            outdir="pass2_multi"
        )

    def run_test(
        self,
        test_set,
        batch_size,
        use_aux=False,
        title="Test Evaluation",
        save_plot_dir=None,
        save_metrics_csv=None,
        max_plots=10,
        per_user=True
    ):

        self.model.set_aux(use_aux)
        self.model.eval()
        preds_all = []
        targets_all = []
        ids_all = []

        with torch.no_grad():
            for batch in torch.utils.data.DataLoader(test_set, batch_size=batch_size):
                inputs = {k: v.to(DEVICE) for k, v in batch.items() if k in self.model.model_input_names}
                labels = batch["labels"].to(DEVICE)
                outputs = self.model(**inputs)

                if outputs.shape[1] < labels.shape[1]:
                    labels = labels[:, -outputs.shape[1]:, :]

                preds_all.append(outputs.cpu().numpy())
                targets_all.append(labels.cpu().numpy())
                
                if 'USUBJID' in batch:
                    ids_all.extend(batch['USUBJID'])

        preds_all = np.concatenate(preds_all, axis=0)
        targets_all = np.concatenate(targets_all, axis=0)

        y_pred = preds_all.reshape(-1)
        y_true = targets_all.reshape(-1)
        
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        print(f"\n{title} Metrics:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE : {mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        # collect ids safely (they are tensors)
        ids_arr = None
        if len(ids_all) > 0:
            if isinstance(ids_all[0], torch.Tensor):
                ids_arr = torch.stack(ids_all).cpu().numpy().ravel()
            else:
                ids_arr = np.array(ids_all).ravel()        

        # === Save metrics to CSV ===
        if save_metrics_csv:
            Path(save_metrics_csv).parent.mkdir(parents=True, exist_ok=True)
            metrics_df = pd.DataFrame([{
                "title": title,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "mape": mape
            }])
            if os.path.exists(save_metrics_csv):
                metrics_df.to_csv(save_metrics_csv, mode='a', header=False, index=False)
            else:
                metrics_df.to_csv(save_metrics_csv, index=False)
            #print(f"[Saved] Metrics saved to {save_metrics_csv}")

        # === Save and Generate multiple plots ===
        if save_plot_dir:
            save_plot_dir = Path(save_plot_dir)
            save_plot_dir.mkdir(parents=True, exist_ok=True)

            if per_user and ids_arr is not None:
                unique_ids = np.unique(ids_arr)
                selected_ids = unique_ids[:max_plots]

                for uid in selected_ids:
                    indices = np.where(ids_arr == uid)[0]
                    if len(indices) == 0:
                        continue
                    i = indices[0]
                    plt.figure(figsize=(10, 4))
                    plt.plot(targets_all[i].squeeze(), label="True")
                    plt.plot(preds_all[i].squeeze(), label="Predicted")
                    plt.title(f"{title} | USUBJID {uid}")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plot_path = save_plot_dir / f"{title.lower().replace(' ', '_')}_user_{uid}.png"
                    plt.savefig(plot_path)
                    plt.close()
            else:
                # Fallback: first N samples
                for i in range(min(max_plots, preds_all.shape[0])):
                    plt.figure(figsize=(10, 4))
                    plt.plot(targets_all[i].squeeze(), label="True")
                    plt.plot(preds_all[i].squeeze(), label="Predicted")
                    plt.title(f"{title} | Sample {i}")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plot_path = save_plot_dir / f"{title.lower().replace(' ', '_')}_sample_{i}.png"
                    plt.savefig(plot_path)
                    plt.close()

        else:
            plt.show()


def run_finetuning_inference():
    
    # Load and prepare your data
    DATA_FILE = "glucose_forecast_data.csv" # Make sure this file exists
    CONTEXT_LENGTH = 512 # Or a smaller value like 512, 768, 1536 for faster iteration
    PREDICTION_LENGTH = 96  # Or a smaller value like 24, 48 {'CL': [512, 1024, 1536], 'FL': [96, 192, 336, 720]}
    VERSION = "_v1.4"
    SAVE_DIR_BASE = f"ttm_finetuned_models/glucose_pipeline{VERSION}"    
    
    df = pd.read_csv(DATA_FILE)
    # Minimal data cleaning from original script
    isolated = [
        911, 930, 943, 976, 334, 484, 491, 549, 558, 1490,
        404, 422, 423, 1425, 1441, 1456, 80, 812, 834, 60,
        227, 248, 25, 254, 1500, 1501, 1520, 1567, 733, 1271,
        288, 29, 290, 312, 317, 48, 756, 761, 767, 95
    ]
    
    # Ensure USUBJID is string if IDs in `isolated` are strings, or vice-versa
    if df['USUBJID'].dtype == 'object' and all(isinstance(x, int) for x in isolated):
        # Attempt to convert USUBJID to numeric if possible, or isolated to string
        try:
            df['USUBJID'] = pd.to_numeric(df['USUBJID'], errors='coerce')
            # Keep only rows where conversion was successful if any isolated IDs are numbers
        except ValueError:
            print("Warning: USUBJID contains non-numeric values. Ensure 'isolated' list matches type or handle mixed types.")
            # As a fallback, convert isolated to string if USUBJID is object type
            isolated = [str(x) for x in isolated]
    
    #exp_users = [1012, 1014, 988]
    
    df = df[~df.USUBJID.isin(isolated)].copy()
    
    #df = df[df.USUBJID.isin(exp_users)].copy()
    df["LBDTC"] = pd.to_datetime(df["LBDTC"])
    df = df.sort_values(by=["USUBJID", "LBDTC"]) # IMPORTANT: Sort data for time series processing
    
    # Preprocess (univariate)
    # Target is LBORRES. Input features for uni are just LBORRES.
    uni_pp = TimeSeriesPreprocessor(
        timestamp_column="LBDTC",
        id_columns=["USUBJID"],
        target_columns=["LBORRES"], # Only glucose
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        scaling=True, # Scaling is generally recommended
        encode_categorical=False, # Though no categoricals defined here for uni
        scaler_type="standard"
    )
    
    splits = {"train": 0.7, "val": 0.2, "test": 0.1}
    
    # Handle potential errors during get_datasets, e.g., if df is too small
    try:
        train_uni_raw, val_uni_raw, test_uni_raw = get_datasets(uni_pp, df.copy(), splits)
        print("\n--- Univariate Datasets Info ---")
        
        print(f"Train dataset length: {len(train_uni_raw)}")
        if len(train_uni_raw) > 0:
            print(f"Train sample 0 keys: {train_uni_raw[0].keys()}")
        
        print(f"Validation dataset length: {len(val_uni_raw)}")
        if len(val_uni_raw) > 0:
            print(f"Validation sample 0 keys: {val_uni_raw[0].keys()}")
        
        print(f"Test dataset length: {len(test_uni_raw)}")
        if len(test_uni_raw) > 0:
            print(f"Test sample 0 keys: {test_uni_raw[0].keys()}")
    
    except Exception as e:
        print(f"Error during univariate dataset creation: {e}")
        print("Ensure your DataFrame has enough data per USUBJID for context and prediction lengths after splitting.")
        raise
    
    
    # Define feature columns for multi-variate (adjust as appropriate)
    feature_cols = [ # Ensure these columns exist in df
        "SEX","PUMP","ACTARMCD","DAY_NIGHT", # Example categoricals
        "SIN_H","COS_H","SIN_DOW","COS_DOW", # Time features
        "NETIOB", "MLDOSE", "MLCAT", # Example numericals / categoricals
        # Add more actual feature columns from your CSV that are not ID/timestamp/target
         "EXCINTSY","SNKBEFEX","PLNEXDUR",
         "RESQCARB","LBORRES_ZSCORE_24","NETIOB_ZSCORE_24"
    ]
    
    # For multivariate, target_columns are what the model will learn to predict from.
    # We still only care about predicting "LBORRES".
    # The other columns in target_columns will serve as exogenous features.
    multivariate_input_features = ["LBORRES"] + feature_cols
    #required_cols = ["LBDTC", "USUBJID"] + multivariate_input_features
    #clean_df = repair_nans(df, required_cols)

    mul_pp = TimeSeriesPreprocessor(
        timestamp_column="LBDTC",
        id_columns=["USUBJID"],
        target_columns=multivariate_input_features,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        scaling=True,
        encode_categorical=True,
        scaler_type="standard"
    )
    
    clean_df = df[["LBDTC", "USUBJID"] + multivariate_input_features].dropna().copy()

    if len(clean_df) == 0:
        print("Error: DataFrame is empty after dropping NaNs for multivariate processing. Cannot proceed.")
        # Handle this case, maybe by skipping multivariate pass or erroring out
        # For now, let's assume it won't be empty for the dummy data.
        train_mul_raw, val_mul_raw, test_mul_raw = None, None, None # Or some empty datasets
    else:    
    
        try:
            train_mul_raw, val_mul_raw, test_mul_raw = get_datasets(mul_pp, clean_df, splits)
        except Exception as e:
            print(f"Error during multivariate dataset creation: {e}")
            print("Ensure your cleaned DataFrame has enough data per USUBJID.")
            train_mul_raw, val_mul_raw, test_mul_raw = train_uni_raw, val_uni_raw, test_uni_raw
            print("Falling back to univariate data for the multivariate pass.")


    user_ids = df['USUBJID'].unique()
    np.random.shuffle(user_ids)
    n = len(user_ids)
    train_ids = user_ids[:int(n * splits['train'])]
    val_ids = user_ids[int(n * splits['train']):int(n * (splits['train'] + splits['val']))]
    test_ids = user_ids[int(n * (splits['train'] + splits['val'])):]
    
    # Pass the DataFrames to your GlucoseDataset
    # Re-check after building datasets
    train_uni = GlucoseDataset(df[df.USUBJID.isin(train_ids)], CONTEXT_LENGTH, PREDICTION_LENGTH)
    val_uni = GlucoseDataset(df[df.USUBJID.isin(val_ids)], CONTEXT_LENGTH, PREDICTION_LENGTH)
    test_uni = GlucoseDataset(df[df.USUBJID.isin(test_ids)], CONTEXT_LENGTH, PREDICTION_LENGTH)

    if len(val_uni) == 0:
        print("[RETRY] Validation set is empty. Moving a user from train to validation.")
        if len(train_ids) > 0:
            # Move one user from train_ids to val_ids
            val_ids = np.append(val_ids, train_ids[-1])
            train_ids = train_ids[:-1]

            val_uni = GlucoseDataset(df[df.USUBJID.isin(val_ids)], CONTEXT_LENGTH, PREDICTION_LENGTH)
            train_uni = GlucoseDataset(df[df.USUBJID.isin(train_ids)], CONTEXT_LENGTH, PREDICTION_LENGTH)
    
    train_multi = GlucoseDataset(df[df.USUBJID.isin(train_ids)], context_length=CONTEXT_LENGTH, prediction_length=PREDICTION_LENGTH, feature_cols=feature_cols, n_time_features=4)
    val_multi = GlucoseDataset(df[df.USUBJID.isin(val_ids)], context_length=CONTEXT_LENGTH, prediction_length=PREDICTION_LENGTH, feature_cols=feature_cols, n_time_features=4)
    test_multi = GlucoseDataset(df[df.USUBJID.isin(test_ids)], context_length=CONTEXT_LENGTH, prediction_length=PREDICTION_LENGTH, feature_cols=feature_cols, n_time_features=4)


    pipeline = TTMGlucosePipeline(
        train_uni, val_uni, train_multi, val_multi,
        save_dir=SAVE_DIR_BASE,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        feature_cols=feature_cols
    )

    pipeline.run_pass1(epochs=4, batch_size=64, lr=5e-4)
    pipeline.run_pass2(epochs=3, batch_size=64, lr=5e-5)
    
    # Univariate test
    pipeline.run_test(
        test_uni,
        batch_size=64,
        use_aux=False,
        title="Univariate Test",
        save_plot_dir=f"{SAVE_DIR_BASE}/outputs{VERSION}/plots/univariate",
        save_metrics_csv=f"{SAVE_DIR_BASE}/outputs{VERSION}/metrics/uni_eval_metrics.csv",
        max_plots=10,
        per_user=True
    )
    # Multivariate test
    pipeline.run_test(
        test_multi,
        batch_size=64,
        use_aux=True,
        title="Multivariate Test",
        save_plot_dir=f"{SAVE_DIR_BASE}/outputs{VERSION}/plots/multivariate",
        save_metrics_csv=f"{SAVE_DIR_BASE}/outputs{VERSION}/metrics/multi_eval_metrics.csv",
        max_plots=10,
        per_user=True
    )


# ==== MAIN SCRIPT ====
if __name__ == "__main__":
    
    set_seed(42)
    run_finetuning_inference()

