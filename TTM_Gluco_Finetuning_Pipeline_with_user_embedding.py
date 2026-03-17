# Optimized TTM Glucose Pipeline (Full Integration)
# High-performance GPU-optimized implementation with proper memory management
# Author: Abiodun Solanke
# Version: 3.1 - Fully optimized

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import torch
import psutil
import math
import logging

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from transformers import (Trainer, TrainingArguments, default_data_collator, 
                         TrainerCallback, EarlyStoppingCallback, set_seed, AutoConfig)
from peft import LoraConfig, get_peft_model, TaskType
from tsfm_public import TimeSeriesPreprocessor, get_datasets
from tsfm_public.toolkit.get_model import get_model
from pathlib import Path
import inspect
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from optimal_finetuning_lr import OptimalLRFinder

import warnings
warnings.filterwarnings("ignore")

# Suppress all Accelerate INFO messages
logging.getLogger("accelerate").setLevel(logging.WARNING)

# GPU Configuration and Optimization
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

class OptimizedGlucoseDataset(Dataset):
    """
    GPU-optimized glucose dataset with pre-allocated tensors and efficient memory usage.
    
    Features:
    - Pre-computed user data arrays for fast access
    - Memory-efficient tensor operations
    - Automatic handling of insufficient data users
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        context_length: int,
        prediction_length: int,
        feature_cols=None,
        n_time_features: int = 4,
        user_id_to_index: dict = None
    ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.feature_cols = feature_cols or []
        self.n_time_features = n_time_features if self.feature_cols else 0
        self.user_id_to_index = user_id_to_index or {}
        
        # Pre-allocate data structures
        self.user_data = {}
        self.window_indices = []
        
        self._build_dataset(df)
        
    def _build_dataset(self, df):
        """Build dataset with optimized memory allocation."""
        grouped = df.groupby("USUBJID", sort=False)
        skipped_users = 0
        
        for user_id, user_df in grouped:
            user_df = user_df.sort_values("LBDTC")
            n = len(user_df)
            
            if n < self.context_length + self.prediction_length:
                skipped_users += 1
                continue
            
            # Pre-convert to numpy arrays for efficiency
            target = user_df["LBORRES"].values.astype(np.float32)
            features = (
                user_df[self.feature_cols].values.astype(np.float32)
                if self.feature_cols else None
            )
            
            user_index = self.user_id_to_index.get(user_id, None)
            if user_index is None:
                # skip if mapping missing (shouldn't happen if built from global df)
                skipped_users += 1
                continue
            
            self.user_data[user_id] = {"target": target, "features": features, "user_index": user_index}
            
            # Generate all valid windows for this user
            for i in range(self.context_length, n - self.prediction_length + 1):
                self.window_indices.append((user_id, i))
        
        if skipped_users > 0:
            print(f"[INFO] Skipped {skipped_users} users with insufficient data")
        
        if len(self.window_indices) == 0:
            raise ValueError("No samples generated - all users have insufficient data")
            
        self._length = len(self.window_indices) # Cache length
    
    def __len__(self) -> int:
        return self._length # Return cached length consistently
    
    def __getitem__(self, idx: int):
        user_id, i = self.window_indices[idx]
        data = self.user_data[user_id]
        target = data["target"]
        features = data["features"]
        
        # Efficient tensor creation with pre-allocated arrays
        past_values = torch.from_numpy(target[i - self.context_length : i]).float().unsqueeze(-1)
        
        future_values = torch.from_numpy(target[i : i + self.prediction_length]).float().unsqueeze(-1)
        
        if features is not None:
            past_feats = torch.from_numpy(
                features[i - self.context_length : i]).float()
            future_feats = torch.from_numpy(features[i : i + self.prediction_length]).float()
        else:
            # Pre-allocate zeros for consistent memory usage
            past_feats = torch.zeros(self.context_length, self.n_time_features, dtype=torch.float32)
            future_feats = torch.zeros(self.prediction_length, self.n_time_features, dtype=torch.float32)
        
        # Pre-allocate boolean masks
        past_mask = torch.ones(self.context_length, 1, dtype=torch.bool)
        future_mask = torch.ones(self.prediction_length, 1, dtype=torch.bool)
        user_index = torch.tensor(data["user_index"], dtype=torch.long)        
        
        return {
            "past_values": past_values,                      # [context, 1]
            "future_values": future_values,                  # [prediction, 1]
            "past_time_features": past_feats,                # [context, n_feat] or [context, 0]
            "future_time_features": future_feats,            # [prediction, n_feat] or [prediction, 0]
            "past_observed_mask": past_mask,
            "future_observed_mask": future_mask,
            "labels": future_values.clone(),                 # Trainer target
            "USUBJID": user_id,
            "user_index": user_index,
        }
    
def compute_optimized_loss(preds, targets):
    """
    Optimized loss computation with efficient tensor operations.
    
    Combines Huber loss with penalty for large errors (Clarke Error Grid inspired).
    """
    # Ensure same device with non_blocking transfer
    if preds.device != targets.device:
        targets = targets.to(preds.device, non_blocking=True)
    
    # Handle shape mismatch efficiently
    if preds.shape != targets.shape:
        if preds.shape[1] < targets.shape[1]:
            targets = targets[:, -preds.shape[1]:, :]
        else:
            raise ValueError(f"Shape mismatch: preds={preds.shape}, targets={targets.shape}")
    
    # Clinically-tuned parameters
    huber_loss = F.smooth_l1_loss(preds, targets, beta=27.0)  # Optimal for glucose range
    
    abs_err = torch.abs(preds - targets)
    large_error_mask = abs_err >= 12.0  # Clinical significance threshold
    
    if large_error_mask.any():
        clarke_penalty = abs_err[large_error_mask].mean()
    else:
        clarke_penalty = torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    
    return 0.7 * huber_loss + 0.3 * clarke_penalty  # Emphasise base loss

class OptimizedCustomTrainer(Trainer):
    """
    GPU-optimized custom trainer with efficient gradient management.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if inputs is None:
            raise ValueError("Received None inputs in compute_loss.")
        
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Missing 'labels' key in inputs.")
        
        labels = labels.to(DEVICE, non_blocking=True)
        
        # Move inputs to device efficiently
        model_inputs = {
            k: v.to(DEVICE, non_blocking=True) 
            for k, v in inputs.items() if k != "labels"
        }
        
        outputs = model(**model_inputs)
        loss = compute_optimized_loss(outputs, labels)
        
        return (loss, {"logits": outputs, "labels": labels}) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **kwargs):
        if inputs is None:
            return None, None, None
        
        labels = inputs.get("labels")
        if labels is None:
            return None, None, None
        
        labels = labels.to(DEVICE, non_blocking=True)
        model_inputs = {
            k: v.to(DEVICE, non_blocking=True) 
            for k, v in inputs.items() if k != "labels"
        }
        
        with torch.no_grad():
            predictions = model(**model_inputs)
            loss = compute_optimized_loss(predictions, labels)
        
        return (loss, predictions, labels)
    
    def evaluate(self, *args, **kwargs):
        """Ensure eval_loss is properly logged."""
        metrics = super().evaluate(*args, **kwargs)
        if "eval_loss" not in metrics and "loss" in metrics:
            metrics["eval_loss"] = metrics["loss"]
        elif "eval_loss" not in metrics:
            raise ValueError("Evaluation did not return 'eval_loss'.")
        return metrics


def inject_optimized_lora(model, target_modules=None):
    """
    Inject LoRA optimised for TTM forecasting models.
    Fixed compatibility with TinyTimeMixer architecture.
    """
    config = LoraConfig(
        r=32,                    # Increased rank for better adaptation
        lora_alpha=32,           # Strong adaptation (α/r = 2.0)
        lora_dropout=0.01,       # Reduced dropout for stability
        target_modules=target_modules or [
            # Core attention layers
            ".q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",           
            # MLP layers
            "mlp.fc1", "mlp.fc2",
            # Add layer norms for better convergence
            "layer_norm", "final_layer_norm"
        ],
        task_type=TaskType.FEATURE_EXTRACTION,  # Correct for time-series models
        inference_mode=False,
        bias="lora-only",            # Critical for glucose baseline adaptation
        
        # Remove advanced features that may cause issues with TTM
        use_rslora=False,       # Disable for compatibility
        use_dora=False,         # Disable for compatibility
        init_lora_weights="pissa",  # Better initialisation if available
        
        modules_to_save=None    # Let PEFT handle automatically
    )
    
    # Apply LoRA with error handling
    try:
        peft_model = get_peft_model(model, config)
        peft_model.print_trainable_parameters()
        return peft_model
    except AttributeError as e:
        print(f"[WARNING] LoRA injection failed: {e}")
        print("[INFO] Falling back to full fine-tuning")
        return model


def build_optimized_training_args(output_dir, lr, batch_size, epochs):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        
        lr_scheduler_type = "cosine",
        
        # Optimised for glucose forecasting
        weight_decay=0.005,          # Reduced for LoRA
        warmup_ratio=0.05,           # Minimal warmup
        
        # Evaluation strategy
        do_eval=True,
        eval_strategy="steps",       # More frequent evaluation
        eval_steps=200,              # Check every 200 steps
        save_strategy="steps",
        save_steps=400,
        logging_steps=50,
        
        # Model selection
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,          # Keep only best models
        
        # Performance optimisation
        fp16=False,                       # Disable for stability initially
        gradient_accumulation_steps=4,    # Reduced for faster updates
        max_grad_norm=0.8,                # Tighter gradient clipping
        optim="adamw_torch_fused",        # Fastest AdamW variant
        
        # Data handling
        dataloader_num_workers=6,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        
        # Reproducibility
        seed=42,
        data_seed=42,
        report_to="none"
    )


class DualHeadTTM(torch.nn.Module):
    """
    Optimized dual-head wrapper for TTM with efficient forward pass.
    """
    
    model_input_names = [
        "past_values",
        "past_time_features", 
        "future_time_features",
        "past_observed_mask",
        "future_observed_mask",
        "user_index",
    ]
    
    def __init__(self, ttm, d_model, pred_len, n_users, user_emb_dim=64):
        super().__init__()
        self.ttm = ttm
        self.pred_len = pred_len
        
        # Personalised embeddings
        self.user_emb = torch.nn.Embedding(num_embeddings=n_users, embedding_dim=user_emb_dim)
        torch.nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.02)

        # Project user embedding to model hidden
        self.user_proj = torch.nn.Linear(user_emb_dim, d_model)
        
        # Initialize heads with proper normalization
        self.uni_head = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(d_model, 1)
        )
        
        self.multi_head = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Dropout(0.1),  
            torch.nn.Linear(d_model, 1)
        )
        
        self.use_aux = False
    
    def forward(self, past_values, past_observed_mask=None,
                past_time_features=None, future_time_features=None,
                future_observed_mask=None, labels=None, user_index=None, **kwargs):
        
        ttm_out = self.ttm(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            past_time_features=past_time_features,
            future_time_features=future_time_features,
            future_observed_mask=future_observed_mask,
            future_values=labels,
            output_hidden_states=True
        )
        
        # Extract hidden states efficiently
        if hasattr(ttm_out, 'backbone_hidden_state') and ttm_out.backbone_hidden_state is not None:
            h = ttm_out.backbone_hidden_state
        elif hasattr(ttm_out, 'hidden_states') and ttm_out.hidden_states is not None:
            h = ttm_out.hidden_states[-1]
        else:
            raise ValueError("No valid hidden state found in TTM output.")
        
        # Handle dimension squeezing
        if h.dim() == 4 and h.shape[1] == 1:
            h = h.squeeze(1)
        
        # Efficient interpolation to prediction length
        if h.shape[1] != self.pred_len:
            h_interp = F.interpolate(
                h.transpose(1, 2), 
                size=self.pred_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        else:
            h_interp = h   # [B, pred_len, d_model]
        
        # Inject personalised embedding
        if user_index is None:
            raise ValueError("user_index is required for personalised embeddings.")
        # user_index: [B]
        u = self.user_emb(user_index)             # [B, user_emb_dim]
        u = self.user_proj(u)                     # [B, d_model]
        u = u.unsqueeze(1).expand(-1, h_interp.size(1), -1)  # [B, pred_len, d_model]

        # Fuse (residual)
        h_fused = h_interp + u

        # Head
        output = self.multi_head(h_fused) if self.use_aux else self.uni_head(h_fused)
        return output
    
    def set_aux(self, value: bool):
        """Switch between univariate and multivariate heads."""
        self.use_aux = value

class OptimizedLossLoggerCallback(TrainerCallback):
    """
    Efficient loss logging with reduced overhead.
    """
    
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
        """Save loss history and generate plots."""
        df_train = pd.DataFrame(self.train_losses)
        df_eval = pd.DataFrame(self.eval_losses)
        
        if df_train.empty and df_eval.empty:
            print("[LossLogger] No logs collected; skipping save.")
            return pd.DataFrame()
        
        # Merge training and evaluation losses
        df = pd.merge_asof(
            df_train.sort_values('step'),
            df_eval.sort_values('step'),
            on='step',
            direction='nearest',
            suffixes=('_train', '_eval')
        )
        
        # Coalesce epoch columns
        epoch_cols = [c for c in ['epoch', 'epoch_train', 'epoch_eval'] if c in df.columns]
        if epoch_cols:
            df['epoch_plot'] = None
            for c in epoch_cols:
                df['epoch_plot'] = df['epoch_plot'].fillna(df[c])
        else:
            df['epoch_plot'] = np.nan
        
        df.to_csv(output_path, index=False)
        
        # Generate loss plot
        if plot_path:
            self._create_loss_plot(df, plot_path)
        
        return df
    
    def _create_loss_plot(self, df, plot_path):
        """Create loss visualization plot."""
        plt.figure(figsize=(10, 4))
        
        x_key = 'epoch_plot' if df['epoch_plot'].notnull().any() else 'step'
        
        if 'loss' in df.columns:
            plt.plot(df[x_key], df['loss'], label='Training Loss', linewidth=2)
        if 'eval_loss' in df.columns:
            plt.plot(df[x_key], df['eval_loss'], label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch' if x_key == 'epoch_plot' else 'Step')
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Curves")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

class OptimizedMemoryMonitorCallback(TrainerCallback):
    """
    Efficient memory monitoring with reduced logging overhead.
    """
    
    def __init__(self, log_every_n_steps: int = 1):
        self.log_every_n_steps = log_every_n_steps
        self._last_memory = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.log_every_n_steps != 0:
            return
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            
            # Only log significant changes
            if abs(allocated - self._last_memory) > 0.1:
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                
                print(f"[Memory] Step {state.global_step}: "
                      f"Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, "
                      f"Peak={max_allocated:.2f}GB")
                
                self._last_memory = allocated
                print(f"Step {state.global_step} - GPU Memory: {allocated:.2f}GB")

def find_optimal_batch_size(model, sample_input, start_size=32, max_size=512):
    """
    Automatically find optimal batch size for current GPU configuration.
    """
    model.eval()
    optimal_size = start_size
    
    for batch_size in [32, 64, 128, 512]:
        if batch_size > max_size:
            break
        
        try:
            torch.cuda.empty_cache()
            
            # Create test batch
            test_batch = {}
            for k, v in sample_input.items():
                if isinstance(v, torch.Tensor):
                    # Expand first dimension to batch_size
                    expanded = v.unsqueeze(0).expand(batch_size, *v.shape)
                    test_batch[k] = expanded.to(DEVICE)
            
            # Test forward pass
            with torch.no_grad():
                _ = model(**test_batch)
            
            optimal_size = batch_size
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[BatchSizeFinder] OOM at batch_size={batch_size}, using {optimal_size}")
                break
            raise e
        except Exception as e:
            print(f"[BatchSizeFinder] Error at batch_size={batch_size}: {e}")
            break
    
    model.train()
    return optimal_size

class OptimizedTTMGlucosePipeline:
    """
    Complete optimized pipeline for TTM glucose forecasting.
    
    Features:
    - Dual-head training (univariate -> multivariate)
    - Automatic batch size optimization
    - Memory-efficient training with gradient checkpointing
    - Comprehensive evaluation and visualization
    """
    
    def __init__(self, train_uni, val_uni, train_multi, val_multi, save_dir, context_length, 
                 prediction_length, batch_size, feature_cols=None, n_users=None):
        
        self.pred_len = prediction_length
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Store datasets
        self.train_uni = train_uni
        self.val_uni = val_uni
        self.train_multi = train_multi
        self.val_multi = val_multi
        self.batch_size = batch_size
        
        self.n_users = n_users
        
        # Initialize model
        self._setup_model(context_length, prediction_length)
        
        # Setup data collator
        self.data_collator = default_data_collator
        
    def _setup_model(self, context_length, prediction_length):
        """Initialize and configure the TTM model."""
        print("[INFO] Setting up TTM model...")
        
        # Load base model
        base_model = get_model(
            "ibm-granite/granite-timeseries-ttm-r2",
            context_length=context_length,
            prediction_length=prediction_length,
        )
        
        print(f"TTM config - context_length: {base_model.config.context_length}")
        print(f"TTM config - prediction_length: {base_model.config.prediction_length}")
        
        # Apply LoRA
        peft_model = inject_optimized_lora(base_model)
        
        # Setup safe forward pass
        base_model_actual = peft_model.base_model.model
        if not hasattr(base_model_actual, '__original_forward__'):
            base_model_actual.__original_forward__ = base_model_actual.forward
            base_model_actual.forward = self._create_safe_forward(base_model_actual)
        
        # Create dual head model
        d_model = peft_model.config.d_model
        self.model = DualHeadTTM(peft_model, d_model, prediction_length, n_users=self.n_users, user_emb_dim=32)
        self.model.to(DEVICE)
        self.model.model_input_names = DualHeadTTM.model_input_names
        
        print(f"[INFO] Model setup complete. Device: {next(self.model.parameters()).device}")
    
    def _create_safe_forward(self, model):
        """Create a safe forward pass that filters kwargs."""
        def super_safe_forward(*args, **kwargs):
            allowed = set(inspect.signature(model.__original_forward__).parameters.keys())
            safe_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            return model.__original_forward__(*args, **safe_kwargs)
        
        return super_safe_forward
    
    def _run_training_pass(self, use_aux, train_set, val_set, epochs, lr, outdir, auto_lr_find=False):
        """
        Execute a single training pass with optimized configuration.
        """
        print(f"\n[INFO] Starting {'multivariate' if use_aux else 'univariate'} training pass")
        print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        try:
            # Set model mode
            self.model.set_aux(use_aux)
            
            # Validate datasets
            if len(train_set) == 0 or len(val_set) == 0:
                raise ValueError(f"Empty dataset: train={len(train_set)}, val={len(val_set)}")
                
            # Automatic learning rate finding
            if auto_lr_find:
                print("[INFO] Running automatic learning rate finder...")

                # Create temporary data loader for LR finding
                temp_loader = DataLoader(
                    train_set, 
                    batch_size=min(32, len(train_set)), 
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True
                )
                                
                lr_finder = OptimalLRFinder()

                # Run LR finder
                lr_results = lr_finder.find_optimal_learning_rate(
                    model=self.model,
                    train_loader=temp_loader,
                    device=DEVICE,
                    min_lr=1e-6,
                    max_lr=1e-2,  # Conservative upper bound for stability
                    num_iter=100,
                    stop_factor=4.0,
                    mode="exponential"
                )

                # Use suggested learning rate
                lr = lr_results['suggested_lr']

                # Save LR finder plot
                lr_plot_path = self.save_dir / outdir / "lr_finder_results.png"
                lr_plot_path.parent.mkdir(parents=True, exist_ok=True)
                lr_finder.plot_lr_finder_results(lr_results, str(lr_plot_path))

                print(f"[INFO] Auto-detected optimal learning rate: {lr:.2e}\n")
            else:
                lr = lr
                print(f"[INFO] Using passed learning rate: {lr:.2e}")

            # Optimize batch size if not provided
            if self.batch_size == "auto":
                sample_input = train_set[0]
                self.batch_size = find_optimal_batch_size(self.model, sample_input)
                print(f"[INFO] Auto-detected optimal batch size: {self.batch_size}")
            
            # Setup training arguments
            output_path = str(self.save_dir / outdir)
            args = build_optimized_training_args(output_path, lr, self.batch_size, epochs)

            # CORRECTED scheduler parameters calculation
            steps_per_epoch = math.ceil(len(train_set) / self.batch_size)  # Use ceil for last batch
            optimizer_steps_per_epoch = math.ceil(steps_per_epoch / args.gradient_accumulation_steps)
            total_optimizer_steps = optimizer_steps_per_epoch * epochs

            # Add safety padding to avoid off-by-one errors
            padded_total_steps = total_optimizer_steps + 2

            # Setup optimizer and scheduler
            optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=args.weight_decay, eps=1e-6)

            scheduler = OneCycleLR(
                optimizer,
                max_lr=lr,  # Use lr directly instead of optimizer.param_groups[0]['lr']
                total_steps=padded_total_steps,  # Use padded calculation
                pct_start=0.08,
                anneal_strategy="cos",
                div_factor=20.0,
                final_div_factor=1e3,
                cycle_momentum=False
            )
            
            # Setup callbacks
            loss_logger = OptimizedLossLoggerCallback()
            callbacks = [
                loss_logger,
                OptimizedMemoryMonitorCallback(log_every_n_steps=1),
                EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=1e-5),
            ]
            
            # Create trainer
            trainer = OptimizedCustomTrainer(
                model=self.model,
                args=args,
                train_dataset=train_set,
                eval_dataset=val_set,
                data_collator=self.data_collator,
                optimizers=(optimizer, scheduler),
                callbacks=callbacks
            )
            
            # Train model
            print(f"[INFO] Starting training with {total_optimizer_steps} total steps...")
            trainer.train()
            
            # Save training metrics
            csv_path = self.save_dir / outdir / "loss_history.csv"
            plot_path = self.save_dir / outdir / "loss_plot.png"
            loss_logger.save(csv_path, plot_path)
            
            print(f"[INFO] Training complete. Results saved to {output_path}")
            
        finally:
            # Cleanup
            if 'trainer' in locals():
                del trainer
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    def run_pass1(self, epochs, lr):
        """Execute univariate training pass."""
        self._run_training_pass(use_aux=False, train_set=self.train_uni, val_set=self.val_uni, epochs=epochs, lr=lr,
                                outdir="pass1_uni")
    
    def run_pass2(self, epochs, lr):
        """Execute multivariate training pass.""" 
        self._run_training_pass(use_aux=True, train_set=self.train_multi, val_set=self.val_multi, epochs=epochs, lr=lr,
                                outdir="pass2_multi")
    
    def run_comprehensive_evaluation(self, test_set, use_aux=False, title="Test Evaluation",
                                     save_plot_dir=None, save_metrics_csv=None, max_plots=10, per_user=True):
        """
        Comprehensive model evaluation with visualization and metrics.
        """
        print(f"\n[INFO] Running {title.lower()}...")
        
        # Set model mode and evaluation
        self.model.set_aux(use_aux)
        self.model.eval()
        
        # Storage for results
        all_predictions = []
        all_targets = []
        all_user_ids = []
        
        # Create DataLoader
        test_loader = DataLoader(
            test_set, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Run inference
        with torch.no_grad():
            for batch in test_loader:
                # Prepare inputs
                inputs = {
                    k: v.to(DEVICE, non_blocking=True) 
                    for k, v in batch.items() 
                    if k in self.model.model_input_names
                }
                labels = batch["labels"].to(DEVICE, non_blocking=True)
                
                # Forward pass
                outputs = self.model(**inputs)
                
                # Handle shape alignment
                if outputs.shape[1] < labels.shape[1]:
                    labels = labels[:, -outputs.shape[1]:, :]
                
                # Store results
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
                
                # Store user IDs if available
                if 'USUBJID' in batch:
                    if isinstance(batch['USUBJID'][0], torch.Tensor):
                        all_user_ids.extend(batch['USUBJID'])
                    else:
                        all_user_ids.extend(batch['USUBJID'])
        
        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Flatten for metrics calculation
        y_pred = predictions.reshape(-1)
        y_true = targets.reshape(-1)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, title)
        
        # Save metrics
        if save_metrics_csv:
            self._save_metrics(metrics, save_metrics_csv)
        
        # Generate plots
        if save_plot_dir:
            self._generate_evaluation_plots(
                predictions, targets, all_user_ids, title, 
                save_plot_dir, max_plots, per_user
            )
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred, title):
        """Calculate comprehensive evaluation metrics."""
        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
        
        # Additional metrics
        r2 = float(1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)))
        
        print(f"\n{title} Metrics:")
        print(f"  MSE  : {mse:.4f}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  MAE  : {mae:.4f}")
        print(f"  MAPE : {mape:.2f}%")
        print(f"  R²   : {r2:.4f}")
        
        return {
            "title": title,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2
        }
    
    def _save_metrics(self, metrics, save_path):
        """Save metrics to CSV file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        metrics_df = pd.DataFrame([metrics])
        
        if save_path.exists():
            metrics_df.to_csv(save_path, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(save_path, index=False)
    
    def _generate_evaluation_plots(self, predictions, targets, user_ids, title, 
                                 save_dir, max_plots, per_user):
        """Generate comprehensive evaluation plots."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if per_user and len(user_ids) > 0:
            self._generate_per_user_plots(predictions, targets, user_ids, title, save_dir, max_plots)
        else:
            self._generate_sample_plots(predictions, targets, title, save_dir, max_plots)
    
    def _generate_per_user_plots(self, predictions, targets, user_ids, title, save_dir, max_plots):
        """Generate per-user evaluation plots."""
        # Handle user IDs
        if isinstance(user_ids[0], torch.Tensor):
            user_ids_array = torch.stack(user_ids).cpu().numpy().ravel()
        else:
            user_ids_array = np.array(user_ids).ravel()
        
        unique_ids = np.unique(user_ids_array)[:max_plots]
        
        for uid in unique_ids:
            indices = np.where(user_ids_array == uid)[0]
            if len(indices) == 0:
                continue
            
            # Use first occurrence of this user
            idx = indices[0]
            
            plt.figure(figsize=(10, 4))
            true_vals = targets[idx].squeeze()
            pred_vals = predictions[idx].squeeze()
            
            plt.plot(true_vals, label="Ground Truth", linewidth=2, alpha=0.8)
            plt.plot(pred_vals, label="Predicted", linewidth=2, alpha=0.8)
            
            plt.title(f"{title} - User {uid}", fontsize=14, fontweight='bold')
            plt.xlabel("Time Steps")
            plt.ylabel("Glucose Level")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = save_dir / f"{title.lower().replace(' ', '_')}_user_{uid}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_sample_plots(self, predictions, targets, title, save_dir, max_plots):
        """Generate sample-based evaluation plots."""
        for i in range(min(max_plots, predictions.shape[0])):
            plt.figure(figsize=(10, 4))
            
            true_vals = targets[i].squeeze()
            pred_vals = predictions[i].squeeze()
            
            plt.plot(true_vals, label="Ground Truth", linewidth=2, alpha=0.8)
            plt.plot(pred_vals, label="Predicted", linewidth=2, alpha=0.8)
            
            plt.title(f"{title} - Sample {i+1}", fontsize=14, fontweight='bold')
            plt.xlabel("Time Steps")
            plt.ylabel("Glucose Level")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = save_dir / f"{title.lower().replace(' ', '_')}_sample_{i+1}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

def prepare_data_splits(df, splits={"train": 0.2, "val": 0.5, "test": 0.3}):
    """
    Prepare data splits ensuring no user appears in multiple splits.
    """
    user_ids = df['USUBJID'].unique()
    np.random.shuffle(user_ids)
    
    n = len(user_ids)
    train_end = int(n * splits['train'])
    val_end = int(n * (splits['train'] + splits['val']))
    
    train_ids = user_ids[:train_end]
    val_ids = user_ids[train_end:val_end]
    test_ids = user_ids[val_end:]
    
    print(f"[INFO] Data split - Train: {len(train_ids)} users, "
          f"Val: {len(val_ids)} users, Test: {len(test_ids)} users")
    
    return train_ids, val_ids, test_ids

# NaN Repair for Multivariate Data
def repair_nans(df, feature_cols, target_col="LBORRES"):
    """
    Advanced NaN handling preserving temporal relationships.
    """
    df = df.sort_values(["USUBJID", "LBDTC"])
    
    for user_id in df['USUBJID'].unique():
        user_mask = df['USUBJID'] == user_id
        user_data = df[user_mask].copy()
        
        # Forward-fill then backward-fill for temporal consistency
        user_data[feature_cols] = user_data[feature_cols].fillna(method='ffill').fillna(method='bfill')
        
        # For remaining NaNs, use median imputation per user
        for col in feature_cols:
            if user_data[col].isna().any():
                user_median = user_data[col].median()
                if pd.notna(user_median):
                    user_data[col].fillna(user_median, inplace=True)
                else:
                    # Fallback to global median
                    global_median = df[col].median()
                    user_data[col].fillna(global_median, inplace=True)
        
        df.loc[user_mask, feature_cols] = user_data[feature_cols]
    
    # Only drop rows where target is NaN
    df = df.dropna(subset=[target_col])
    return df


def run_optimized_pipeline():
    """
    Main function to run the complete optimized pipeline.
    """
    print("="*80)
    print("OPTIMIZED TTM GLUCOSE FORECASTING PIPELINE")
    print("="*80)
    
    # Configuration
    DATA_FILE = "glucose_forecast_data.csv"
    CONTEXT_LENGTH = 512
    PREDICTION_LENGTH = 96
    VERSION = "_v3.2."
    SAVE_DIR_BASE = f"ttm_finetuned_models/glucose_pipeline{VERSION}"
    # Pass 1: Univariate training (foundation learning)
    UNIVARIATE_LR = 5e-2  # Conservative for stable foundation

    # Pass 2: Multivariate training (refinement)
    MULTIVARIATE_LR = 5e-3 # Lower for fine-tuning with additional features

    
    # Load and clean data
    print(f"[INFO] Loading data from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    
    # Remove isolated users (from your original list)
    isolated_users = [
        911, 930, 943, 976, 334, 484, 491, 549, 558, 1490,
        404, 422, 423, 1425, 1441, 1456, 80, 812, 834, 60,
        227, 248, 25, 254, 1500, 1501, 1520, 1567, 733, 1271,
        288, 29, 290, 312, 317, 48, 756, 761, 767, 95
    ]
    
    df = df[~df.USUBJID.isin(isolated_users)].copy()
    df["LBDTC"] = pd.to_datetime(df["LBDTC"])
    df = df.sort_values(by=["USUBJID", "LBDTC"])
    
    print(f"[INFO] Data loaded - {len(df)} records, {df['USUBJID'].nunique()} unique users")
    
    # Define feature columns for multivariate training
    feature_cols = [# Core temporal features
                    "SIN_H", "COS_H", "SIN_DOW", "COS_DOW",

                     # Physiological features
                     "NETIOB", "MLDOSE", "MLCAT", "RESQCARB",

                     # Historical context (most important)
                     "LBORRES_ZSCORE_24", "NETIOB_ZSCORE_24",

                     # Essential demographic/device
                     "SEX", "PUMP", "DAY_NIGHT",

                     # Remove potentially noisy features
                     #"ACTARMCD", "EXCINTSY", "SNKBEFEX", "PLNEXDUR"  # Consider removing
                    ]
    # Prepare data splits
    train_ids, val_ids, test_ids = prepare_data_splits(df)
    
    # Build stable USUBJID -> index mapping
    unique_users = np.sort(df['USUBJID'].unique())
    user_id_to_index = {uid: idx for idx, uid in enumerate(unique_users)}
    n_users_total = len(unique_users)
    print(f"[INFO] Personalisation: {n_users_total} users for embeddings")
        
    # Create datasets
    print("[INFO] Creating datasets...")
    train_uni = OptimizedGlucoseDataset(df[df.USUBJID.isin(train_ids)], CONTEXT_LENGTH, PREDICTION_LENGTH,
                                        feature_cols=None, n_time_features=0, user_id_to_index=user_id_to_index)
    val_uni = OptimizedGlucoseDataset(df[df.USUBJID.isin(val_ids)], CONTEXT_LENGTH, PREDICTION_LENGTH, 
                                         feature_cols=None, n_time_features=0, user_id_to_index=user_id_to_index)
    test_uni = OptimizedGlucoseDataset(df[df.USUBJID.isin(test_ids)], CONTEXT_LENGTH, PREDICTION_LENGTH,
                                       feature_cols=None, n_time_features=0, user_id_to_index=user_id_to_index)

    # Handle empty validation set
    if len(val_uni) == 0:
        print("[WARNING] Validation set empty, moving one user from training")
        if len(train_ids) > 0:
            val_ids = np.append(val_ids, train_ids[-1])
            train_ids = train_ids[:-1]
            val_uni = OptimizedGlucoseDataset(
                df[df.USUBJID.isin(val_ids)], CONTEXT_LENGTH, PREDICTION_LENGTH
            )
            train_uni = OptimizedGlucoseDataset(
                df[df.USUBJID.isin(train_ids)], CONTEXT_LENGTH, PREDICTION_LENGTH
            )
    
    # For multivariate, target_columns are what the model will learn to predict from.
    # We still only care about predicting "LBORRES".
    # The other columns in target_columns will serve as exogenous features.
    multivariate_input_features = ["LBORRES"] + feature_cols
    required_cols = ["LBDTC", "USUBJID"] + multivariate_input_features
    multi_df = repair_nans(df, required_cols)
    
    #multi_df = df[["LBDTC", "USUBJID"] + multivariate_input_features].dropna().copy()
    
    # Create multivariate datasets
    train_multi = OptimizedGlucoseDataset(multi_df[multi_df.USUBJID.isin(train_ids)], CONTEXT_LENGTH, PREDICTION_LENGTH,
                                          feature_cols=feature_cols, n_time_features=4, user_id_to_index=user_id_to_index)
    val_multi = OptimizedGlucoseDataset(multi_df[multi_df.USUBJID.isin(val_ids)], CONTEXT_LENGTH, PREDICTION_LENGTH,
                                        feature_cols=feature_cols, n_time_features=4, user_id_to_index=user_id_to_index)
    test_multi = OptimizedGlucoseDataset(multi_df[multi_df.USUBJID.isin(test_ids)], CONTEXT_LENGTH, PREDICTION_LENGTH,
                                         feature_cols=feature_cols, n_time_features=4, user_id_to_index=user_id_to_index)    
    print(f"[INFO] Dataset sizes:")
    print(f"  Train: {len(train_uni)} samples")
    print(f"  Val:   {len(val_uni)} samples")
    print(f"  Test:  {len(test_uni)} samples")
    
    # Initialize pipeline
    print("[INFO] Initializing pipeline...")
    pipeline = OptimizedTTMGlucosePipeline(
        train_uni, val_uni, train_multi, val_multi,
        save_dir=SAVE_DIR_BASE,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        batch_size="auto",
        feature_cols=feature_cols,
        n_users=n_users_total
    )
    
    # Training Phase
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)
    
    # Pass 1: Univariate training
    pipeline.run_pass1(epochs=3, lr=UNIVARIATE_LR)
    
    # Pass 2: Multivariate training  
    pipeline.run_pass2(epochs=2, lr=MULTIVARIATE_LR)
    
    # Evaluation Phase
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)
    
    # Univariate evaluation
    pipeline.run_comprehensive_evaluation(
        test_uni,
        use_aux=False,
        title="Univariate Test",
        save_plot_dir=f"{SAVE_DIR_BASE}/outputs{VERSION}/plots/univariate",
        save_metrics_csv=f"{SAVE_DIR_BASE}/outputs{VERSION}/metrics/evaluation_results.csv",
        max_plots=10,
        per_user=True
    )
    
    # Multivariate evaluation
    pipeline.run_comprehensive_evaluation(
        test_multi,
        use_aux=True,
        title="Multivariate Test",
        save_plot_dir=f"{SAVE_DIR_BASE}/outputs{VERSION}/plots/multivariate",
        save_metrics_csv=f"{SAVE_DIR_BASE}/outputs{VERSION}/metrics/evaluation_results.csv",
        max_plots=10,
        per_user=True
    )
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print(f"Results saved to: {SAVE_DIR_BASE}")
    print("="*80)

if __name__ == "__main__":
    # Set random seeds for reproducibility
    set_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the optimized pipeline
    run_optimized_pipeline()
