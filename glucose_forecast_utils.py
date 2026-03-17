import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from datetime import datetime, timedelta, timezone
from dateutil import parser
import json
import numpy as np

import sys

import math
import os
from pathlib import Path
import re
import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed, AutoConfig

import tempfile
from tsfm_public import(
    TimeSeriesForecastingPipeline,
    TimeSeriesPreprocessor,
    TinyTimeMixerForPrediction,
    TrackingCallback,
    count_parameters,
    get_datasets,
)
from tsfm_public.toolkit.time_series_preprocessor import prepare_data_splits
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions

from peft import LoraConfig, get_peft_model, PeftModel, set_peft_model_state_dict


import warnings
warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:.2f}'.format
class TTM_glucose_forecaster:
    def __init__(self, data, subjid, timestamp_column, target_columns, task, context_length, prediction_length, batch_size, scaler, fewshot_percent, epochs, sample_type, seed=42, evaluate_after_init=False):
        self.data = data
        self.subjid = subjid
        self.context_length = context_length
        self.prediction_length = prediction_length
        set_seed(seed)
        self.seed = seed
        self.batch_size = batch_size
        self.scaler = scaler
        self.fewshot_percent = fewshot_percent
        self.epochs = epochs
        self.sample_type = sample_type
        self.timestamp_column = timestamp_column
        self.target_columns = target_columns
        self.task = task

        self.TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
        self.TARGET_DATASET = f"{self.task}_data"
        self.OUT_DIR = "ttm_finetuned_models/"
        self.TTM_MODEL_REVISION = "main"

        self.split_config = {"train": 0.4, "val": 0.1, "test": 0.5}

        #self.train_df, self.valid_df, self.test_df = prepare_data_splits(
            #self.data,
            #context_length=self.context_length,
            #split_config=self.split_config
        #)

        self.column_specifiers = {
            "timestamp_column": self.timestamp_column,
            "id_columns": [],
            "target_columns": self.target_columns,
            "control_columns": [],
            "conditional_columns": []
        }

        if self.sample_type == 'zeroshot':
            self.zeroshot_eval()
        elif self.sample_type == 'fewshot':
            self.fewshot_finetune_eval()
        else:
            self.load_finetuned_ttm_model()
        
        if evaluate_after_init:
            self.evaluate_plot()

    def zeroshot_eval(self):
        tsp = TimeSeriesPreprocessor(
            **self.column_specifiers,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            scaling=False,
            encode_categorical=False,
            scaler_type=self.scaler,
        )
        self.dset_train, self.dset_valid, self.dset_test = get_datasets(tsp, self.data, self.split_config)

        print(f"[Zeroshot] Train: {len(self.dset_train)}, Valid: {len(self.dset_valid)}, Test: {len(self.dset_test)}")

        zeroshot_model = get_model(
            self.TTM_MODEL_PATH,
            context_length=self.context_length,
            prediction_length=self.prediction_length
        )

        self.forecast_trainer = Trainer(
            model=zeroshot_model,
            args=TrainingArguments(
                output_dir=tempfile.mkdtemp(),
                per_device_eval_batch_size=self.batch_size,
                seed=self.seed,
                report_to="none"
            ),
        )

        print(f"{'-'*20} Zero-shot Evaluation Initialised {'-'*20}")

    def fewshot_finetune_eval(self, learning_rate=None, freeze_backbone=True, loss="mse", quantile=0.5, use_lora=False, lora_config_params=None, adapter_name="default_adapter"):
        
        out_dir = os.path.join(self.OUT_DIR, self.TARGET_DATASET, adapter_name)
        print(f"{'-'*20} Few-shot Finetuning ({self.fewshot_percent}%) [{adapter_name}] {'-'*20}")

        tsp = TimeSeriesPreprocessor(
            **self.column_specifiers,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            scaling=False,
            encode_categorical=False,
            scaler_type=self.scaler,
        )

        self.dset_train, self.dset_val, self.dset_test = get_datasets(
            tsp,
            self.data,
            self.split_config,
            fewshot_fraction=self.fewshot_percent / 100,
            fewshot_location="last",
            seed=self.seed
        )

        print(f"[Fewshot] Train: {len(self.dset_train)}, Valid: {len(self.dset_val)}, Test: {len(self.dset_test)}")

        model = get_model(
            self.TTM_MODEL_PATH,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            num_input_channels=tsp.num_input_channels,
            decoder_mode="mix_channel",
            prediction_channel_indices=tsp.prediction_channel_indices,
            loss=loss,
            head_dropout=0.4,
        )

        if use_lora:
            if lora_config_params is None:
                lora_config_params = {
                    "r": 4,
                    "lora_alpha": 8,
                    "lora_dropout": 0.1,
                    "target_modules": ["q_proj", "v_proj"]
                }

            lora_config = LoraConfig(
                task_type="FEATURE_EXTRACTION",
                inference_mode=False,
                **lora_config_params
            )
            model = get_peft_model(model, lora_config)
            print("LoRA adapters added to TinyTimeMixer. Trainable params:")
            model.print_trainable_parameters()

        if freeze_backbone and hasattr(model, "backbone"):
            for param in model.backbone.parameters():
                param.requires_grad = False
            print(f"Model Parameters after freezing: {count_parameters(model)}")

        if learning_rate is None:
            learning_rate, model = optimal_lr_finder(model, self.dset_train, batch_size=self.batch_size)
            print(f"Optimal Learning Rate Found: {learning_rate:.6f}")

        training_args = TrainingArguments(
            output_dir=os.path.join(out_dir, "output"),
            overwrite_output_dir=True,
            learning_rate=learning_rate,
            num_train_epochs=self.epochs,
            do_eval=True,
            evaluation_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            dataloader_num_workers=4,
            report_to="none",
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,
            logging_dir=os.path.join(out_dir, "logs"),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=self.seed,
        )

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        scheduler = OneCycleLR(
            optimizer,
            learning_rate,
            epochs=self.epochs,
            steps_per_epoch=math.ceil(len(self.dset_train) / self.batch_size),
        )

        self.forecast_trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.dset_train,
            eval_dataset=self.dset_val,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=1e-5),
                TrackingCallback()
            ],
            optimizers=(optimizer, scheduler),
        )

        self.forecast_trainer.train()
        model_save_path = os.path.join(out_dir, "saved_model")
        self.forecast_trainer.model.save_pretrained(model_save_path)
        with open(os.path.join(model_save_path, "training_args.json"), "w") as f:
            json.dump(self.forecast_trainer.args.to_dict(), f, indent=2)

        print(f"{'-'*20} Finetuning Complete [{adapter_name}] {'-'*20}")
        print(f"[Saved] Weights and training config saved to {model_save_path}")
    
    def load_finetuned_ttm_model(self, adapter_name="default_adapter", use_lora=False):
        
        model_path = os.path.join(self.OUT_DIR, self.TARGET_DATASET, adapter_name, "saved_model")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist. Ensure fine-tuned model is saved.")

        base_model = get_model(
            self.TTM_MODEL_PATH,
            context_length=self.context_length,
            prediction_length=self.prediction_length
        )

        if use_lora:
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = TinyTimeMixerForPrediction.from_pretrained(model_path)

        model.config.context_length = self.context_length
        model.config.prediction_length = self.prediction_length

        tsp = TimeSeriesPreprocessor(
            **self.column_specifiers,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            scaling=False,
            encode_categorical=False,
            scaler_type="standard",
        )

        self.dset_train, self.dset_val, self.dset_test = get_datasets(tsp, self.data, {"train": 0.4, "val": 0.1, "test": 0.5})
        print(f"[LoadModel] Test Samples: {len(self.dset_test)}")

        self.forecast_trainer = Trainer(model=model, args=TrainingArguments(output_dir="./tmp", report_to="none"))
       

    def evaluate_plot(self):

        print("Evaluating on test set...")
        eval_output = self.forecast_trainer.evaluate(self.dset_test)
        print("Trainer Evaluation Output:", eval_output)

        predictions_dict = self.forecast_trainer.predict(self.dset_test)
        
        y_pred = predictions_dict.predictions[0] if isinstance(predictions_dict.predictions, tuple) else predictions_dict.predictions
        
        y_pred_target = y_pred[:, :, 0]  # assuming LBORRES is at index 0

        # --- Extract LBORRES ground truth from dataset
        y_true_batches = []
        for i in range(len(self.dset_test)):
            item = self.dset_test[i]

            if isinstance(item, dict):
                # Check for known target keys
                if "future_values" in item:
                    target_tensor = item["future_values"]
                elif "targets" in item:
                    target_tensor = item["targets"]
                else:
                    continue

                # Convert tensor to numpy
                if isinstance(target_tensor, torch.Tensor):
                    target_tensor = target_tensor.cpu().numpy()

                # Only select LBORRES (channel 0)
                y_true_batches.append(target_tensor[:, 0])

        if not y_true_batches:
            print(f"Could not extract {self.task} ground truth from test dataset.")
            return

        y_true_target = np.stack(y_true_batches)

        # Align sequence lengths
        min_len = min(y_pred_target.shape[1], y_true_target.shape[1])
        y_pred_flat = y_pred_target[:, :min_len].flatten()
        y_true_flat = y_true_target[:, :min_len].flatten()

        # --- Compute Evaluation Metrics
        mse = np.mean((y_true_flat - y_pred_flat) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true_flat - y_pred_flat))
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100

        print(f"\n--- Evaluation Metrics for {self.task} ---")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"MAE  : {mae:.4f}")
        print(f"MAPE : {mape:.2f}%")
        print("----------------------------------------\n")

        # --- Plot predictions (visual inspection)
        plot_predictions(
            model=self.forecast_trainer.model,
            dset=self.dset_test,
            plot_dir=os.path.join(self.OUT_DIR, self.TARGET_DATASET),
            plot_prefix=self.sample_type,
            timestamp_column=self.timestamp_column,
            channel=0
        )
        
if __name__ =='__main__':
    
    df = pd.read_csv('glucose_forecast_data.csv')
    isolated_users = [911,  930,  943,  976, 334, 484,  491,  549,  558, 1490,
                      404,  422,  423, 1425, 1441, 1456, 80,  812,  834, 60,
                      227,  248,   25,  254, 1500, 1501, 1520, 1567, 733, 1271,
                      288,   29,  290,  312,  317, 48,  756,  761,  767, 95]

    training_data = df[~df['USUBJID'].isin(isolated_users)]
    
    training_data["LBDTC"] = pd.to_datetime(training_data["LBDTC"])
    training_data = training_data.sort_values("LBDTC")

    #training_data = training_data.set_index('LBDTC')[['LBORRES', 'NETIOB', 'ACTARMCD', 'DAY_NIGHT', 'RESQCARB']].resample("5min").mean().interpolate().reset_index()

    timestamp_column = 'LBDTC'
    target_columns = ['LBORRES', 'NETIOB', 'ACTARMCD', 'DAY_NIGHT', 'RESQCARB']

    task = 'glucose'

    TTM_glucose_forecaster(training_data, subjid=None, timestamp_column=timestamp_column, target_columns=target_columns,
                            task=task, context_length=1536, prediction_length=144, batch_size=64, scaler='standard', 
                            fewshot_percent=100, epochs=50, sample_type='fewshot', evaluate_after_init=True)