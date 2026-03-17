import os
import numpy as np
import pandas as pd
import torch
import psutil
import math
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt

class OptimalLRFinder:
    """
    Learning Rate Range Test implementation based on Leslie N. Smith's paper.
    
    Automatically finds optimal learning rates by gradually increasing LR
    and monitoring loss behavior.
    """
    
    def __init__(self):
        print("[LRFinder] Initialised")
        
    
    def find_optimal_learning_rate(
        self,
        model,
        train_loader,
        device,
        min_lr=1e-7,
        max_lr=1.0,
        num_iter=100,
        stop_factor=4.0,
        mode="exponential"
    ):
        """
        Automatically find optimal learning rate using Learning Rate Range Test.
        
        Based on Leslie N. Smith's paper "Cyclical Learning Rates for Training Neural Networks"
        and the fastai implementation.
        
        Args:
            model: PyTorch model to test
            train_loader: DataLoader with training data
            device: Computing device
            min_lr: Minimum learning rate to test
            max_lr: Maximum learning rate to test
            num_iter: Number of iterations to test over
            stop_factor: Stop if loss > stop_factor * best_loss
            mode: 'exponential' or 'linear' increase strategy
            
        Returns:
            dict: {
                'suggested_lr': float,  # Recommended learning rate
                'lrs': list,           # All tested learning rates
                'losses': list,        # Corresponding losses
                'best_loss': float     # Minimum loss found
            }
        """
        print(f"[LRFinder] Starting learning rate range test...")
        print(f"[LRFinder] Range: {min_lr:.2e} to {max_lr:.2e} over {num_iter} iterations")
        
        # Store original state
        self.model = model
        self.model.train()
        original_state = model.state_dict()
        
        # Create temporary optimizer for LR test
        temp_optimizer = torch.optim.AdamW(self.model.parameters(), lr=min_lr)
        
        # Setup learning rate schedule
        if mode == "exponential":
            lr_lambda = lambda iteration: min_lr * (max_lr / min_lr) ** (iteration / (num_iter - 1))
        else:  # linear
            lr_lambda = lambda iteration: min_lr + (max_lr - min_lr) * (iteration / (num_iter - 1))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(temp_optimizer, lr_lambda)
        
        # Storage for results
        lrs = []
        losses = []
        best_loss = float('inf')
        
        try:
            # Get iterator for training data
            train_iter = iter(train_loader)
            
            for iteration in range(num_iter):
                try:
                    # Get next batch
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        # Reset iterator if we run out of data
                        train_iter = iter(train_loader)
                        batch = next(train_iter)
                    
                    # Prepare inputs - handle different batch formats
                    if isinstance(batch, dict):
                        # Handle model inputs efficiently
                        inputs = {
                            k: v.to(device, non_blocking=True)
                            for k, v in batch.items()
                            if k in self.model.model_input_names
                        }
                        
                        labels = batch.get("labels")
                        if labels is not None:
                            labels = labels.to(device, non_blocking=True)
                    else:
                        # Handle tuple/list format
                        inputs, labels = batch
                        inputs = inputs.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                    
                    # Forward pass
                    temp_optimizer.zero_grad()
                    
                    if isinstance(inputs, dict):
                        outputs = self.model(**inputs)
                    else:
                        outputs = self.model(inputs)
                    
                    # Compute loss using the same loss function as training
                    # Import the loss function from the main module
                    from TTM_Gluco_Finetuning_Pipeline_Optimised import compute_optimized_loss
                    
                    loss = compute_optimized_loss(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    temp_optimizer.step()
                    scheduler.step()
                    
                    # Record results
                    current_lr = scheduler.get_last_lr()[0]
                    current_loss = loss.item()
                    
                    lrs.append(current_lr)
                    losses.append(current_loss)
                    
                    # Track best loss
                    if current_loss < best_loss:
                        best_loss = current_loss
                    
                    # Early stopping if loss explodes
                    if current_loss > stop_factor * best_loss and iteration > 10:
                        print(f"[LRFinder] Early stopping at iteration {iteration} (loss explosion)")
                        break
                    
                    # Progress update
                    if iteration % (num_iter // 10) == 0:
                        print(f"[LRFinder] Progress: {iteration}/{num_iter}, LR: {current_lr:.2e}, Loss: {current_loss:.4f}")
                        
                except Exception as e:
                    print(f"[LRFinder] Error at iteration {iteration}: {e}")
                    break
        
        finally:
            # Restore original model state
            self.model.load_state_dict(original_state)
            torch.cuda.empty_cache()
        
        # Analyze results and suggest optimal learning rate
        if len(losses) < 10:
            print("[LRFinder] Warning: Too few samples collected for reliable analysis")
            suggested_lr = min_lr
        else:
            suggested_lr = self._analyze_lr_results(lrs, losses)
        
        print(f"[LRFinder] Suggested learning rate: {suggested_lr:.2e}")
        
        return {'suggested_lr': suggested_lr, 'lrs': lrs, 'losses': losses, 'best_loss': best_loss}
    
    def _analyze_lr_results(self, lrs, losses):
        """
        Analyze LR finder results to suggest optimal learning rate.
        
        Uses the steepest gradient method recommended by Leslie Smith.
        """
        # Smooth losses to reduce noise
        if len(losses) > 10:
            # Simple moving average
            window = min(5, len(losses) // 10)
            smoothed_losses = []
            for i in range(len(losses)):
                start_idx = max(0, i - window // 2)
                end_idx = min(len(losses), i + window // 2 + 1)
                smoothed_losses.append(np.mean(losses[start_idx:end_idx]))
            losses = smoothed_losses
        
        # Find the learning rate where loss decreases fastest (steepest negative gradient)
        gradients = []
        for i in range(1, len(losses)):
            if lrs[i] > 0 and lrs[i-1] > 0:  # Avoid log(0) issues
                gradient = (losses[i] - losses[i-1]) / (np.log10(lrs[i]) - np.log10(lrs[i-1]))
                gradients.append(gradient)
        
        if not gradients:
            return lrs[len(lrs) // 2]  # Fallback to middle value
        
        # Find steepest descent point
        min_gradient_idx = np.argmin(gradients)
        
        # The suggested LR is typically at the steepest descent point
        suggested_lr = lrs[min_gradient_idx + 1]  # +1 because gradients array is offset
        
        return suggested_lr
    
    def plot_lr_finder_results(self, lr_results, save_path=None):
        """
        Plot learning rate finder results with suggestion highlighted.
        """
        lrs = lr_results['lrs']
        losses = lr_results['losses']
        suggested_lr = lr_results['suggested_lr']
        
        plt.figure(figsize=(10, 4))
        plt.plot(lrs, losses, 'b-', linewidth=2, label='Loss', alpha=0.8)
        
        # Highlight suggested learning rate
        plt.axvline(x=suggested_lr, color='red', linestyle='--', linewidth=3,
                   label=f'Suggested LR: {suggested_lr:.2e}')
        
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Learning Rate Range Test Results', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[LRFinder] Plot saved to {save_path}")
        
        plt.show()
        plt.close()

# Convenience function for standalone usage
def find_optimal_lr_standalone(model, train_loader, device, **kwargs):
    """
    Standalone function to find optimal learning rate.
    
    Usage:
        lr_finder = OptimalLRFinder()
        results = lr_finder.find_optimal_learning_rate(model, train_loader, device)
        lr_finder.plot_lr_finder_results(results)
    """
    finder = OptimalLRFinder()
    results = finder.find_optimal_learning_rate(model, train_loader, device, **kwargs)
    finder.plot_lr_finder_results(results)
    return results

if __name__ == "__main__":
    print("OptimalLRFinder module loaded successfully!")
    print("Usage:")
    print("  lr_finder = OptimalLRFinder()")
    print("  results = lr_finder.find_optimal_learning_rate(model, train_loader, device)")
    print("  lr_finder.plot_lr_finder_results(results, 'lr_finder_plot.png')")
