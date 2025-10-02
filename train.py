#!/usr/bin/env python3
"""
Main Training Script for Cell2Gene HEST Spatial Gene Expression Prediction

author: Jingkun Yu
"""

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

from dataset import HESTSpatialDataset, collate_fn_hest_graph
from trainer import train_hest_graph_model, setup_optimizer_and_scheduler, setup_model
from utils import get_fold_samples, evaluate_model_metrics, save_evaluation_results, plot_training_curves, setup_device


def main():
    """Main training workflow - 10-fold cross validation"""
    
    # Configuration parameters
    hest_data_dir = "/data/yujk/hovernet2feature/HEST/hest_data"
    graph_dir = "/data/yujk/hovernet2feature/hest_graphs_dinov3"
    
    # All 49 samples (for 10-fold cross validation)
    all_samples = ['TENX152', 'MISC73', 'MISC72', 'MISC71', 'MISC70',
                   'MISC69', 'MISC68', 'MISC67', 'MISC66', 'MISC65', 
                   'MISC64', 'MISC63', 'MISC62', 'MISC58', 'MISC57',
                   'MISC56', 'MISC51', 'MISC50', 'MISC49', 'MISC48',
                   'MISC47', 'MISC46', 'MISC45', 'MISC44', 'MISC43',
                   'MISC42', 'MISC41', 'MISC40', 'MISC39', 'MISC38',
                   'MISC37', 'MISC36', 'MISC35', 'MISC34', 'MISC33',
                   'TENX92', 'TENX91', 'TENX90', 'TENX89', 'TENX49',
                   'TENX29', 'ZEN47', 'ZEN46', 'ZEN45', 'ZEN44',
                   'ZEN43', 'ZEN42', 'ZEN39', 'ZEN38']
    
    # Specify gene file
    gene_file = "/data/yujk/hovernet2feature/HEST/tutorials/SA_process/common_genes_misc_tenx_zen_897.txt"
    
    batch_size = 16
    num_epochs = 60
    learning_rate = 3e-6
    weight_decay = 1e-5
    feature_dim = 128
    
    # Early stopping parameters
    patience = 50
    min_delta = 1e-6
    
    # Start training from specified fold
    start_fold = 0  
    
    print("=== HEST Spatial Supervised Training - 10-fold Cross Validation ===")
    print("✓ Using direct file reading (no HEST API required)")
    print("✓ Using 897 intersection genes")
    print("✓ Sample-level 10-fold cross validation")
    print(f"✓ Total samples: {len(all_samples)}")
    print(f"✓ Gene file: {gene_file}")
    print(f"✓ Starting from Fold {start_fold + 1}")
    
    # Setup device
    device = setup_device(device_id=1)
    
    # Create results directory
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    
    # Store all fold results
    all_fold_results = {}
    
    # Try to load temporary results
    temp_results_file = "./logs/temp_fold_results.json"
    if os.path.exists(temp_results_file):
        try:
            with open(temp_results_file, 'r') as f:
                all_fold_results = json.load(f)
                # Convert string keys back to int
                all_fold_results = {int(k): v for k, v in all_fold_results.items()}
                print(f"✓ Loaded temporary results: {len(all_fold_results)} folds")
                print(f"  Completed folds: {sorted(all_fold_results.keys())}")
        except Exception as e:
            print(f"Warning: Could not load temporary results file: {e}")
    
    # Execute 10-fold cross validation (starting from specified fold)
    for fold_idx in range(start_fold, 10):
        print(f"\n{'='*50}")
        print(f"Starting Fold {fold_idx + 1}/10")
        print(f"{'='*50}")
        
        # Get current fold train and test samples
        train_samples, test_samples = get_fold_samples(fold_idx, all_samples)
        print(f"Training samples ({len(train_samples)}): {train_samples}")
        print(f"Test samples ({len(test_samples)}): {test_samples}")
        
        # Create datasets (only load required samples)
        train_dataset = HESTSpatialDataset(
            hest_data_dir=hest_data_dir,
            graph_dir=graph_dir,
            sample_ids=train_samples,  # Only pass training samples
            feature_dim=feature_dim,
            mode='train',
            gene_file=gene_file
        )
        
        test_dataset = HESTSpatialDataset(
            hest_data_dir=hest_data_dir,
            graph_dir=graph_dir,
            sample_ids=test_samples,   # Only pass test samples
            feature_dim=feature_dim,
            mode='test',
            gene_file=gene_file
        )
        
        # Create data loaders (reduce memory pressure)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=collate_fn_hest_graph,
            num_workers=0,  # Avoid multi-process memory issues
            pin_memory=False,  # Reduce memory usage
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=collate_fn_hest_graph,
            num_workers=0,  # Avoid multi-process memory issues
            pin_memory=False,  # Reduce memory usage
        )
        
        num_genes = train_dataset.num_genes
        
        print(f"\n=== Fold {fold_idx + 1} Dataset Info ===")
        print(f"Device: {device}")
        print(f"Gene count: {num_genes}")
        print(f"Training spots: {len(train_dataset)}")
        print(f"Test spots: {len(test_dataset)}")
        
        # Create model
        model = setup_model(feature_dim, num_genes, device)
        if model is None:
            print("Failed to setup model, skipping this fold")
            continue
        
        # Setup optimizer and scheduler
        optimizer, scheduler = setup_optimizer_and_scheduler(
            model, learning_rate, weight_decay, num_epochs
        )
        
        print(f"\n=== Fold {fold_idx + 1} Training Configuration ===")
        print(f"Model: StaticGraphTransformerPredictor + GAT")
        print(f"Optimizer: AdamW, Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size}, Epochs: {num_epochs}")
        print(f"Early stopping: patience={patience}, min_delta={min_delta}")
        
        # Train model
        train_losses, test_losses = train_hest_graph_model(
            model, train_loader, test_loader, optimizer, scheduler,
            num_epochs=num_epochs, device=device, patience=patience, min_delta=min_delta, fold_idx=fold_idx
        )
        
        # Load best model for evaluation
        if os.path.exists("best_hest_graph_model.pt"):
            model.load_state_dict(torch.load("best_hest_graph_model.pt", map_location=device))
            print("Loaded best model weights for evaluation")
        
        # Evaluate model performance
        eval_results, predictions, targets = evaluate_model_metrics(model, test_loader, device)
        
        # Save current fold's best model
        fold_model_path = f"./checkpoints/best_hest_graph_model_fold_{fold_idx}.pt"
        if os.path.exists("best_hest_graph_model.pt"):
            os.rename("best_hest_graph_model.pt", fold_model_path)
        
        # Save current fold's complete evaluation metrics
        fold_evaluation_results = {
            'fold_idx': fold_idx,
            'train_samples': train_samples,
            'test_samples': test_samples,
            'num_train_spots': len(train_dataset),
            'num_test_spots': len(test_dataset),
            'num_genes': num_genes,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_test_loss': test_losses[-1] if test_losses else None,
            'eval_results': eval_results
        }
        
        # Store in all results
        all_fold_results[fold_idx] = fold_evaluation_results
        
        # Save results
        save_evaluation_results(eval_results, predictions, targets, fold_idx, "./logs")
        plot_training_curves(train_losses, test_losses, fold_idx, "./logs")
        
        # Save temporary results (in case of interruption)
        with open(temp_results_file, 'w') as f:
            json.dump(all_fold_results, f, indent=2)
        
        print(f"\n=== Fold {fold_idx + 1} Completed ===")
        print(f"Final test loss: {test_losses[-1] if test_losses else 'N/A'}")
        print(f"Overall correlation: {eval_results['overall_correlation']:.6f}")
        print(f"Mean gene correlation: {eval_results['mean_gene_correlation']:.6f}")
        
        # Clean up memory
        del model, train_dataset, test_dataset, train_loader, test_loader
        torch.cuda.empty_cache()
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"10-FOLD CROSS VALIDATION COMPLETED")
    print(f"{'='*60}")
    
    # Calculate overall statistics
    overall_correlations = []
    mean_gene_correlations = []
    final_test_losses = []
    
    for fold_idx, results in all_fold_results.items():
        eval_results = results['eval_results']
        overall_correlations.append(eval_results['overall_correlation'])
        mean_gene_correlations.append(eval_results['mean_gene_correlation'])
        final_test_losses.append(results['final_test_loss'])
    
    print(f"Average overall correlation: {np.mean(overall_correlations):.6f} ± {np.std(overall_correlations):.6f}")
    print(f"Average gene correlation: {np.mean(mean_gene_correlations):.6f} ± {np.std(mean_gene_correlations):.6f}")
    print(f"Average final test loss: {np.mean(final_test_losses):.6f} ± {np.std(final_test_losses):.6f}")
    
    # Save final results
    final_results_file = "./logs/final_10fold_results.json"
    with open(final_results_file, 'w') as f:
        json.dump(all_fold_results, f, indent=2)
    
    print(f"\nFinal results saved to: {final_results_file}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()