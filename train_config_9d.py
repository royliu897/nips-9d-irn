import os
import torch
import random

# Get scratch directory for loading data
# This ensures we look in the high-speed storage where curate_data_9d.py saved the file
scratch_dir = os.environ.get('SCRATCH', 'data') 

CONFIG = {
    # Point to the curated .pt file
    'data_path': os.path.join(scratch_dir, 'simulation_data', 'surrogate_training', 'nips_9d_data_gen_curated.pt'),
    
    # The 9 physics parameters in the exact order they were saved
    'param_keys': ['Ax', 'Az', 'J1a', 'J1b', 'J2a', 'J2b', 'J3a', 'J3b', 'J4'],
    
    # Dimensions
    'num_q': 2500, 
    'num_w': 150, 
    
    # --- ARCHITECTURE CONFIGS ---
    'param_input_dim': 9, # Size of params
    'dim_in': 13,         # Legacy key (kept for compatibility)
    'dim_hidden': 256,
    'num_layers': 4,      # Depth of Synthesis Network
    'w0_initial': 30.0,
    'output_dim': 1,
    
    # Normalization Bounds (Must match generate_data_9d.jl exactly)
    'input_normalizer': {
        'h': {'min': -0.5, 'max': 0.5},
        'k': {'min': -0.5, 'max': 0.5},
        'l': {'min': -0.5, 'max': 0.5},
        'energy': {'min': 0, 'max': 150},
        'Ax': {'min': -0.02, 'max': 0.0},
        'Az': {'min': 0.0, 'max': 0.42},
        'J1a': {'min': -5.4, 'max': 0.0},
        'J1b': {'min': -4.0, 'max': 0.0},
        'J2a': {'min': 0, 'max': 0.4},
        'J2b': {'min': 0, 'max': 0.4},
        'J3a': {'min': 0, 'max': 27.8},
        'J3b': {'min': 0, 'max': 27.8},
        'J4': {'min': -0.76, 'max': 0.0},
    },

    # Training Hyperparameters
    'batch_size': 4,       # Safe for A100 with 12M points per batch
    'learning_rate': 1e-4,
    
    # TRAINING DURATION CONTROLS
    'num_epochs': 500,           # Safety ceiling (reduced from 50,000)
    'early_stopping_patience': 20, # Stop if no improvement for 20 epochs
    
    'train_split': 0.8,
    'val_split': 0.15,
    'test_split': 0.05,
    
    # Logging
    'log_interval': 10,
    'save_interval': 20,
    'checkpoint_dir': 'checkpoints_9d', 
    'wandb_project': 'nips-9d-siren-film',
    'wandb_entity': None, 
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': random.randint(0, 10000)
}
