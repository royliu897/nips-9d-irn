import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import wandb
from tqdm import tqdm
from train_config_9d import CONFIG
from siren_model import SurrogateModel

# -----------------------------------------------------------------------------
# OPTIMIZED DATASET CLASS (Lazy Loading)
# -----------------------------------------------------------------------------
class SpectrumDataset(Dataset):
    def __init__(self, data_path, param_keys, num_q, num_w):
        print(f"Loading dataset from {data_path}...")
        # Load data to RAM (CPU)
        data = torch.load(data_path, map_location='cpu')
        
        self.params_tensor = data['params_tensor'] # (Ns, 9)
        self.q = data['q']                         # (Ns, Nq, 3)
        self.w = data['w']                         # (Nw,)
        self.sqw = data['sqw']                     # (Ns, Nq, Nw)
        
        self.param_keys = param_keys
        self.normalizer = CONFIG['input_normalizer']
        
        # Pre-calculate normalization bounds (keep on CPU for __getitem__)
        self.p_mins = torch.tensor([self.normalizer[k]['min'] for k in param_keys]).float()
        self.p_maxs = torch.tensor([self.normalizer[k]['max'] for k in param_keys]).float()
        
        # Coordinates bounds for Q and W separate
        # Q bounds (h,k,l)
        self.q_mins = torch.tensor([self.normalizer['h']['min'], self.normalizer['k']['min'], self.normalizer['l']['min']]).float()
        self.q_maxs = torch.tensor([self.normalizer['h']['max'], self.normalizer['k']['max'], self.normalizer['l']['max']]).float()
        
        # Energy bounds
        self.w_min = torch.tensor(self.normalizer['energy']['min']).float()
        self.w_max = torch.tensor(self.normalizer['energy']['max']).float()

    def __len__(self):
        return len(self.params_tensor)

    def normalize(self, tensor, min_v, max_v):
        # Normalize to [-1, 1]
        return 2 * ((tensor - min_v) / (max_v - min_v)) - 1

    def __getitem__(self, idx):
        """
        Returns COMPACT tensors.
        We do NOT expand the meshgrid here. We save PCIe bandwidth by expanding on GPU.
        """
        # 1. Normalize Params (9,)
        p_raw = self.params_tensor[idx]
        p_norm = self.normalize(p_raw, self.p_mins, self.p_maxs)
        
        # 2. Normalize Q (Nq, 3)
        q_raw = self.q[idx] 
        q_norm = self.normalize(q_raw, self.q_mins, self.q_maxs)
        
        # 3. Normalize W (Nw,) - Note: w is global, but usually fast to just grab
        w_raw = self.w
        w_norm = self.normalize(w_raw, self.w_min, self.w_max)
        
        # 4. Target (Nq, Nw) - flattened later
        target = self.sqw[idx]
        
        return p_norm.float(), q_norm.float(), w_norm.float(), target.float()

# -----------------------------------------------------------------------------
# GPU-ACCELERATED MESHGRID & LOSS
# -----------------------------------------------------------------------------
def construct_grid_on_gpu(q_norm, w_norm):
    """
    Constructs the full coordinate meshgrid on the GPU.
    q_norm: (Batch, Nq, 3)
    w_norm: (Batch, Nw)
    Returns: coords (Batch, Nq*Nw, 4)
    """
    B, Nq, _ = q_norm.shape
    B_w, Nw = w_norm.shape
    assert B == B_w
    
    # 1. Expand Q to (Batch, Nq, Nw, 3)
    # Unsqueeze dim 2 to broadcast over Nw
    q_expanded = q_norm.unsqueeze(2).expand(B, Nq, Nw, 3)
    
    # 2. Expand W to (Batch, Nq, Nw, 1)
    # Unsqueeze dim 1 to broadcast over Nq, dim 3 for feature
    w_expanded = w_norm.view(B, 1, Nw, 1).expand(B, Nq, Nw, 1)
    
    # 3. Concatenate to (Batch, Nq, Nw, 4)
    coords = torch.cat([q_expanded, w_expanded], dim=-1)
    
    # 4. Flatten to (Batch, Nq*Nw, 4)
    return coords.reshape(B, -1, 4)

def weighted_mse_loss(output, target):
    # Standard weighted MSE
    weights = 1.0 + 9.0 * target
    loss = weights * (output - target) ** 2
    return torch.mean(loss)

# -----------------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------------
def train_epoch(model, train_loader, optimizer, device, epoch, scaler):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch_idx, (params, q_norm, w_norm, target) in enumerate(pbar):
        # Move compact tensors to GPU (Fast!)
        params = params.to(device, non_blocking=True)
        q_norm = q_norm.to(device, non_blocking=True)
        w_norm = w_norm.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Construct Inputs on GPU (Massive compute parallelization)
        coords = construct_grid_on_gpu(q_norm, w_norm)
        target_flat = target.reshape(target.shape[0], -1, 1)
        
        optimizer.zero_grad(set_to_none=True) # Saves memory
        
        # Mixed Precision Context
        with torch.amp.autocast('cuda'):
            # Forward
            output = model(params, coords)
            # Loss
            loss = weighted_mse_loss(output, target_flat)
        
        # Scaled Backward Pass
        scaler.scale(loss).backward()
        
        # Unscale & Step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        if batch_idx % CONFIG['log_interval'] == 0:
            wandb.log({'train/batch_loss': loss.item(), 'epoch': epoch})
    
    return total_loss / num_batches

def validate_epoch(model, val_loader, device, epoch):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        for params, q_norm, w_norm, target in pbar:
            params = params.to(device, non_blocking=True)
            q_norm = q_norm.to(device, non_blocking=True)
            w_norm = w_norm.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Construct inputs on GPU
            coords = construct_grid_on_gpu(q_norm, w_norm)
            target_flat = target.reshape(target.shape[0], -1, 1)
            
            # AMP inference
            with torch.amp.autocast('cuda'):
                output = model(params, coords)
                loss = weighted_mse_loss(output, target_flat)
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / num_batches

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    wandb.init(project=CONFIG['wandb_project'], entity=CONFIG['wandb_entity'], config=CONFIG)
    
    # Enable TF32 for A100 (Better precision/speed trade-off)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    full_dataset = SpectrumDataset(CONFIG['data_path'], CONFIG['param_keys'], CONFIG['num_q'], CONFIG['num_w'])
    
    dataset_size = len(full_dataset)
    train_size = int(CONFIG['train_split'] * dataset_size)
    val_size = int(CONFIG['val_split'] * dataset_size)
    test_size = dataset_size - train_size - val_size
    if train_size + val_size + test_size != dataset_size:
        test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(CONFIG['seed'])
    )

    # Increased Num Workers for parallel loading of small vectors
    # Persistent workers keeps the processes alive
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True 
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True
    )

    model = SurrogateModel(CONFIG).to(CONFIG['device'])
    print(f"Model Parameters: {model.count_parameters():,}")
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Initialize Gradient Scaler for AMP
    scaler = torch.amp.GradScaler('cuda')

    CONFIG['checkpoint_dir'] = os.path.join(wandb.run.dir, "checkpoints")
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0 # Initialize Early Stopping Counter
    
    try:
        for epoch in range(CONFIG['num_epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, CONFIG['device'], epoch, scaler)
            val_loss = validate_epoch(model, val_loader, CONFIG['device'], epoch)
            
            wandb.log({'train/epoch_loss': train_loss, 'val/epoch_loss': val_loss, 'epoch': epoch})
            print(f"Epoch {epoch+1} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            
            # --- EARLY STOPPING LOGIC ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0 # Reset counter
                torch.save(model.state_dict(), os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth'))
                print(f"  -> New Best Model (Loss: {best_val_loss:.6f})")
            else:
                patience_counter += 1
                print(f"  -> No improvement. Patience: {patience_counter}/{CONFIG['early_stopping_patience']}")
                
            if patience_counter >= CONFIG['early_stopping_patience']:
                print("\n!!! Early Stopping Triggered !!!")
                print(f"Validation loss has not improved for {CONFIG['early_stopping_patience']} epochs.")
                break
            # ---------------------------
                
            if (epoch + 1) % CONFIG['save_interval'] == 0:
                torch.save(model.state_dict(), os.path.join(CONFIG['checkpoint_dir'], f'ckpt_{epoch+1}.pth'))

    except KeyboardInterrupt:
        print("Training interrupted.")
        torch.save(model.state_dict(), os.path.join(CONFIG['checkpoint_dir'], 'interrupted.pth')) 
