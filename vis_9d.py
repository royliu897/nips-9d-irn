import os
import torch
import h5py
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from train_config_9d import CONFIG
from siren_model import SurrogateModel

# Use non-interactive backend for TACC
mpl.use('Agg') 

# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------
def normalize(tensor, min_v, max_v):
    return 2 * ((tensor - min_v) / (max_v - min_v)) - 1

def find_latest_model():
    """
    Scans the wandb directory to find the most recently modified best_model.pth
    """
    default_path = os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth')
    if os.path.exists(default_path):
        return default_path

    search_pattern = "wandb/*/checkpoints/best_model.pth"
    candidates = glob.glob(search_pattern)
    
    if not candidates:
        search_pattern = "wandb/*/*/checkpoints/best_model.pth"
        candidates = glob.glob(search_pattern)

    if not candidates:
        return None

    latest_ckpt = max(candidates, key=os.path.getmtime)
    print(f"Found latest checkpoint at: {latest_ckpt}")
    return latest_ckpt

def load_model(device):
    model = SurrogateModel(CONFIG).to(device)
    ckpt_path = find_latest_model()
    
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
    else:
        print("WARNING: No checkpoint found! Using random weights.")
    model.eval()
    return model

def prepare_inputs(params_dict, q_points, energy_points, device):
    # 1. Normalize Parameters
    norm_cfg = CONFIG['input_normalizer']
    p_vec = [params_dict[k] for k in CONFIG['param_keys']]
    
    p_tens = torch.tensor(p_vec).float()
    p_mins = torch.tensor([norm_cfg[k]['min'] for k in CONFIG['param_keys']]).float()
    p_maxs = torch.tensor([norm_cfg[k]['max'] for k in CONFIG['param_keys']]).float()
    
    p_norm = normalize(p_tens, p_mins, p_maxs).to(device).unsqueeze(0)

    # 2. Normalize Coordinates
    n_path = len(q_points)
    n_w = len(energy_points)
    
    q_tens = torch.tensor(q_points).float()
    q_mins = torch.tensor([norm_cfg['h']['min'], norm_cfg['k']['min'], norm_cfg['l']['min']]).float()
    q_maxs = torch.tensor([norm_cfg['h']['max'], norm_cfg['k']['max'], norm_cfg['l']['max']]).float()
    q_norm = normalize(q_tens, q_mins, q_maxs)
    
    w_tens = torch.tensor(energy_points).float()
    w_min = torch.tensor(norm_cfg['energy']['min']).float()
    w_max = torch.tensor(norm_cfg['energy']['max']).float()
    w_norm = normalize(w_tens, w_min, w_max)
    
    # 3. Construct Meshgrid
    q_exp = q_norm.unsqueeze(1).expand(n_path, n_w, 3)
    w_exp = w_norm.view(1, n_w, 1).expand(n_path, n_w, 1)
    
    coords = torch.cat([q_exp, w_exp], dim=-1).unsqueeze(0).reshape(1, -1, 4).to(device)
    
    return p_norm, coords

# -----------------------------------------------------------------------------
# TASK A: DENSE COMPARISON (Heatmap vs Heatmap)
# -----------------------------------------------------------------------------
def run_dense_comparison(model, device):
    gt_path = os.path.join(os.environ['SCRATCH'], "nips_viz", "dense_gt_slice.h5")
    
    if not os.path.exists(gt_path):
        print(f"Skipping: {gt_path} not found.")
        return

    print(f"\n--- Running Dense Comparison ---")
    with h5py.File(gt_path, 'r') as f:
        # Load Raw Data
        gt_data_raw = f["data"][:] 
        energies = f["energies"][:]
        q_path_raw = f["q_path"][:] 
        
        params = {}
        for k in f["params"].keys():
            params[k] = float(f["params"][k][()])
            
    # --- FIX 1: Handle Q-Path Dimensions ---
    # We expect (N_points, 3). Julia saves (3, N_points). Python reads (N_points, 3).
    # Sometimes h5py reads it as (3, N_points). We detect and fix.
    if q_path_raw.shape[0] == 3 and q_path_raw.shape[1] != 3:
        # It's (3, N), transpose to (N, 3)
        q_path = q_path_raw.T
    else:
        q_path = q_path_raw
        
    print(f"Q-Path Shape: {q_path.shape}")

    # --- FIX 2: Handle Ground Truth Orientation ---
    # Julia (Energies x Path) -> Python (Path x Energies)
    # We want (Energies x Path) for plotting [Energies on Y-axis]
    # Check if we need to transpose back.
    if gt_data_raw.shape[0] == len(q_path):
        print("Detected Transposed GT (Path x Energy). Transposing to (Energy x Path)...")
        gt_data = gt_data_raw.T
    else:
        gt_data = gt_data_raw
        
    print(f"GT Data Shape: {gt_data.shape}") # Should be (200, 298)

    # Run Prediction
    with torch.no_grad():
        p_in, c_in = prepare_inputs(params, q_path, energies, device)
        output = model(p_in, c_in)
        
        # Reshape: (1, N_path*N_w, 1) -> (N_path, N_w) -> Transpose to (N_w, N_path)
        pred_img = output.reshape(len(q_path), len(energies)).cpu().numpy().T
        pred_img = np.maximum(0, pred_img)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    extent = [0, len(q_path), energies[0], energies[-1]]
    
    # Ground Truth
    im1 = ax[0].imshow(gt_data, origin='lower', aspect='auto', cmap='inferno', extent=extent, vmin=0, vmax=np.max(gt_data))
    ax[0].set_title("Ground Truth (Sunny.jl)")
    ax[0].set_ylabel("Energy (meV)")
    ax[0].set_xlabel("Path Index")
    plt.colorbar(im1, ax=ax[0])
    
    # Prediction
    im2 = ax[1].imshow(pred_img, origin='lower', aspect='auto', cmap='inferno', extent=extent, vmin=0, vmax=np.max(gt_data))
    ax[1].set_title("SIREN Prediction")
    ax[1].set_xlabel("Path Index")
    plt.colorbar(im2, ax=ax[1])
    
    plt.tight_layout()
    plt.savefig("viz_dense_comparison.png")
    print("Saved viz_dense_comparison.png")

# -----------------------------------------------------------------------------
# TASK B: STATISTICAL CHECK
# -----------------------------------------------------------------------------
def run_statistical_check(model, device):
    print(f"\n--- Running Statistical Check ---")
    data = torch.load(CONFIG['data_path'], map_location='cpu')
    idx = 100
    
    real_q = data['q'][idx]
    real_sqw = data['sqw'][idx]
    real_w = data['w']
    p_tensor = data['params_tensor'][idx]
    
    params = {k: float(p_tensor[i]) for i, k in enumerate(CONFIG['param_keys'])}

    with torch.no_grad():
        p_in, c_in = prepare_inputs(params, real_q.numpy(), real_w.numpy(), device)
        output = model(p_in, c_in)
        pred_sqw = output.reshape(2500, 150).cpu().numpy()
        pred_sqw = np.maximum(0, pred_sqw)

    gt_curve = real_sqw.mean(dim=0).numpy()
    pred_curve = pred_sqw.mean(axis=0)

    plt.figure(figsize=(8, 6), dpi=120)
    plt.plot(real_w, gt_curve, 'k--', label='Ground Truth', linewidth=2)
    plt.plot(real_w, pred_curve, 'r-', label='SIREN Prediction', linewidth=2)
    plt.title(f"Statistical Validation (Sample {idx})")
    plt.xlabel("Energy (meV)")
    plt.ylabel("Integrated Intensity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("viz_statistical_check.png")
    print("Saved viz_statistical_check.png")

if __name__ == "__main__":
    device = CONFIG['device']
    model = load_model(device)
    run_dense_comparison(model, device)
    run_statistical_check(model, device)
