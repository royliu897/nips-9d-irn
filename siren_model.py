import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModulatedSineLayer(nn.Module):
    """
    A Sine layer that accepts frequency (gamma) and phase (beta) modulations.
    Modulations are based on Hamiltonian parameteres, so we have
    f(x|h) = sin(omega_0 * (gamma * (Wx + b) + beta)), where x is our
    momentum-energy coordinates and h is our hamiltonian paramters.
    """
    def __init__(self, in_features, out_features, omega_0=30, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, x, gamma, beta):
        # x: coordinates
        # gamma: scale
        # beta: shift
        
        out = self.linear(x)
        
        # Apply FiLM modulation
        # allows Hamiltonian parameters to stretch/move the sin function
        out = gamma * out + beta
        
        return torch.sin(self.omega_0 * out)


class MappingNetwork(nn.Module):
    """
    Maps 9D Hamiltonian parameters to the modulation space of the SIREN.
    Maps Ax,Ay... to gamma, beta
    Uses a standard ReLU MLP, not siren layers, for efficiency.
    """
    def __init__(self, in_features, hidden_features, num_layers, out_features):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU(inplace=True))
            
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ModulatedSiren(nn.Module):
    def __init__(self, 
                 coords_dim=4,       # (h, k, l, w)
                 params_dim=9,       # (J1, J2...)
                 hidden_features=256, 
                 hidden_layers=3, 
                 out_features=1, 
                 omega_0=30.0):
        super().__init__()
        
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        
        # --- SYNTHESIS NETWORK (The SIREN) ---
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(ModulatedSineLayer(coords_dim, hidden_features, omega_0=omega_0, is_first=True))
        
        # Hidden layers
        for _ in range(hidden_layers):
            self.layers.append(ModulatedSineLayer(hidden_features, hidden_features, omega_0=omega_0))
            
        # Final layer (Standard Linear, no modulation usually, or minimal)
        self.final_linear = nn.Linear(hidden_features, out_features)
        
        with torch.no_grad():
            bound = math.sqrt(6 / hidden_features) / omega_0
            self.final_linear.weight.uniform_(-bound, bound)
            self.final_linear.bias.zero_()

        # --- MAPPING NETWORK ---
        # We need to generate gamma and beta for every hidden feature in every modulated layer.
        # Total Modulations = (Num_Layers + 1) * Hidden_Features * 2 (for gamma & beta)
        self.num_modulations = (hidden_layers + 1) * hidden_features * 2
        
        self.mapping_net = MappingNetwork(
            in_features=params_dim,
            hidden_features=256,
            num_layers=3, 
            out_features=self.num_modulations
        )

    def forward(self, params, coords):
        """
        params: (Batch, 9)
        coords: (Batch, Points, 4)
        """
        # 1. Generate Modulations
        # (Batch, Total_Mods)
        mods = self.mapping_net(params)
        
        # Reshape to (Batch, 1, Total_Mods) for broadcasting over points
        mods = mods.unsqueeze(1) 
        
        # Split mods into chunks for each layer
        # Each layer needs 2 * hidden_features (gamma + beta)
        mods_split = torch.split(mods, 2 * self.hidden_features, dim=-1)
        
        # 2. Run Synthesis Network
        x = coords
        for layer, layer_mods in zip(self.layers, mods_split):
            # Split scale (gamma) and shift (beta)
            gamma, beta = torch.chunk(layer_mods, 2, dim=-1)
            
            # If gamma is 0 then Gamma*x=0, so Gamma starts at 1.0
            gamma = gamma + 1.0 
            
            x = layer(x, gamma, beta)
            
        return self.final_linear(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class SurrogateModel(ModulatedSiren):
    def __init__(self, config):
        super().__init__(
            coords_dim=4, # h, k, l, w
            params_dim=config['param_input_dim'], # 9
            hidden_features=config['dim_hidden'],
            hidden_layers=config['num_layers'],
            out_features=1,
            omega_0=config['w0_initial']
        )
