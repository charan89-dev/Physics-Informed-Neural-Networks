"""
ThermalPINN — Multi-Vd Universal Surrogate (Academic 3-Panel Validation)
Phase 2 (Data Pipeline), Phase 3 (Model Architecture), Phase 4 (Physics Loss), Phase 5 (Training)
"""

import os
import glob
import re

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE = 1e-3
PDE_LAMBDA = 1e-6     # Lowered to allow Joule heating at the drain
BC_LAMBDA = 10.0      # Lowered to prevent gradient explosions at the 300K contacts
NUM_EPOCHS = 5000     # Increased to allow the physics time to converge
PRINT_EVERY = 500

DATA_DIR = "./pinn_csv_data"          
FILE_PATTERN = "n*_full_2D.csv"
SILICON_DOPING_THRESHOLD = 1e10

# ---------------------------------------------------------------------------
# Phase 2: Data Loading & Tensor Setup
# ---------------------------------------------------------------------------
NODE_TO_VD = {
    "n59": 0.1, "n159": 1.0, "n184": 2.0, "n209": 3.0, "n1686": 0.5, "n2036": 1.5, "n2386": 2.5,
    "n351": 0.1, "n353": 1.0, "n355": 2.0, "n357": 3.0, "n1736": 0.5, "n2086": 1.5, "n2436": 2.5,
    "n576": 0.1, "n578": 1.0, "n580": 2.0, "n582": 3.0, "n1786": 0.5, "n2136": 1.5, "n2486": 2.5,
    "n801": 0.1, "n803": 1.0, "n805": 2.0, "n807": 3.0, "n1836": 0.5, "n2186": 1.5, "n2536": 2.5,
    "n1026": 0.1, "n1028": 1.0, "n1030": 2.0, "n1032": 3.0, "n1886": 0.5, "n2236": 1.5, "n2586": 2.5,
    "n1251": 0.1, "n1253": 1.0, "n1255": 2.0, "n1257": 3.0, "n1936": 0.5, "n2286": 1.5, "n2636": 2.5,
    "n1476": 0.1, "n1478": 1.0, "n1480": 2.0, "n1482": 3.0, "n1986": 0.5, "n2336": 1.5, "n2686": 2.5
}

class TCADDataset:
    def __init__(self, data_dir: str = DATA_DIR, file_pattern: str = FILE_PATTERN):
        csv_paths = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        node_pattern = re.compile(r"(n\d+)_full_2D\.csv$")
        frames = []
        
        for path in csv_paths:
            match = node_pattern.search(os.path.basename(path))
            if match and match.group(1) in NODE_TO_VD:
                df = pd.read_csv(path)
                df["Vd"] = NODE_TO_VD[match.group(1)]
                frames.append(df)

        self.df = pd.concat(frames, ignore_index=True)

        # Delta-T Transformation: Shift absolute temp to temperature rise
        self.df["LatticeTemperature"] = self.df["LatticeTemperature"] - 300.0

        # ROBUST BOUNDARY MASK: Lock physical metal contacts
        delta_t_phys = self.df["LatticeTemperature"].to_numpy()
        self.contact_mask = torch.tensor(delta_t_phys < 0.05, dtype=torch.bool)

        self.mins, self.maxs, normalized = {}, {}, {}
        for col in ["X", "Y", "Vd", "LatticeTemperature"]:
            self.mins[col], self.maxs[col] = self.df[col].min(), self.df[col].max()
            denom = self.maxs[col] - self.mins[col]
            normalized[col] = np.zeros(len(self.df)) if denom == 0 else (self.df[col].to_numpy() - self.mins[col]) / denom

        # Extract Physical Spans for Anisotropy Correction
        self.L_x = self.maxs["X"] - self.mins["X"]
        self.L_y = self.maxs["Y"] - self.mins["Y"]
        self.T_scale = self.maxs["LatticeTemperature"] - self.mins["LatticeTemperature"]

        xyv_norm = np.stack([normalized["X"], normalized["Y"], normalized["Vd"]], axis=1)
        self.X_in = torch.tensor(xyv_norm, dtype=torch.float32, requires_grad=True)
        self.T_true = torch.tensor(normalized["LatticeTemperature"], dtype=torch.float32).unsqueeze(1)
        
        doping_mag = self.df["DopingConcentration"].to_numpy()
        self.silicon_mask = torch.tensor(np.abs(doping_mag) > SILICON_DOPING_THRESHOLD, dtype=torch.bool)

    def denormalize(self, col: str, values: torch.Tensor) -> torch.Tensor:
        return values * (self.maxs[col] - self.mins[col]) + self.mins[col]

# ---------------------------------------------------------------------------
# Phase 3: Neural Network Architecture
# ---------------------------------------------------------------------------
class ThermalPINN(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_width: int = 50, hidden_layers: int = 5):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]
        layers += [nn.Linear(hidden_width, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, xyv: torch.Tensor) -> torch.Tensor:
        return self.net(xyv)

# ---------------------------------------------------------------------------
# Phase 4: Hybrid Physics Loss (Anisotropy Chain-Rule Fix)
# ---------------------------------------------------------------------------
def compute_pde_residual(T_pred: torch.Tensor, X_in: torch.Tensor, L_x: float, L_y: float, T_scale: float) -> torch.Tensor:
    grad_T = torch.autograd.grad(outputs=T_pred, inputs=X_in, grad_outputs=torch.ones_like(T_pred), create_graph=True, retain_graph=True)[0]
    dT_dx_norm, dT_dy_norm = grad_T[:, 0:1], grad_T[:, 1:2]

    d2T_dx2_norm = torch.autograd.grad(outputs=dT_dx_norm, inputs=X_in, grad_outputs=torch.ones_like(dT_dx_norm), create_graph=True, retain_graph=True)[0][:, 0:1]
    d2T_dy2_norm = torch.autograd.grad(outputs=dT_dy_norm, inputs=X_in, grad_outputs=torch.ones_like(dT_dy_norm), create_graph=True, retain_graph=True)[0][:, 1:2]

    # Apply Chain Rule to scale normalized derivatives back to physical micrometers
    d2T_dx2_phys = d2T_dx2_norm * (T_scale / (L_x ** 2))
    d2T_dy2_phys = d2T_dy2_norm * (T_scale / (L_y ** 2))

    return d2T_dx2_phys + d2T_dy2_phys

def compute_losses(model: nn.Module, dataset: TCADDataset):
    T_pred = model(dataset.X_in)
    
    data_loss = torch.mean((T_pred - dataset.T_true) ** 2)
    
    pde_residual = compute_pde_residual(T_pred, dataset.X_in, dataset.L_x, dataset.L_y, dataset.T_scale)
    pde_loss = torch.mean(pde_residual[dataset.silicon_mask] ** 2)

    # Use the robust physical contact mask for boundary locking
    boundary_loss = torch.mean((T_pred[dataset.contact_mask] - dataset.T_true[dataset.contact_mask]) ** 2)

    total_loss = data_loss + (PDE_LAMBDA * pde_loss) + (BC_LAMBDA * boundary_loss)
    return total_loss, data_loss, pde_loss, boundary_loss

# ---------------------------------------------------------------------------
# Phase 5: Training & Academic 3-Panel Visualization
# ---------------------------------------------------------------------------
def train(model: nn.Module, dataset: TCADDataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(1, NUM_EPOCHS + 1):
        optimizer.zero_grad()
        total_loss, data_loss, pde_loss, boundary_loss = compute_losses(model, dataset)
        total_loss.backward()
        optimizer.step()
        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Data: {data_loss.item():.4e} | PDE: {pde_loss.item():.4e} | BC: {boundary_loss.item():.4e}")

def visualize_and_verify_2_5v(model: nn.Module, dataset: TCADDataset, device: torch.device):
    model.eval()
    target_vd = 2.5
    
    # Isolate 2.5V data
    vd_mask = np.isclose(dataset.df["Vd"].to_numpy(), target_vd, atol=1e-5)
    vd_mask_tensor = torch.tensor(vd_mask, dtype=torch.bool, device=device)
    
    xyv_25 = dataset.X_in[vd_mask_tensor]
    T_true_25_norm = dataset.T_true[vd_mask_tensor]
    silicon_mask_25 = dataset.silicon_mask[vd_mask_tensor]
    
    T_pred_25_norm = model(xyv_25)
    
    xy_norm = xyv_25.detach().cpu()
    T_pred_norm = T_pred_25_norm.detach().cpu().squeeze(1)
    T_true_norm = T_true_25_norm.detach().cpu().squeeze(1)
    silicon_mask_cpu = silicon_mask_25.detach().cpu().numpy().astype(int) 
    
    # Denormalize & Add 300K Baseline back
    x_phys = dataset.denormalize("X", xy_norm[:, 0]).numpy()
    y_phys = dataset.denormalize("Y", xy_norm[:, 1]).numpy()
    T_pred_phys = dataset.denormalize("LatticeTemperature", T_pred_norm).numpy() + 300.0
    T_true_phys = dataset.denormalize("LatticeTemperature", T_true_norm).numpy() + 300.0

    # Calculate Error Map
    error_map = np.abs(T_true_phys - T_pred_phys)

    # 3-Panel Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin = min(np.min(T_true_phys), np.min(T_pred_phys))
    vmax = max(np.max(T_true_phys), np.max(T_pred_phys))

    # Panel 1: TCAD
    cf0 = axes[0].tricontourf(y_phys, x_phys, T_true_phys, levels=100, cmap="jet", vmin=vmin, vmax=vmax)
    axes[0].tricontour(y_phys, x_phys, silicon_mask_cpu, levels=[0.5], colors='black', linewidths=1.0)
    axes[0].invert_yaxis()
    axes[0].set_title("Commercial Solver (Sentaurus TCAD)")
    axes[0].set_xlabel("Y (µm)")
    axes[0].set_ylabel("X (µm)")
    fig.colorbar(cf0, ax=axes[0], label="Temperature (K)")

    # Panel 2: PINN
    cf1 = axes[1].tricontourf(y_phys, x_phys, T_pred_phys, levels=100, cmap="jet", vmin=vmin, vmax=vmax)
    axes[1].tricontour(y_phys, x_phys, silicon_mask_cpu, levels=[0.5], colors='black', linewidths=1.0)
    axes[1].invert_yaxis()
    axes[1].set_title("SimNet / PINN Surrogate")
    axes[1].set_xlabel("Y (µm)")
    fig.colorbar(cf1, ax=axes[1], label="Temperature (K)")

    # Panel 3: Error
    cf2 = axes[2].tricontourf(y_phys, x_phys, error_map, levels=100, cmap="viridis")
    axes[2].tricontour(y_phys, x_phys, silicon_mask_cpu, levels=[0.5], colors='white', linewidths=1.0, linestyles='dashed')
    axes[2].invert_yaxis()
    axes[2].set_title(f"Absolute Difference (Max Error: {np.max(error_map):.3f} K)")
    axes[2].set_xlabel("Y (µm)")
    fig.colorbar(cf2, ax=axes[2], label="Error (K)")

    plt.tight_layout()
    plt.savefig("thermal_pinn_academic_comparison.png", dpi=300)
    print("\nAcademic validation plot saved to thermal_pinn_academic_comparison.png")

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dataset = TCADDataset()
    model = ThermalPINN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    dataset.X_in = dataset.X_in.to(device).requires_grad_(True)
    dataset.T_true = dataset.T_true.to(device)
    dataset.contact_mask = dataset.contact_mask.to(device)
    dataset.silicon_mask = dataset.silicon_mask.to(device)
    
    print("\nStarting Training Phase (Anisotropy Fix & Robust BC)...")
    train(model, dataset)
    visualize_and_verify_2_5v(model, dataset, device)

    torch.save(model.state_dict(), "thermal_pinn_brain.pth")