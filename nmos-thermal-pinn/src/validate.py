import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import glob

# 1. Define the architecture to load the brain
class ThermalPINN(nn.Module):
    def __init__(self, input_dim=3, hidden_width=50, hidden_layers=5):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]
        layers += [nn.Linear(hidden_width, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, xyv: torch.Tensor) -> torch.Tensor:
        return self.net(xyv)

# 2. Setup Normalization Limits (Must match training limits exactly)
print("Loading Normalization Data...")
all_files = glob.glob("./pinn_csv_data/n*_full_2D.csv")
df_list = [pd.read_csv(f) for f in all_files]
full_df = pd.concat(df_list, ignore_index=True)

x_min, x_max = full_df["X"].min(), full_df["X"].max()
y_min, y_max = full_df["Y"].min(), full_df["Y"].max()
T_min, T_max = full_df["LatticeTemperature"].min(), full_df["LatticeTemperature"].max()
vd_min, vd_max = 0.1, 3.0

# 3. Load the 2.5V Ground Truth TCAD Data
target_vd = 2.5
print(f"Loading TCAD Ground Truth for Vd = {target_vd}V...")
df = pd.read_csv("./pinn_csv_data/n2436_full_2D.csv")

x_phys = df["X"].to_numpy()
y_phys = df["Y"].to_numpy()
T_true_phys = df["LatticeTemperature"].to_numpy()

# Normalize inputs for the AI
x_norm = (x_phys - x_min) / (x_max - x_min)
y_norm = (y_phys - y_min) / (y_max - y_min)
vd_norm = np.full_like(x_norm, (target_vd - vd_min) / (vd_max - vd_min))

# 4. Load the AI Brain
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ThermalPINN().to(device)
model.load_state_dict(torch.load("thermal_pinn_brain.pth", map_location=device, weights_only=True))
model.eval()

xyv_tensor = torch.tensor(np.stack([x_norm, y_norm, vd_norm], axis=1), dtype=torch.float32).to(device)

# 5. Predict the Temperatures
with torch.no_grad():
    T_pred_norm = model(xyv_tensor).cpu().squeeze().numpy()
T_pred_phys = T_pred_norm * (T_max - T_min) + T_min

# =========================================================================
# 6. MATHEMATICAL VALIDATION METRICS
# =========================================================================

# A. Mean Absolute Error (MAE)
absolute_errors = np.abs(T_pred_phys - T_true_phys)
mae = np.mean(absolute_errors)
max_err = np.max(absolute_errors)

# B. Boundary Condition Check
# Dynamically find the lowest temperature in the TCAD mesh (the 300K metal contacts)
true_min_temp = np.min(T_true_phys)
boundary_mask = np.isclose(T_true_phys, true_min_temp, atol=0.05)
ai_boundary_temps = T_pred_phys[boundary_mask]

# Safety check to prevent the zero-size array crash
if len(ai_boundary_temps) > 0:
    boundary_mae = np.mean(np.abs(ai_boundary_temps - true_min_temp))
    boundary_min = np.min(ai_boundary_temps)
    boundary_max = np.max(ai_boundary_temps)
    
    print("\n==================================================")
    print(f"   MATHEMATICAL VALIDATION METRICS (Vd = {target_vd}V)")
    print("==================================================")
    print(f"Overall Mean Absolute Error:       {mae:.4f} K")
    print(f"Overall Maximum Error:             {max_err:.4f} K")
    print("--------------------------------------------------")
    print(f"PHYSICAL BOUNDARY CHECK (Target = {true_min_temp:.4f} K)")
    print(f"AI Average at Metal Contacts:      {np.mean(ai_boundary_temps):.4f} K")
    print(f"AI Min/Max at Metal Contacts:      {boundary_min:.4f} K / {boundary_max:.4f} K")
    print(f"Boundary Error (MAE):              {boundary_mae:.4f} K")
    print("==================================================\n")
else:
    print("Error: Tolerance too strict. Could not locate boundary nodes in the TCAD mesh.")