import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# 1. Define the exact same brain architecture so PyTorch knows how to load it
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

# 2. CHOOSE YOUR UNSEEN VOLTAGE HERE
# You can change this to any number between 0.1 and 3.0!
target_vd = 1.85  
print(f"--- Simulating Unseen State: Vd = {target_vd}V in milliseconds ---")

# 3. Quickly grab the normalization limits from the raw data
all_files = glob.glob("./pinn_csv_data/n*_full_2D.csv")
df_list = [pd.read_csv(f) for f in all_files]
full_df = pd.concat(df_list, ignore_index=True)

x_min, x_max = full_df["X"].min(), full_df["X"].max()
y_min, y_max = full_df["Y"].min(), full_df["Y"].max()
T_min, T_max = full_df["LatticeTemperature"].min(), full_df["LatticeTemperature"].max()
vd_min, vd_max = 0.1, 3.0

# 4. Load the pre-trained brain
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ThermalPINN().to(device)
model.load_state_dict(torch.load("thermal_pinn_brain.pth", map_location=device, weights_only=True))
model.eval()

# 5. Build the artificial 1.85V input tensor based on the physical mesh
df_single = pd.read_csv("./pinn_csv_data/n59_full_2D.csv")
x_phys = df_single["X"].to_numpy()
y_phys = df_single["Y"].to_numpy()
silicon_mask = (np.abs(df_single["DopingConcentration"].to_numpy()) > 1e10).astype(int)

x_norm = (x_phys - x_min) / (x_max - x_min)
y_norm = (y_phys - y_min) / (y_max - y_min)
vd_norm = np.full_like(x_norm, (target_vd - vd_min) / (vd_max - vd_min))

xyv_tensor = torch.tensor(np.stack([x_norm, y_norm, vd_norm], axis=1), dtype=torch.float32).to(device)

# =========================================================================
# 6. INSTANT PREDICTION (No training, no epochs)
# =========================================================================
with torch.no_grad():
    T_pred_norm = model(xyv_tensor).cpu().squeeze().numpy()

# Denormalize the temperature back to Kelvin
T_pred_phys = T_pred_norm * (T_max - T_min) + T_min

print(f"\n===========================================")
print(f"AI PREDICTION RESULTS FOR Vd = {target_vd}V")
print(f"Absolute Peak Temperature: {np.max(T_pred_phys):.5f} K")
print(f"===========================================\n")

# =========================================================================
# 7. DRAW THE DEVICE (SENTAURUS TCAD STYLE)
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# A. Swap Y and X to match TCAD's orientation. 
# B. Use 'jet' colormap for the Blue->Red rainbow.
# C. Set vmin=300.0 so the 300K boundary edges are forced to deep blue.
# Replace the contour line in predict.py:
contour = ax.tricontourf(y_phys, x_phys, T_pred_phys, levels=150, cmap="jet", 
                         vmin=np.min(T_pred_phys), vmax=np.max(T_pred_phys))
fig.colorbar(contour, ax=ax, label="Predicted Lattice Temperature (K)")

# Draw solid black borders around the Silicon/Oxide regions to match SVisual
ax.tricontour(y_phys, x_phys, silicon_mask, levels=[0.5], colors='black', linewidths=1.5)

# Invert the vertical axis so X=0 is at the top, just like Sentaurus
ax.invert_yaxis()

# Add text labels mapped to the newly rotated coordinates
ax.text(-0.05, 0.030, "SOURCE", color='black', fontweight='bold', fontsize=11, ha='center')
ax.text(-0.05, 0.250, "DRAIN\n(Hotspot)", color='white', fontweight='bold', fontsize=11, ha='center')
ax.text(0.06, 0.130, "GATE &\nOXIDE", color='black', fontweight='bold', fontsize=11, ha='center')

ax.set_xlabel("Y (physical units)")
ax.set_ylabel("X (physical units)")
ax.set_title(f"PINN Surrogate (TCAD Style) — Unseen Vd = {target_vd}V")

# Force the aspect ratio to be physically accurate
ax.set_aspect("equal")
plt.tight_layout()

# Save the high-res image
plt.savefig(f"thermal_pinn_tcad_style.png", dpi=300)
print(f"TCAD-style image saved to thermal_pinn_tcad_style.png")