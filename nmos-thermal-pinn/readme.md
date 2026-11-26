# 2D nMOS Thermal Surrogate Model (PINN)

A 2D Physics-Informed Neural Network (PINN) built in PyTorch to predict steady-state spatial lattice temperature distributions in an nMOS transistor across unseen drain voltages ($V_d$), trained on stitched Synopsys Sentaurus TCAD datasets.

## Overview
Running full 3D or dense multi-voltage thermal simulations in Sentaurus TCAD is computationally expensive, and exporting raw spatial meshes directly from Sentaurus Visual is limited to 1D/2D slices. This project solves both problems with an end-to-end TCAD-to-PyTorch pipeline:
1. **TCAD Slice Stitching:** Extracts 1D vertical thermal slices from 49 Sentaurus TCAD simulations ($V_d \in [0.1\text{ V}, 3.0\text{ V}]$) and stitches them into complete 2D device meshes in Python.
2. **Physics-Informed Training:** Trains a 3-input neural network $(X, Y, V_d) \rightarrow \Delta T$ using a hybrid loss combining TCAD ground-truth data, a continuous 2D Fourier heat-diffusion PDE residual (`torch.autograd`), and 300 K metal contact boundary constraints.
3. **Instant Inference:** Loads saved weights (`thermal_pinn_brain.pth`) to predict full 2D heat maps and peak hotspot temperatures for unseen drain voltages in milliseconds.

## Repository Structure
```text
nmos-thermal-pinn/
├── data/
│   └── sample_n59_full_2D.csv       # Sample stitched 2D TCAD mesh
├── results/
│   ├── tcad_vs_pinn_comparison.png  # 3-panel validation plot (TCAD vs PINN vs Error)
│   └── unseen_1.85V_prediction.png  # Predicted thermal map for unseen Vd = 1.85V
├── src/
│   ├── stitch_tcad_slices.py        # Stitches Sentaurus vertical slices into 2D CSVs
│   ├── train_pinn.py                # Trains the (X, Y, Vd) PINN with Fourier PDE loss
│   ├── predict_voltage.py           # Runs instant inference for any unseen Vd
│   └── validate_metrics.py          # Computes MAE, max error, and boundary error
├── weights/
│   └── thermal_pinn_brain.pth       # Trained PyTorch model weights
├── .gitignore
├── requirements.txt
└── README.md
