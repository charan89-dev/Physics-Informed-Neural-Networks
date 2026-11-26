2D nMOS Thermal Surrogate Model (PINN)A PyTorch-based Physics-Informed Neural Network (PINN) surrogate for predicting steady-state 2D lattice temperature distributions in nMOS transistors across arbitrary drain voltages ($V_d$). Trained on stitched Synopsys Sentaurus TCAD datasets, the model embeds 2D Fourier heat diffusion physics directly into its loss function to enable millisecond inference for unseen operating voltages.How It Was Built (Engineering Workflow)Direct full-mesh point extraction from Sentaurus TCAD (both 3D and 2D) presents export limitations when building continuous ML datasets. To overcome this, the project uses an end-to-end automated pipeline:TCAD Simulation & Slice Extraction:Simulated a 2D nMOS device in Sentaurus Workbench across 49 operating/structural variations covering 7 drain voltages ($V_d \in \{0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0\}\text{ V}$). Vertical 1D cuts were exported as CSV slices containing spatial coordinates ($X, Y$), DopingConcentration, and LatticeTemperature.2D Mesh Stitching (src/stitch_tcad_slices.py):A custom Python script parses the raw slice filenames to inject the $Y$-coordinate and stitches all slices back into 49 complete 2D mesh datasets (n*_full_2D.csv).Physics-Informed Neural Network (src/train_pinn.py):Architecture: 5-layer MLP (50 neurons/layer, Tanh activations) mapping normalized $(X, Y, V_d) \rightarrow \Delta T$.$\Delta T$ Baseline Shift: Temperatures are shifted relative to the $300.0\text{ K}$ ambient baseline ($\Delta T = T - 300.0\text{ K}$) to improve gradient sensitivity.Hybrid Physics Loss: Combines Data MSE, Metal Contact Boundary Loss ($\Delta T < 0.05\text{ K}$ mask), and the 2D steady-state Fourier heat diffusion PDE residual computed via torch.autograd:
$$\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0$$
The PDE residual is masked strictly to the active silicon region ($\vert{}\text{DopingConcentration}\vert{} > 10^{10}$) and scaled via the chain rule to correct for coordinate anisotropy between physical $X$ and $Y$ spans.Unseen Voltage Inference & Validation (src/predict_voltage.py, src/validate_metrics.py):The trained weights (weights/thermal_pinn_brain.pth) predict full 2D thermal maps for unseen voltages (e.g., $V_d = 1.85\text{ V}$) in milliseconds without re-running TCAD, and quantify node-by-node accuracy against Sentaurus ground truth.Repository Structurenmos-thermal-pinn/
├── data/
│   └── sample_n59_full_2D.csv       # Sample stitched 2D TCAD mesh
├── results/
│   ├── tcad_vs_pinn_comparison.png  # 3-panel TCAD vs. PINN vs. Error comparison
│   └── unseen_1.85V_prediction.png  # Predicted 2D thermal map for unseen Vd = 1.85V
├── src/
│   ├── stitch_tcad_slices.py        # Stitches 1D TCAD slices into 2D CSV meshes
│   ├── train_pinn.py                # Trains the (X, Y, Vd) PINN and saves weights
│   ├── predict_voltage.py           # Runs instant inference for any unseen Vd
│   └── validate_metrics.py          # Computes MAE, Max Error, and Boundary Error
├── weights/
│   └── thermal_pinn_brain.pth       # Pre-trained PyTorch model checkpoint
├── .gitignore
├── requirements.txt
└── README.md
Validation ResultsThe model was evaluated on unseen voltage inference ($V_d = 1.85\text{ V}$) and quantitatively benchmarked against Sentaurus TCAD ground truth at $V_d = 2.5\text{ V}$:MetricValueNotesUnseen Voltage Prediction ($V_d = 1.85\text{ V}$)$303.0858\text{ K}$Peak predicted lattice temperature (instant inference)Overall MAE ($V_d = 2.5\text{ V}$)$3.1898\text{ K}$Node-wise Mean Absolute Error across the 2D meshOverall Max Error ($V_d = 2.5\text{ V}$)$4.0509\text{ K}$Maximum pointwise deviation from TCADContact Boundary MAE ($V_d = 2.5\text{ V}$)$3.0539\text{ K}$Target contact temp: $300.1227\text{ K}$ (AI avg: $303.1766\text{ K}$)Visual Outputs(Add results/tcad_vs_pinn_comparison.png here once generated from train_pinn.py.)How to Use1. Setup EnvironmentClone the repository and install the required Python packages:git clone https://github.com/charan89-dev/nmos-thermal-pinn.git
cd nmos-thermal-pinn
pip install -r requirements.txt
2. Predict an Unseen Voltage (Quick Start)You do not need to retrain the model to run inference. Open src/predict_voltage.py, set target_vd to any value between $0.1\text{ V}$ and $3.0\text{ V}$ (default is 1.85), and run:python src/predict_voltage.py
This loads weights/thermal_pinn_brain.pth, prints the predicted peak temperature, and saves a TCAD-style 2D contour plot.3. Train From Scratch (Optional)If you have the stitched 2D CSV files inside ./pinn_csv_data/:# Step 1: Stitch raw Sentaurus 1D slices (only if starting from raw slice exports)
python src/stitch_tcad_slices.py

# Step 2: Train the PINN (5,000 epochs) and generate the 3-panel validation plot
python src/train_pinn.py

# Step 3: Run numerical error metrics against TCAD ground truth
python src/validate_metrics.py
Known Limitations & RoadmapBoundary Drift & Joule Heating: Pure Laplace regularization ($\nabla^2 T = 0$) suppresses localized drain Joule heating ($Q > 0$), and high ambient-to-rise ratios can cause baseline drift at $300\text{ K}$ contacts.Next Phase: Extending the surrogate from steady-state 2D $(X, Y, V_d)$ to a 4D Spatiotemporal Transient Solver $(X, Y, Z, t, V_d)$ using transient 3D Sentaurus TCAD datasets to model live thermal wave propagation.
