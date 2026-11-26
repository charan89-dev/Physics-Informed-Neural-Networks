import os
import glob
import pandas as pd

# 1. Point to your extracted folder on Windows
data_dir = r"C:\PINN\pinn_csv_data"

# Find every single CSV file in the folder
all_files = glob.glob(os.path.join(data_dir, "n*_y*.csv"))

# 2. Extract the unique simulation names (e.g., 'n59', 'n1026', 'n1480')
prefixes = set()
for f in all_files:
    basename = os.path.basename(f)
    # Split at '_y' to get the prefix before the coordinate
    prefix = basename.split('_y')[0]
    prefixes.add(prefix)

print(f"Found {len(prefixes)} distinct simulations to process.")

# 3. Loop through every single simulation and stitch its slices
for target_sim in prefixes:
    slice_files = glob.glob(os.path.join(data_dir, f"{target_sim}_y*.csv"))
    
    frames = []
    for file in slice_files:
        # Extract the Y coordinate from the filename
        filename = os.path.basename(file)
        y_str = filename.replace(f"{target_sim}_y", "").replace(".csv", "")
        y_val = float(y_str)
        
        # Read the CSV, strictly skipping row index 1 (the 'none,SELECTED' string row)
        df = pd.read_csv(file, skiprows=[1])
        
        # Inject the Y coordinate as a new column
        df['Y'] = y_val
        frames.append(df)
        
    # Stitch all slices for this specific simulation together
    if frames:
        full_2d_mesh = pd.concat(frames, ignore_index=True)
        
        # Reorder columns so X and Y are neatly at the front
        cols = ['X', 'Y'] + [c for c in full_2d_mesh.columns if c not in ['X', 'Y']]
        full_2d_mesh = full_2d_mesh[cols]
        
        # Save the perfectly stitched 2D mesh
        output_file = os.path.join(data_dir, f"{target_sim}_full_2D.csv")
        full_2d_mesh.to_csv(output_file, index=False)
        print(f"Saved: {target_sim}_full_2D.csv | Nodes: {len(full_2d_mesh)}")

print("\nAll simulations successfully stitched and ready for PyTorch!")