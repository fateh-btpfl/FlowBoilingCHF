import pandas as pd
import numpy as np
import joblib

def load_and_predict():
    """
    Loads the trained Extra Trees PAML model and predicts the Critical Heat Flux 
    for a given set of cryogenic flow boiling parameters.
    """
    # 1. Load the pre-trained Extra Trees model
    model_path = 'et_paml_model.pkl'
    try:
        et_model = joblib.load(model_path)
        print("Model loaded successfully.\n")
    except FileNotFoundError:
        print(f"Error: Could not find '{model_path}'. Please ensure it is in the same directory.")
        return

    # 2. Define the input parameters (Example: Liquid Helium in Microgravity)
    # These must exactly match the features used during training.
    input_data = {
        'P_r': [0.15],
        'x_in': [0.05],
        'Re_fo': [15000],
        'Re_go': [85000],
        'Pr_f': [0.95],
        'Pr_g': [0.85],
        'We_fo': [250.0],
        'We_go': [1200.0],
        'Fr_fo': [45.0],
        'Fr_go': [300.0],
        'Su_f': [1.2e6],
        'Su_g': [4.5e5],
        'Bd': [0.005],           # Very low Bond number for microgravity/small diameter
        'visc_ratio': [8.5],
        'Ja': [0.12],
        'rho_ratio': [7.2],
        'Ld_ratio': [50.0],
        'theta': [0.0],          # Horizontal
        'Fluid_Helium': [1],     # One-hot encoded fluid flags
        'Fluid_Methane': [0],
        'Fluid_Nitrogen': [0],
        'Fluid_Hydrogen': [0]
    }

    # The baseline physics prediction (e.g., from Ganesan or Shah)
    # Clipped to 1.0 to avoid log(0) errors, matching training methodology
    q_physics_baseline = max(15000.0, 1.0) 

    # Convert to DataFrame
    df_input = pd.DataFrame(input_data)

    # 3. Predict the Log-Residual
    # The ET model predicts: log(Actual_CHF) - log(Physics_CHF)
    predicted_log_residual = et_model.predict(df_input)[0]

    # 4. PAML Reconstruction Step
    # Reconstruct real CHF: CHF = Physics * exp(log_residual)
    final_chf = q_physics_baseline * np.exp(predicted_log_residual)

    # 5. Output Results
    print("-" * 40)
    print("Cryogenic CHF Prediction Results")
    print("-" * 40)
    print(f"Fluid:             Liquid Helium")
    print(f"Physics Baseline:  {q_physics_baseline:.2f} W/m^2")
    print(f"PAML Multiplier:   {np.exp(predicted_log_residual):.4f}")
    print(f"Final Predicted CHF: {final_chf:.2f} W/m^2")
    print("-" * 40)

if __name__ == "__main__":
    load_and_predict()