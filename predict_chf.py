import pandas as pd
import numpy as np
import joblib

FEATURES = [
    "P_r", "x_in", "Re_fo", "Re_go", "Pr_f", "Pr_g",
    "We_fo", "We_go", "Fr_fo", "Fr_go", "Su_f", "Su_g",
    "Bd", "visc_ratio", "Ja", "rho_ratio", "Ld_ratio", "theta",
    "Fluid_Helium", "Fluid_Methane", "Fluid_Nitrogen", "Fluid_Hydrogen",
]

def predict_chf(input_data: dict, q_physics: float) -> float:
    """
    Predict Critical Heat Flux using the Extra Trees PAML model.

    PAML reconstruction:
        CHF = expm1( log1p(q_physics) + model.predict(X) )

    Args:
        input_data: dict with keys matching FEATURES (22 dimensionless groups +
                    one-hot fluid flags).  Values are scalars or single-element lists.
        q_physics:  Physics-correlation CHF baseline [W/m²] (clip to ≥ 1.0).

    Returns:
        Predicted CHF [W/m²].
    """
    model = joblib.load("et_paml_model.pkl")
    q = max(float(q_physics), 1.0)
    df = pd.DataFrame({k: [v] if not isinstance(v, list) else v for k, v in input_data.items()})
    log_residual = model.predict(df[FEATURES])[0]
    return float(np.expm1(np.log1p(q) + log_residual))


def load_and_predict():
    """Demo: Liquid Helium in Microgravity."""
    input_data = {
        "P_r":            0.15,
        "x_in":           0.05,
        "Re_fo":          15000,
        "Re_go":          85000,
        "Pr_f":           0.95,
        "Pr_g":           0.85,
        "We_fo":          250.0,
        "We_go":          1200.0,
        "Fr_fo":          45.0,
        "Fr_go":          300.0,
        "Su_f":           1.2e6,
        "Su_g":           4.5e5,
        "Bd":             0.005,
        "visc_ratio":     8.5,
        "Ja":             0.12,
        "rho_ratio":      7.2,
        "Ld_ratio":       50.0,
        "theta":          0.0,
        "Fluid_Helium":   1,
        "Fluid_Methane":  0,
        "Fluid_Nitrogen": 0,
        "Fluid_Hydrogen": 0,
    }
    q_physics_baseline = 15000.0

    chf = predict_chf(input_data, q_physics_baseline)
    log_res = np.log(chf / q_physics_baseline)

    print("-" * 45)
    print("Cryogenic CHF Prediction (Extra Trees PAML)")
    print("-" * 45)
    print(f"Fluid:              Liquid Helium")
    print(f"Physics Baseline:   {q_physics_baseline:.2f} W/m²")
    print(f"PAML Multiplier:    {np.exp(log_res):.4f}")
    print(f"Predicted CHF:      {chf:.2f} W/m²")
    print("-" * 45)


if __name__ == "__main__":
    load_and_predict()
