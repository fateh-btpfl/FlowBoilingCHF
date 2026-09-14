"""
Physics-Assisted Extra Trees predictor for cryogenic flow boiling critical heat
flux.

The model predicts a logarithmic residual on a baseline correlation rather than
CHF itself:

    r     = log1p(CHF) - log1p(q_physics)
    CHF   = expm1( log1p(q_physics) + r_hat )

so a baseline correlation value must be supplied alongside the 19 dimensionless
inputs. See README.md for the definition of each input and for how the Froude
and Bond numbers are to be evaluated.

Reference: IJHMT HMT-D-26-03728.
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "et_paml_model.pkl")
CARD_PATH = os.path.join(HERE, "model_card.json")

# The 19 inputs, in the order the model expects. Fluid identity is NOT an input:
# the model is applied to any of the four cryogens through the dimensionless
# groups alone.
FEATURES = [
    "P_r", "x_in", "Re_fo", "Re_go", "Pr_f", "Pr_g", "We_fo", "We_go",
    "Fr_fo", "Fr_go", "Su_f", "Su_g", "Bd", "visc_ratio", "Ja", "rho_ratio",
    "Ld_ratio", "theta", "g_ratio",
]

_MODEL = None


def load_model():
    """Load the trained ensemble once and keep it for subsequent calls."""
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"{os.path.basename(MODEL_PATH)} not found next to this script. "
                "Download it from the repository release and place it here.")
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def _frame(inputs):
    """Accept a dict of scalars, a list of dicts, or a DataFrame. Returns the
    feature matrix in model order, having checked that nothing is missing."""
    if isinstance(inputs, pd.DataFrame):
        df = inputs.copy()
    elif isinstance(inputs, dict):
        df = pd.DataFrame({k: np.atleast_1d(v) for k, v in inputs.items()})
    else:
        df = pd.DataFrame(list(inputs))
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise KeyError(f"missing inputs: {', '.join(missing)}")
    if df[FEATURES].isna().any().any():
        bad = df[FEATURES].columns[df[FEATURES].isna().any()].tolist()
        raise ValueError(f"inputs contain missing values: {', '.join(bad)}")
    return df[FEATURES]


def predict_chf(inputs, q_physics):
    """Predict CHF in W/m2.

    Args:
        inputs:    dict, list of dicts, or DataFrame carrying the 19 features.
        q_physics: baseline correlation CHF in W/m2, scalar or array of matching
                   length. Values below 1 W/m2 are clipped, as in training.

    Returns:
        float for a single input, otherwise a numpy array.
    """
    X = _frame(inputs)
    q = np.clip(np.atleast_1d(np.asarray(q_physics, dtype=float)), 1.0, None)
    if len(q) == 1 and len(X) > 1:
        q = np.repeat(q, len(X))
    if len(q) != len(X):
        raise ValueError(f"q_physics has length {len(q)} but {len(X)} rows were given")
    residual = load_model().predict(X)
    chf = np.expm1(np.log1p(q) + residual)
    return float(chf[0]) if len(chf) == 1 else chf


def predict_csv(path, out=None):
    """Predict for every row of a CSV holding the 19 features plus q_physics."""
    df = pd.read_csv(path)
    if "q_physics" not in df.columns:
        raise KeyError("the CSV must carry a q_physics column in W/m2")
    df["CHF_predicted"] = predict_chf(df, df["q_physics"].to_numpy())
    dest = out or os.path.splitext(path)[0] + "_predicted.csv"
    df.to_csv(dest, index=False)
    print(f"wrote {dest} ({len(df)} rows)")
    return df


# A single measured microgravity LN2 condition, carried as a worked example so a
# new user can check that their installation reproduces a known result.
DEMO = {
    "P_r": 0.121823, "x_in": -0.031021, "Re_fo": 73227.43, "Re_go": 1078712.55,
    "Pr_f": 1.810014, "Pr_g": 0.939380, "We_fo": 1414.788, "We_go": 60580.293,
    "Fr_fo": 156550.27, "Fr_go": 287034253.4, "Su_f": 3790147.2, "Su_g": 19207909.0,
    "Bd": 0.0088262, "visc_ratio": 14.730991, "Ja": 0.0, "rho_ratio": 42.819332,
    "Ld_ratio": 80.0, "theta": 0.0, "g_ratio": 1.0e-4,
}
DEMO_Q_PHYSICS = 107560.0
DEMO_MEASURED = 98775.3


def demo():
    chf = predict_chf(DEMO, DEMO_Q_PHYSICS)
    print("-" * 58)
    print("Cryogenic CHF prediction, Physics-Assisted Extra Trees")
    print("-" * 58)
    print(f"Condition            LN2, microgravity (g/ge = {DEMO['g_ratio']:.0e})")
    print(f"Baseline correlation {DEMO_Q_PHYSICS:12,.1f} W/m2")
    print(f"PAML multiplier      {chf / DEMO_Q_PHYSICS:12.4f}")
    print(f"Predicted CHF        {chf:12,.1f} W/m2")
    print(f"Measured CHF         {DEMO_MEASURED:12,.1f} W/m2")
    print(f"Relative error       {100 * (chf - DEMO_MEASURED) / DEMO_MEASURED:12.2f} %")
    print("-" * 58)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="CSV of conditions to predict, one row each")
    ap.add_argument("--out", help="where to write the predictions")
    ap.add_argument("--card", action="store_true", help="print the model card and exit")
    a = ap.parse_args()
    if a.card:
        print(json.dumps(json.load(open(CARD_PATH)), indent=2))
    elif a.csv:
        predict_csv(a.csv, a.out)
    else:
        demo()


if __name__ == "__main__":
    main()
