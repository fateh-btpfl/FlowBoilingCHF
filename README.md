# Physics-Assisted Machine Learning for Cryogenic Flow Boiling Critical Heat Flux

Trained model and inference code for the Physics-Assisted Extra Trees predictor
of flow boiling critical heat flux (CHF) in liquid helium, hydrogen, nitrogen and
methane, across terrestrial and reduced gravity.

Reference: *Comparative Analysis of Physics-Assisted Machine Learning Frameworks
for Predicting Cryogenic Flow Boiling Critical Heat Flux under Terrestrial,
Martian, Lunar and Microgravity*, International Journal of Heat and Mass
Transfer (HMT-D-26-03728).

This repository carries the trained model and the code needed to run it. The
underlying CHF database is the PU-BTPFL compilation and is not distributed here;
the sources it draws on are cited in the paper.

## 1. What the model predicts

The model does not predict CHF directly. It predicts a logarithmic residual on a
baseline empirical correlation,

    r     = log1p(q_CHF) - log1p(q_physics)

and the prediction is reconstructed as

    q_CHF = expm1( log1p(q_physics) + r_hat )

where `log1p(x) = ln(1 + x)` and `expm1(x) = exp(x) - 1`. A baseline correlation
value `q_physics` in W/m2 must therefore be supplied with every prediction. The
baseline used in training is the Ganesan et al. correlation, with values below
1 W/m2 clipped to 1 W/m2. Substituting a different correlation changes what the
residual means and is not supported.

## 2. Contents

| File | Description |
|---|---|
| `predict_chf.py` | Inference: single conditions, batches, or a CSV |
| `et_paml_model.pkl` | Trained Extra Trees ensemble, 897 trees, fitted on all 2438 terrestrial datapoints |
| `model_card.json` | Hyperparameters, input list, training set size and held-out accuracy |
| `requirements.txt` | Pinned dependencies |

## 3. Installation

Python 3.9 or later.

```bash
pip install -r requirements.txt
```

The serialized model is a scikit-learn estimator. It depends on scikit-learn
alone, but pickles are version-sensitive; install the pinned versions if the
model fails to load.

## 4. Usage

Run the worked example, a measured microgravity nitrogen condition:

```bash
python predict_chf.py
```

Predict a batch from a CSV holding the 19 inputs plus a `q_physics` column:

```bash
python predict_chf.py --csv conditions.csv --out predictions.csv
```

Or from Python:

```python
from predict_chf import predict_chf

conditions = {
    "P_r": 0.121823, "x_in": -0.031021, "Re_fo": 73227.43, "Re_go": 1078712.55,
    "Pr_f": 1.810014, "Pr_g": 0.939380, "We_fo": 1414.788, "We_go": 60580.293,
    "Fr_fo": 156550.27, "Fr_go": 287034253.4, "Su_f": 3790147.2, "Su_g": 19207909.0,
    "Bd": 0.0088262, "visc_ratio": 14.730991, "Ja": 0.0, "rho_ratio": 42.819332,
    "Ld_ratio": 80.0, "theta": 0.0, "g_ratio": 1.0e-4,
}
chf = predict_chf(conditions, q_physics=107560.0)   # W/m2
```

## 5. Inputs

Nineteen dimensionless inputs, in this order. Fluid identity is **not** an
input: the four cryogens are distinguished by their property groups alone, so
no fluid flag is set anywhere.

| Input | Description |
|---|---|
| `P_r` | Reduced pressure, P / P_crit |
| `x_in` | Thermodynamic equilibrium quality at inlet |
| `Re_fo`, `Re_go` | Liquid-only and vapor-only Reynolds numbers |
| `Pr_f`, `Pr_g` | Liquid and vapor Prandtl numbers |
| `We_fo`, `We_go` | Liquid-only and vapor-only Weber numbers |
| `Fr_fo`, `Fr_go` | Liquid-only and vapor-only Froude numbers |
| `Su_f`, `Su_g` | Liquid and vapor Suratman numbers |
| `Bd` | Bond number |
| `visc_ratio` | Dynamic viscosity ratio, liquid / vapor |
| `rho_ratio` | Density ratio, liquid / vapor |
| `Ja` | Jakob number |
| `Ld_ratio` | Heated length to diameter ratio, L_H / D |
| `theta` | Flow orientation angle, degrees; 0 horizontal, 90 vertical upflow |
| `g_ratio` | Local gravitational acceleration divided by 9.81 m/s2 |

Two points govern whether a reduced gravity prediction is meaningful:

* **The Froude and Bond numbers must be evaluated at the local gravitational
  acceleration**, not at 9.81 m/s2. A microgravity condition and a terrestrial
  condition that are otherwise identical must therefore differ in `Fr_fo`,
  `Fr_go` and `Bd` as well as in `g_ratio`.
* `g_ratio` carries the gravity level explicitly, as the ratio g / 9.81. Use
  1.0 for terrestrial, 0.38 for Martian, 0.166 for Lunar, and a small value such
  as 1e-4 for microgravity.

Thermophysical properties are evaluated at saturation conditions. The values
used in training came from CoolProp.

## 6. Accuracy and range of validity

On a held-out 20% test partition drawn at random from the database, the model
attains a mean absolute percentage error of 13.93%, against 25.61% for the
Ganesan et al. correlation on the same data, with 88.7% of predictions within
30% of measurement.

That figure applies to conditions of the kind the database represents. Accuracy
measured under stricter protocols in the paper is lower, and these bounds should
be read before the model is relied upon:

| Situation | MAPE | Correlation, same test |
|---|---|---|
| Conditions and campaigns represented in the database | 13.9% | 25.6% |
| An experimental campaign not represented in the database | 27.3% | 27.2% |
| A cryogen absent from the database | 45.7% | 27.2% |
| The 11 measured reduced gravity conditions, none seen in training | 9.3% | 7.5% |

In short: the model improves substantially on the correlation within the range
of conditions the database covers, is no better than the correlation for a new
experimental campaign, and should not be used for a cryogen other than the four
it was trained on.

The training data span the following conditions. Predictions outside them are
extrapolation.

| Parameter | Range |
|---|---|
| Fluids | LN2 (968 points), LH2 (729), LHe (688), LCH4 (53) |
| Inlet pressure | 8.35 kPa to 4.08 MPa |
| Mass flux | 2.15 to 8204 kg/m2s |
| Tube diameter | 0.47 to 14.1 mm |
| Heated length | 0.02 to 1.65 m |
| Inlet quality | -2.06 to 0.95 |
| Orientation | 0 to 315 degrees |
| Measured CHF | 46.1 W/m2 to 8.20 MW/m2 |

The model returns a point estimate with no uncertainty interval. The paper finds
that the predictive intervals of the distributional models are over-confident,
so no interval is published here; a design margin must be set by the user.

## 7. Citation

```bibtex
@article{Mehmood2026CryoCHF,
  title   = {Comparative Analysis of Physics-Assisted Machine Learning Frameworks
             for Predicting Cryogenic Flow Boiling Critical Heat Flux under
             Terrestrial, Martian, Lunar and Microgravity},
  author  = {Mehmood, Muhammad Fateh and Prinster, Owen Mee Hyang and Kim, Sunjae
             and Damle, Nishad and Hartwig, Jason and Mudawar, Issam},
  journal = {International Journal of Heat and Mass Transfer},
  year    = {2026}
}
```

## 8. License

MIT, see `LICENSE`.
