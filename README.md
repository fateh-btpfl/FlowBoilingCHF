# Physics-Assisted Machine Learning (PAML) Framework for Cryogenic Flow Boiling Critical Heat Flux

## 1. Overview
This repository contains the trained Extra Trees Machine Learning ensemble and the associated inference script for predicting Critical Heat Flux (CHF) in cryogenic flow boiling. 

Standard empirical correlations and purely data-driven Machine Learning (ML) models frequently fail to generalize across diverse thermophysical fluid properties and varying gravitational environments. To address this limitation, this repository implements a Physics-Assisted Machine Learning (PAML) framework. The provided model universally scales across multiple cryogenic fluids (Liquid Nitrogen, Liquid Helium, Liquid Hydrogen, and Liquid Methane) and gravity regimes (ranging from Terrestrial to Microgravity) by predicting the residual error of a baseline physics correlation.

## 2. Mathematical Framework
Unlike direct ML prediction methodologies that map fluid properties directly to a dimensional heat flux, this PAML architecture is designed to learn the non-linear discrepancies between experimental data and established physical laws.

During training, the target variable was defined as the logarithmic residual between the measured CHF and a baseline physical prediction (e.g., the Ganesan or Shah correlation):

    y_target = ln(q''_CHF,actual) - ln(q''_CHF,physics)

Consequently, the Extra Trees model contained in this repository predicts this dimensionless log-residual. The provided Python script automatically performs the necessary mathematical reconstruction to yield the final dimensional CHF value using the following relation:

    q''_CHF,final = q''_CHF,physics * exp(y_predicted)

## 3. Repository Contents
* `predict_chf.py`: The primary execution script that loads the model, defines the input parameter array, evaluates the log-residual, and reconstructs the final CHF.
* `et_paml_model.pkl`: The serialized (trained) Extra Trees regression model containing the ensemble architecture and nodal weights.

## 4. System Requirements and Installation
The framework is built using standard Python data science libraries. A Python environment (version 3.8 or higher) is required.

Install the necessary dependencies via pip:

```bash
pip install scikit-learn pandas numpy joblib

```

## 5. Input Parameters

To execute a prediction, the user must define the thermophysical and system state of the flow boiling system. The input array must strictly adhere to the following 22 features, ordered exactly as expected by the model.

Note: Gravitational acceleration (g) is not provided as an explicit input, as its influence is fundamentally captured via the relevant dimensionless force balances (Froude and Bond numbers).

### Thermodynamic and Operating Conditions

* `P_r`: Reduced pressure (P/P_crit)
* `x_in`: Thermodynamic equilibrium quality at the inlet
* `Ja`: Jakob number
* `theta`: Flow orientation angle (Degrees; e.g., 0.0 for horizontal, 90.0 for vertical upward)

### Dimensionless Force and Property Ratios

* `Re_fo`: Liquid-only Reynolds number
* `Re_go`: Vapor-only Reynolds number
* `Pr_f`: Liquid Prandtl number
* `Pr_g`: Vapor Prandtl number
* `We_fo`: Liquid-only Weber number
* `We_go`: Vapor-only Weber number
* `Fr_fo`: Liquid-only Froude number
* `Fr_go`: Vapor-only Froude number
* `Su_f`: Liquid Suratman number
* `Su_g`: Vapor Suratman number
* `Bd`: Bond number
* `visc_ratio`: Dynamic viscosity ratio (Liquid / Vapor)
* `rho_ratio`: Density ratio (Liquid / Vapor)
* `Ld_ratio`: Heated length-to-diameter ratio (L_H / D)

### Categorical Fluid Identifiers (One-Hot Encoded)

The user must activate the specific working fluid by assigning a value of `1` to the target fluid and `0` to the remaining three.

* `Fluid_Helium`
* `Fluid_Methane`
* `Fluid_Nitrogen`
* `Fluid_Hydrogen`

### Baseline Physics Prediction

* `q_physics_baseline`: The CHF value (in W/m^2) calculated from the chosen baseline empirical correlation for the given flow conditions.

## 6. Usage Instructions

1. Ensure `predict_chf.py` and `et_paml_model.pkl` are located in the same working directory.
2. Open `predict_chf.py` in a standard text editor or IDE.
3. Locate the `input_data` dictionary within the script and modify the scalar values to reflect your specific flow boiling conditions.
4. Update the `q_physics_baseline` variable with your calculated baseline correlation value.
5. Execute the script via the command line:

```bash
python predict_chf.py

```

The script will output the specified flow conditions, the baseline prediction, the PAML multiplier, and the final corrected CHF value in standard SI units (W/m^2).

