# Smart EV Battery Management Using Adaptive Neuro-Fuzzy Inference Systems (ANFIS)

## 📌 Project Overview
Electric Vehicles (EVs) rely heavily on accurate Battery Management Systems (BMS) to optimize battery performance, guarantee operational safety, prevent overcharging/deep discharging, and maximize pack lifespan. Estimating critical battery metrics such as **State of Charge (SOC)** and **State of Health (SOH)** is notoriously challenging due to non-linear electrochemical dynamics, temperature sensitivity, aging effects, and unpredictable driver load profiles.

This BTech Capstone project introduces an intelligent soft-computing framework utilizing **Adaptive Neuro-Fuzzy Inference Systems (ANFIS)**, combining the transparent rule-based reasoning of Fuzzy Logic with the learning capabilities of Artificial Neural Networks. To establish a rigorous academic benchmark, the ANFIS approach is benchmarked against established Machine Learning baselines: **Artificial Neural Networks (ANN / Multi-Layer Perceptron)** and **Random Forest**.

Finally, the project features an interactive desktop dashboard built with **Flet** allowing real-time battery telemetry simulation, parameter tuning, model comparison, and visual diagnostics.

---

## 🎯 Key Objectives
1. **Accurate State Estimation**: Predict battery State of Charge (SOC) and State of Health (SOH) across varying temperatures, currents, and discharge cycles.
2. **Soft Computing Synergy**: Implement ANFIS to leverage linguistic fuzzy rules combined with data-driven backpropagation/hybrid parameter tuning.
3. **Comparative Benchmarking**: Systematically evaluate ANFIS against classical machine learning baselines (Random Forest, ANN) using standardized regression metrics ($RMSE$, $MAE$, $R^2$, $MAPE$).
4. **Interactive BMS Monitoring**: Deliver an intuitive, responsive **Flet** dashboard displaying live battery telemetry, pack health, remaining range, and comparative model predictions.
5. **Academic Rigor**: Maintain a reproducible, modular, test-driven codebase adhering to software engineering and academic research best practices.

---

## 🛠️ Technology Stack
- **Language**: Python (3.10+)
- **Fuzzy & Soft Computing**: Adaptive Neuro-Fuzzy Inference System (ANFIS), Takagi-Sugeno-Kang (TSK) fuzzy models
- **Machine Learning**: Scikit-Learn (Random Forest Regressor, Multi-Layer Perceptron / ANN)
- **Data Analysis & Scientific Computing**: NumPy, Pandas, SciPy
- **Data Visualization**: Matplotlib
- **Model Persistence**: Joblib
- **User Interface / Dashboard**: Flet (Flutter-powered Python framework)
- **Testing**: Pytest / Unittest

---

## 🏗️ Project Architecture & Planned Modules

```
smart_ev_battery_anfis_final/
├── data/
│   ├── raw/                  # Original, immutable battery discharge & cycle datasets
│   └── processed/            # Cleaned, normalized, and feature-engineered datasets
├── docs/                     # Project documentation, specifications, and development plan
├── models/                   # Serialized trained model weights and parameter artifacts
├── results/
│   ├── graphs/               # Exported plots (actual vs. predicted, loss curves, MF plots)
│   ├── metrics/              # Tabulated benchmark results (JSON/CSV)
│   └── predictions/          # Model inference dumps and test set predictions
├── src/
│   ├── __init__.py
│   ├── dashboard/            # Flet GUI components, controls, and reactive state management
│   ├── data/                 # Ingestion, cleaning, noise filtering, and feature extraction
│   ├── evaluation/           # Performance metrics (RMSE, MAE, R²) and visualization helpers
│   └── models/               # Model architectures: ANFIS, Random Forest, ANN
├── tests/                    # Unit and integration test suites
├── requirements.txt          # Project dependencies
└── README.md                 # Project overview and instructions
```

### Module Breakdown
- **`src/data/`**:
  - Handles parsing of experimental battery cycle data (voltage, current, temperature, time, Ah).
  - Outlier filtering, cycle segmentation, normalization, and train-val-test splitting.
- **`src/models/`**:
  - **ANFIS Model**: Fuzzification layer, membership function definitions (e.g., Gaussian/Bell-shaped), fuzzy rule evaluation, defuzzification, and hybrid learning.
  - **Random Forest Baseline**: Ensemble decision trees tuned for non-linear regression.
  - **ANN / MLP Baseline**: Feedforward neural network with backpropagation.
- **`src/evaluation/`**:
  - Computes evaluation metrics ($R^2$, Root Mean Square Error, Mean Absolute Error).
  - Generates publication-ready figures (error distributions, predicted vs. actual SOC curves, membership function evolution).
- **`src/dashboard/`**:
  - Interactive Flet desktop interface.
  - Battery gauge dials, dynamic input sliders (Voltage, Current, Cell Temperature, Cycle Index), live SOC readout, and model comparison toggles.

---

## 📊 Project Status & Implementation Roadmap

| Stage | Description | Status | Key Deliverable |
| :--- | :--- | :---: | :--- |
| **Step 1** | Workspace & Architecture Setup | ✅ Completed | Modular directory layout, strict `.gitignore`, clean environment |
| **Step 2** | Dataset Audit & Validation | ✅ Completed | [docs/DATASET_AUDIT.md](docs/DATASET_AUDIT.md) — 600 cycles audited, 580 labeled cycles verified |
| **Step 3** | Feature Extraction & Dataset Preparation | ✅ Completed | [docs/FEATURE_EXTRACTION.md](docs/FEATURE_EXTRACTION.md) — 7 domain features extracted to `data/processed/` |
| **Step 4** | ANFIS & Baseline Model Development | ✅ Completed | [docs/MODEL_DEVELOPMENT.md](docs/MODEL_DEVELOPMENT.md) — Zero-order Sugeno ANFIS, RF, and MLP trained |
| **Step 5** | Interactive Desktop Dashboard | ✅ Completed | [src/dashboard/app.py](src/dashboard/app.py) — Single-window Smart EV Battery SOH Predictor with preset loading & inference |

---

## 🔬 Benchmark Results (Test Set: Cycles 465–580)

All models were evaluated on the chronologically split, held-out test partition (unseen aging cycles 465 to 580, strictly chronological to prevent data leakage):

| Model Architecture | Parameters / Rules | Test MAE (% SOH) | Test RMSE (% SOH) | Max Error (% SOH) | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ANFIS** (Sugeno Zero-Order) | 5 rules (70 premise, 5 consequent = 75 params) | **2.4473%** | **2.4935%** | **3.5266%** | -19.7706 |
| **Random Forest Regressor** | 50 trees, max depth 12 (4,546 nodes) | 3.4362% | 3.4969% | 4.5944% | -39.8522 |
| **MLP / ANN** | 2 hidden layers (64, 32 ReLU, Adam) (2,625 weights) | 1.0933% | 1.1202% | 1.9251% | -3.1920 |

*Key finding*: On the training partition, ANFIS achieves **0.3813% RMSE** ($R^2 = 0.9836$) with only 75 parameters. Under strict chronological holdout extrapolation (cycles 465–580 where battery SOH degrades below training range to 83.04%), MLP ANN achieves 1.1202% RMSE via continuous gradient extrapolation, ANFIS achieves 2.4935% RMSE with smooth Gaussian bounding and target centering, while Random Forest suffers from tree-leaf saturation, yielding 3.4969% RMSE. Negative $R^2$ values are mathematically expected under temporal out-of-domain extrapolation where local variance in the test window ($\sigma \approx 0.54\%$) is smaller than the cumulative aging drop.

---

## 🖥️ Smart EV Battery Management Using ANFIS Interface

The application features a clean, professional, single-window light-theme interface built with **Flet** ([src/dashboard/app.py](src/dashboard/app.py)):

1. **Header & Context**:
   - Window Title: **Smart EV Battery Management Using ANFIS**.
   - Subtitle: *Adaptive Neuro-Fuzzy Inference System for State of Health (SOH) Estimation*.
2. **Three Predefined Cycle Example Buttons**:
   - Dedicated buttons: **Healthy Battery** (Fresh cell, Cycle 1, ~97.50% SOH), **Moderately Degraded Battery** (Mid-life, Cycle 313, ~87.94% SOH), and **More Degraded Battery** (Aged cell, Cycle 580, ~83.10% SOH).
   - Clicking an example populates the 7 operational feature fields only; stored predictions are **never** pre-inserted.
   - Reference SOH and cycle identifiers are stored separately as benchmark ground truth and are **never** passed into the ANFIS model.
3. **Seven Standard Battery Feature Inputs (with Units)**:
   - `tv_ch`: Charging duration from 3.20 V to 4.15 V (s)
   - `tv_d`: Discharging duration from active onset to 3.60 V (s)
   - `tI_ch`: Current-related duration until current tapers $\le 500\text{ mA}$ (s)
   - `Tavg`: Average cell surface temperature (°C)
   - `tTpeak_ch`: Elapsed time from charging onset to peak temperature (s)
   - `Eavg_ch`: Average cell terminal voltage during charging (V)
   - `Eavg_d`: Average cell terminal voltage during discharging (V)
4. **Action Controls**:
   - **Predict Battery Health**: Prominent primary button that validates inputs against physical bounds, applies the training-fitted `StandardScaler`, and executes the pre-trained ANFIS model.
   - **Reset**: Clears all input fields, validation messages, and displayed results back to the initial state.
5. **Prediction Result Card**:
   - Displays computed SOH percentage, color-coded health classification badge (*Optimal / Healthy*, *Moderate Degradation*, *Degraded / End-of-Life*), qualitative interpretation, dataset reference comparison (if an authentic preset was loaded), and an operational applicability note.
6. **Compact Model Evaluation Benchmark Section**:
   - Displays the verified held-out chronological test set metrics for the active **First-Order Sugeno ANFIS** model:
     - **Test MAE**: `0.8246 %`
     - **Test RMSE**: `0.8574 %`
     - **Test R²**: `-1.4560`
     - **Architecture**: 2 Linear Rules (44 parameters)
   - Includes an explicit academic caveat highlighting that the negative test $R^2$ reflects narrow test-set target variance ($\sigma = 0.55\%$) under chronological out-of-domain extrapolation, demonstrating robust downward degradation tracking rather than certified high predictive accuracy across diverse operating conditions.
7. **Research Prototype Disclaimer**:
   - Clarifies that the tool is developed for academic demonstration and is not an ISO 26262 automotive-certified BMS.

---

## 🚀 Execution Guide

### 1. Prerequisites & Environment Setup
Python 3.10+ (tested with Python 3.12). A dedicated virtual environment is strongly recommended:

```powershell
# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 2. Launching the SOH Predictor Application
To launch the desktop interface:

```powershell
.\.venv\Scripts\python.exe -m src.dashboard.app
```
*(Or `python -m src.dashboard.app` with an active virtual environment)*

### 3. Running Model Training & Benchmarking Pipeline
To re-run the complete training, evaluation, and visualization pipeline:

```powershell
# Zero-Order ANFIS & ML Baselines Benchmark
.\.venv\Scripts\python.exe -m src.evaluation.evaluate_all

# First-Order ANFIS Training & Trajectory Evaluation
.\.venv\Scripts\python.exe -m src.evaluation.evaluate_first_order
```

### 4. Running the Automated Test Suite
To execute all 42 automated unit and integration tests across data extraction, zero-order modeling, first-order modeling, preset loading, and dashboard safeguards:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

---

## ⚠️ Scientific Limitations & Safety Disclaimers

1. **Laboratory Controlled Cycling vs. Dynamic Drive Profiles**:
   - The models were trained and benchmarked on standardized CC-CV (constant-current constant-voltage) charging and constant discharge cycling of Samsung INR18650-35E cylindrical cells under controlled ambient conditions.
   - Dynamic real-world automotive drive cycles (e.g., UDDS, US06, WLTP) feature high-frequency regenerative braking spikes and varying thermal conditions that require dynamic state-space or time-series filtering before feature aggregation.

2. **Unseen Battery Generalization**:
   - All experimental data was collected from a single cell specimen (Battery 1). Generalization across independent cells, cell batches, differing form factors, or alternative battery chemistries (e.g., LFP, sodium-ion) has not been empirically verified.

3. **Test-Set Blinding Transparency**:
   - While the test set rows were strictly excluded from model parameter optimization (LSE and gradient backpropagation) and feature scaling, exploratory scratch experimentation during development printed test-set error metrics. The test set was therefore not completely blinded during preliminary hyperparameter search.

4. **Chronological Extrapolation & Degradation Knee**:
   - The test partition strictly evaluates cycles 465 to 580. While ANFIS maintains smooth bounding characteristics, deep battery aging typically exhibits a sharp non-linear capacity drop ("knee point") that requires online recursive parameter adaptation in production environments.

5. **Academic Simulation vs. Certified On-Board BMS**:
   - This project is developed as an academic soft-computing capstone study and interactive research diagnostic tool.
   - **It does not interface with live automotive CAN-bus hardware** and is **not certified for functional safety** under ISO 26262 / ASIL-D standards. Displayed values are statistical and neuro-fuzzy model estimates.

6. **Strict Leakage Prevention Guarantee**:
   - Cycle counters (`CN`, `global_cycle`), raw reference capacity, and ground-truth SOH are strictly excluded from model feature vectors. Inference relies solely on the 7 electrochemical operational features.

