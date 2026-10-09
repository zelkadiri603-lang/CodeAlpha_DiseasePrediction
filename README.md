# Heart Disease Risk Classification (UCI Cleveland)

An end-to-end machine learning pipeline built on the UCI Cleveland dataset to predict binary heart disease risk. The focus of this workspace is data preprocessing, clinical feature interaction, and evaluating trade-offs between precision and recall for diagnostic tasks.

## Key Outcomes

* **Top Model**: Logistic Regression
* **Metrics**: 90.16% Accuracy | 92.86% Recall | 0.9513 ROC-AUC
* **Core Insight**: Feature engineering improved ROC-AUC across all candidate models, with engineered interaction terms (`exercise_angina_st`) showing significant feature attribution alongside fluoroscopy vessel count (`ca`).

## Workflow

1. **Preprocessing & Cleaning**:
   - Imputed missing values in `ca` and `thal` using median statistics.
   - Applied `log1p` transformation to reduce skewness on `oldpeak` and `chol`.
   - Encoded categorical variables via one-hot encoding.

2. **Feature Engineering**:
   - `heart_rate_ratio`: `thalach` / `age`
   - `bp_chol_prod`: `trestbps` * `chol`
   - `exercise_angina_st`: `exang` * `oldpeak`

3. **Scaling & Modeling**:
   - Scaled continuous features using `StandardScaler`.
   - Benchmarked standard models against engineered variants.
   - Employed `CalibratedClassifierCV` for probability estimates on SVM.

## Benchmark Results

| Model | Baseline Acc | Baseline AUC | FE Acc | FE Recall | FE AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 85.25% | 0.9123 | **90.16%** | **92.86%** | **0.9513** |
| **Random Forest** | 77.05% | 0.8772 | 83.61% | 89.29% | 0.9421 |
| **Gradient Boosting** | 77.05% | 0.8864 | 81.97% | 92.86% | 0.9405 |
| **Support Vector Machine** | 85.25% | 0.9091 | 90.16% | 92.86% | 0.9361 |

## Project Structure

```text
.
├── data/                  # Raw dataset files
├── models/                # Saved model & scaler pipelines (.pkl)
├── notebooks/             # Exploratory analysis & training
├── requirements.txt
└── README.md