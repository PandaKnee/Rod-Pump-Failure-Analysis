import numpy as np
import pandas as pd
import patsy
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import re

# generated with chatGPT

# =====================================================================================
# Helper Functions
# =====================================================================================

def clean_col(s):
    """Clean a column name so it plays nicely with Patsy formulas."""
    s = re.sub(r"[^0-9a-zA-Z_]", "_", s)
    return re.sub(r"_+", "_", s)


def is_skewed(x):
    return abs(pd.Series(x).skew()) > 1


# =====================================================================================
# Load & Prepare Raw Data
# =====================================================================================

df = pd.read_csv("./training data/knn.csv")

drop_cols = ["FAILURETYPE", "UWI", "tbguid", "lifetime_start"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

categorical_cols = ["bha_configuration", "ROUTE"]
df[categorical_cols] = df[categorical_cols].astype(str)
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df.columns = [clean_col(c) for c in df.columns]

numeric_cols_all = df.select_dtypes(include=[np.number]).columns
df[numeric_cols_all] = df[numeric_cols_all].fillna(df[numeric_cols_all].median())

exclude = ["FAILED", "lifetime_duration_days", "sample_weight"]


# =====================================================================================
# Cross-validation Setup
# =====================================================================================

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)
cv_c_indices = []

print("\n==================== STARTING K-FOLD CV ====================\n")

# =====================================================================================
# MAIN CV LOOP
# =====================================================================================

for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
    print(f"\n===== Fold {fold+1}/{K} =====")

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    # -----------------------------------
    # Identify usable numeric columns
    # -----------------------------------
    numeric_cols = [c for c in train.select_dtypes(include=[np.number]).columns if c not in exclude]

    # Drop zero variance
    zero_var = [c for c in numeric_cols if train[c].var() == 0]
    train.drop(columns=zero_var, inplace=True)
    test.drop(columns=zero_var, inplace=True, errors="ignore")

    numeric_cols = [c for c in train.select_dtypes(include=[np.number]).columns if c not in exclude]

    # -----------------------------------
    # Scale numeric columns
    # -----------------------------------
    scaler = StandardScaler()
    train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
    test[numeric_cols] = scaler.transform(test[numeric_cols])

    # -----------------------------------
    # Log-transform skewed features
    # -----------------------------------
    for col in numeric_cols:
        if is_skewed(train[col]):
            train[col] = np.sign(train[col]) * np.log1p(np.abs(train[col]))
            test[col] = np.sign(test[col]) * np.log1p(np.abs(test[col]))

    # =================================================================================
    # PATSY DESIGN MATRIX (FULL FEATURE SET)
    # =================================================================================

    continuous_cols = [c for c in numeric_cols if train[c].nunique() >= 10]
    variances = train[continuous_cols].var().sort_values(ascending=False)
    top_spline_vars = list(variances.head(20).index)

    spline_terms = " + ".join([f"bs({col}, df=4)" for col in top_spline_vars])
    base_terms = " + ".join([c for c in train.columns if c not in exclude])

    full_formula = spline_terms + " + " + base_terms

    # ---- Build design matrices using Patsy (automatic dummy + spline expansion)
    X_train = patsy.dmatrix(full_formula + " - 1", train, return_type="dataframe")
    X_test = patsy.dmatrix(full_formula + " - 1", test, return_type="dataframe")

    y_train = train[["FAILED", "lifetime_duration_days"]]
    y_test = test[["FAILED", "lifetime_duration_days"]]

    # =================================================================================
    # LASSO FEATURE SELECTION (Cox model with L1 penalty)
    # =================================================================================
    cox_lasso = CoxPHFitter(penalizer=1.0, l1_ratio=1.0)

    cox_lasso.fit(
        pd.concat([X_train, y_train], axis=1),
        duration_col="lifetime_duration_days",
        event_col="FAILED",
    )

    coef_df = cox_lasso.summary
    coef_df["abs_z"] = coef_df["z"].abs()

    top_selected = coef_df.sort_values("abs_z", ascending=False).head(30).index.tolist()

    # =================================================================================
    # FINAL TRAIN/TEST MATRICES USING SELECTED FEATURES ONLY
    # =================================================================================
    X_train_final = X_train[top_selected].copy()
    X_test_final = X_test[top_selected].copy()

    # =================================================================================
    # FINAL RIDGE COX MODEL
    # =================================================================================
    final_cox = CoxPHFitter(penalizer=0.5, l1_ratio=0.0)

    final_cox.fit(
        pd.concat([X_train_final, y_train], axis=1),
        duration_col="lifetime_duration_days",
        event_col="FAILED",
    )

    # =================================================================================
    # PREDICT TEST SET RISK + C-INDEX
    # =================================================================================
    risk_test = final_cox.predict_partial_hazard(X_test_final)

    c_idx = concordance_index(
        y_test["lifetime_duration_days"],
        -risk_test,  # negative because high risk = shorter survival
        y_test["FAILED"]
    )

    cv_c_indices.append(c_idx)
    print(f"Fold {fold+1} C-index = {c_idx:.4f}")

# =====================================================================================
# Final CV Results
# =====================================================================================

print("\n==================== CV COMPLETE ====================")
print("Fold C-index values:", [round(x, 4) for x in cv_c_indices])
print("Mean C-index:", np.mean(cv_c_indices))
print("Std deviation:", np.std(cv_c_indices))
print("=====================================================\n")
