import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

train_data = pd.read_parquet("UNSW_NB15_training-set.parquet")
test_data  = pd.read_parquet("UNSW_NB15_testing-set.parquet")

print("Train shape:", train_data.shape)
print("Test shape :", test_data.shape)
drop_cols = ['id', 'attack_cat']
train_data.drop(columns=[c for c in drop_cols if c in train_data.columns], inplace=True)
test_data.drop(columns=[c for c in drop_cols if c in test_data.columns], inplace=True)
target_col = 'label'
y_train = train_data[target_col]
y_test  = test_data[target_col]
X_train = train_data.drop(target_col, axis=1)
X_test  = test_data.drop(target_col, axis=1)

for df in [X_train, X_test]:
    df['bytes_ratio']    = df['sbytes'] / (df['dbytes'] + 1)
    df['total_bytes']    = df['sbytes'] + df['dbytes']
    df['pkts_ratio']     = df['spkts'] / (df['dpkts'] + 1)
    df['total_pkts']     = df['spkts'] + df['dpkts']
    df['bytes_per_pkt']  = df['total_bytes'] / (df['total_pkts'] + 1)
    df['dur_per_pkt']    = df['dur'] / (df['total_pkts'] + 1)
    df['src_load_ratio'] = df['sload'] / (df['dload'] + 1)
    df['src_loss_ratio'] = df['sloss'] / (df['dloss'] + 1)
    df['tcp_rtt']        = df['synack'] + df['ackdat']

print(f"Features after engineering: {X_train.shape[1]}")

# ================================
# ENCODING
# ================================
X_train = pd.get_dummies(X_train)
X_test  = pd.get_dummies(X_test)
X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)
print(f"Total features after encoding: {X_train.shape[1]}")

# ================================
# SAVE COLUMN NAMES
# ================================
columns = X_train.columns.tolist()
joblib.dump(columns, "columns.pkl")
print("✅ columns.pkl saved —", len(columns), "features")

# ================================
# SCALING
# ================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")
print("✅ scaler.pkl saved")

# ================================
# CLASS WEIGHT
# ================================
normal_count = (y_train == 0).sum()
attack_count = (y_train == 1).sum()
scale_pos_weight = normal_count / attack_count
print(f"\nnormal={normal_count}, attack={attack_count}")
print(f"scale_pos_weight = {scale_pos_weight:.4f}")

# ================================
# MODEL 1 — XGBoost
# ================================
print("\n--- Training XGBoost ---")
xgb_model = XGBClassifier(
    n_estimators=2000,
    max_depth=12,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    colsample_bylevel=0.8,
    min_child_weight=1,
    gamma=0.05,
    reg_alpha=0.05,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    tree_method='hist',
    early_stopping_rounds=100,
    n_jobs=-1,
    random_state=42
)
xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=100
)
xgb_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_acc = accuracy_score(y_test, (xgb_pred_proba > 0.5).astype(int))
print(f"XGBoost Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")

# ================================
# MODEL 2 — LightGBM
# ================================
print("\n--- Training LightGBM ---")
lgbm_model = LGBMClassifier(
    n_estimators=2000,
    max_depth=12,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.05,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=100,
    n_jobs=-1,
    random_state=42,
    verbose=-1
)
lgbm_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)]
)
lgbm_pred_proba = lgbm_model.predict_proba(X_test_scaled)[:, 1]
lgbm_acc = accuracy_score(y_test, (lgbm_pred_proba > 0.5).astype(int))
print(f"LightGBM Accuracy: {lgbm_acc:.4f} ({lgbm_acc*100:.2f}%)")

# ================================
# ENSEMBLE — average both probabilities
# combining two models always beats one
# ================================
print("\n--- Ensemble (XGBoost + LightGBM) ---")
ensemble_proba = (xgb_pred_proba + lgbm_pred_proba) / 2
ensemble_pred  = (ensemble_proba > 0.5).astype(int)

acc = accuracy_score(y_test, ensemble_pred)
print(f"\nEnsemble Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
print("\nClassification Report:\n", classification_report(y_test, ensemble_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, ensemble_pred))

# ================================
# SAVE BOTH MODELS
# ================================
joblib.dump(xgb_model,  "model.pkl")       # primary model
joblib.dump(lgbm_model, "lgbm_model.pkl")  # secondary model
print("\n✅ model.pkl saved (XGBoost)")
print("✅ lgbm_model.pkl saved (LightGBM)")
print("All saved files:", [f for f in os.listdir() if f.endswith('.pkl')])