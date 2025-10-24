import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm.callback import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

# -----------------------------
# 1. Load data
# -----------------------------
X = pd.read_csv('trainingData.txt', header=None)
y = pd.read_csv('trainingTruth.txt', header=None, names=['label']).squeeze()
test_data = pd.read_csv('testData.txt', header=None)

# -----------------------------
# 2. Preprocess
# -----------------------------
# Replace empty strings with NaN and convert to numeric
X = X.replace('', np.nan).apply(pd.to_numeric, errors='coerce')
test_data = test_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')

# Remove rows where y is null (align X and y)
valid_mask = ~y.isna()
X = X[valid_mask]
y = y[valid_mask]

# Impute missing values with median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Labels to zero-based for LightGBM
y = y - 1

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# -----------------------------
# 3. Create LightGBM datasets
# -----------------------------
train_dataset = lgb.Dataset(X_train, label=y_train)
valid_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

# -----------------------------
# 4. Set parameters for multiclass
# -----------------------------
params = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'verbose': -1,
    'seed': 42
}

# -----------------------------
# 5. Train model with early stopping and logging callbacks
# -----------------------------
num_rounds = 500
model = lgb.train(
    params,
    train_dataset,
    num_boost_round=num_rounds,
    valid_sets=[valid_dataset],
    callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=20)]
)

# -----------------------------
# 6. Validation predictions and metrics
# -----------------------------
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
y_val_pred_labels = np.argmax(y_val_pred, axis=1)
accuracy = np.mean(y_val_pred_labels == y_val)
print(f'\nValidation Accuracy: {accuracy:.4f}')

# Calculate AUC for each class
print('\nClass-wise AUC scores:')
for i in range(4):
    y_true_bin = (y_val == i).astype(int)
    auc = roc_auc_score(y_true_bin, y_val_pred[:, i])
    print(f'Class {i+1} AUC: {auc:.4f}')

# -----------------------------
# 7. Train on full data for test prediction
# -----------------------------
full_train = lgb.Dataset(X_imputed, label=y)
final_model = lgb.train(params, full_train, num_boost_round=model.best_iteration)

# Preprocess test data
test_imputed = imputer.transform(test_data)

# Make predictions
test_pred = final_model.predict(test_imputed)
test_labels = np.argmax(test_pred, axis=1) + 1  # back to 1-based labels

# Save final predictions: 4 probability columns + predicted label
output = np.column_stack([test_pred, test_labels])
np.savetxt('testLabel_lightgbm.txt', output, fmt='%.6f\t%.6f\t%.6f\t%.6f\t%d', delimiter='\t')
print("\nTest predictions saved to 'testLabel_lightgbm.txt'")