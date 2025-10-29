import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm.callback import early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ENHANCED LIGHTGBM MULTI-CLASS CLASSIFICATION (NOISE-ROBUST)")
print("="*80)

# -----------------------------
# 1. Load data
# -----------------------------
print("\n[1/8] Loading data...")
X = pd.read_csv('trainingData.txt', header=None)
y = pd.read_csv('trainingTruth.txt', header=None, names=['label']).squeeze()
test_data = pd.read_csv('testData.txt', header=None)

print(f"Training data shape: {X.shape}")
print(f"Test data shape: {test_data.shape}")

# -----------------------------
# 2. Data Preprocessing (Noise-Aware)
# -----------------------------
print("\n[2/8] Preprocessing data (noise-robust approach)...")

# Replace empty strings with NaN and convert to numeric
X = X.replace('', np.nan).apply(pd.to_numeric, errors='coerce')
test_data = test_data.apply(pd.to_numeric, errors='coerce')

# Remove rows where y is null (align X and y)
valid_mask = ~y.isna()
X = X[valid_mask].reset_index(drop=True)
y = y[valid_mask].reset_index(drop=True)

print(f"After cleaning: {X.shape[0]} samples, {X.shape[1]} features")

# Analyze missing data in training
missing_percentage = (X.isna().sum() / len(X)) * 100
features_with_missing = (missing_percentage > 0).sum()
print(f"Features with missing values: {features_with_missing}/{X.shape[1]}")
if features_with_missing > 0:
    print(f"  Max missing %: {missing_percentage.max():.2f}%")
    print(f"  Mean missing %: {missing_percentage[missing_percentage > 0].mean():.2f}%")

# Check class distribution
print("\nClass distribution:")
class_counts = y.value_counts().sort_index()
print(class_counts)
for cls in class_counts.index:
    print(f"  Class {int(cls)}: {class_counts[cls]} samples ({100*class_counts[cls]/len(y):.2f}%)")

# Check for class imbalance
is_imbalanced = (class_counts.max() / class_counts.min()) > 1.5
if is_imbalanced:
    print("⚠️  Dataset appears imbalanced - using balanced weights")

# Labels to zero-based for LightGBM
y = y - 1

# -----------------------------
# 3. Noise-Robust Imputation Strategy
# -----------------------------
print("\n[3/8] Applying noise-robust imputation...")

# Strategy: Use median imputation for robustness against noise
# Median is less sensitive to outliers/noise than mean
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
test_imputed = imputer.transform(test_data)

# Optional: Apply RobustScaler for outlier resistance
# Note: Tree-based models don't require scaling, but can help with noise
# Uncommenting this may help if noise manifests as extreme outliers
# scaler = RobustScaler()
# X_imputed = scaler.fit_transform(X_imputed)
# test_imputed = scaler.transform(test_imputed)

print(f"Final feature count: {X_imputed.shape[1]}")

# Detect potential outliers/noise
for i in range(min(5, X_imputed.shape[1])):  # Check first 5 features
    q1, q99 = np.percentile(X_imputed[:, i], [1, 99])
    outlier_count = np.sum((X_imputed[:, i] < q1) | (X_imputed[:, i] > q99))
    if outlier_count > len(X_imputed) * 0.05:
        print(f"  Feature {i}: {outlier_count} potential outliers detected ({100*outlier_count/len(X_imputed):.1f}%)")

# -----------------------------
# 4. Noise-Robust Hyperparameters
# -----------------------------
print("\n[4/8] Configuring noise-robust hyperparameters...")

# Parameters optimized for noisy data:
# - Higher min_data_in_leaf: prevents overfitting to noise
# - Lower learning_rate: more gradual learning
# - Regularization (lambda_l1, lambda_l2): reduces overfitting
# - max_depth limitation: prevents learning noise patterns
# - Feature/bagging fraction: adds randomness to combat noise

param_configs = [
    {
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': 7,
        'min_data_in_leaf': 30,  # Higher to resist noise
        'feature_fraction': 0.75,  # Lower for more robustness
        'bagging_fraction': 0.75,
        'bagging_freq': 5,
        'lambda_l1': 1.0,  # Strong L1 regularization
        'lambda_l2': 1.0,  # Strong L2 regularization
        'min_gain_to_split': 0.01,  # Prevent weak splits
        'verbose': -1,
        'is_unbalance': is_imbalanced,
        'seed': 42
    },
    {
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 50,
        'max_depth': 8,
        'min_data_in_leaf': 25,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
        'lambda_l1': 0.5,
        'lambda_l2': 1.5,
        'min_gain_to_split': 0.005,
        'verbose': -1,
        'is_unbalance': is_imbalanced,
        'seed': 123
    },
    {
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.025,
        'num_leaves': 40,
        'max_depth': 9,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 3,
        'lambda_l1': 0.3,
        'lambda_l2': 0.7,
        'min_gain_to_split': 0.01,
        'verbose': -1,
        'is_unbalance': is_imbalanced,
        'seed': 456
    },
    {
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.015,
        'num_leaves': 25,
        'max_depth': 6,
        'min_data_in_leaf': 40,  # Very conservative
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 6,
        'lambda_l1': 1.5,
        'lambda_l2': 1.5,
        'min_gain_to_split': 0.02,
        'verbose': -1,
        'is_unbalance': is_imbalanced,
        'seed': 789
    }
]

# -----------------------------
# 5. Cross-Validation Training
# -----------------------------
print("\n[5/8] Training with 5-fold cross-validation...")

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store results for each configuration
all_config_scores = []

for config_idx, params in enumerate(param_configs):
    print(f"\n--- Configuration {config_idx + 1}/{len(param_configs)} ---")
    print(f"LR: {params['learning_rate']}, Leaves: {params['num_leaves']}, "
          f"Depth: {params['max_depth']}, Min_data: {params['min_data_in_leaf']}")
    
    fold_scores = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_imputed, y)):
        X_train, X_val = X_imputed[train_idx], X_imputed[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create datasets
        train_dataset = lgb.Dataset(X_train, label=y_train)
        valid_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)
        
        # Train model with more conservative early stopping
        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=1500,  # More rounds with lower LR
            valid_sets=[valid_dataset],
            callbacks=[
                early_stopping(stopping_rounds=100),  # More patience for noisy data
                log_evaluation(period=0)
            ]
        )
        
        # Validate
        y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        y_val_pred_labels = np.argmax(y_val_pred, axis=1)
        accuracy = accuracy_score(y_val, y_val_pred_labels)
        
        fold_scores.append(accuracy)
        fold_models.append(model)
        
        print(f"  Fold {fold + 1}: Accuracy = {accuracy:.4f}, Best iteration = {model.best_iteration}")
    
    # Calculate average CV score
    avg_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"  → CV Score: {avg_score:.4f} ± {std_score:.4f}")
    
    all_config_scores.append((avg_score, std_score, config_idx, fold_models))

# Select best configuration
best_score, best_std, best_config_idx, best_fold_models = max(all_config_scores, key=lambda x: x[0])
best_params = param_configs[best_config_idx]

print(f"\n✓ Best configuration: Config {best_config_idx + 1}")
print(f"  CV Score: {best_score:.4f} ± {best_std:.4f}")

# -----------------------------
# 6. Validation Metrics
# -----------------------------
print("\n[6/8] Detailed validation metrics...")

all_val_preds = []
all_val_true = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_imputed, y)):
    X_val = X_imputed[val_idx]
    y_val = y.iloc[val_idx]
    
    model = best_fold_models[fold]
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    
    all_val_preds.append(y_val_pred)
    all_val_true.extend(y_val.values)

all_val_preds = np.vstack(all_val_preds)
all_val_true = np.array(all_val_true)

val_pred_labels = np.argmax(all_val_preds, axis=1)
accuracy = accuracy_score(all_val_true, val_pred_labels)

print(f"\nOverall Validation Accuracy: {accuracy:.4f}")

print("\nClass-wise AUC scores:")
auc_scores = []
for i in range(4):
    y_true_bin = (all_val_true == i).astype(int)
    auc = roc_auc_score(y_true_bin, all_val_preds[:, i])
    auc_scores.append(auc)
    print(f"  Class {i+1} AUC: {auc:.4f}")

macro_auc = np.mean(auc_scores)
print(f"  Macro-average AUC: {macro_auc:.4f}")

print("\nClassification Report:")
print(classification_report(all_val_true, val_pred_labels, 
                          target_names=[f'Class {i+1}' for i in range(4)],
                          digits=4))

# -----------------------------
# 7. Feature Importance
# -----------------------------
print("\n[7/8] Analyzing feature importance...")

importance = best_fold_models[0].feature_importance(importance_type='gain')
importance_df = pd.DataFrame({
    'feature_idx': range(len(importance)),
    'importance': importance
}).sort_values('importance', ascending=False)

print("\nTop 20 most important features:")
for idx, row in importance_df.head(20).iterrows():
    print(f"  Feature {int(row['feature_idx'])}: {row['importance']:.2f}")

# Check if some features are being ignored (possible noise features)
zero_importance = (importance == 0).sum()
if zero_importance > 0:
    print(f"\n⚠️  {zero_importance} features have zero importance (likely noise)")

# -----------------------------
# 8. Ensemble Prediction on Test Set
# -----------------------------
print("\n[8/8] Generating ensemble predictions on test set...")

# Strategy for noisy data: Use larger ensemble for stability
ensemble_test_preds = []

# Use best config with different seeds + some diversity
seeds = [42, 123, 456, 789, 2024, 2025, 3141, 9876]

for seed_idx, seed in enumerate(seeds):
    print(f"  Training ensemble model {seed_idx + 1}/{len(seeds)} (seed={seed})...")
    
    params_with_seed = best_params.copy()
    params_with_seed['seed'] = seed
    
    # Use average best iteration from CV
    avg_best_iter = int(np.mean([m.best_iteration for m in best_fold_models]))
    
    full_train = lgb.Dataset(X_imputed, label=y)
    final_model = lgb.train(
        params_with_seed, 
        full_train, 
        num_boost_round=avg_best_iter
    )
    
    test_pred = final_model.predict(test_imputed)
    ensemble_test_preds.append(test_pred)

# Average ensemble predictions (robust to noise)
test_pred_final = np.mean(ensemble_test_preds, axis=0)

# Also calculate median for extra robustness (optional sanity check)
test_pred_median = np.median(ensemble_test_preds, axis=0)

# Use mean predictions (more stable for probability outputs)
test_labels = np.argmax(test_pred_final, axis=1) + 1

# Calculate prediction confidence and agreement
prediction_confidence = np.max(test_pred_final, axis=1)
ensemble_agreement = np.array([
    np.mean([np.argmax(pred[i]) == np.argmax(test_pred_final[i]) 
             for pred in ensemble_test_preds])
    for i in range(len(test_pred_final))
])

print(f"\nPrediction confidence statistics:")
print(f"  Mean: {prediction_confidence.mean():.4f}")
print(f"  Median: {np.median(prediction_confidence):.4f}")
print(f"  Min: {prediction_confidence.min():.4f}")
print(f"  Max: {prediction_confidence.max():.4f}")

print(f"\nEnsemble agreement statistics:")
print(f"  Mean agreement: {ensemble_agreement.mean():.4f}")
print(f"  High agreement (>0.8): {(ensemble_agreement > 0.8).sum()} samples")
print(f"  Low agreement (<0.5): {(ensemble_agreement < 0.5).sum()} samples")

# Show class distribution in predictions
print("\nPredicted class distribution:")
pred_counts = pd.Series(test_labels).value_counts().sort_index()
for cls in pred_counts.index:
    print(f"  Class {int(cls)}: {pred_counts[cls]} samples ({100*pred_counts[cls]/len(test_labels):.2f}%)")

# Compare with training distribution
print("\nClass distribution comparison:")
print("Training vs Test predictions:")
for cls in sorted(class_counts.index):
    train_pct = 100 * class_counts[cls] / len(y)
    test_pct = 100 * pred_counts.get(cls, 0) / len(test_labels)
    diff = test_pct - train_pct
    print(f"  Class {int(cls)}: {train_pct:.1f}% → {test_pct:.1f}% (Δ {diff:+.1f}%)")

# -----------------------------
# 9. Save Results
# -----------------------------
output = np.column_stack([test_pred_final, test_labels])
np.savetxt('testLabel_lightgbm.txt', output, 
           fmt='%.6f\t%.6f\t%.6f\t%.6f\t%d', 
           delimiter='\t')

# Also save ensemble agreement scores for analysis
np.savetxt('testLabel_confidence.txt', 
           np.column_stack([test_labels, prediction_confidence, ensemble_agreement]),
           fmt='%d\t%.6f\t%.6f',
           header='predicted_label\tconfidence\tensemble_agreement',
           comments='')

print("\n" + "="*80)
print("✓ COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nTest predictions saved to 'testLabel_lightgbm.txt'")
print(f"Confidence metrics saved to 'testLabel_confidence.txt'")
print(f"  - {len(test_labels)} predictions generated")
print(f"  - Ensemble of {len(seeds)} models")
print(f"  - Expected validation accuracy: {best_score:.4f} ± {best_std:.4f}")
print(f"  - Expected macro-average AUC: {macro_auc:.4f}")
print("\n" + "="*80)
