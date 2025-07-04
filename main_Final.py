import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from scipy.stats import randint
import warnings
import random
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)
warnings.filterwarnings("ignore")

# Load the Excel file and remove the column named "x6"
df = pd.read_excel("proje.xlsx")
df.drop("x6", axis=1, inplace=True)
df_train = df.iloc[:100].copy() # First 100 rows are for training
df_test = df.iloc[100:].copy() # Last 20 rows are for prediction


# Create segment labels and prepare data for classification
def segment_label(y):
    if y < 0:
        return "NEG"
    elif y < 1000:
        return "LOW"
    else:
        return "HIGH"

df_train["Segment"] = df_train["Y"].apply(segment_label)
X_train_cls = df_train.drop(columns=["Y", "Segment"])
y_train_cls = df_train["Segment"]
X_test = df_test.drop(columns=["Y"]) #all x in last 20 data


# Split the data to verify RandomForestClassifier's performance on segment prediction
X_cls_train, X_cls_val, y_cls_train, y_cls_val = train_test_split(
    X_train_cls, y_train_cls, test_size=0.2, random_state=42, stratify=y_train_cls)
X_cls_train_res, y_cls_train_res = X_cls_train, y_cls_train
print(y_cls_train.value_counts())


# Train the RandomForestClassifier model on the training data
segment_clf = RandomForestClassifier(random_state=42, class_weight="balanced")
segment_clf.fit(X_cls_train_res, y_cls_train_res)


# Evaluate model on the validation set & 5-Fold Cross-Validation
y_val_pred = segment_clf.predict(X_cls_val)
print("\n📊 SegmentClassifier Accuracy (Validation Set):", round(accuracy_score(y_cls_val, y_val_pred) * 100, 2), "%")
print(classification_report(y_cls_val, y_val_pred))
segment_clf_cv = RandomForestClassifier(random_state=42, class_weight="balanced")
cv_scores = cross_val_score(segment_clf_cv, X_train_cls, y_train_cls, cv=5, scoring='accuracy')
print("\n📊 SegmentClassifier 5-Fold CV Accuracy (Mean):", round(np.mean(cv_scores) * 100, 2), "%")
print("Fold Achievements:", [round(score * 100, 2) for score in cv_scores])

#Predict the segment labels for the last 20 rows (test data)
predicted_segments = segment_clf.predict(X_test)


# Data augmentation function (used only for LOW segment)
def apply_noise_augmentation(X, y, noise_level=0.05, augmentation_factor=3):
    if augmentation_factor <= 1 or X.shape[0] == 0:
        return X, y
    X_clean = np.nan_to_num(X)
    y_clean = np.nan_to_num(y)
    X_aug = [X_clean]
    y_aug = [y_clean]
    for i in range(X_clean.shape[0]):
        for _ in range(augmentation_factor - 1):
            noise = np.random.normal(0, noise_level, X_clean[i].shape)
            xi_noisy = X_clean[i] * (1 + noise)
            X_aug.append([xi_noisy])
            y_aug.append([y_clean[i]])
    return np.vstack(X_aug), np.concatenate(y_aug)

#5-Fold Cross-Validation (LOW segment, log transform + optimization)
print("\n🔁 LOW Segment - 5-Fold Cross-Validation:")
low_mask = df_train["Segment"] == "LOW"
X_low = df_train.loc[low_mask, ["x1", "x2", "x3", "x4", "x5"]].values
y_low = df_train.loc[low_mask, "Y"].values
y_low_log = np.log1p(y_low) # Log transformation
X_low_aug, y_low_aug = apply_noise_augmentation(X_low, y_low_log, noise_level=0.05, augmentation_factor=3) # Data increase



# Define the hyperparameter search space for RandomForest
param_dist = {
    'n_estimators': randint(10, 100),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}
# Set up 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
low_maes = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_low_aug), 1):
    X_tr, X_val = X_low_aug[train_idx], X_low_aug[val_idx]
    y_tr, y_val = y_low_aug[train_idx], y_low_aug[val_idx]

    base_model = RandomForestRegressor(random_state=42)
    search = RandomizedSearchCV(base_model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42)
    search.fit(X_tr, y_tr)
    best_model = search.best_estimator_

    preds_log = best_model.predict(X_val)
    preds = np.expm1(preds_log)
    actuals = np.expm1(y_val)

    mae = mean_absolute_error(actuals, preds)
    print(f">> Fold {fold} MAE: {round(mae, 2)}")

    df_fold = pd.DataFrame({
        "Fold": fold,
        "Real Y": actuals,
        "Predict Y": preds,
        "Error (abs)": np.abs(actuals - preds).round(2)
    })
    print(df_fold.to_string(index=False))
    low_maes.append(mae)

print(f">> LOW Segment MAE (Mean of 5-Fold): {round(np.mean(low_maes), 2)}")

#NEG SEGMENT
print("\n🔁 NEG Segment - 5-Fold Cross-Validation:")

neg_mask = df_train["Segment"] == "NEG"
X_neg = df_train.loc[neg_mask, ["x1", "x2", "x3", "x4", "x5"]].values
y_neg = df_train.loc[neg_mask, "Y"].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_maes = []
neg_r2s = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_neg), 1):
    X_tr, X_val = X_neg[train_idx], X_neg[val_idx]
    y_tr, y_val = y_neg[train_idx], y_neg[val_idx]

    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)

    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    print(f">> Fold {fold} MAE: {round(mae, 2)}, R²: {round(r2, 4)}")

    neg_maes.append(mae)
    neg_r2s.append(r2)

print(f">> NEG Segment MAE (Mean of 5-Fold): {round(np.mean(neg_maes), 2)}")
print(f">> NEG Segment R²  (Mean of 5-Fold): {round(np.mean(neg_r2s), 4)}")


# === Hold-out Test ===
print("\n🔍 Segment Based Hold-out Performance:")
segment_models = {}
segment_holdout_mae = {}
segment_holdout_results = []

X_train_full = df_train.drop(columns=["Y", "Segment"])
y_train_full = df_train["Y"]
segment_labels = df_train["Segment"]

for segment in ["NEG", "LOW", "HIGH"]:
    idx = segment_labels == segment
    X_seg = X_train_full[idx]
    y_seg = y_train_full[idx]

    if segment == "HIGH":
        Q1, Q3 = y_seg.quantile(0.25), y_seg.quantile(0.75)
        IQR = Q3 - Q1
        mask = (y_seg >= Q1 - 1.5 * IQR) & (y_seg <= Q3 + 1.5 * IQR)
        X_seg = X_seg[mask]
        y_seg = y_seg[mask]
        model = GradientBoostingRegressor(random_state=42)

    elif segment == "LOW":
        X_seg = X_seg[["x1", "x2", "x3", "x4", "x5"]]
        X_tr, X_val, y_tr, y_val = train_test_split(X_seg, y_seg, test_size=0.2, random_state=42)


        y_tr_log = np.log(y_tr + 10) # New log transformation

        # Data augmentation (only on X_tr and y_tr_log)
        X_tr_aug, y_tr_aug = apply_noise_augmentation(X_tr.values, y_tr_log.values, 0.05, 3)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_tr_aug, y_tr_aug)

        # Prediction and recycling
        preds_log = model.predict(X_val.values)
        preds = np.exp(preds_log) - 10

        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        print(f"{segment}: MAE = {round(mae, 2)}, MSE = {round(mse, 2)}, R² = {round(r2, 4)}")
        segment_models[segment] = model
        segment_holdout_mae[segment] = mae

        df_compare = pd.DataFrame({
            "Segment": segment,
            "Real Y": y_val,
            "Predict Y": preds,
            "Error (abs)": np.abs(y_val - preds).round(2)
        })
        segment_holdout_results.append(df_compare)
        continue

    else:
        model = XGBRegressor(objective='reg:squarederror', random_state=42)

     #Predict Segment for hold-out
    X_tr, X_val, y_tr, y_val = train_test_split(X_seg, y_seg, test_size=0.2, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)

    mae = mean_absolute_error(y_val, preds)

    mse = mean_squared_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    print(f"{segment}: MAE = {round(mae, 2)}, MSE = {round(mse, 2)}, R² = {round(r2, 4)}")
    segment_models[segment] = model
    segment_holdout_mae[segment] = mae

    df_compare = pd.DataFrame({
        "Segment": segment,
        "Real Y": y_val,
        "Predict Y": preds,
        "Error (abs)": np.abs(y_val - preds).round(2)
    })
    segment_holdout_results.append(df_compare)

    print("\n📊 Hold-out Segment MAE Value:")
    for seg, mae in segment_holdout_mae.items():
        print(f"{seg}: MAE = {round(mae, 2)}")

    print("\n📋 Hold-out Comparison Table:")
    if segment_holdout_results:
        full_df = pd.concat(segment_holdout_results, ignore_index=True)
        print(full_df.to_string(index=False))


    # ✅ Retrain final model for real use (full segment data)
    X_final = X_seg.values
    y_final = y_seg.values

    if segment == "LOW":
        y_final_log = np.log(y_final + 10)
        X_final_aug, y_final_aug = apply_noise_augmentation(X_final, y_final_log, 0.05, 3)
        final_model = RandomForestRegressor(random_state=42)
        final_model.fit(X_final_aug, y_final_aug)
        segment_models[segment] = final_model
    elif segment == "HIGH":
        final_model = GradientBoostingRegressor(random_state=42)
        final_model.fit(X_final, y_final)
        segment_models[segment] = final_model
    else:  # NEG
        final_model = XGBRegressor(objective='reg:squarederror', random_state=42)
        final_model.fit(X_final, y_final)
        segment_models[segment] = final_model




    #OVERALL HOLD-OUT SCORES (all segments combined)
    from sklearn.metrics import mean_squared_error, r2_score

    all_actuals = full_df["Real Y"].values
    all_preds = full_df["Predict Y"].values

    overall_mae = mean_absolute_error(all_actuals, all_preds)
    overall_mse = mean_squared_error(all_actuals, all_preds)
    overall_r2 = r2_score(all_actuals, all_preds)

    print("\n📈 Overall Hold-out Performance (All Segments):")
    print(f"MAE = {round(overall_mae, 2)}")
    print(f"MSE = {round(overall_mse, 2)}")
    print(f"R²  = {round(overall_r2, 4)}")

# Last 20 data estimates
final_preds = []
for i in range(len(X_test)):
    seg = predicted_segments[i]
    model = segment_models[seg]

    if seg == "LOW":
        xi = X_test.iloc[i:i + 1][["x1", "x2", "x3", "x4", "x5"]].values
        pred_log = model.predict(xi)[0]
        pred_y = np.exp(pred_log) - 10  # reverse log transformation
    else:
        xi = X_test.iloc[i:i + 1].values
        pred_y = model.predict(xi)[0]

    final_preds.append(pred_y)

print("\n🧾 Last 20 Data Estimates:")
df_test_results = pd.DataFrame({
    "Data ID": list(range(101, 121)),
    "Predict Segment": predicted_segments,
    "Predict Y": np.round(final_preds, 6)
})
print(df_test_results.to_string(index=False))
