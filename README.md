# Machine_Learning_Segmented_Model_Selection
Term project for the "Machine Learning and Applications" course. Predicting unlabeled Y values for 20 samples using a segmented regression model trained on 100 labeled samples with 6 numerical features.

# 🤖 Machine Learning Segmented Model Selection – CE475 Term Project

📚 **Course**: CE475 – Machine Learning and Applications  
👨‍💻 **Student**: Mehmet Arda Uçar  
🎓 **Department**: Computer Engineering – İzmir University of Economics  
📅 **Project Type**: Term Project + Final Presentation  
📌 **Goal**: Predict unknown Y values using segment-specific regression models

---

## 🧩 Problem Description

We are provided with:

- 📄 **100 labeled rows**: each with 6 numerical features (`x1` to `x6`) and a known continuous target value `Y`
- ❓ **20 unlabeled rows**: same 6 features, but `Y` is missing

The goal is to **predict the Y values for these 20 unlabeled instances** using a robust machine learning strategy.

---

## 🧠 Methodology

Due to the **highly heterogeneous distribution** of `Y`, we apply a **segmentation-based modeling strategy**:

### 📌 Step 1: Segment Classification
- A `RandomForestClassifier` is used to assign each instance into one of three segments:
  - 🔴 `NEG` (Y < 0)
  - 🟡 `LOW` (0 ≤ Y < 1000)
  - 🟢 `HIGH` (Y ≥ 1000)

### 📌 Step 2: Segment-Specific Regression Models
| Segment | Model | Notes |
|--------|--------------------------|------------------------------|
| 🔴 NEG | `XGBoostRegressor`        | Handles irregular patterns |
| 🟡 LOW | `RandomForestRegressor`  | + log transform + noise augmentation |
| 🟢 HIGH | `GradientBoostingRegressor` | + IQR-based outlier filtering |

---

## 🧪 Tools & Libraries

- `Python 3.13`
- `scikit-learn`
- `xgboost`
- `pandas`, `numpy`, `scipy`
- Cross-validation, MAE, R² scoring
- Log transform, data augmentation, IQR filtering

---

## 📊 Results Summary

| Segment | MAE | R² |
|--------|----------|--------|
| 🔴 NEG | **7.71** | 0.7637 |
| 🟡 LOW | **107.14** | 0.4978 |
| 🟢 HIGH | **714.51** | 0.9054 |

✅ **Overall Hold-Out Performance**  
- MAE: `185.93`  
- R²: `0.9668`

---

## 📌 Notes

- This project was presented as part of the CE475 course final evaluation.
- All predictions and evaluations were conducted using a train-test split strategy and cross-validation.
- The segmentation approach significantly improved model interpretability and reduced error.

---

## 🧠 Lessons Learned

✅ Segmenting complex regression problems can drastically improve performance  
✅ Hybrid modeling = specialized solutions for specialized data  
✅ Data preprocessing (log, noise, outlier filtering) is just as important as the model choice  

---

## 📬 Contact

Feel free to reach out for feedback or collaboration.  
Thanks for reading! 🙌
