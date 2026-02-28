# 📱 Vodafone Age Prediction Service

**Current Status:** Sprint 2 Completed (Advanced EDA & Baseline Models)  
**Task:** Multiclass Classification (6 Age Groups)

## 🎯 Business Goal
Development of a machine learning model to predict the age group of a mobile operator's subscribers. 
The model will allow the business to:
* Personalize marketing campaigns and special offers.
* Optimize advertisement targeting.
* Reduce customer churn rate by preventing irrelevant service offerings.

**Target Variable:** `target` (6 classes: 1 — youngest, 6 — oldest).

---

## 📊 Sprint 1: Data Understanding & Cleaning

### 1. Data Overview
The initial dataset (`train.csv`) contains **210,000 records** and **1294 features**. 
It is a classic "wide" telecom dataset that includes:
* **Profile Features:** Region, device brand, operating system.
* **Behavioral Features:** Application activity (Instagram, TikTok, WOG), SMS from specific services, and call durations.

### 2. Data Cleaning Strategy
A Baseline approach was applied to handle the raw data:
* **Behavioral Data (Sparse):** Missing values (`NaN`) in activity columns (e.g., `tinder_volume`, `banks_sms_count`) were filled with **`0`**. 
  * *Rationale:* The absence of a billing record means the absence of an event (the subscriber did not use the app). Using a median would create artificial noise.
* **Categorical Data:** Missing geographical and technical data were filled with the `'Unknown'` marker.
* **Data Quality Fixes:** * Removed the technical identifier `user_hash`.
  * Fixed an anomalous `'0'` value in the `device_brand` column (replaced with `'Unknown'`).
  * Dropped zero-variance (constant) columns that carried no predictive information (e.g., `AKCIYA`).
* **Final Dataset Shape:** `(210000, 1288)`.

### 3. Key EDA Findings
* **Class Imbalance:** Age groups are unevenly distributed. Groups 4 & 5 (older demographics) dominate the dataset (~60k each), while Groups 1 & 2 (youth) are the minority (~5-10k).
* **Generational "Digital Signatures" (App Usage):**
  * *Instagram:* Primary youth marker. Peaks in Group 2 and sharply declines by Group 6.
  * *Twitch:* Exclusively used by the youngest demographics (Groups 1 & 2).
  * *LinkedIn:* Peaks in Group 3 (young professionals building careers).
  * *Viber & Facebook:* Have a more uniform distribution and "age" much slower than other social networks.
* **Device Brand Insights:** Nokia ranks 3rd in overall popularity and serves as a strong predictor for senior groups (5-6) using feature phones. Apple usage strongly correlates with Groups 2, 3, and 4.

---

## 🚀 Sprint 2: Advanced EDA & Baseline Models

### 1. Feature Selection (Mutual Information)
To combat the "Curse of Dimensionality" associated with 1288 features, we performed a dependency analysis using **Mutual Information (MI)** to capture non-linear relationships.
* **Top Predictors:** `lifetime` (account maturity), `instagram_volume`, and `DATA_VOLUME_WEEKDAYS` showed the highest predictive power.
* **Dimensionality Reduction:** By plotting the MI scores, we observed a clear "Elbow" curve. We applied a cutoff at **K=200** features, preserving the most valuable data while safely discarding ~1000 noisy features.

### 2. Multi-Model Baseline Evaluation
We built robust `scikit-learn` Pipelines with `StandardScaler` and `LabelEncoder` (to ensure `XGBoost` compatibility). We evaluated 7 different algorithms to find the best architectural fit for our data.

| Model | Train F1 (W) | Test F1 (W) | Overfit Gap | Time (sec) |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | 0.633 | **0.463** | 0.169 | 11.5 |
| XGBoost | 0.700 | 0.457 | 0.242 | 15.2 |
| Gradient Boosting | 0.493 | 0.453 | 0.040 | 407.1 |
| Random Forest | 0.999 | 0.445 | 0.555 | 13.5 |
| Logistic Regression | 0.426 | 0.412 | 0.014 | 9.9 |
| kNN | 0.561 | 0.341 | 0.220 | 22.7 |
| Decision Tree | 0.999 | 0.339 | 0.661 | 5.6 |

### 3. Model Diagnostics (Bias-Variance Tradeoff)
Analyzing the performance gap revealed key algorithm behaviors:
* **High Variance (Overfitting):** `Decision Tree` and `Random Forest` showed massive overfit gaps (>0.55), effectively memorizing the training data instead of learning patterns.
* **High Bias (Underfitting):** `Logistic Regression` was too rigid to capture non-linear telecom behavior, resulting in low scores on both sets.
* **The Champion:** `LightGBM` provided the best balance of speed and generalization.



### 4. A/B Test: Feature Selection Validation
We conducted an experiment by training models on the **Full Dataset (1288 features)** vs. our **Reduced Dataset (200 features)**:
* **Linear Models (LR):** Training time increased by **10x**, while performance decreased due to the high noise-to-signal ratio.
* **Tree Ensembles (RF):** Accuracy degraded as the random feature selection process was overwhelmed by 1000+ non-informative columns.
* **LightGBM Insight:** While LightGBM's F1 slightly increased to 0.476 on the full set, it was solely due to **Categorical Features** (Region, Device Brand) being included, not the noisy numeric ones.
* **Decision:** For future sprints, we will use a **"Golden Dataset"**: Top 200 numeric features + encoded categorical variables.

---

## 🛠 Tech Stack
* **Language:** Python 3.10+
* **Libraries:** Pandas, NumPy, Seaborn, Matplotlib, Scikit-Learn, LightGBM
* **Environment:** AWS SageMaker (Jupyter Lab)
* **Version Control:** Git / GitHub

---

## 🗺 Roadmap (Next Steps)
**Sprint 3 Focus (Model Tuning & Improvement):**
1. **Handling Class Imbalance:** Implement SMOTE, class weights (`class_weight='balanced'`), or Focal Loss to improve Recall for minority groups (youth).
2. **Hyperparameter Tuning:** Use Optuna or GridSearchCV to optimize the LightGBM architecture.
3. **Advanced Feature Engineering:** Create aggregated meta-features (e.g., `Social_Media_to_Total_Traffic_Ratio`) to provide the model with stronger signals.
