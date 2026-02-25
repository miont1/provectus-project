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
* **Dimensionality Reduction:** By plotting the MI scores, we observed a clear "Elbow" curve. We applied a cutoff at **K=200** features, preserving the most valuable data while safely discarding ~1000+ noisy features. The reduced dataset was saved as `vodafone_age_top200_sprint2.csv`.

### 2. Baseline Models Evaluation
We built `scikit-learn` Pipelines integrating `StandardScaler` to prevent data leakage during the Train/Test split (80/20). Because of the strict class imbalance, we prioritized **Weighted Precision, Recall, and F1-Score**.

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1 Score (Weighted) |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | **0.464** | **0.468** | **0.464** | **0.463** |
| Random Forest | 0.447 | 0.458 | 0.447 | 0.444 |
| Logistic Regression | 0.416 | 0.419 | 0.416 | 0.412 |

*Note: Random guessing for 6 classes yields ~16.6% accuracy. Our baseline significantly outperforms this without any hyperparameter tuning.*

### 3. Model Insights & Unsupervised Analysis
* **Why LightGBM wins:** Tree-based ensembles naturally handle the non-linear relationships and heavy-tailed distributions typical of behavioral telecom data.
* **The Minority Challenge:** While LightGBM performs well overall, the detailed classification report shows it struggles to identify the youth segments (Group 1 Recall: ~0.31) due to the heavy dataset imbalance, while easily isolating distinct senior segments (Group 6 Precision: 0.57).
* **PCA (Unsupervised):** A 2D Principal Component Analysis projection confirmed that classes overlap heavily. Age prediction relies on continuous behavioral blending rather than strict linear boundaries.

### 4. A/B Test: Feature Selection vs. Full Dataset
To validate our dimensionality reduction approach, an experiment was conducted by training the baseline models on the complete dataset (1288 features) including encoded categorical variables.

* **Logistic Regression & Random Forest:** Performance degraded (F1 dropped, and LR training time increased by 10x). The inclusion of ~1000 noisy features overwhelmed the linear optimization and degraded the random subspace selection in RF.
* **LightGBM:** Handled the noise perfectly and improved its F1-Score from 0.463 to 0.476. This 1.3% boost is attributed directly to the inclusion of encoded categorical features (like `device_brand` and Region), not the remaining sparse numeric columns.
* **Strategic Decision:** For the next sprint, we will construct a "Golden Dataset" combining the Top 200 numeric features with the key categorical features to maximize predictive power while avoiding the curse of dimensionality.

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
