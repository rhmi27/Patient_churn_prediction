# Patient_churn_prediction using Machine Learning

# Project Overview
This project aims to address the critical issue of patient churn within healthcare settings. Patient churn, defined as patients who discontinue care or switch providers, poses significant challenges. It can disrupt the continuity of patient care, negatively impact operational efficiency due to fluctuating patient volumes, and lead to substantial financial losses for healthcare providers. Understanding and mitigating churn is essential for maintaining a stable patient base and delivering high-quality, continuous care.


# Overall Goal
The primary goal of this project is to develop a robust machine learning solution capable of accurately predicting which patients are at high risk of churning. By proactively identifying these at-risk individuals, healthcare providers can implement targeted interventions to improve patient satisfaction and retention.


# Dataset
The project utilizes the patient_churn_dataset.csv file, sourced locally from Google Drive. This dataset comprises 2000 rows and initially 21 columns, with PatientID serving as a unique identifier before being dropped. It contains a diverse set of features related to patient demographics, service usage, satisfaction, and financial interactions, aimed at predicting patient Churned status.

# Key Features Include:
- **Demographic Information**: Age, Gender.
- **Service Usage**: Tenure_Months, Visits_Last_Year, Missed_Appointments, Days_Since_Last_Visit.
- **Satisfaction Metrics**: Overall_Satisfaction, Wait_Time_Satisfaction, Staff_Satisfaction, Provider_Rating.
- **Financial Aspects**: Avg_Out_Of_Pocket_Cost, Billing_Issues.
- **Other**: Portal_Usage, Referrals_Made, Distance_To_Facility_Miles.
- **Target Variable**: Churned (binary: 0 for non-churned, 1 for churned).

# Initial Preprocessing Steps:
- The PatientID column was dropped as it's an identifier and not a predictive feature.
- Gender (Male/Female) was numerically encoded to 0 and 1, respectively.
- The State column was dropped.
- Categorical features Specialty and Insurance_Type were one-hot encoded using pd.get_dummies to convert them into a numerical format suitable for machine learning models. drop_first=True was used to avoid multicollinearity.
- The Last_Interaction_Date column was processed to derive time-based features (e.g., for Recency_Score) and then dropped.


# Exploratory Data Analysis (EDA)
The EDA phase provided crucial insights into the dataset's characteristics and its relationship with patient churn, revealing several key patterns:
- **Churn Distribution**: The dataset exhibits a significant class imbalance, with approximately 68.35% (1367 out of 2000) of patients having churned, as shown by the value counts for the 'Churned' column. This highlights the importance of addressing imbalance during model training.

- **Demographic Insights (Age and Gender)**:
    - The **Distribution of Patient Age** histogram showed a broad age range, with a slight concentration around the 60s. The Age Distribution by Churn Status box plot indicated that while age isn't a primary           discriminator, there might be subtle age-related patterns influencing churn.
    - The **Distribution of Churn by Gender** count plot illustrated that both genders experience churn. Although more female patients churned, this generally aligns with their higher representation in the              dataset, suggesting no significant gender-specific churn propensity on its own.

- **Service Usage Patterns**: Histograms for Tenure_Months, Visits_Last_Year, Missed_Appointments, and Days_Since_Last_Visit showed varied distributions. Notably:
    - **Tenure_Months**: Non-churned patients had a slightly longer average tenure (64.85 months) compared to churned patients (58.83 months), indicating newer patients might be more susceptible to churn.
    - **Missed Appointments**: Higher Missed_Appointments and the engineered Missed_rate were positively correlated with churn, suggesting disengagement. Longer Days_Since_Last_Visit (lower Recency_Score) also            showed a correlation with churn.

- **Satisfaction Metric**s: Box plots of Overall_Satisfaction, Wait_Time_Satisfaction, Staff_Satisfaction, and Provider_Rating against churn status consistently revealed that churned patients generally reported lower satisfaction scores across all these categories. This strongly implies that patient dissatisfaction is a significant driver of churn.

- **Financial Aspects**: Both Avg_Out_Of_Pocket_Cost and Billing_Issues showed a positive correlation with churn. Patients with higher out-of-pocket costs or who experienced billing issues were more likely to churn, as observed in the correlation heatmap and count plots.

- **Correlation Analysis**: A comprehensive Correlation Matrix of Numerical Features heatmap identified strong relationships. For example, satisfaction metrics and Tenure_Months were negatively correlated with churn, while Avg_Out_Of_Pocket_Cost, Missed_Appointments, and Days_Since_Last_Visit were positively correlated.


# Methodology
The project followed a structured approach, encompassing data preprocessing, model selection, and rigorous training and evaluation:
**Data Preprocessing**:
- **Cleaning and Transformation**: The initial dataset underwent several transformations:
    - The PatientID column was dropped as it's an identifier and not a predictive feature.
    - The Gender column was converted from categorical ('Male', 'Female') to numerical (0, 1).
    - The State column was dropped.
    - Specialty and Insurance_Type columns were one-hot encoded using pd.get_dummies with drop_first=True to create new binary features, transforming them into a format suitable for machine learning algorithms.
    - The Last_Interaction_Date was used for creating new features (like Recency_Score) and subsequently dropped, as the date object itself was not directly used in modeling.

- **Feature Engineering**: Several new features were engineered to capture more predictive signals:
    - **Missed_rate**: Ratio of Missed_Appointments to Visits_Last_Year.
    - **visits_freq**: Frequency of visits per Tenure_Months.
    - **Recency_Score**: Normalized Days_Since_Last_Visit to represent how recently a patient interacted.
    - **Satisfaction_Variance**: Standard deviation across all satisfaction-related columns (Overall_Satisfaction, Wait_Time_Satisfaction, Staff_Satisfaction, Provider_Rating).
    - **Wait_vs_Staff_Gap**: The difference between Wait_Time_Satisfaction and Staff_Satisfaction.

- **Train-Test Split**: The processed dataset was divided into training (80%) and testing (20%) sets to ensure model generalization and avoid overfitting.

- **Handling Class Imbalance**: Due to the significant class imbalance observed in the 'Churned' target variable (more churned patients than non-churned), the Synthetic Minority Over-sampling Technique (SMOTE) was applied to the training data. This created synthetic samples for the minority class, resulting in a balanced training set (X_resampled, y_resampled) and preventing models from being biased towards the majority class.

- **Feature Scaling**: Numerical features in both the resampled training data and the test data were scaled using StandardScaler. This standardized the features to have a mean of 0 and a standard deviation of 1, which is crucial for algorithms sensitive to feature magnitudes and can improve convergence and performance.

**Model Selection**:
Four diverse machine learning classification models were selected for their ability to handle different data characteristics and provide robust predictions:
  - **Logistic Regression**: A linear model, chosen for its interpretability and baseline performance.
  - **Random Forest Classifier**: An ensemble tree-based model, known for its robustness, ability to capture non-linear relationships, and built-in feature importance capabilities.
  - **XGBoost Classifier**: A highly efficient and powerful gradient boosting algorithm, often achieving state-of-the-art performance in structured data prediction tasks.
  - **CatBoost Classifier**: Another advanced gradient boosting library, specifically designed to handle categorical features effectively without extensive preprocessing and known for its high accuracy and              robustness.

**Training and Evaluation**:
- **Training**: Each selected model was trained on the SMOTE-resampled training data (X_resampled, y_resampled).
- **Prediction**: After training, models were used to predict churn on the unseen test set (X_test).
- **Evaluation Metrics**: Model performance was comprehensively evaluated using a suite of metrics, with a particular focus on the recall of the churned class (class 1) due to its importance in identifying as many actual churners as possible (minimizing false negatives). Other metrics included:
    - **Accuracy**: Overall correctness of predictions.
    - **Precision**: The proportion of positive identifications that were actually correct.
    - **F1-Score**: The harmonic mean of precision and recall.
    - **Confusion Matrix**: To visualize the counts of true positive, true negative, false positive, and false negative predictions.

This rigorous evaluation process ensured that the chosen model not only performed well overall but was also effective at identifying the critical 'churned' cases, aligning with the project's primary objective.


# Results
After training and evaluating four different machine learning models, the performance on the test set, with a critical focus on **recall for the churned class (class 1)**, is summarized below:

| Model                  | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Accuracy | Confusion Matrix (TN, FP, FN, TP) |
| :--------------------- | :------------------ | :--------------- | :----------------- | :------- | :-------------------------------- |
| **Logistic Regression**| 0.78                | 0.79             | 0.79               | 0.69     | TN: 41, FP: 65, FN: 61, TP: 233    |
| **Random Forest**      | 0.77                | **0.82**         | 0.80               | 0.69     | TN: 35, FP: 71, FN: 52, TP: 242    |
| **XGBoost**            | 0.78                | 0.79             | 0.78               | 0.68     | TN: 39, FP: 67, FN: 63, TP: 231    |
| **CatBoost**           | 0.77                | 0.79             | 0.78               | 0.67     | TN: 36, FP: 70, FN: 63, TP: 231    |

**Key Findings:**
*   **Best Performing Model**: The **Random Forest Classifier** achieved the highest recall for the churned class at **0.82**. This is particularly important for churn prediction, as it indicates the model's effectiveness in identifying a large proportion of actual churners, thus minimizing false negatives (patients who churn but are predicted not to).
*   **Overall Accuracy**: All models exhibited competitive overall accuracies, ranging from 0.67 to 0.69.
*   **Trade-offs**: While Logistic Regression, XGBoost, and CatBoost also showed strong recall (0.79), Random Forest marginally surpassed them. Random Forest also offers a good balance between predictive power and the ability to capture complex non-linear relationships, making it a suitable choice for this problem. XGBoost and CatBoost, while powerful, might require more fine-tuning for optimal performance, and their interpretability can be more challenging compared to Logistic Regression.
*   **Suitability**: Given the business objective of proactively identifying and retaining patients, the **high recall for the churned class** is paramount. Therefore, the Random Forest Classifier is deemed the most suitable model for this task due to its ability to effectively identify at-risk patients.


# Conclusion
This project successfully developed and evaluated machine learning models to predict patient churn, identifying key factors that contribute to patients leaving a healthcare facility. Through comprehensive EDA and model training, we achieved the following main outcomes:

*   **Effective Churn Prediction**: The **Random Forest Classifier** emerged as the top-performing model, demonstrating a **recall of 0.82 for the churned class**. This indicates its strong ability to accurately identify patients at risk of churning, which is crucial for proactive retention efforts.

*   **Key Churn Drivers Identified**: Feature importance analysis consistently highlighted several critical drivers across the tree-based models:
    *   **Patient Satisfaction**: `Overall_Satisfaction`, `Wait_Time_Satisfaction`, `Staff_Satisfaction`, and `Provider_Rating` are paramount. Lower satisfaction correlates strongly with higher churn.
    *   **Operational Factors**: High `Missed_rate` and `Missed_Appointments`, longer `Days_Since_Last_Visit` (low `Recency_Score`), and `Distance_To_Facility_Miles` are significant indicators.
    *   **Financial Aspects**: `Avg_Out_Of_Pocket_Cost` and `Billing_Issues` also play a substantial role, suggesting financial burdens influence churn decisions.
    *   **Demographics and Service Type**: `Tenure_Months` (shorter tenure is riskier), `Age`, `Specialty` (e.g., Orthopedics, Neurology), and `Insurance_Type` also influence churn patterns.

**Practical Implications of Findings:**
The insights derived from this project have significant practical implications for healthcare providers aiming to improve patient retention:
-  **Targeted Intervention**: The predictive model can be deployed to regularly identify high-risk patients. Interventions can then be personalized based on the specific churn drivers identified for each patient. For example, a patient with low `Wait_Time_Satisfaction` might receive an offer for expedited scheduling, while one with `Billing_Issues` could be offered financial counseling.
-  **Resource Optimization**: By understanding the primary churn drivers, healthcare providers can strategically allocate resources. Instead of broad-stroke initiatives, efforts can be focused on improving specific areas that have the most impact on retention, such as enhancing satisfaction in identified areas or addressing common billing concerns.
-  **Proactive Engagement**: The project underscores the importance of proactive engagement. Early detection of dissatisfaction (e.g., through satisfaction surveys, monitoring appointment attendance) allows for timely interventions before a patient decides to churn.
-  **Strategic Planning**: The identified key churn drivers can inform long-term strategic planning, leading to systemic improvements in patient experience, operational efficiency, and financial transparency to foster loyalty and reduce churn rates.



# How to Run the Project
1.  **Prerequisites**:
    *   Python 3.8+ (or your specific version, e.g., Python 3.10)
    *   Git

2.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/patient_churn_prediction.git
    cd patient_churn_prediction
    ```

3.  **Set up a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `.\venv\Scripts\activate`
    ```

4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Jupyter Notebook**:
    ```bash
    jupyter notebook notebooks/patient_churn_analysis.ipynb
    ```
