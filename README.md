# HR Employee Attrition Prediction

An end-to-end machine learning project that demonstrates how HR analytics can be used to predict employee attrition. This project uses Python and synthetic employee data to build, evaluate, and interpret classification models.

## Project Overview

Employee attrition occurs when employees leave an organization. Understanding potential attrition patterns can help HR teams explore retention strategies and provide timely support.

This project builds a complete machine learning pipeline to predict whether a fictional employee will stay or resign within the following year.

**Note:** This is an educational project using synthetic data. It does not use real employee records or establish the actual causes of employee resignations.

## Objectives

- Create and analyse a synthetic HR dataset.
- Identify and handle missing values and duplicate records.
- Perform exploratory data analysis (EDA).
- Prepare numerical and categorical features for machine learning.
- Address class imbalance using SMOTENC.
- Train and compare Logistic Regression classification models.
- Evaluate model performance using standard classification metrics.
- Interpret predictions and discuss limitations.
- Save the trained model components and analysis results.

## Dataset

The notebook automatically generates 1,000 fictional employee records and adds 20 duplicate rows for data-cleaning practice.

The dataset contains the following features:

| Feature | Description |
|---|---|
| EmployeeID | Fictional employee identifier |
| Department | Employee's department |
| WorkMode | Office, Hybrid, or Remote |
| ExperienceYears | Years of work experience |
| MonthlySalary | Monthly salary in INR |
| SatisfactionScore | Satisfaction score from 1 to 5 |
| OvertimeHours | Overtime hours in a snapshot month |
| Absences | Absence days in a snapshot month |
| Resigned | Target variable: 0 = Stayed, 1 = Resigned |

Missing values are deliberately introduced into selected input columns to practise data cleaning and preprocessing.

## Technologies Used

- **Python** — programming language
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical calculations
- **Matplotlib** — data visualisation
- **Scikit-learn** — preprocessing, Logistic Regression and model evaluation
- **imbalanced-learn** — SMOTENC for handling class imbalance
- **Joblib** — saving the fitted model components
- **Google Colab / Jupyter Notebook** — development environment

## Machine Learning Workflow

1. **Data generation:** Create a synthetic HR dataset.
2. **Data inspection:** Examine data types, missing values and duplicates.
3. **Data cleaning:** Remove duplicate records and validate feature ranges.
4. **Feature selection:** Separate model inputs from the target variable.
5. **Train-test split:** Divide the data into 80% training and 20% testing records.
6. **Exploratory data analysis:** Examine distributions, summary statistics and class imbalance using training data.
7. **Missing-value treatment:** Impute numerical values using the median and categorical values using the mode.
8. **Feature scaling:** Standardise numerical features.
9. **Class imbalance handling:** Apply SMOTENC to the training data only.
10. **Categorical encoding:** Use one-hot encoding to prepare categorical features.
11. **Model training:** Train Logistic Regression models with and without oversampling.
12. **Model evaluation:** Compare both models on the same untouched test set.
13. **Interpretation:** Examine model coefficients, false positives and false negatives.
14. **Prediction:** Demonstrate prediction for a new fictional employee.
15. **Model saving:** Export the fitted components and analysis outputs.

## Model Evaluation

The project compares two Logistic Regression models:

- Baseline model trained without oversampling.
- Model trained using SMOTENC to address class imbalance.

Performance is evaluated using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted resignations that are actual resignations.
- **Recall:** Proportion of actual resignations correctly identified.
- **F1-score:** Balance between precision and recall.
- **Confusion matrix:** Breakdown of correct and incorrect predictions.

Actual results should be added after running the notebook. The impact of SMOTENC must be determined from the observed metrics rather than assumed.

## How to Run the Project

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File → Upload notebook**.
3. Upload `HR_Attrition_Monday_Complete_Pipeline.ipynb`.
4. Use a standard CPU runtime.
5. Run the notebook cells from top to bottom.
6. Review the generated dataset, visualisations, evaluation metrics and saved files.

The notebook installs its required machine learning libraries and generates the synthetic dataset automatically.

## Output Files

The notebook is designed to generate:

- `hr_attrition_raw.csv` — synthetic dataset before duplicate removal.
- `hr_attrition_cleaned.csv` — cleaned dataset.
- `model_comparison.csv` — comparison of model evaluation metrics.
- `hr_attrition_teaching_model.joblib` — saved preprocessing components and trained model.

## Limitations and Responsible Use

- The dataset is fictional and does not represent real organisational behaviour.
- The model's performance on synthetic data does not establish its performance on real employees.
- Predictive associations do not prove why employees leave.
- The model score is not automatically a calibrated probability.
- Real-world use would require representative data, validation, fairness assessments, privacy safeguards and ongoing monitoring.

The project is intended for learning and experimentation, not for making hiring, promotion or termination decisions.

## Future Improvements

- Test additional classification algorithms.
- Apply cross-validation and more robust model validation.
- Assess model calibration and fairness across employee groups.
- Evaluate the model on suitable, authorised real-world data.
- Explore how prediction results could inform supportive employee-retention initiatives.

## Author

Created as an educational project in HR analytics and machine learning.
