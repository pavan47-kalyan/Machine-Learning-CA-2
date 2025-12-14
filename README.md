Bank Customer Churn Prediction using Machine Learning
A supervised machine learning project that predicts Customer Churn (Exited vs. Retained) using demographic, financial, and account behavior indicators from a bank customer dataset.
📌 Project Overview
Customer retention is critical for the banking industry. This project applies predictive analytics and supervised learning to analyze customer data and classify whether a customer is likely to leave the bank (churn) or stay. By identifying at-risk customers, banks can take proactive measures to improve retention.
🎯 Objectives
Clean and preprocess bank customer data (handling categorical variables like Geography and Gender).
Perform Exploratory Data Analysis (EDA) to identify key factors driving customer churn.
Build and compare multiple supervised learning models.
Evaluate models using standard classification metrics (Accuracy, Precision, Recall, F1-Score).
Identify the best-performing model for prediction.
🧭 Approach
Data Loading: Imported the dataset containing customer details (Credit Score, Balance, Age, etc.).
Data Cleaning & Preprocessing:
Removed non-predictive columns (CustomerId, Surname).
Encoded categorical variables (Gender via Label Encoding, Geography via One-Hot Encoding).
Scaled numerical features using StandardScaler for distance-based algorithms.
Exploratory Data Analysis (EDA): Visualized distributions and correlations to understand churn behavior.
Model Training: Split data into training (80%) and testing (20%) sets and trained five different algorithms.
Model Comparison: Compared models based on accuracy and confusion matrices.
🧠 Machine Learning Models Used
Logistic Regression
K-Nearest Neighbors (KNN) ⭐
Naive Bayes
Decision Tree Classifier
Linear Support Vector Machine (SVM)
🛠️ Tools & Technologies
Python
Pandas & NumPy (Data Manipulation)
Matplotlib & Seaborn (Data Visualization)
Scikit-learn (Machine Learning)
Jupyter Notebook
📊 Evaluation Metrics
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
🏆 Best Model
The K-Nearest Neighbors (KNN) classifier delivered the best overall performance, achieving the highest accuracy of ~85%. It effectively grouped similar customer profiles to predict churn, outperforming linear models and the basic Decision Tree on this dataset.
🔮 Key Findings
Imbalanced Data: About 80% of customers stayed, while 20% churned.
Age Factor: Older customers showed a higher likelihood of churning compared to younger ones.
Geography: Customers in Germany had a higher churn rate compared to France and Spain.
Activity: "Active Members" were significantly less likely to exit than inactive ones.
📈 Output
Visualization of Model Accuracy Comparison.
Confusion Matrix showing True Positives (correctly predicted churners) and True Negatives.
Classification Report detailing precision and recall for each class.
