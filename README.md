# Gold-Price-Prediction-Using-Machine-Learning-Random-Forest-
This project focuses on predicting gold prices (GLD) using historical financial and economic indicators with the help of machine learning. A Random Forest Regressor model is trained on real-world data to capture complex, non-linear relationships between gold prices and correlated market variables.

The project demonstrates the complete data science workflow — from data exploration and visualization to model training, evaluation, and result interpretation.

🗂 Dataset Description

The dataset contains 2290 records with the following features:

Date – Trading date

SPX – S&P 500 Index

USO – Oil price ETF

SLV – Silver price ETF

EUR/USD – Euro to Dollar exchange rate

GLD – Gold price ETF (Target Variable)

🔍 Exploratory Data Analysis (EDA)

Checked dataset structure, null values, and statistical summary

Analyzed correlations between variables using correlation matrix & heatmap

Identified strong positive correlation between GLD and SLV

Visualized gold price distribution using histogram & KDE plot

🤖 Machine Learning Model

Algorithm Used: Random Forest Regressor

Train-Test Split: 80% training, 20% testing

Evaluation Metric: R² Score

📊 Model Performance:

R² Score: 0.989 → indicates excellent prediction accuracy

📈 Results & Visualization

Compared actual vs predicted gold prices

Visualized prediction performance using line plots

Achieved high model reliability with minimal prediction error

🛠 Tools & Technologies

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

🚀 Conclusion

This project highlights how machine learning can be effectively used in financial forecasting. The Random Forest model successfully captured market trends and produced highly accurate gold price predictions.
