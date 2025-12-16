# =========================================
# Gold Price Prediction Using Machine Learning
# =========================================

# Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# -------------------------
# Data Collection & Processing
# -------------------------
data = pd.read_csv('gld_price_data.csv')

# -------------------------
# Basic Data Exploration
# -------------------------
print("Dataset Shape:", data.shape)
print("\nDataset Info:")
print(data.info())

print("\nNull Values:\n", data.isnull().sum())
print("\nStatistical Summary:\n", data.describe())

# -------------------------
# Correlation Analysis
# -------------------------
correlation = data.select_dtypes(include='number').corr()

plt.figure(figsize=(8, 8))
sns.heatmap(
    correlation,
    cbar=True,
    square=True,
    fmt='.1f',
    annot=True,
    annot_kws={'size': 8},
    cmap='Blues'
)
plt.title("Correlation Heatmap")
plt.show()

# Correlation values of GLD
print("\nCorrelation with GLD:\n", correlation['GLD'])

# -------------------------
# Distribution of Gold Price
# -------------------------
sns.histplot(data['GLD'], kde=True, color='green')
plt.title("GLD Price Distribution")
plt.show()

# -------------------------
# Splitting Features & Target
# -------------------------
X = data.drop(['Date', 'GLD'], axis=1)
Y = data['GLD']

# -------------------------
# Train-Test Split
# -------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# -------------------------
# Model Training (Random Forest Regressor)
# -------------------------
regressor = RandomForestRegressor(n_estimators=100, random_state=2)
regressor.fit(X_train, Y_train)

# -------------------------
# Model Evaluation
# -------------------------
test_data_prediction = regressor.predict(X_test)

# R-squared Error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error:", error_score)

# -------------------------
# Actual vs Predicted Plot
# -------------------------
Y_test = list(Y_test)

plt.plot(Y_test, color='blue', label='Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of Values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()
