import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset from a CSV file
data = pd.read_csv('salary.csv')

# Features and target variable
X = data[['Experience']]  # Feature: Years of Experience
y = data['Salary']        # Target: Salary

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Take user input for prediction
user_experience = float(input("Enter years of experience: "))

# Predict salary for the user input
predicted_salary = model.predict([[user_experience]])[0]
print(f'Predicted Salary for {user_experience} years of experience: ${predicted_salary:.2f}')

# Plotting the data and the best-fit line
plt.scatter(X, y, color='blue', label='Data Points')  # Plot the original data points
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Best-Fit Line')  # Plot the best-fit line
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs. Years of Experience')
plt.legend()
plt.show()