# %% [markdown]
# Import the neccessary libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,r2_score

# %% [markdown]
# Data Loading

# %%
print("Loading data from 'data.csv'.......")
try:
    df = pd.read_csv("data.csv")
    print("Data loaded successfully")
except FileNotFoundError:
    print("Error: The file 'data.csv' was not found. Please ensure it is in same directory")
    exit()

print("Display the first 5 rows and information")
print(df.head().to_string())
print(df.info())

# %% [markdown]
# Data Preprocessing and Feature Engineering

# %%
# we select multiple features (independent variables) and the target (dependent variable)

# Check for missing values
df_missing = df.isnull().sum()
print("Missing values")
print(df_missing)

# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("Duplicated values")
print(df_duplicated)

# Define the features (X) and target (y)
# We will predict the close_eurgpb using other price columns of the same currency pair

X = df[["open_eurgbp","high_eurgbp","low_eurgbp"]] # Independent variables (features)
y = df["close_eurgbp"] # Dependent variable (target)

print("Shape of features (X):",X.shape)
print("Shape of target (y):",y.shape)

# %% [markdown]
# Data Splitting

# %%
# We split the data into a training set and a testing set. The model learns from the 
#  training data and is then evaluated on the testig data. We'll use the 70/30 split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# %% [markdown]
# Model Training and Evaluation

# %%
# We will train two models for comparision to demosnstrate why Gradient Boosting is 
# a "best fit" algorithm for this type of problem

print("Training and Evaluating Linear Regression Model")
linear_model = LinearRegression()
linear_model.fit(X_train,y_train)

y_pred_linear = linear_model.predict(X_test)
r2_linear = r2_score(y_test,y_pred_linear)
print(f"Linear Regression R-squared: {r2_linear:.4f}")


print("Training and Evaluating Gradient Boosting Regressor")
# The n-estimators parameter controls the number of boosting stages, and learning_rate
# controls the contribution of each tree
gbm_model = GradientBoostingRegressor()
gbm_model.fit(X_train,y_train)

y_pred_gbm = gbm_model.predict(X_test)
r2_gbm = r2_score(y_test,y_pred_gbm)
print(F'Gradient Boosting Regression R-squared: {r2_gbm}')

# %% [markdown]
# Comparison and Conclusion

# %%
print("Model Comparison")
print(f"Linear Regression R-squared: {r2_linear}")
print(f"Gradient Boosting R-squared: {r2_gbm}")

if r2_gbm > r2_linear:
    print("Conclusion: The Gradient Boosting Regressor model performed better due to its ablity to capture more complex relationships and correct errors iteratively.")
else:
    print("Conclusion: The Linear Regression model was more accurate (or similiar performance), suggesting a very strong linear relationship in this dataset.")

# %% [markdown]
# Making a New Prediction using the Linear Regression model

# %%
# This part of the code prompts the user to input values for all features to make a prediction
print("Interactive Prediction with Linear Regression Model")
print("Enter the following values to predict the Close price of EUR/GBP.")

while True:
    try:
        open_price_input = input("Enter the Open price (or type 'exit' to quit):")
        if open_price_input == "exit":
            break
        open_price = float(open_price_input)
        high_price = float(input("Enter the High price:"))
        low_price = float(input("Enter the Low price:"))

        # We must reshape the input to a 2D array for a single sample.
        new_prices = np.array([[open_price,high_price,low_price]])

        predicted_close = linear_model.predict(new_prices)

        print(f"For the given prices, the predicted Close price is: {predicted_close[0]:.5f}")
    except ValueError:
        print("Invalid input. Please enter valid numbers for all three fields")

# %% [markdown]
# Making a New Prediction using the Gradient Boosting model

# %%
# This part of the code prompts the user to input values for all features to make a prediction
print("Interactive Prediction with Gradient Boosting Model")
print("Enter the following values to predict the Close price of EUR/GBP.")

while True:
    try:
        gbm_open_price_input = input("Enter the Open price (or type 'exit' to quit):")
        if gbm_open_price_input == "exit":
            break
        gbm_open_price = float(gbm_open_price_input)
        gbm_high_price = float(input("Enter the High price:"))
        gbm_low_price = float(input("Enter the Low price:"))

        # We must reshape the input to a 2D array for a single sample.
        gbm_new_prices = np.array([[gbm_open_price,gbm_high_price,gbm_low_price]])

        gbm_predicted_close = linear_model.predict(gbm_new_prices)

        print(f"For the given prices, the predicted Close price is: {gbm_predicted_close[0]:.5f}")
    except ValueError:
        print("Invalid input. Please enter valid numbers for all three fields")


