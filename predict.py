import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

file_path = 'ufc-cleaned.csv'  
df = pd.read_csv(file_path)

# Ensure Winner column is binary (1 for Red win, 0 for Blue win)
df['WinnerBinary'] = df['Winner'].apply(lambda x: 1 if x == 'Red' else 0)

# Create Reach Difference 
df['ReachDiff'] = df['RedReachCms'] - df['BlueReachCms']

# Drop rows with missing values in relevant columns 
df.dropna(subset=['ReachDiff', 'WinnerBinary'], inplace=True)


X = df[['ReachDiff']]
y = df['WinnerBinary']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)
mse = mean_squared_error(y_test, y_pred)

print("Linear Regression Model:")
print("Mean Squared Error:", mse)
print("Accuracy:", accuracy)
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

def predict_winner(red_reach, blue_reach):
    reach_diff = red_reach - blue_reach

    input_data = pd.DataFrame([[reach_diff]], columns=['ReachDiff'])  
    prediction = model.predict(input_data)  
    return 'Red' if prediction[0] >= 0.5 else 'Blue'

def main():
    try:
        red_reach = float(input("Enter red reach (in cm): "))
        blue_reach = float(input("Enter blue reach (in cm): "))
        
        winner = predict_winner(red_reach, blue_reach)
        print(f"The predicted winner is: {winner}")

    except ValueError:
        print("Please enter valid numbers for reach.")

if __name__ == "__main__":
    main()
