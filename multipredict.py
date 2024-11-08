import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

file_path = 'ufc-cleaned.csv'  
df = pd.read_csv(file_path)

# Ensure Winner column is binary (1 for Red win, 0 for Blue win)
df['WinnerBinary'] = df['Winner'].apply(lambda x: 1 if x == 'Red' else 0)

# Create Reach Difference and Avg Significant Strikes Landed Difference 
df['ReachDiff'] = df['RedReachCms'] - df['BlueReachCms']
df['AvgSigStrDiff'] = df['RedAvgSigStrLanded'] - df['BlueAvgSigStrLanded']

# Drop rows with missing values in relevant columns 
df.dropna(subset=['ReachDiff', 'AvgSigStrDiff', 'WinnerBinary'], inplace=True)

# Define variables
X = df[['ReachDiff', 'AvgSigStrDiff']]
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

def predict_winner(red_reach, blue_reach, red_avg_sig_str, blue_avg_sig_str):
    reach_diff = red_reach - blue_reach
    avg_sig_str_diff = red_avg_sig_str - blue_avg_sig_str

    input_data = pd.DataFrame([[reach_diff, avg_sig_str_diff]], columns=['ReachDiff', 'AvgSigStrDiff'])  
    prediction = model.predict(input_data)  
    return prediction


def main():
    try:
        red_reach = float(input("Enter red reach (in cm): "))
        blue_reach = float(input("Enter blue reach (in cm): "))
        red_avg_sig_str = float(input("Enter red average significant strikes landed: "))
        blue_avg_sig_str = float(input("Enter blue average significant strikes landed: "))
        
        winner = predict_winner(red_reach, blue_reach, red_avg_sig_str, blue_avg_sig_str)
        print(f"The predicted winner is: {winner}")

    except ValueError:
        print("Please enter valid numbers for reach and average significant strikes landed.")

if __name__ == "__main__":
    main()
