import pandas as pd

def calculate_correlations(file_path):
    try:
        df = pd.read_csv(file_path)

        if 'RedOdds' not in df.columns or 'BlueOdds' not in df.columns:
            print("Error: The 'RedOdds' or 'BlueOdds' columns are not found in the dataset.")
            return

        # Create 'oddsdiff' column (RedOdds - BlueOdds)
        df['oddsdiff'] = df['RedOdds'] - df['BlueOdds']

        # Select only numeric columns for correlation calculation
        numeric_df = df.select_dtypes(include=['float64', 'int64'])

        # Calculate Pearson correlation with 'oddsdiff'
        correlations = numeric_df.corr()['oddsdiff']

        # Remove self-correlation and display sorted results
        print("Pearson Correlation Coefficients with 'oddsdiff':\n")
        print(correlations.drop('oddsdiff').sort_values(ascending=False))

    except FileNotFoundError:
        print("Error: The specified file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Update this path to the location of your CSV file
    file_path = 'ufc-cleaned.csv'
    calculate_correlations(file_path)
