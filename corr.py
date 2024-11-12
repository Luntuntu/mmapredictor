import pandas as pd

def calculate_correlations(file_path):
    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        # Ensure 'WinnerBinary' column exists
        if 'Winner' not in df.columns:
            print("Error: The 'Winner' column is not found in the dataset.")
            return

        # Create 'WinnerBinary' column (1 for Red win, 0 for Blue win)
        df['WinnerBinary'] = df['Winner'].apply(lambda x: 1 if x == 'Red' else 0)

        # Select only numeric columns for correlation calculation
        numeric_df = df.select_dtypes(include=['float64', 'int64'])

        # Calculate Pearson correlation with 'WinnerBinary'
        correlations = numeric_df.corr()['WinnerBinary']

        # Remove self-correlation and display sorted results
        print("Pearson Correlation Coefficients with 'WinnerBinary':\n")
        print(correlations.drop('WinnerBinary').sort_values(ascending=False))

    except FileNotFoundError:
        print("Error: The specified file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Update this path to the location of your CSV file
    file_path = 'ufc-cleaned.csv'
    calculate_correlations(file_path)
