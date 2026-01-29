import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def get_streak(row, color):
    """Return the larger value between win and lose streak"""
    win_streak = row[f'{color}CurrentWinStreak']
    lose_streak = row[f'{color}CurrentLoseStreak']
    return win_streak if win_streak >= lose_streak else -lose_streak

def main():
    try:
        print("Loading data...")
        df = pd.read_csv('binarymaster.csv')
        
        # Create difference features
        print("Creating difference features...")
        feature_diffs = {
            'sig_str_diff': 'RedAvgSigStrLanded - BlueAvgSigStrLanded',
            'td_diff': 'RedAvgTDLanded - BlueAvgTDLanded',
            'reach_diff': 'RedReachCms - BlueReachCms',
            'age_diff': 'RedAge - BlueAge',
            'sub_att_diff': 'RedAvgSubAtt - BlueAvgSubAtt'
        }
        
        for new_col, formula in feature_diffs.items():
            df[new_col] = df.eval(formula)
        
        # Add streak features
        print("Adding streak features...")
        df['red_streak'] = df.apply(lambda row: get_streak(row, 'Red'), axis=1)
        df['blue_streak'] = df.apply(lambda row: get_streak(row, 'Blue'), axis=1)
        df['streak_diff'] = df['red_streak'] - df['blue_streak']
        
        # Prepare features and target
        features = list(feature_diffs.keys()) + ['streak_diff']
        X = df[features]
        y = df['Winner']
        
        # Split the data
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        print("Training decision tree model...")
        dt = DecisionTreeClassifier(random_state=42, max_depth=5)
        dt.fit(X_train, y_train)
        
        # Make predictions
        print("Making predictions...")
        y_pred = dt.predict(X_test)
        
        # Evaluate the model
        print("\nModel Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Blue Win', 'Red Win']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': dt.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        for idx, row in feature_importance.iterrows():
            print(f"{row['feature']}: {row['importance']:.3f}")
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance['feature'], feature_importance['importance'])
        plt.title('Feature Importance in UFC Winner Prediction')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("\nFeature importance plot saved as 'feature_importance.png'")
        
    except FileNotFoundError:
        print("Error: Could not find the input file 'binarymaster.csv'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
