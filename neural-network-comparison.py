import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

def get_streak(row, color):
    """Return the larger value between win and lose streak"""
    win_streak = row[f'{color}CurrentWinStreak']
    lose_streak = row[f'{color}CurrentLoseStreak']
    return win_streak if win_streak >= lose_streak else -lose_streak

def create_mlp_deep():
    """Create a deep MLP with multiple hidden layers"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(7,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_mlp_wide():
    """Create a wide MLP with larger hidden layers"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(7,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_rnn():
    """Create a simple RNN model"""
    model = Sequential([
        SimpleRNN(64, input_shape=(1, 7)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    try:
        # Load and prepare data
        print("Loading data...")
        df = pd.read_csv('binarymaster.csv')
        
        # Create features
        print("Creating features...")
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
        df['streak_diff'] = df.apply(lambda row: get_streak(row, 'Red') - get_streak(row, 'Blue'), axis=1)
        
        # Prepare features and target
        features = list(feature_diffs.keys()) + ['streak_diff']
        X = df[features]
        y = df['Winner']
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Convert target to categorical
        y_cat = to_categorical(y)
        
        # Define models
        models = {
            'Deep MLP': create_mlp_deep,
            'Wide MLP': create_mlp_wide,
            'RNN': create_rnn
        }
        
        # Prepare for cross-validation
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        results = {name: [] for name in models.keys()}
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        # Perform cross-validation
        print("\nPerforming 10-fold cross-validation...")
        for name, create_model in models.items():
            print(f"\nEvaluating {name}...")
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
                print(f"Fold {fold + 1}/10")
                
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y_cat[train_idx], y_cat[val_idx]
                
                # Reshape input for RNN
                if name == 'RNN':
                    X_train = X_train.reshape(-1, 1, 7)
                    X_val = X_val.reshape(-1, 1, 7)
                
                # Create and train model
                model = create_model()
                history = model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Evaluate model
                val_pred = model.predict(X_val, verbose=0)
                val_pred_classes = np.argmax(val_pred, axis=1)
                val_true_classes = np.argmax(y_val, axis=1)
                accuracy = accuracy_score(val_true_classes, val_pred_classes)
                fold_scores.append(accuracy)
            
            results[name] = fold_scores
            print(f"{name} - Mean accuracy: {np.mean(fold_scores):.3f} (+/- {np.std(fold_scores) * 2:.3f})")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.boxplot([results[name] for name in models.keys()], labels=models.keys())
        plt.title('Neural Network Architectures Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('neural_network_comparison.png')
        
        # Print detailed results
        print("\nDetailed Results:")
        for name, scores in results.items():
            scores = np.array(scores)
            print(f"\n{name}:")
            print(f"Mean accuracy: {scores.mean():.3f}")
            print(f"Standard deviation: {scores.std():.3f}")
            print(f"Min accuracy: {scores.min():.3f}")
            print(f"Max accuracy: {scores.max():.3f}")
        
    except FileNotFoundError:
        print("Error: Could not find the input file 'file'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
