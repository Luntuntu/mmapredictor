import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
from psycopg2 import Error

def load_data_from_postgres(
    #define these main
    host="localhost",
    database="your_database",
    user="your_username",
    password=None,
    table_name="your_table"
):
    """
    Load UFC fight data from PostgreSQL database.
    """
    try:
        # Connection parameters
        conn_params = {
            'host': host,
            'database': database,
            'user': user
        }
        
        # Add password only if provided
        if password:
            conn_params['password'] = password
            
        # Establish database connection
        conn = psycopg2.connect(**conn_params)
        
        # Query to fetch result and all predictor columns (change depending on used data)
        query = f"""
        SELECT result, precomp_avg_takedowns_attempts, age_differential, precomp_avg_takedowns_landed,
               precomp_avg_knockdowns, precomp_avg_sub_attempts, precomp_avg_reversals, precomp_avg_control, 
               precomp_avg_sig_strikes_landed, precomp_avg_sig_strikes_attempts, precomp_avg_total_strikes_landed, 
               precomp_avg_total_strikes_attempts, precomp_avg_head_strikes_landed, precomp_avg_body_strikes_landed, 
               age_diff_direct
        FROM {table_name};
        """
        
        # Read data into pandas DataFrame
        df = pd.read_sql_query(query, conn)
        
        # Close connection
        conn.close()
        
        print(f"Successfully loaded {len(df)} records from PostgreSQL")
        
        return df
        
    except (Exception, Error) as error:
        print(f"Error connecting to PostgreSQL: {error}")
        return None

def prepare_data(df):
    """
    Prepare data for modeling.
    """
    # Handle missing values
    df = df.dropna()
    
    # Define features and target
    X = df.drop('result', axis=1)
    y = df['result']
    
    # Get feature names for later use
    features = X.columns.tolist()
    
    return X, y, features

def train_xgboost_model(X, y):
    """
    Train an XGBoost model with the data.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize XGBoost with parameters 
    model = XGBClassifier(
        learning_rate=0.3,
        n_estimators=200,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train model
    model.fit(
        X_train_scaled, 
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=True
    )
    
    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_model(model, X_test_scaled, y_test, features, X, y):
    """
    Evaluate model performance with various metrics and visualizations.
    """
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('ufc_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance plot
    plt.figure(figsize=(12, 6))
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('ufc_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ufc_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    print("\nCross-validation ROC-AUC scores:", cv_scores)
    print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Create correlation heatmap of features
    plt.figure(figsize=(10, 8))
    X_df = pd.DataFrame(X, columns=features)
    correlation_matrix = X_df.corr()
    
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={"shrink": .5}
    )
    
    plt.title('Feature Correlation Heatmap', pad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('ufc_feature_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_df

def main():
    try:
        # Load data from PostgreSQL
        print("Loading data from PostgreSQL...")
        df = load_data_from_postgres(
            #insert own login here
        )
        
        if df is None:
            print("Failed to load data. Exiting.")
            return
            
        # Check data shapes and basic statistics
        print(f"\nDataFrame shape: {df.shape}")
        print("\nData types:")
        print(df.dtypes)
        print("\nClass distribution:")
        print(df['result'].value_counts())
        
        # Prepare data
        print("\nPreparing data for modeling...")
        X, y, features = prepare_data(df)
        
        # Train model
        print("\nTraining XGBoost model...")
        model, scaler, X_train_scaled, X_test_scaled, y_train, y_test = train_xgboost_model(X, y)
        
        # Evaluate model
        print("\nEvaluating model...")
        feature_importance = evaluate_model(model, X_test_scaled, y_test, features, X, y)
        
        # Print top features
        print("\nTop 5 most important features:")
        print(feature_importance.head(5))
        
        print("\nModel training and evaluation complete! Visualization files saved.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
