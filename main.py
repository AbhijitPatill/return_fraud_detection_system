import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import RocCurveDisplay
from xgboost import XGBClassifier
import joblib
import json
from flask import Flask, request, jsonify
import logging
import datetime
from flask_cors import CORS

# ========== DATA LOADING ==========
data = pd.read_csv(r"C:\Users\91976\Downloads\archive (5)\Fraudulent_E-Commerce_Transaction_Data_2.csv")

# ========== FEATURE ENGINEERING FUNCTION ==========
def feature_engineering(df):
    """Apply feature engineering transformations"""
    amount_threshold = df['Transaction Amount'].quantile(0.95)
    df['High_Amount'] = (df['Transaction Amount'] > amount_threshold).astype(int)
    df['Is_Night'] = df['Transaction Hour'].apply(lambda x: 1 if 0 <= x <= 5 else 0)
    df['New_Account'] = (df['Account Age Days'] < 30).astype(int)
    df['Address_Mismatch'] = (df['Shipping Address'] != df['Billing Address']).astype(int)
    
    # One-hot encode categorical variables
    categoricals = ['Payment Method', 'Device Used', 'Product Category']
    df = pd.get_dummies(df, columns=categoricals, drop_first=True)
    
    # Drop unnecessary columns
    drop_cols = ['Transaction ID', 'Customer ID', 'IP Address', 'Shipping Address', 
                 'Billing Address', 'Transaction Date', 'Customer Location']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    return df

# ========== TRAINING PIPELINE ==========
print("Starting training pipeline...")

# Apply feature engineering
data_fe = feature_engineering(data)

# Split features and target
X = data_fe.drop('Is Fraudulent', axis=1)
y = data_fe['Is Fraudulent']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE for class imbalance
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Train XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_sm, y_train_sm)

# Make predictions
y_pred_xgb = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)[:, 1]

# ========== MODEL EVALUATION ==========
print("\n=== XGBoost Classification Report (Default Threshold) ===")
print(classification_report(y_test, y_pred_xgb, digits=4))

# ROC Curve - save instead of show
RocCurveDisplay.from_estimator(xgb, X_test, y_test)
plt.savefig('roc_curve.png')
plt.close()
print("ROC curve saved as roc_curve.png")

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
ap_score = average_precision_score(y_test, y_proba)
print(f"Average Precision Score: {ap_score:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(thresholds, recall[:-1], label='Recall')
plt.plot(thresholds, precision[:-1], label='Precision')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.title('Precision and Recall vs Decision Threshold')
plt.savefig('precision_recall_curve.png')
plt.close()
print("Precision-Recall curve saved as precision_recall_curve.png")

# Fixed threshold evaluation
fixed_threshold = 0.3
y_pred_fixed = (y_proba >= fixed_threshold).astype(int)
print(f"\nClassification report at fixed threshold = {fixed_threshold}:")
print(classification_report(y_test, y_pred_fixed, digits=4))
print(confusion_matrix(y_test, y_pred_fixed))

# ========== SAVE MODEL AND ARTIFACTS ==========
joblib.dump(xgb, 'xgb_fraud_model.joblib')
print("XGBoost model saved to xgb_fraud_model.joblib")

joblib.dump(scaler, 'scaler.joblib')
print("Scaler saved to scaler.joblib")

# Save feature columns for inference alignment
feature_columns = X.columns.tolist()
with open('feature_columns.json', 'w') as f:
    json.dump(feature_columns, f)
print("Feature columns saved to feature_columns.json")

# ========== FLASK API SETUP ==========
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    filename='fraud_predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def log_prediction(input_data, probability, prediction):
    """Log prediction details"""
    logging.info(f"Input: {input_data} | Probability: {probability:.4f} | Prediction: {prediction}")

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Fraud Detection API is active',
        'endpoint': '/predict',
        'method': 'POST'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get JSON data from request
        input_data = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Apply feature engineering
        df = feature_engineering(df)
        
        # Load saved feature columns
        with open('feature_columns.json', 'r') as f:
            feature_columns = json.load(f)
        
        # Reindex to match training features
        df = df.reindex(columns=feature_columns, fill_value=0)
        
        # Scale features
        X_scaled = scaler.transform(df)
        
        # Predict probability
        proba = xgb.predict_proba(X_scaled)[:, 1][0]
        
        # Apply threshold
        prediction = int(proba >= 0.3)
        
        # Log prediction
        log_prediction(input_data, proba, prediction)
        
        # Return response
        return jsonify({
            'fraud_prediction': prediction,
            'probability': float(proba),
            'threshold': 0.3,
            'status': 'success'
        })
    
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Flask API server...")
    print("API will be available at: http://127.0.0.1:5000")
    print("Prediction endpoint: http://127.0.0.1:5000/predict")
    print("="*50 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)
