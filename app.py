from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from google.generativeai import GenerativeModel, configure
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Load trained models
MODEL_PATH = 'models/xgb_model.joblib'
SCALER_PATH = 'models/scaler.joblib'
LABEL_ENCODER_PATH = 'models/label_encoder.joblib'

try:
    import xgboost
    print(f"XGBoost version: {xgboost.__version__}")
    if xgboost.__version__ < '1.5':
        raise ImportError("XGBoost version must be 1.5 or higher for GPU support.")
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    xgb_model = joblib.load(MODEL_PATH)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models or checking XGBoost version: {e}")
    raise

# Configure Gemini API
configure(api_key='AIzaSyDtySa9vyC_l9EmFof-_0UJbBmM_QVzdrU')# Replace with your actual API key
gemini_model = GenerativeModel('gemini-1.5-flash')

# Feature columns expected by the model (from training)
feature_columns = [
    'Age', 'Income', 'Savings', 'Existing Investments', 'Existing Policies',
    'Website Activity Score', 'Past Claims', 'Coverage Amount', 'Interest in Add-on Benefits',
    'Income_Log', 'Age_Income_Ratio', 'Gender_Male', 'Location_Suburban', 'Location_Urban',
    'Life Stage_Married', 'Life Stage_Parents', 'Life Stage_Partnered', 'Life Stage_Single',
    'Premium Term_20', 'Premium Term_5', 'Premium Term_Whole', 'Age_Group_Middle', 'Age_Group_Senior'
]

def preprocess_input_data(customer_data):
    # Create a DataFrame with raw input
    df = pd.DataFrame([customer_data])
    
    # Ensure numeric columns are numeric and fill missing values with defaults
    numeric_cols = ['Age', 'Income', 'Savings', 'Existing Investments', 'Existing Policies',
                    'Website Activity Score', 'Past Claims', 'Coverage Amount', 'Interest in Add-on Benefits']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace('', np.nan).fillna(0)
    
    # Feature engineering (matching training code)
    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], 
                                labels=['Young', 'Middle', 'Senior'], 
                                include_lowest=True).astype(str)
    if 'Income' in df.columns:
        df['Income_Log'] = np.log1p(df['Income'])  # Log transform Income
        df['Age_Income_Ratio'] = df['Age'] / (df['Income'] + 1)  # Avoid division by zero
    
    # Map categorical variables to numeric (matching training code)
    categorical_mappings = {
        'Gender': {'Male': 1, 'Female': 0, 'Unknown': 0},
        'Location': {'Suburban': 1, 'Urban': 0, 'Rural': 0, 'Unknown': 0},
        'Life Stage': {'Single': 0, 'Married': 1, 'Partnered': 2, 'Retired': 3, 'Unknown': 0},
        'Premium Term': {'20': 1, '5': 0, 'Whole': 0, 'Unknown': 0}
    }
    
    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(lambda x: mapping.get(x, mapping['Unknown']))
    
    # One-hot encode categorical features (matching training code)
    df = pd.get_dummies(df, columns=['Gender', 'Location', 'Life Stage', 'Premium Term', 'Age_Group'], 
                        drop_first=True)
    
    # Ensure all expected columns are present with zeros for missing features
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training order
    df = df[feature_columns]
    
    # Scale the data
    X_scaled = scaler.transform(df)
    return X_scaled

@app.route('/')
def root():
    return redirect(url_for('home'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/pcr')
def index():
    return render_template('pcr.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        try:
            # Capture form data (ensure all fields match training features)
            customer_data = {
                'Name': request.form.get('name', 'Customer'),
                'Age': float(request.form.get('age', 0)),
                'Gender': request.form.get('gender', 'Unknown'),
                'Location': request.form.get('location', 'Unknown'),
                'Income': float(request.form.get('income', 0)),
                'Savings': float(request.form.get('savings', 0)),
                'Existing Investments': float(request.form.get('existing_investments', 0)),
                'Existing Policies': float(request.form.get('existing_policies', 0)),
                'Website Activity Score': float(request.form.get('website_activity', 0)),
                'Past Claims': float(request.form.get('past_claims', 0)),
                'Coverage Amount': float(request.form.get('coverage_amount', 0)),
                'Premium Term': request.form.get('premium_term', 'Unknown'),
                'Life Stage': request.form.get('life_stage', 'Unknown'),
                'Interest in Add-on Benefits': float(request.form.get('add_on_interest', 0))
            }

            # Validate critical fields
            if customer_data['Age'] < 18 or customer_data['Age'] > 100:
                raise ValueError("Age must be between 18-100 years")
            if customer_data['Income'] < 0:
                raise ValueError("Income cannot be negative")

            # Preprocess and predict
            X_scaled = preprocess_input_data(customer_data)
            prediction_encoded = xgb_model.predict(X_scaled)[0]
            prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

            # First response with personalized greeting and point-wise format
            user_name = customer_data['Name']
            prompt = (
                f"Hii {user_name}! I'm Salahkar, your insurance advisor from SBI Life Insurance. "
                f"Based on your profile, I recommend the {prediction_label} insurance policy. "
                f"Provide a short explanation in 2-3 bullet points (use dashes '-') why this plan suits the customer. "
                f"Do not mention any other policy types. Keep it simple, friendly, and end with 'How can I assist you further?'"
            )

            response = gemini_model.generate_content(
                prompt,
                generation_config={'max_output_tokens': 500}
            ).text.strip()

            # Store in session
            session['prediction'] = prediction_label
            session['customer_data'] = customer_data
            session['initial_response'] = response

            return redirect(url_for('chatbot'))
        except ValueError as e:
            return render_template('index.html', error=f"Invalid input: {str(e)}")
        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {str(e)}")
    return render_template('index.html', error=None)

@app.route('/chatbot')
def chatbot():
    prediction = session.get('prediction', 'No recommendation available.')
    customer_data = session.get('customer_data', {})
    initial_response = session.get('initial_response', "Welcome! I'm Salahkar, your insurance advisor from SBI Life Insurance. How can I assist you?")
    return render_template('chatbot.html', prediction=prediction, customer_data=customer_data, initial_response=initial_response)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    prediction = session.get('prediction', 'No recommendation available.')

    # Follow-up response without name, in point-wise format
    prompt = (
        f"I'm Salahkar, your insurance advisor from SBI Life Insurance. "
        f"The recommended policy is {prediction}. "
        f"For your message: '{user_message}', provide a short response in 2-3 bullet points (use dashes '-') if possible. "
        f"Do not mention policies other than {prediction}. Keep it concise and end with 'How can I assist you further?'"
    )

    response = gemini_model.generate_content(
        prompt,
        generation_config={'max_output_tokens': 750}
    ).text.strip()

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
