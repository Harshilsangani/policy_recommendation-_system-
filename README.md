# Policy Recommendation System

## Overview
This project is a Policy Recommendation System that utilizes machine learning models to recommend policies based on user data.

## Features
- **Machine Learning Model**: Uses an XGBoost model for predictions.
- **Data Processing**: Includes label encoding and scaling.
- **Web Application**: Likely a Flask-based app (if `app.py` runs a server).

## Project Structure
```
policy_recommendation_system/
│── app.py  # Main application script
│── training.ipynb  # Model training notebook
│── requirements.txt  # Dependencies
│── README.md  # Project description
│── .gitignore  # Ignore unnecessary files
│
├── data/
│   ├── realistic_synthetic_insurance_data_2.csv
│
├── models/
│   ├── label_encoder.joblib
│   ├── scaler.joblib
│   ├── xgb_model.joblib
│
├── static/
│   ├── client_avatar.png
│   ├── logo.png
│   ├── salesman_avatar.png
│   ├── style.css
├── templates/
│   ├── home.html
│   ├── pcr.html
│   ├── chatbot.html
```

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Harshilsangani/policy_recommendation-_system-.git
   cd policy_recommendation-_system-
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```

## License
MIT License

