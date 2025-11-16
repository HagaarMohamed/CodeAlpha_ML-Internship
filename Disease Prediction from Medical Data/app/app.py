from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = 'heart-disease-prediction-2025-secret-key'

# Load trained model and preprocessing objects
model = joblib.load("model/heart_model.joblib")
label_encoder = joblib.load("model/label_encoder.joblib")
scaler = joblib.load("model/scaler.joblib")
feature_names = joblib.load("model/feature_names.joblib")

# All categorical columns that need label encoding
categorical_columns = [
    'State', 'Sex', 'GeneralHealth', 'LastCheckupTime', 'PhysicalActivities',
    'RemovedTeeth', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer',
    'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
    'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
    'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing',
    'DifficultyErrands', 'SmokerStatus', 'ECigaretteUsage', 'ChestScan',
    'RaceEthnicityCategory', 'AgeCategory', 'AlcoholDrinkers', 'HIVTesting',
    'FluVaxLast12', 'PneumoVaxEver', 'TetanusLast10Tdap', 'HighRiskLastYear',
    'CovidPos'
]

# Default values for missing columns
default_values = {
    'State': 'Alabama',
    'LastCheckupTime': 'Within past year (anytime less than 12 months ago)',
    'RemovedTeeth': 'None of them',
    'HadAngina': 'No',
    'HadStroke': 'No',
    'HadAsthma': 'No',
    'HadSkinCancer': 'No',
    'HadCOPD': 'No',
    'HadDepressiveDisorder': 'No',
    'HadKidneyDisease': 'No',
    'HadArthritis': 'No',
    'DeafOrHardOfHearing': 'No',
    'BlindOrVisionDifficulty': 'No',
    'DifficultyConcentrating': 'No',
    'DifficultyWalking': 'No',
    'DifficultyDressingBathing': 'No',
    'DifficultyErrands': 'No',
    'ECigaretteUsage': 'Never used e-cigarettes in my entire life',
    'ChestScan': 'No',
    'RaceEthnicityCategory': 'White only, Non-Hispanic',
    'AlcoholDrinkers': 'No',
    'HIVTesting': 'No',
    'FluVaxLast12': 'No',
    'PneumoVaxEver': 'No',
    'TetanusLast10Tdap': 'No, did not receive any tetanus shot in the past 10 years',
    'HighRiskLastYear': 'No',
    'CovidPos': 'No'
}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect user input from form
            user_input = {
                'PhysicalHealthDays': float(request.form['PhysicalHealthDays']),
                'MentalHealthDays': float(request.form['MentalHealthDays']),
                'SleepHours': float(request.form['SleepHours']),
                'HeightInMeters': float(request.form['HeightInMeters']),
                'WeightInKilograms': float(request.form['WeightInKilograms']),
                'BMI': float(request.form['BMI']),
                'Sex': request.form['Sex'],
                'GeneralHealth': request.form['GeneralHealth'],
                'PhysicalActivities': request.form['PhysicalActivities'],
                'HadDiabetes': request.form['HadDiabetes'],
                'SmokerStatus': request.form['SmokerStatus'],
                'AgeCategory': request.form['AgeCategory']
            }

            # Add default values for missing columns
            complete_input = {}
            for col in feature_names:
                if col in user_input:
                    complete_input[col] = user_input[col]
                elif col in default_values:
                    complete_input[col] = default_values[col]
                else:
                    complete_input[col] = 0

            # Create DataFrame
            df = pd.DataFrame([complete_input])
            df = df[feature_names]

            # Apply Label Encoding
            df_encoded = df.copy()
            for col in categorical_columns:
                if col in df_encoded.columns:
                    try:
                        df_encoded[col] = label_encoder.transform(df_encoded[col])
                    except ValueError:
                        df_encoded[col] = 0

            # Apply scaling
            df_scaled = scaler.transform(df_encoded)

            # Make prediction
            prediction = model.predict(df_scaled)[0]
            probability = model.predict_proba(df_scaled)[0]

            # Save results in session
            session['prediction'] = int(prediction)
            session['probability'] = float(probability[1] * 100)
            session['risk_low'] = float(probability[0] * 100)

            return redirect(url_for('result'))

        except Exception as e:
            error_message = f"Error: {str(e)}"
            return render_template("predict.html", error=error_message)
    
    return render_template("predict.html")

@app.route("/result")
def result():
    # Check if prediction exists
    if 'prediction' not in session:
        return redirect(url_for('predict'))
    
    # Get results from session
    prediction = session.get('prediction')
    probability = session.get('probability')
    risk_low = session.get('risk_low')
    
    return render_template("result.html", 
                         prediction=prediction,
                         probability=probability,
                         risk_low=risk_low)

if __name__ == "__main__":
    app.run(debug=True)
