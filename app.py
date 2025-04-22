# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import re

app = Flask(__name__)

# Load and process data
def prepare_data():
    # Read the data
    df = pd.read_csv('data.csv')
    
    # Clean Engine column - extract numeric value from strings like "200 CC"
    def extract_engine_size(engine_str):
        if pd.isna(engine_str):
            return np.nan
        match = re.search(r'(\d+)', str(engine_str))
        if match:
            return float(match.group(1))
        return np.nan
    
    df['Engine'] = df['Engine'].apply(extract_engine_size)
    
    # Drop rows with NaN values in Engine column
    df = df.dropna(subset=['Engine'])
    
    # Remove upper outliers from Selling_Price using 1.5 IQR rule
    Q1 = df['Selling_Price'].quantile(0.25)
    Q3 = df['Selling_Price'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    
    df = df[df['Selling_Price'] <= upper_bound]
    
    # Prepare features and target
    X = df[['Year', 'Engine']]
    y = df['Selling_Price']
    
    return X, y, df

# Train model and calculate Relative MAE
def train_model():
    X, y, _ = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate MAE
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate Relative MAE (as a percentage)
    y_test_mean = np.mean(y_test)
    relative_mae = (mae / y_test_mean) * 100
    
    return model, mae, relative_mae

# Global variables for model and MAE values
model, mae, relative_mae = train_model()

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            year = float(request.form['year'])
            engine = float(request.form['engine'])
            
            # Make prediction
            prediction = model.predict([[year, engine]])[0]
            prediction = f"{prediction:,.2f}"
        except Exception as e:
            # For debugging
            print(f"Error: {e}")
            prediction = "Error in input data"
    
    return render_template('index.html', prediction=prediction, mae=f"{mae:,.2f}", relative_mae=f"{relative_mae:.2f}")

if __name__ == '__main__':
    app.run(debug=True)