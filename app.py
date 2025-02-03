# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os

app = Flask(__name__)

def initialize_model(sheet_id, sheet_gid):
    """Initialize and train the model on startup"""
    try:
        url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={sheet_gid}'
        df = pd.read_csv(url)
        df.dropna(inplace=True)
        
        # Remove outliers
        df = df[(np.abs((df[['52w low', '52 high', 'Current value']] - 
                df[['52w low', '52 high', 'Current value']].mean()) / 
                df[['52w low', '52 high', 'Current value']].std()) < 3).all(axis=1)]
        
        X = df[['52w low', '52 high']]
        y = df['Current value']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Save model
        with open('model.pkl', 'wb') as file:
            pickle.dump(model, file)
        
        return model
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None

def get_model():
    """Get the trained model"""
    try:
        if not os.path.exists('model.pkl'):
            model = initialize_model('1cHxHplghQcUINvoBlpqebRm5CcyHqVDeeuS8PCqtgPY', '750909688')
        else:
            with open('model.pkl', 'rb') as file:
                model = pickle.load(file)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        low = float(data.get('low'))
        high = float(data.get('high'))
        
        if low > high:
            return jsonify({'success': False, 'error': '52-week low cannot be greater than 52-week high'})
        
        model = get_model()
        if model is None:
            return jsonify({'success': False, 'error': 'Model not available'})
            
        prediction = model.predict([[low, high]])[0]
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2)
        })
    except ValueError:
        return jsonify({'success': False, 'error': 'Please enter valid numbers'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Initialize model on startup
    get_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)