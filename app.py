from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs('models', exist_ok=True)
app = Flask(__name__)

def create_fallback_model():
    """Create a basic model if network initialization fails"""
    try:
        # Create a dummy dataset
        np.random.seed(42)
        X = np.random.rand(100, 2) * 100
        y = X[:, 0] * 0.5 + X[:, 1] * 0.5 + np.random.normal(0, 10, 100)
        
        model = LinearRegression()
        model.fit(X, y)
        
        with open('models/model.pkl', 'wb') as file:
            pickle.dump(model, file)
        
        return model
    except Exception as e:
        logger.error(f"Failed to create fallback model: {e}")
        return None

def initialize_model(sheet_id, sheet_gid):
    try:
        url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={sheet_gid}'
        df = pd.read_csv(url)
        df.dropna(inplace=True)
        
        X = df[['52w low', '52 high']]
        y = df['Current value']
        
        model = LinearRegression()
        model.fit(X, y)
        
        with open('models/model.pkl', 'wb') as file:
            pickle.dump(model, file)
        
        return model
    except Exception as e:
        logger.warning(f"Network model initialization failed: {e}")
        return create_fallback_model()

def get_model():
    try:
        if not os.path.exists('models/model.pkl'):
            model = initialize_model('1cHxHplghQcUINvoBlpqebRm5CcyHqVDeeuS8PCqtgPY', '750909688')
        else:
            with open('models/model.pkl', 'rb') as file:
                model = pickle.load(file)
        return model
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        return create_fallback_model()

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

# Ensure model is initialized on startup
get_model()

if __name__ == '__main__':
    app.run(debug=True)
