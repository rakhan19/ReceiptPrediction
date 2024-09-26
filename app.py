from flask import Flask, request, jsonify, render_template
import torch
from receiptPrediction import loadModel

app = Flask(__name__)

model, maxCount = loadModel()
#routing for web
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    month = request.form.get('month', type=int)
    if month is None or month < 1 or month > 12:
        return jsonify({"error": "Invalid month"}), 400
    
    # Prepare input for the model
    input_tensor = torch.tensor([[month]], dtype=torch.float32)
    with torch.no_grad():
        normalPrediction = model(input_tensor).item()
        prediction = normalPrediction * maxCount
    return jsonify({'predicted_receipts': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)