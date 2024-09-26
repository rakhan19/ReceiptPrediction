from flask import Flask, request, jsonify, render_template
import torch
from receiptPrediction import loadModel

app = Flask(__name__)

model = loadModel()
#routing for web
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    month = data.get('month', None)

    if month is None:
        return jsonify({'error': 'Please select a month'}), 400
    
    input = torch.tensor([[month]], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input).item

    return jsonify({'Predicted number of receipts': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)