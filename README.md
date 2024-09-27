# ReceiptPrediction

## Summary

Application predicts the total number of receipts for an entered month using a trained NN model.

## Requirements

Docker must be installed on the machine.

## Setup

1. **Clone the Repository**:
   Open a terminal and run the following command to clone the repository:

   ```bash
   git clone https://github.com/rakhan19/ReceiptPrediction.git
   cd ReceiptPrediction

2. **Build Docker Image**:
    After navigating to project directory, build the image by running: 

    ```bash
    docker build -t receipt-prediction-app .

3. **Run Docker Container**:
    Once the image is built. run the container:

    ```bash
    docker run -p 5000:5000 receipt-prediction-app

4. **Access Web App**:
    Open web browser and go to:

    http://localhost:5000

    Homepage should be displayed.

## Usage

Enter your desired month (1-12) into the input box, and teh app will return a prediction for teh total number of receipts for that month.

An invalid input includes months outside the range (1-12), and the app will return an error.
