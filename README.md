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

## Thought Process

Looking at the data, first thought led to summing up all the days in a month, and having all sums shown in a switch case for each month. Due to the lack of extra variables, this made sense at the time.

However, if more variables are added, the relationships become more complex. Meaning, that a neural network would be a good step towards future iterations. If other columns are added to the data, the model can be changed to handle this new data. Switch case would not be able to handle such a change. 

With the data given (days and number of receipts), many other external factors are not represented. External factors could include the state of the economy, holidays, inclement weather, inflation, or even app competitors.

