# Predictive Maintenance Using LSTM

This project demonstrates the application of Long Short-Term Memory (LSTM) neural networks for predictive maintenance in supply chain assets. By analyzing time-series sensor data from equipment and maintenance logs, the model predicts potential failures, allowing for proactive maintenance strategies. The code integrates data from both cloud and on-premises sources, preprocesses this data for neural network training, and employs an LSTM model for failure prediction.

## Project Structure

- **Data Integration**: Utilizes SQLAlchemy to connect to cloud and on-premises databases, fetching historical sensor readings and maintenance records.
- **Data Preprocessing**: Normalizes sensor data and transforms it into sequences suitable for LSTM input.
- **Model Training and Evaluation**: Defines an LSTM model, trains it on the preprocessed data, and evaluates its performance.

## Setup and Requirements

### Requirements
- Python 3.x
- SQLAlchemy
- pandas
- scikit-learn
- TensorFlow 2.x

### Database Connections
Update the `cloud_db_string` and `on_prem_db_string` placeholders with your actual database connection strings.

### Installation
Ensure you have the required libraries installed:
```bash
pip install sqlalchemy pandas scikit-learn tensorflow
```

## Usage

1. **Data Fetching**: Execute the SQL queries to retrieve sensor readings and maintenance records. Ensure that your database schema matches the queries or adjust them accordingly.
2. **Data Preprocessing**:
    - Normalize sensor readings using `StandardScaler`.
    - Create sequences from sensor readings to serve as input to the LSTM model.
3. **Model Training**:
    - Split the data into training and testing sets.
    - Define and compile the LSTM model.
    - Train the model on the training set and validate it using the testing set.
4. **Evaluation**: Assess the model's accuracy and adjust parameters as necessary for optimization.

## LSTM Model

The LSTM model is designed with a single LSTM layer followed by a dense output layer. The model aims to predict equipment failures based on sequential sensor data, which is crucial for implementing effective predictive maintenance strategies.

## Notes

- The success of the LSTM model highly depends on the quality and quantity of the data. Consider collecting a diverse set of sensor readings over a significant period to improve model accuracy.
- Adjust the `sequence_length` parameter based on the frequency of sensor readings and the nature of the equipment to capture relevant temporal patterns.
- Experiment with different LSTM architectures, adding more layers or units, to find the optimal model configuration for your specific use case.
