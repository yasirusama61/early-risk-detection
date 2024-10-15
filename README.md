# Early Risk Detection System for Battery Cell Manufacturing

## Overview

This project establishes a **Minimum Viable Product (MVP)** for an early risk detection system in **battery cell manufacturing**. It proactively identifies potential quality issues in real-time through machine learning and integrates with the **Manufacturing Execution System (MES)** for real-time monitoring.

## Features

- **MVP Development**: Initial risk detection system focusing on core functionalities.
- **Concept Verification**: Validates feasibility and algorithm performance.
- **Iterative Refinement**: Continuous improvement using the PDCA cycle.
- **MES Integration**: Integrated with MES using custom APIs and OPC-UA protocols.
- **Validation Testing**: Rigorous testing across various conditions.

## Tools and Technologies

- **Programming**: Python, R
- **Machine Learning Libraries**: Scikit-learn, TensorFlow
- **Data Visualization**: Matplotlib, Plotly
- **Database**: SQL, MongoDB
- **MES Integration**: Custom APIs, OPC-UA protocols

## Project Structure

- `data/`: Raw and processed data for model training and testing.
- `scripts/`:
  - `data_preprocessing.py`: Preprocess data and handle missing values.
  - `model_training.py`: Train machine learning models for risk detection.
  - `mes_integration.py`: Communicate with MES using custom API.
  - `validation_testing.py`: Evaluate the model on new test data.
- `results/`: Metrics, plots, and evaluation results from model testing.
- `models/`: Trained models stored for further use.
- `README.md`: Project documentation.
- `requirements.txt`: Dependencies required to run the project.

## Getting Started

1. Clone the repository:
    `git clone https://github.com/yasirusama61/early-risk-detection.git`
    `cd early-risk-detection`
    

2. Install dependencies:
   
    `pip install -r requirements.txt`
    

3. Run data preprocessing:
    
    `python scripts/data_preprocessing.py`
    

4. Train the model:
    
    `python scripts/model_training.py`
    

5. Validate the model:
    
    `python scripts/validation_testing.py`
    
## Feature and Target Definition

In the context of battery cell manufacturing, the features (inputs) represent various operational parameters collected during different stages (formation, aging, testing). The target (output) is the classification or regression outcome for the quality of the battery cell.

### 1. Features (Inputs)

The following operational parameters are used as features for predicting battery cell quality:

- **Voltage**: Voltage readings during battery operation.
- **Current**: Current drawn or supplied by the battery.
- **Temperature**: Internal and ambient temperatures during operation.
- **SOC (State of Charge)**: Estimated remaining charge.
- **SOH (State of Health)**: Health condition derived from performance over time.
- **Internal Resistance**: Resistance within the battery.
- **Cycle Count**: Number of charge/discharge cycles.
- **Discharge Time**: Time taken to discharge from full charge.
- **Charge Time**: Time taken to reach full charge.
- **Formation Energy**: Energy input during the formation process.
- **Aging Time**: Time spent in the aging process.
- **Ambient Temperature**: External temperature during testing.
- **Pressure**: Pressure readings during the assembly process.
- **Liquid Level**: Coolant level around cells (for immersion-cooled systems).

code for defining the features:

 `features = ['voltage', 'current', 'temperature', 'soc', 'soh', 'internal_resistance', 'cycle_count', 'discharge_time', 'charge_time', 'formation_energy', 'aging_time', 'ambient_temperature', 'pressure', 'liquid_level']`  #use PCA to incorportate important features

## Results

The early risk detection system has achieved the following outcomes:

- **Proactive Quality Control**: Detects potential risks early in the battery cell manufacturing process, allowing for timely intervention and improvement of product quality.
- **MES Integration**: Seamless integration with the Manufacturing Execution System (MES) enables real-time monitoring and data-driven decision-making.
- **Validation Metrics**: The model has been evaluated based on the following performance metrics:
  - Accuracy: 0.92
  - Precision: 0.88
  - Recall: 0.85
  - F1 Score: 0.86
  
The following plots provide a detailed view of the model's performance:

1. **Confusion Matrix**  
   ![Confusion Matrix](results/confusion_matrix.png)  
   This plot shows the classification performance between high-risk and low-risk battery cells.  
   - The model correctly predicted 25 high-risk cases and 25 low-risk cases.
   - It misclassified 31 high-risk cells as low-risk and 19 low-risk cells as high-risk.
   - The matrix gives insight into the overall classification accuracy, as well as areas where the model could be improved (i.e., reducing false positives and false negatives).

2. **ROC Curve**  
   ![ROC Curve](results/roc_curve.png)  
   - The ROC curve illustrates the trade-off between the true positive rate and false positive rate, with an improved AUC score of 0.84, showing better classification performance.
   - The curve shows how well the model distinguishes between the high-risk and low-risk classes at various thresholds.

3. **Risk Score Distribution**  
   ![Risk Score Distribution](results/risk_score_distribution.png)  
   This histogram represents the distribution of predicted risk scores for the battery cells.  
   - The x-axis represents the risk score, with higher scores indicating higher risk.
   - Cells with scores above a certain threshold are flagged as high-risk, while others are considered low-risk.
   - This distribution helps in identifying clusters of cells that require further testing or quality control actions.

4. **Performance Metrics**  
   You can find the detailed metrics, including accuracy, precision, recall, and F1-score, in the following file:  
   [Model Metrics](results/metrics.txt)


## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Author

- Usama Yasir Khan
- LinkedIn: [Usama Yasir Khan](https://www.linkedin.com/in/usama-yasir-khan-856803173)
