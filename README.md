## Auto Insurance Fraud Detection – ML Inference Demonstration

### Overview
This project demonstrates an end-to-end machine learning workflow for motor insurance fraud detection, from exploratory data analysis and feature engineering through model development, evaluation, explainability and reusable inference.
The project uses a public motor insurance claims dataset. Its primary purpose is to explore how additional contextual and temporal information can improve fraud-risk modelling, while demonstrating how a trained machine-learning model can be separated from the development environment and reused for inference on new data.
The final model is an XGBoost binary classifier. The trained model and its required feature schema are saved as reusable artefacts and subsequently loaded by a separate Python inference script.

### Analytical Approach
The project investigates whether fraud detection can be strengthened by engineering information not directly represented in the original raw variables.
Two areas were explored in particular.

#### External Economic Data
West Texas Intermediate (WTI) crude oil price data was integrated with the insurance dataset by incident date.
This demonstrates integrating an external time-series data source into an existing analytical dataset and explores whether broader economic conditions contain useful information for fraud-risk modelling.

#### Cyclical Representation of Incident Time
Incident hour was transformed using sine and cosine cyclical encoding:
```math
sin(2πt/24)
```

```math
cos(2πt/24)
```
This represents time as a continuous daily cycle rather than a simple linear variable. For example, incidents at 23:00 and 00:00 are treated as temporally close rather than as opposite ends of a numeric scale.
This allows the model to investigate potential differences in fraud risk across different periods of the day while preserving the cyclical nature of time.

### Model Development
The modelling workflow includes:
- data cleaning and preprocessing;
- integration of external economic data;
- feature engineering;
- stratified train/test splitting;
- XGBoost classification;
- feature selection using permutation importance;
- cross-validation of alternative feature sets;
- model evaluation using PR-AUC, ROC-AUC, precision, recall and F1;
- model explainability using SHAP and permutation importance; and
- decision-threshold analysis, including False Discovery Rate (FDR).
The final XGBoost model is trained using the selected stable feature set.

### Model Integration
After training, the final model is saved as an XGBoost model artefact:
```
model/fraud_xgb_model.json
```

The exact feature schema expected by the model is stored separately:
```
model/features.json
```

This separates model training from inference and ensures that new observations are supplied to the model using the same feature structure used during training.
`predict.py` loads both artefacts independently of the training notebook, validates that the required features are present, and generates fraud probabilities for new observations.

### Sample Inference
A random subset of the held-out test data is provided in:
```
data/sample_inference.csv
```
The sample is used only to demonstrate the inference process and was not used to train the final model.
Run:
```
python predict.py
```

The script:
1. loads the saved XGBoost model;
2. loads the required feature schema;
3. reads the sample inference dataset;
4. checks that all required model features are available;
5. orders the input according to the saved feature schema; and
6. generates fraud probabilities using the trained model.
Example output:
```
   fraud_probability
0           0.742...
1           0.083...
2           0.316...
```

### Project Structure
```
.
├── Auto Insurance Fraud Detection For ASD.ipynb
├── predict.py
├── requirements.txt
├── model/
│   ├── fraud_xgb_model.json
│   └── features.json
└── data/
    └── sample_inference.csv
```
The Jupyter notebook contains the exploratory analysis, feature engineering, model development, evaluation and explainability workflow.
predict.py provides the standalone inference component.

### Installation
Clone the repository and install the required Python packages:
```
pip install -r requirements.txt
```

Then run:
```
python predict.py
```

### Scope and Limitations
This project is a small demonstration of machine-learning model development and reusable inference, not a production MLOps system.
The inference script expects preprocessed, model-ready features. Productionising the complete upstream data ingestion and feature-engineering pipeline is outside the scope of this demonstration.
A production implementation would also require additional controls such as automated testing, model and data monitoring, logging, version management, deployment infrastructure and appropriate security controls.

### Data
The machine-learning model uses a public motor insurance fraud dataset for analysis and demonstration.
External WTI crude-oil price data is sourced from the Federal Reserve Economic Data (FRED) series DCOILWTICO – Crude Oil Prices: West Texas Intermediate (WTI), Cushing, Oklahoma.

### Technologies
Python, pandas, NumPy, scikit-learn, XGBoost, SHAP and Jupyter Notebook.
