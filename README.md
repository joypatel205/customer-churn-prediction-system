# Telco Customer Churn Prediction (Baseline Model)

A machine learning project for predicting customer churn in telecommunications using baseline features with calibrated probability outputs.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Schema](#data-schema)
- [Model Architecture](#model-architecture)
- [API Usage](#api-usage)
- [Results](#results)
- [Configuration](#configuration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements a customer churn prediction system for telecommunications companies using machine learning. The system predicts the probability that a customer will churn (cancel their service) based on their usage patterns, service subscriptions, and demographic information.

### Key Capabilities
- **Binary Classification**: Predicts churn probability (0-1 scale)
- **Model Calibration**: Uses Platt scaling and Isotonic regression for reliable probabilities
- **Time-aware Splitting**: Respects temporal order in train/validation/test splits
- **REST API**: Production-ready FastAPI endpoint for real-time predictions
- **Batch Scoring**: Efficient processing of large customer datasets

## ✨ Features

- **Multiple Models**: Logistic Regression and Random Forest with calibration
- **Robust Preprocessing**: Handles missing values, categorical encoding, and feature scaling
- **Model Evaluation**: Comprehensive metrics including ROC-AUC, PR-AUC, and Brier Score
- **Calibration Analysis**: Reliability diagrams and calibration curves
- **Production Ready**: FastAPI service with health checks and error handling
- **Reproducible**: Fixed random seeds and versioned dependencies

## 📁 Project Structure

```
telco-churn-baseline/
├── src/                          # Source code
│   ├── api.py                   # FastAPI REST API
│   ├── config.py                # Configuration settings
│   ├── data_schema.py           # Data validation and types
│   ├── evaluate.py              # Model evaluation and calibration
│   ├── metrics.py               # Performance metrics
│   ├── preprocess.py            # Data preprocessing pipeline
│   ├── score.py                 # Batch scoring functionality
│   └── train.py                 # Model training pipeline
├── data/                        # Data directory
│   └── Telco_customer_churn.xlsx
├── artifacts/                   # Generated artifacts
│   ├── models/                  # Trained model files
│   │   ├── logreg_platt.joblib
│   │   └── rf_isotonic.joblib
│   ├── preprocessors/           # Preprocessing pipelines
│   │   └── baseline_preprocessor.joblib
│   └── reports/                 # Evaluation reports
│       ├── test_metrics.json
│       ├── reliability_diagram_test.png
│       ├── calibration_curve_test.csv
│       └── batch_scores_*.csv
├── notebooks/                   # Analysis notebooks
│   └── churn_reasons_eda.py
├── Makefile                     # Build automation
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
└── README.md                    # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Customer-Churn-Prediction-project/telco-churn-baseline
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify data file**
   Ensure `Telco_customer_churn.xlsx` is in the `data/` directory.

## ⚡ Quick Start

### Training and Evaluation
```bash
# Train models
make train

# Evaluate calibration
make eval

# Generate batch scores
make score

# Run all steps
make all
```

### Alternative Commands
```bash
# Direct Python execution
python -m src.train
python -m src.evaluate
python -m src.score
```

### API Server
```bash
# Start FastAPI server
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health
```

## 📊 Data Schema

### Input Features (19 features)

#### Categorical Features (16)
- **Demographics**: Gender, Senior Citizen, Partner, Dependents
- **Services**: Phone Service, Multiple Lines, Internet Service
- **Add-ons**: Online Security, Online Backup, Device Protection, Tech Support
- **Entertainment**: Streaming TV, Streaming Movies
- **Billing**: Contract, Paperless Billing, Payment Method

#### Numeric Features (3)
- **Tenure Months**: Customer relationship duration (0-72 months)
- **Monthly Charges**: Monthly service cost ($18.25-$118.75)
- **Total Charges**: Cumulative charges ($18.8-$8684.8)

#### Target Variable
- **Churn Label**: Binary (0=No Churn, 1=Churn)

### Data Preprocessing
- **Missing Values**: Imputed using median (numeric) and mode (categorical)
- **Categorical Encoding**: One-hot encoding for nominal variables
- **Feature Scaling**: StandardScaler for numeric features
- **Time-aware Split**: 70% train, 15% validation, 15% test

## 🤖 Model Architecture

### Supported Models

#### 1. Logistic Regression + Platt Scaling
- **Base Model**: L2-regularized logistic regression (C=1.0)
- **Calibration**: Sigmoid (Platt) scaling on validation set
- **Use Case**: Interpretable baseline with reliable probabilities

#### 2. Random Forest + Isotonic Regression
- **Base Model**: 400 trees, sqrt features, unlimited depth
- **Calibration**: Isotonic regression on validation set
- **Use Case**: Non-linear patterns with monotonic calibration

### Model Selection
- **Primary Metric**: PR-AUC (Precision-Recall Area Under Curve)
- **Secondary Metrics**: ROC-AUC, Brier Score
- **Calibration Quality**: Reliability diagrams and calibration curves

## 🔌 API Usage

### Start Server
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Prediction Endpoint

**POST** `/predict`

#### Request Format
```json
{
  "records": [
    {
      "Gender": "Female",
      "Senior Citizen": "0",
      "Partner": "Yes",
      "Dependents": "No",
      "Tenure Months": 12,
      "Phone Service": "Yes",
      "Multiple Lines": "No",
      "Internet Service": "DSL",
      "Online Security": "No",
      "Online Backup": "Yes",
      "Device Protection": "No",
      "Tech Support": "No",
      "Streaming TV": "No",
      "Streaming Movies": "No",
      "Contract": "Month-to-month",
      "Paperless Billing": "Yes",
      "Payment Method": "Electronic check",
      "Monthly Charges": 49.95,
      "TotalCharges": 599.4
    }
  ]
}
```

#### Response Format
```json
{
  "probabilities": [0.7234]
}
```

#### cURL Example
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "records": [{
         "Gender": "Female",
         "Tenure Months": 12,
         "Monthly Charges": 49.95,
         "Contract": "Month-to-month"
       }]
     }'
```

## 📈 Results

### Model Performance (Test Set)

| Model | ROC-AUC | PR-AUC | Brier Score |
|-------|---------|--------|-------------|
| **Logistic Regression + Platt** | **0.853** | **0.682** | **0.128** |
| Random Forest + Isotonic | 0.830 | 0.606 | 0.139 |

### Key Insights
- **Best Model**: Logistic Regression with Platt scaling
- **Churn Rate**: ~26.5% in test set
- **High-Risk Threshold**: Probability > 0.7 identifies top 15% risk customers
- **Calibration Quality**: Well-calibrated probabilities (reliability diagram available)

### Generated Artifacts
- **Model Files**: `artifacts/models/*.joblib`
- **Metrics**: `artifacts/reports/test_metrics.json`
- **Calibration Plot**: `artifacts/reports/reliability_diagram_test.png`
- **Batch Scores**: `artifacts/reports/batch_scores_*.csv`

## ⚙️ Configuration

### Key Parameters (`src/config.py`)
```python
RANDOM_STATE = 42                    # Reproducibility seed
TEST_SIZE_TIME_FRACTION = 0.15       # Test set size
VAL_SIZE_TIME_FRACTION = 0.15        # Validation set size
BASELINE_MODELS = ["logreg", "random_forest"]  # Models to train
```

### Environment Variables
- `MODEL_NAME`: Model file for API (default: `logreg_platt.joblib`)

### Model Hyperparameters

#### Logistic Regression
- Penalty: L2 regularization
- C: 1.0 (inverse regularization strength)
- Solver: liblinear
- Max iterations: 1000

#### Random Forest
- Estimators: 400 trees
- Max depth: Unlimited
- Min samples split: 2
- Min samples leaf: 1
- Max features: sqrt(n_features)

## 🛠️ Development

### Code Structure
- **Modular Design**: Separate modules for training, evaluation, and serving
- **Type Hints**: Full type annotations for better IDE support
- **Error Handling**: Comprehensive exception handling and logging
- **Testing**: Unit tests for core functionality (add `tests/` directory)

### Adding New Models
1. Update `BASELINE_MODELS` in `config.py`
2. Add model training logic in `train.py`
3. Update API schema if needed in `api.py`

### Custom Preprocessing
Modify `preprocess.py` to add:
- Feature engineering
- Advanced imputation strategies
- Custom transformations

## 🔧 Troubleshooting

### Common Issues

#### 1. Missing Data File
```
Error: FileNotFoundError: Telco_customer_churn.xlsx not found
```
**Solution**: Ensure data file is in `data/` directory

#### 2. Import Errors
```
Error: ModuleNotFoundError: No module named 'src'
```
**Solution**: Run commands from project root directory

#### 3. API Startup Errors
```
Error: RuntimeError: Model not found
```
**Solution**: Train models first using `make train`

#### 4. Memory Issues
```
Error: MemoryError during training
```
**Solution**: Reduce `n_estimators` in Random Forest or use smaller dataset

### Performance Optimization
- **Batch Size**: Process predictions in batches of 1000-10000 records
- **Parallel Processing**: Use `n_jobs=-1` for Random Forest
- **Memory Management**: Clear intermediate variables in large datasets

### Logging
Enable detailed logging by setting environment variable:
```bash
export PYTHONPATH=.
export LOG_LEVEL=DEBUG
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Check the troubleshooting section above
- Review the code documentation in `src/` modules

---

**Last Updated**: November 2024  
**Version**: 1.0.0
