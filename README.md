# Machine Learning–Based Detection of Schizophrenia Using EEG Signals: A PCA-Optimized Classification Framework

## Overview
This project provides an end-to-end machine learning pipeline for detecting Schizophrenia using EEG (Electroencephalography) signals. It utilizes Principal Component Analysis (PCA) to optimize a set of extracted time and frequency domain features, which are then classified using models such as Support Vector Machines (SVM) and Random Forest.

An interactive **Streamlit** dashboard is also provided to let users simulate EEG data, visualize the signal, and see real-time predictions.

## Features
- **Data Simulation**: Generates synthetic multi-channel EEG signals for both Schizophrenia patients (SZ) and Healthy Controls (HC) to allow immediate testing without downloading massive physiological datasets.
- **Signal Preprocessing**: Implements standard techniques like baseline correction.
- **Feature Extraction**: Extracts clinically relevant features including Delta, Theta, Alpha, Beta, and Gamma band powers, as well as Log-Energy and Shannon Entropy.
- **Dimensionality Reduction**: Utilizes PCA to maintain 95% of data variance while reducing feature space dimensionality, combating the curse of dimensionality common with EEG data.
- **Machine Learning Classification**: Offers SVM and Random Forest frameworks to categorize the signals based on the optimized PCA components.
- **Interactive UI**: A sleek Streamlit application for end-user interaction.

## Project Structure
```text
schizophrenia-detection/
│
├── src/
│   ├── __init__.py
│   ├── data_simulation.py       # Simulates multi-channel EEG data
│   ├── preprocessing.py         # Handles data standardization and cleaning
│   ├── feature_extraction.py    # Extracts frequency and entropy features
│   ├── pca_optimizer.py         # Implements PCA and scaling pipelines
│   └── classifier.py            # Model training & classification logic
│
├── models/                      # Directory to store trained models
│
├── train.py                     # Main script to train and evaluate the full ML pipeline
├── app.py                       # Streamlit web application
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Getting Started

### 1. Setup Virtual Environment
It is recommended to use a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Train the Model
You can execute the entire pipeline (data generation, feature extraction, PCA, training, evaluation) by running:
```bash
python train.py
```
This script will output accuracy, classification reports, and save the trained PCA object and models into the `models/` directory.

### 4. Run the Web Application
To start the interactive Streamlit UI, run:
```bash
streamlit run app.py
```
This will open a local web server (typically `http://localhost:8501`) where you can interact with the system.

## Methodology Highlights
### Principal Component Analysis (PCA)
EEG signals typically output high-dimensional data which can lead to overfitting ("Curse of Dimensionality"). In this framework, PCA ensures that highly correlated features (redundant information across adjacent electrodes) are condensed into orthogonal components that capture the majority of the data's variance.

### Classification
Support Vector Machine (SVM) was selected as a default strong baseline due to its effectiveness in high-dimensional spaces. A Random Forest classifier is also available for comparative ensemble-based evaluation.
