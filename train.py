import os
import joblib
from sklearn.model_selection import train_test_split
from src.data_loader import load_real_eeg_data
from src.preprocessing import baseline_correction
from src.feature_extraction import extract_features
from src.pca_optimizer import PCAOptimizer
from src.classifier import train_svm, evaluate_model

def main():
    print("--- Machine Learning Framework for Schizophrenia Detection ---")
    
    # 1. Load Real Data
    print("\n1. Loading Real EEG Dataset...")
    dataset_path = "dataset_text (3)"
    if not os.path.exists(dataset_path):
        print("Dataset directory not found!")
        return
        
    X_raw, y, sfreq = load_real_eeg_data(dataset_path)
    print(f"Loaded data shape: {X_raw.shape}")
    print(f"Normal: {sum(y==0)}, Schizophrenic: {sum(y==1)}")
    
    # 2. Preprocessing
    print("2. Preprocessing Signals (Baseline Correction)...")
    X_preprocessed = baseline_correction(X_raw)
    
    # 3. Feature Extraction
    print("3. Extracting Features (Band Power & Entropy)...")
    X_features = extract_features(X_preprocessed, sfreq)
    print(f"Extracted feature matrix shape: {X_features.shape}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    
    # 4. PCA Optimization
    print("4. Optimizing Features with PCA...")
    pca_opt = PCAOptimizer(n_components=0.95) # Keep 95% variance
    X_train_pca = pca_opt.fit_transform(X_train)
    X_test_pca = pca_opt.transform(X_test)
    print(f"Reduced features from {X_train.shape[1]} to {X_train_pca.shape[1]}")
    
    # 5. Classification
    print("5. Training SVM Classifier...")
    svm_model = train_svm(X_train_pca, y_train, C=1.0, kernel='rbf')
    
    # Evaluation
    print("\n--- Evaluation Results ---")
    acc, report, cm = evaluate_model(svm_model, X_test_pca, y_test)
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(report)
    
    # 6. Save Models
    print("\n6. Saving Pipeline...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(pca_opt, 'models/pca_optimizer.pkl')
    joblib.dump(svm_model, 'models/svm_model.pkl')
    print("Saved 'pca_optimizer.pkl' and 'svm_model.pkl' to models directory.")
    print("Pipeline Execution Complete!")

if __name__ == "__main__":
    main()
