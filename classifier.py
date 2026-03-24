from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_svm(X_train, y_train, C=1.0, kernel='rbf', random_state=42):
    """
    Trains a Support Vector Machine classifier.
    """
    model = SVC(C=C, kernel=kernel, probability=True, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Trains a Random Forest classifier.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model.
    Returns:
        accuracy: float
        report: string
        cm: ndarray
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Schizophrenia (1)'])
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, cm
