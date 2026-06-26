# modelling.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib
import warnings

warnings.filterwarnings('ignore')

# Basic Kriteria 2 → wajib autolog
mlflow.sklearn.autolog()


def load_data():

    # Dataset sudah dipreprocessing pada Kriteria 1
    df = pd.read_csv('namadataset_preprocessing/titanic_preprocessing.csv')

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, X_test, y_test):

    mlflow.set_experiment("eksperimen_model")

    with mlflow.start_run():

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc_test = accuracy_score(y_test, y_pred)

        print("✅ Model trained successfully!")
        print(f"📊 Test Accuracy: {acc_test:.4f}")

        joblib.dump(model, "model.pkl")

    return model


if __name__ == "__main__":

    print("🚀 Starting Titanic Model Training...")

    X_train, X_test, y_train, y_test = load_data()

    model = train_model(
        X_train,
        y_train,
        X_test,
        y_test
    )

    print("✅ All files saved successfully!")
    print("📁 Files created:")
    print("   - model.pkl")