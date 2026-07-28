import joblib
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from xgboost import plot_importance

from src.config import MODEL_DIR, REPORT_DIR, CLASS_NAMES


def evaluate():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    X = np.load("data/X.npy")
    y = np.load("data/y.npy")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    X_test = scaler.transform(X_test)

    svm = joblib.load(MODEL_DIR / "svm_model.pkl")
    rf = joblib.load(MODEL_DIR / "random_forest.pkl")
    xgb = joblib.load(MODEL_DIR / "xgboost_model.pkl")

    models = {
        "SVM": svm,
        "Random Forest": rf,
        "XGBoost": xgb,
    }

    scores = {}

    best_pred = None

    for name, model in models.items():

        print(f"\nEvaluating {name}")

        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)

        scores[name] = acc

        print(f"Accuracy : {acc:.4f}")

        if name == "XGBoost":
            best_pred = pred

            report = classification_report(
                y_test,
                pred,
                target_names=CLASS_NAMES,
            )

            with open(
                REPORT_DIR / "classification_report.txt",
                "w",
            ) as f:
                f.write(report)

    print("\nGenerating Confusion Matrix...")

    cm = confusion_matrix(y_test, best_pred)

    plt.figure(figsize=(8, 6))

    plt.imshow(cm)

    plt.title("Confusion Matrix")

    plt.colorbar()

    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)

    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(REPORT_DIR / "confusion_matrix.png")

    plt.close()

    print("Generating Model Comparison...")

    plt.figure(figsize=(6, 4))

    plt.bar(scores.keys(), scores.values())

    plt.ylabel("Accuracy")

    plt.ylim(0, 1)

    for i, value in enumerate(scores.values()):
        plt.text(i, value + 0.01, f"{value:.3f}", ha="center")

    plt.tight_layout()

    plt.savefig(REPORT_DIR / "model_comparison.png")

    plt.close()

    print("Generating Feature Importance...")

    plt.figure(figsize=(10, 8))

    plot_importance(
        xgb,
        max_num_features=20,
        importance_type="gain",
    )

    plt.tight_layout()

    plt.savefig(REPORT_DIR / "feature_importance.png")

    plt.close()

    best_model = max(scores, key=scores.get)

    with open(
        REPORT_DIR / "training_summary.txt",
        "w",
    ) as f:

        f.write("IMAGE CLASSIFICATION PROJECT\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Dataset Size : {len(X)}\n")

        f.write(f"Feature Count : {X.shape[1]}\n\n")

        f.write("Model Accuracies\n")

        for model, score in scores.items():
            f.write(f"{model:<20}: {score:.4f}\n")

        f.write(f"\nBest Model : {best_model}\n")

    print("\nReports Generated Successfully!")

    print(REPORT_DIR)


if __name__ == "__main__":
    evaluate()