import joblib
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.config import MODEL_DIR


def train():

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    X = np.load("data/X.npy")
    y = np.load("data/y.npy")

    print(f"Feature Matrix : {X.shape}")
    print(f"Labels         : {y.shape}")

    # ==========================================================
    # Train Test Split
    # ==========================================================

    print("\nSplitting Dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    # ==========================================================
    # Feature Scaling
    # ==========================================================

    print("\nScaling Features...")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        scaler,
        MODEL_DIR / "scaler.pkl",
    )

    # ==========================================================
    # SVM
    # ==========================================================

    print("\n" + "=" * 60)
    print("Training SVM")
    print("=" * 60)

    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        random_state=42,
    )

    svm.fit(X_train, y_train)

    svm_pred = svm.predict(X_test)

    svm_acc = accuracy_score(y_test, svm_pred)

    print(f"\nSVM Accuracy : {svm_acc:.4f}\n")

    print(classification_report(y_test, svm_pred))

    print("Confusion Matrix")

    print(confusion_matrix(y_test, svm_pred))

    joblib.dump(
        svm,
        MODEL_DIR / "svm_model.pkl",
    )

    # ==========================================================
    # Random Forest
    # ==========================================================

    print("\n" + "=" * 60)
    print("Random Forest Hyperparameter Tuning")
    print("=" * 60)

    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=20,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
        verbose=2,
    )

    search.fit(X_train, y_train)

    rf_best = search.best_estimator_

    print("\nBest Parameters")

    print(search.best_params_)

    rf_pred = rf_best.predict(X_test)

    rf_acc = accuracy_score(
        y_test,
        rf_pred,
    )

    print(f"\nRandom Forest Accuracy : {rf_acc:.4f}\n")

    print(classification_report(y_test, rf_pred))

    print("Confusion Matrix")

    print(confusion_matrix(y_test, rf_pred))

    joblib.dump(
        rf_best,
        MODEL_DIR / "random_forest.pkl",
    )

    # ==========================================================
    # XGBoost
    # ==========================================================

    print("\n" + "=" * 60)
    print("Training XGBoost")
    print("=" * 60)

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y)),
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
    )

    xgb.fit(X_train, y_train)

    xgb_pred = xgb.predict(X_test)

    xgb_acc = accuracy_score(
        y_test,
        xgb_pred,
    )

    print(f"\nXGBoost Accuracy : {xgb_acc:.4f}\n")

    print(classification_report(y_test, xgb_pred))

    print("Confusion Matrix")

    print(confusion_matrix(y_test, xgb_pred))

    joblib.dump(
        xgb,
        MODEL_DIR / "xgboost_model.pkl",
    )

    # ==========================================================
    # Model Comparison
    # ==========================================================

    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)

    results = {
        "SVM": svm_acc,
        "Random Forest": rf_acc,
        "XGBoost": xgb_acc,
    }

    for model, score in sorted(
        results.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{model:<20}: {score:.4f}")

    best_model_name = max(
        results,
        key=results.get,
    )

    print("\n" + "=" * 60)
    print(f"🏆 Best Model : {best_model_name}")
    print("=" * 60)

    if best_model_name == "SVM":
        best_model = svm

    elif best_model_name == "Random Forest":
        best_model = rf_best

    else:
        best_model = xgb

    joblib.dump(
        best_model,
        MODEL_DIR / "best_model.pkl",
    )

    print("\nSaved Models:")

    print("✓ scaler.pkl")
    print("✓ svm_model.pkl")
    print("✓ random_forest.pkl")
    print("✓ xgboost_model.pkl")
    print("✓ best_model.pkl")

    print("\nTraining Completed Successfully!")


if __name__ == "__main__":
    train()