import os
import time
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize


def evaluate_model(model, X_test, y_test,
                   model_name='model',
                   save_dir='../reports',
                   model_dir='../models'):
    # Ensure folders exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Prediction Time
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    # Accuracy, F1, Confusion Matrix
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    f1_macro = class_report['macro avg']['f1-score']
    f1_weighted = class_report['weighted avg']['f1-score']
    cm = confusion_matrix(y_test, y_pred)

    # Save Confusion Matrix Plot
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Save Classification Report
    report_path = os.path.join(save_dir, f"{model_name}_classification_report.json")
    with open(report_path, "w") as f:
        json.dump(class_report, f, indent=4)

    # Binarize labels for ROC AUC
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)

    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            raise ValueError("Model does not support probability estimation")

        # ROC AUC (macro)
        roc_auc = roc_auc_score(y_test_bin, y_score, average='macro', multi_class='ovr')

        # Plot ROC curve for a few selected classes
        plt.figure(figsize=(10, 6))
        for i in [0, 3, 5, 8]:  # Pick interesting digits
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            plt.plot(fpr, tpr, label=f'Class {i}')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.title(f"ROC Curves - {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        roc_path = os.path.join(save_dir, f"{model_name}_roc_curve.png")
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()
    except Exception as e:
        print(f"[!] ROC AUC Error: {e}")
        roc_auc = None
        roc_path = None

    # Save the Model
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    model_size_kb = os.path.getsize(model_path) / 1024

    # Return summary
    metrics = {
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'F1 Macro': round(f1_macro, 4),
        'F1 Weighted': round(f1_weighted, 4),
        'ROC AUC (macro)': round(roc_auc, 4) if roc_auc else 'N/A',
        'Prediction Time (s)': round(predict_time, 4),
        'Model Size (KB)': round(model_size_kb, 2),
        'Confusion Matrix Path': cm_path,
        'Classification Report Path': report_path,
        'Model Path': model_path,
        'ROC Curve Path': roc_path or "N/A"
    }

    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics
