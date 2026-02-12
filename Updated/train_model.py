# train_model_with_plots.py
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize

def main():
    parser = argparse.ArgumentParser(description="Train RandomForest and plot Accuracy, ROC, Confusion Matrix.")
    parser.add_argument("--csv", type=str, default="es.csv",
                        help="Path to the input CSV file.")
    parser.add_argument("--target", type=str, default="Optimal for Turbine",
                        help="Name of the target column.")
    parser.add_argument("--model_out", type=str, default="wind_turbine_model_complex.pkl",
                        help="Where to save the trained model.")
    parser.add_argument("--acc_png", type=str, default="accuracy.png",
                        help="Where to save the accuracy bar chart.")
    parser.add_argument("--roc_png", type=str, default="roc_curve.png",
                        help="Where to save the ROC curve plot.")
    parser.add_argument("--cm_png", type=str, default="confusion_matrix.png",
                        help="Where to save the confusion matrix plot.")
    parser.add_argument("--report_txt", type=str, default="classification_report.txt",
                        help="Where to save the classification report.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test size fraction for train/test split.")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed.")
    args = parser.parse_args()

    # === 1) Load data ===
    print(f"Loading data from: {args.csv}")
    data = pd.read_csv(args.csv)

    if args.target not in data.columns:
        raise ValueError(f"Target column '{args.target}' not found. Available: {list(data.columns)}")

    X = data.drop(columns=[args.target])
    y = data[args.target]

    # One-hot encode any categorical feature columns
    X = pd.get_dummies(X, drop_first=True)

    # === 2) Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y if len(np.unique(y))>1 else None
    )

    # === 3) Train model (similar specs) ===
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # === 4) Accuracy ===
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.4f}")

    # Accuracy plot
    plt.figure()
    plt.bar(["Accuracy"], [acc])
    plt.ylim(0, 1)
    plt.title("Test Accuracy")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(args.acc_png, dpi=150)
    plt.close()
    print(f"Saved accuracy plot to: {args.acc_png}")

    # === 5) ROC Curve (robust for binary & multiclass) ===
    classes = np.array(model.classes_)
    n_classes = len(classes)

    print("Classes seen by model:", classes)

    plt.figure()
    if n_classes == 2:
        # Binary case
        if hasattr(model, "predict_proba"):
            # By convention, column 1 corresponds to classes[1]
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        pos_label = classes[1]
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)

        # Shape sanity
        print("Binary ROC shapes:")
        print("y_test shape:", np.array(y_test).shape)
        print("y_score shape:", np.array(y_score).shape)

        plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")

    else:
        # Multiclass micro-average
        y_test_bin = label_binarize(y_test, classes=classes)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)  # (n_samples, n_classes)
            y_score = np.asarray(y_score)
        else:
            y_score = model.decision_function(X_test)

        # Shape sanity
        print("Multiclass ROC shapes:")
        print("y_test_bin shape:", np.array(y_test_bin).shape)
        print("y_score shape:", np.array(y_score).shape)

        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f"Micro-average ROC (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Multiclass, micro-average)")
        plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(args.roc_png, dpi=150)
    plt.close()
    print(f"Saved ROC plot to: {args.roc_png}")

    # === 6) Confusion Matrix ===
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(args.cm_png, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to: {args.cm_png}")
    
    # Print confusion matrix to console
    print("Confusion Matrix:")
    print(cm)

    # === 7) Classification Report ===
    report = classification_report(y_test, y_pred, target_names=[str(c) for c in classes])
    
    # Save to file
    with open(args.report_txt, 'w') as f:
        f.write("Classification Report\n")
        f.write("=====================\n\n")
        f.write(report)
    
    # Print to console
    print("\nClassification Report:")
    print(report)
    print(f"Saved classification report to: {args.report_txt}")

    # === 8) Save model ===
    joblib.dump(model, args.model_out)
    print(f"Saved model to: {args.model_out}")

if __name__ == "__main__":
    main()