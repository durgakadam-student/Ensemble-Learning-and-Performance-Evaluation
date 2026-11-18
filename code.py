%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import (
    StackingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load Dataset
RS = 42

data = load_breast_cancer()
X, y = data.data, data.target

print("Dataset Shape:", X.shape)
print("Class Counts:", np.bincount(y))
print("Features:", X.shape[1])

# 2. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RS
)

# 3. Base Classifiers
lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=RS)
)

knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=7)
)

svm = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", probability=True, random_state=RS)
)

dt = DecisionTreeClassifier(max_depth=4, random_state=RS)

nb = GaussianNB()

base = [
    ("LR", lr),
    ("KNN", knn),
    ("SVM", svm),
    ("DT", dt),
    ("NB", nb)
]

# 4. Ensemble Classifiers
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RS
)

stack = StackingClassifier(
    estimators=base,
    final_estimator=LogisticRegression(
        max_iter=1000,
        random_state=RS
    ),
    cv=cv
)

vote_hard = VotingClassifier(
    estimators=base,
    voting="hard"
)

ada_stump = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0,
    random_state=RS
)

ada_dt3 = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=100,
    learning_rate=0.5,
    random_state=RS
)

# 5. Define All Models
models = {
    "Logistic Regression": lr,
    "KNN": knn,
    "SVM (RBF)": svm,
    "Decision Tree": dt,
    "Naive Bayes": nb,
    "Voting (Hard)": vote_hard,
    "AdaBoost (Stump)": ada_stump,
    "AdaBoost (DT depth 3)": ada_dt3,
    "Stacking (LR meta)": stack
}

# 6. Train and Evaluate Models
rows = []
preds = {}

print("\n========== MODEL TRAINING ==========\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    preds[name] = prediction

    # ROC-AUC
    if hasattr(model, "predict_proba"):
        try:
            auc = roc_auc_score(
                y_test,
                model.predict_proba(X_test)[:, 1]
            )
        except (AttributeError, ValueError):
            auc = np.nan
    else:
        auc = np.nan

    # Cross Validation
    cvs = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    # Store Results
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(y_test, prediction),
        "Recall": recall_score(y_test, prediction),
        "F1": f1_score(y_test, prediction),
        "ROC_AUC": auc,
        "CV_Mean": cvs.mean(),
        "CV_Std": cvs.std()
    })

    print(name, "completed")

# 7. Display Results
df = pd.DataFrame(rows).round(4)

print("\n========== MODEL PERFORMANCE COMPARISON ==========\n")

display(df)

# Save CSV
df.to_csv("results.csv", index=False)

# 8. Graph 1: Metrics Comparison
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(df))
w = 0.2

for i, metric in enumerate(
    ["Accuracy", "Precision", "Recall", "F1"]
):
    ax.bar(
        x + i * w,
        df[metric],
        w,
        label=metric
    )

ax.set_xticks(x + 1.5 * w)
ax.set_xticklabels(
    df["Model"],
    rotation=30,
    ha="right"
)

ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_xlabel("Machine Learning Models")
ax.set_title("Base vs Ensemble Models - Test Set Metrics")
ax.legend(ncol=4)

plt.tight_layout()
plt.savefig("metrics_comparison.png", dpi=150)
plt.show()
plt.close()

# 9. Graph 2: Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

selected_models = [
    "Voting (Hard)",
    "AdaBoost (Stump)",
    "Stacking (LR meta)"
]

for ax, name in zip(axes, selected_models):

    cm = confusion_matrix(y_test, preds[name])

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=data.target_names
    ).plot(
        ax=ax,
        colorbar=False,
        cmap="Blues"
    )

    ax.set_title(name)

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()
plt.close()

# 10. Graph 3: Cross Validation Accuracy
fig, ax = plt.subplots(figsize=(10, 6))

d2 = df.sort_values("CV_Mean")

ax.barh(
    d2["Model"],
    d2["CV_Mean"],
    xerr=d2["CV_Std"],
    color="steelblue"
)

ax.set_xlim(0.85, 1.0)
ax.set_xlabel("5-Fold CV Accuracy")
ax.set_ylabel("Machine Learning Models")
ax.set_title("Cross-Validation Accuracy (Mean ± Std)")

plt.tight_layout()
plt.savefig("cv_accuracy.png", dpi=150)
plt.show()
plt.close()

# 11. Save Confusion Matrix Values
cm_results = {
    name: confusion_matrix(y_test, prediction).tolist()
    for name, prediction in preds.items()
}

with open("cm.json", "w") as file:
    json.dump(cm_results, file, indent=4)

print("\n========== CONFUSION MATRIX VALUES ==========\n")

for name, cm in cm_results.items():
    print(name, ":", cm)

# 12. Final Output
print("\n========== PROJECT COMPLETED ==========")

print("All models trained and evaluated successfully.")
print("All three graphs displayed successfully.")

print("\nGenerated Files:")
print("1. results.csv")
print("2. metrics_comparison.png")
print("3. confusion_matrices.png")
print("4. cv_accuracy.png")
print("5. cm.json")
