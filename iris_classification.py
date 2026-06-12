#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset
df = pd.read_csv('Iris.csv')
df.drop(columns=['Id'], inplace=True)


#with some data exploration and visualization

print("=" * 60)
print("STEP 1: DATA EXPLORATION")
print("=" * 60)
print(f"\n• Shape  : {df.shape[0]} rows, {df.shape[1]} columns")
print(f"• Columns: {list(df.columns)}")
print("\n• First 5 rows:")
print(df.head())
print("\n• Class distribution (how many flowers per species):")
print(df["Species"].value_counts())
print("\n• Any missing values?", df.isnull().sum().sum(), "← 0 means none ")
print("\n• Basic statistics (min, max, mean, std):")
print(df.describe().round(2))

# Visualize the data

features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
palette  = {"Iris-setosa": "#E63946", "Iris-versicolor": "#457B9D",
            "Iris-virginica": "#2A9D8F"}

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Feature Distributions by Species", fontsize=16, fontweight="bold")

for ax, feat in zip(axes.flat, features):
    for species, color in palette.items():
        ax.hist(df[df["Species"] == species][feat], bins=15, alpha=0.6,
                label=species, color=color)
    ax.set_title(feat)
    ax.set_xlabel("cm")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("plot_distributions.png", dpi=150)
plt.close()
print("\n Saved: plot_distributions.png")

# Pair-plot style scatter matrix
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("Pairwise Feature Scatter Plots", fontsize=16, fontweight="bold")
pairs = [(f1, f2) for i, f1 in enumerate(features)
                for f2 in features[i+1:]]
for ax, (f1, f2) in zip(axes.flat, pairs):
    for species, color in palette.items():
        subset = df[df["Species"] == species]
        ax.scatter(subset[f1], subset[f2], label=species, color=color,
                alpha=0.7, s=30)
    ax.set_xlabel(f1, fontsize=8)
    ax.set_ylabel(f2, fontsize=8)
    ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig("plot_scatter.png", dpi=150)
plt.close()
print("Saved: plot_scatter.png")

#process the data and prepare for training

X = df[features].values          # shape: (150, 4)
y = df["Species"].values          # shape: (150,)

# Encode species names → integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)   # [0, 0, ..., 1, 1, ..., 2, 2, ...]
print("\n" + "=" * 60)
print("STEP 3: PREPROCESSING")
print("=" * 60)
print(f"\nLabel encoding map: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Train / Test split  (random_state fixes the randomness so results are reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")
# stratify=y_encoded ensures each class keeps its 50/50/50 ratio in both sets

# Scale features
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)   # learn mean & std from TRAIN only
X_test  = scaler.transform(X_test)        # apply SAME scale to TEST
print(" Features scaled with StandardScaler ")


#train and evaluate multiple models

models = {
    "K-Nearest Neighbours" : KNeighborsClassifier(n_neighbors=5),
    "Decision Tree"        : DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest"        : RandomForestClassifier(n_estimators=100, random_state=42),
}

print("\n" + "=" * 60)
print("STEP 4 & 5: TRAINING & EVALUATION")
print("=" * 60)

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)               # ← THE LEARNING HAPPENS HERE
    y_pred   = model.predict(X_test)          # predict on unseen test data
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {"model": model, "y_pred": y_pred, "accuracy": accuracy}
    print(f"\n{'─'*50}")
    print(f"MODEL: {name}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print("  Classification Report:")
    report = classification_report(y_test, y_pred,target_names=le.classes_, digits=3)
    # indent for readability
    for line in report.splitlines():
        print("    " + line)


#visualize confusion matrices for each model


fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold")

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=0.5, cbar=False)
    ax.set_title(f"{name}\n({res['accuracy']*100:.1f}% acc)", fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

plt.tight_layout()
plt.savefig("plot_confusion_matrices.png", dpi=150)
plt.close()
print("\n Saved: plot_confusion_matrices.png")

# Accuracy comparison bar chart
fig, ax = plt.subplots(figsize=(8, 5))
names    = list(results.keys())
accs     = [results[n]["accuracy"] * 100 for n in names]
colors   = ["#E63946", "#457B9D", "#2A9D8F"]
bars     = ax.bar(names, accs, color=colors, width=0.5, edgecolor="white")
ax.set_ylim(90, 102)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.savefig("plot_accuracy_comparison.png", dpi=150)
plt.close()
print(" Saved: plot_accuracy_comparison.png")

# Feature importance (Random Forest only)
rf_model    = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
indices     = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(4), importances[indices], color="#2A9D8F", edgecolor="white")
ax.set_xticks(range(4))
ax.set_xticklabels([features[i] for i in indices], rotation=20, ha="right")
ax.set_ylabel("Importance Score")
ax.set_title("Random Forest — Feature Importances\n(which features matter most?)",
            fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_feature_importance.png", dpi=150)
plt.close()
print(" Saved: plot_feature_importance.png")


#predict and use the best model to predict new samples

best_name  = max(results, key=lambda n: results[n]["accuracy"])
best_model = results[best_name]["model"]

new_flowers = np.array([
    [5.1, 3.5, 1.4, 0.2],   # looks like setosa
    [6.5, 3.0, 5.2, 2.0],   # looks like virginica
    [5.9, 2.8, 4.5, 1.5],   # looks like versicolor
])

new_flowers_scaled = scaler.transform(new_flowers)
predictions        = best_model.predict(new_flowers_scaled)
probabilities      = best_model.predict_proba(new_flowers_scaled)

print("\n" + "=" * 60)
print(f"STEP 6: PREDICTIONS (using {best_name})")
print("=" * 60)
header = f"{'SepalL':>8} {'SepalW':>8} {'PetalL':>8} {'PetalW':>8}  {'Prediction':<20} Confidence"
print("\n" + header)
print("─" * len(header))
for flower, pred, prob in zip(new_flowers, predictions, probabilities):
    species    = le.inverse_transform([pred])[0]
    confidence = prob.max() * 100
    vals       = "  ".join(f"{v:>6.1f}" for v in flower)
    print(f"  {vals}  {species:<20} {confidence:.1f}%")

print("\n All done! Check the saved PNG plots for visuals.")
print(f" Best model: {best_name}")
