import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from sktime.datasets import load_italy_power_demand
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.dictionary_based import BOSSEnsemble


from sktime_ext.classification.deep_learning.tspulse import TSPulseClassifier


print("=" * 80)
print("TSPulse vs sktime Benchmark")
print("=" * 80)

# -------------------------------------------------
# 1. LOAD DATASET
# -------------------------------------------------
print("\n[1/4] Loading dataset...")

X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")


# -------------------------------------------------
# 2. TRAIN YOUR MODEL
# -------------------------------------------------
print("\n[2/4] Training TSPulseClassifier...")

tspulse = TSPulseClassifier(
    n_epochs=30,        # keep small for speed
    batch_size=16,
    freeze_backbone=False,
    learning_rate=1e-4,
    verbose=True
)

tspulse.fit(X_train, y_train)
y_pred_tpulse = tspulse.predict(X_test)


# -------------------------------------------------
# 3. TRAIN BASELINE MODELS
# -------------------------------------------------
print("\n[3/4] Training baseline models...")

# KNN (multivariate safe)
knn = KNeighborsTimeSeriesClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Rocket (strong baseline)
rocket = RocketClassifier()
rocket.fit(X_train, y_train)
y_pred_rocket = rocket.predict(X_test)


# -------------------------------------------------
# 4. EVALUATION
# -------------------------------------------------
print("\n[4/4] Results")
print("=" * 80)

def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name}")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Macro F1: {f1:.4f}")
    print("-" * 40)


evaluate("TSPulseClassifier", y_test, y_pred_tpulse)
evaluate("KNN", y_test, y_pred_knn)
evaluate("Rocket", y_test, y_pred_rocket)

print("=" * 80)