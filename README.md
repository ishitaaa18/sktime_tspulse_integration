# TSPulse Classifier - Proper sktime Integration

**Status:** ✅ Complete - True sktime integration (inherits from BaseClassifier)

---

## Quick Start

### 1. Test Integration (2 minutes)
```bash
python test_sktime_integration.py
```
Verifies proper sktime integration.

### 2. Test on Real Data (10 minutes)
```bash
python test_real_datasets.py
```
Tests on GunPoint, ItalyPowerDemand, ArrowHead datasets.

### 3. Full Training (20 minutes)
```bash
python train_real_dataset.py
```
100 epochs with early stopping on real datasets.

---

## What Makes This a Proper Integration

### Before (Wrapper):
```python
class TSPulseClassifier:  # Standalone class
    def fit(self, X, y):
        # Manual everything
```

### After (Proper Integration):
```python
class TSPulseClassifier(BaseClassifier):  # Inherits from sktime
    def _fit(self, X, y):
        # BaseClassifier handles infrastructure
```

**Key Differences:**
- ✅ Inherits from `BaseClassifier`
- ✅ Implements `_fit`, `_predict`, `_predict_proba`
- ✅ Automatic label encoding/decoding
- ✅ Compatible with sktime pipelines & CV
- ✅ Follows sktime protocols

---

## Installation

### 1. Clone granite-tsfm
```bash
cd ..
git clone https://github.com/ibm-granite/granite-tsfm.git
cd tpulse-sktime-integration
```

### 2. Install dependencies
```bash
pip install torch transformers scikit-learn pandas numpy scipy
pip install sktime
```

---

## Usage

### Basic Usage:
```python
from sktime.classification.deep_learning import TSPulseClassifier
from sktime.datasets import load_gunpoint

# Load data
X_train, y_train = load_gunpoint(split="train", return_X_y=True)
X_test, y_test = load_gunpoint(split="test", return_X_y=True)

# Create classifier
clf = TSPulseClassifier(
    n_epochs=50,
    batch_size=32,
    freeze_backbone=True,
    early_stopping=True,
    early_stopping_patience=10,
    verbose=True,
    tsfm_path='../granite-tsfm'
)

# Train and predict
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
```

### NEW: Works with sktime!
```python
# Pipelines
from sktime.pipeline import make_pipeline
pipeline = make_pipeline(TSPulseClassifier(n_epochs=50))
pipeline.fit(X_train, y_train)

# Cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=5)
```

---

## Architecture

### Inheritance Chain:
```
TSPulseClassifier
    └── BaseClassifier (sktime.classification.base)
        └── BasePanelMixin
            └── BaseEstimator
```

### Model:
- **Base:** TSPulseForClassification from tsfm_public
- **No custom modifications** (architecturally correct)
- **Follows** official IBM Granite patterns
- **Parameters:** ~1M (pre-trained)
- **Context length:** 512 timesteps (auto-padded)

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_epochs` | 50 | Training epochs |
| `batch_size` | 32 | Batch size |
| `learning_rate` | None | LR (auto-detected if None) |
| `freeze_backbone` | True | Freeze backbone (train head only) |
| `early_stopping` | False | Enable early stopping |
| `early_stopping_patience` | 10 | Patience for early stopping |
| `random_state` | 42 | Random seed |
| `verbose` | True | Print training progress |
| `tsfm_path` | None | Path to granite-tsfm repo |

---

## Testing

### 1. Integration Test
```bash
python test_sktime_integration.py
```
**Checks:**
- Inheritance from BaseClassifier
- Required methods implemented
- sktime tags set correctly
- Training works
- Predictions work

### 2. Real Dataset Test
```bash
python test_real_datasets.py
```
**Tests on:**
- GunPoint (Easy): Expected 95-100%
- ItalyPowerDemand (Medium): Expected 90-97%
- ArrowHead (Hard): Expected 75-85%

### 3. Full Training
```bash
python train_real_dataset.py
```
**Features:**
- 100 epochs with early stopping
- Train/val loss monitoring
- Macro F1 tracking
- Full evaluation metrics

---

## Expected Performance

| Dataset | Difficulty | Expected Accuracy |
|---------|-----------|-------------------|
| GunPoint | Easy | 95-100% |
| ItalyPowerDemand | Medium | 90-97% |
| ArrowHead | Hard (3 classes) | 75-85% |
| ECG200 | Hard | 85-92% |

---

## File Structure

```
tpulse-sktime-integration/
├── sktime/classification/deep_learning/
│   ├── __init__.py
│   └── tspulse.py              ← MAIN FILE (580 lines)
├── test_sktime_integration.py  ← Integration test
├── test_real_datasets.py       ← Real data test
├── train_real_dataset.py       ← Full training
├── quick_test.py               ← Quick validation
├── README.md                   ← This file
└── TESTING_GUIDE.md           ← Testing instructions
```

---

## Integration Checklist

- [x] Inherits from BaseClassifier
- [x] Implements `_fit(X, y)`
- [x] Implements `_predict(X)`
- [x] Implements `_predict_proba(X)`
- [x] Sets `_tags` dictionary
- [x] Calls `super().__init__()`
- [x] Returns integers from `_predict()`
- [x] Handles numpy3D input
- [x] Compatible with sktime pipelines
- [x] Compatible with sklearn CV
- [x] Automatic label encoding/decoding
- [x] Uses TSPulseForClassification directly
- [x] NO custom backbone modifications

---

## Troubleshooting

### Low Accuracy?
- Increase epochs: `n_epochs=100`
- Unfreeze backbone: `freeze_backbone=False`
- More data or data augmentation

### Training Slow?
- Reduce batch size: `batch_size=16`
- Reduce epochs: `n_epochs=20`
- Use GPU (automatic if available)

### Import Errors?
- Check `tsfm_path` points to granite-tsfm
- Install dependencies: `pip install torch transformers`

---

## References

- **TSPulse Paper:** https://arxiv.org/abs/2505.13033
- **IBM Tutorial:** https://www.ibm.com/think/tutorials/time-series-analysis-granite-time-tspulse
- **HuggingFace Model:** https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1
- **GitHub:** https://github.com/ibm-granite/granite-tsfm
- **sktime:** https://www.sktime.net/

---

## Summary

This is a **proper sktime integration**, not a wrapper:
- ✅ Inherits from BaseClassifier
- ✅ Follows sktime protocols
- ✅ Compatible with sktime ecosystem
- ✅ Uses TSPulseForClassification directly
- ✅ Ready for production
- ✅ Ready for contribution to sktime

**Test it:** `python test_real_datasets.py`
