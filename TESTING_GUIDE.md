# Testing Guide - TSPulse Classifier

## 🎯 Available Tests

We provide 4 different testing scripts for various use cases:

---

## 1. Quick Test (Recommended First)

**File:** `quick_test.py`

**Purpose:** Fast validation that the model is learning

```bash
python quick_test.py
```

**Details:**
- **Time:** 2-3 minutes
- **Epochs:** 10
- **Dataset:** Challenging synthetic (3 classes, high noise)
- **Expected Accuracy:** 95-98%

**When to use:**
- First time setup verification
- After making code changes
- Quick sanity check

---

## 2. Full Training (Main Testing)

**File:** `train_real_dataset.py`

**Purpose:** Comprehensive training on real datasets with 100 epochs

```bash
python train_real_dataset.py
```

**Details:**
- **Time:** 10-20 minutes
- **Epochs:** Up to 100 (early stopping enabled)
- **Datasets:** Tries multiple real datasets in order:
  1. **ItalyPowerDemand** (Medium) - 90-97% expected
  2. **ArrowHead** (Hard, 3 classes) - 75-85% expected
  3. **GunPoint** (Easy) - 95-100% expected
  4. **BasicMotions** - ~100% expected
  5. **Challenging Synthetic** (fallback) - 60-75% expected

**Features:**
- ✅ Early stopping with patience=10
- ✅ Train/Val loss tracking
- ✅ Macro F1 monitoring
- ✅ Prediction distribution check
- ✅ Full classification report
- ✅ Confusion matrix

**When to use:**
- Final validation before production
- Performance benchmarking
- Demonstrating to mentors

---

## 3. UCR Benchmark Suite

**File:** `test_ucr_datasets.py`

**Purpose:** Test on multiple standard UCR archive datasets

```bash
python test_ucr_datasets.py
```

**Details:**
- **Time:** 15-30 minutes
- **Epochs:** 50 per dataset
- **Datasets:**
  - GunPoint (Easy)
  - ItalyPowerDemand (Medium)
  - ArrowHead (Hard)
  - ECG200 (Hard)

**Output:**
- Performance summary table
- Average accuracy across all datasets
- Comparison with expected ranges

**When to use:**
- Research/publication
- Comparing with other methods
- Comprehensive benchmarking

---

## 4. Simple Production Test

**File:** `test_tspulse.py`

**Purpose:** Quick architecture compliance check

```bash
python test_tspulse.py
```

**Details:**
- **Time:** 1-2 minutes
- **Epochs:** 5
- **Dataset:** Simple synthetic
- **Expected Accuracy:** 80%+

**When to use:**
- Architecture verification
- Quick sanity check
- Installation validation

---

## 📊 Expected Performance Ranges

### Real UCR Datasets:

| Dataset | Difficulty | Expected Accuracy | Notes |
|---------|-----------|-------------------|-------|
| **GunPoint** | Easy | 95-100% | Binary, clear patterns |
| **BasicMotions** | Easy | ~100% | 4 classes, distinct |
| **ItalyPowerDemand** | Medium | 90-97% | Power consumption patterns |
| **ArrowHead** | Hard | 75-85% | 3 classes, subtle differences |
| **ECG200** | Hard | 85-92% | ECG heartbeat classification |

### Synthetic Datasets:

| Dataset | Difficulty | Expected Accuracy | Notes |
|---------|-----------|-------------------|-------|
| **Simple** (quick_test.py old) | Too Easy | ~100% | Frequency-based, 2 classes |
| **Challenging** (train_real_dataset.py) | Hard | 60-75% | 3 classes, high noise, overlapping |

---

## 🚀 Recommended Testing Workflow

### For First-Time Setup:
```bash
# 1. Quick validation (2 min)
python quick_test.py

# 2. If successful, run full training (15 min)
python train_real_dataset.py
```

### For Production Deployment:
```bash
# Run UCR benchmark suite
python test_ucr_datasets.py
```

### For Development:
```bash
# After code changes, quick check
python test_tspulse.py
```

---

## 🔍 What to Look For

### ✅ Good Signs:
1. **Train loss decreasing** (e.g., 0.7 → 0.01)
2. **Val loss decreasing** (e.g., 0.6 → 0.05)
3. **Both classes predicted** (not 100% one class)
4. **Macro F1 > 0.60** (for hard datasets)
5. **Macro F1 > 0.90** (for easy datasets)

### ⚠️ Warning Signs:
1. **Constant predictions** (only predicting one class)
2. **Loss not decreasing** (stuck at ~0.69)
3. **Very low F1 score** (<0.50 on easy datasets)
4. **Val loss increasing** (overfitting)

---

## 🐛 Troubleshooting

### Issue: "100% accuracy on synthetic data"
**Solution:** The dataset is too easy. Use `train_real_dataset.py` which has:
- 3 classes (not 2)
- High noise (0.5 std)
- Overlapping patterns
- Random walks and outliers

Expected accuracy drops to 60-75% (appropriate difficulty).

### Issue: "Model predicts only one class"
**Solution:** Already fixed! The improvements include:
- Macro F1 monitoring
- Better synthetic patterns
- Train/val loss tracking
- Prediction distribution checks

### Issue: "Accuracy too low (<50%)"
**Possible causes:**
1. Need more epochs (try 100-200)
2. Need to unfreeze backbone: `freeze_backbone=False`
3. Learning rate too high/low: `learning_rate=1e-4`
4. Dataset might be genuinely hard

---

## 📈 Performance Tuning

### To Improve Accuracy:

1. **Increase epochs:**
   ```python
   clf = TSPulseClassifier(n_epochs=200, ...)
   ```

2. **Unfreeze backbone:**
   ```python
   clf = TSPulseClassifier(freeze_backbone=False, ...)
   ```

3. **Adjust learning rate:**
   ```python
   clf = TSPulseClassifier(learning_rate=1e-4, ...)
   ```

4. **Increase batch size** (if memory allows):
   ```python
   clf = TSPulseClassifier(batch_size=64, ...)
   ```

### To Speed Up Training:

1. **Reduce epochs:**
   ```python
   clf = TSPulseClassifier(n_epochs=20, ...)
   ```

2. **Increase batch size:**
   ```python
   clf = TSPulseClassifier(batch_size=64, ...)
   ```

3. **Use GPU** (if available):
   - Automatic detection
   - 3-5x faster than CPU

---

## 🎯 Recommendations by Use Case

### For Demos/Presentations:
- Use `quick_test.py` (fast, impressive results)
- Then show `train_real_dataset.py` output (real data)

### For Mentor Review:
- Run `train_real_dataset.py` (comprehensive)
- Show training history and metrics
- Demonstrate early stopping and monitoring

### For Research/Papers:
- Run `test_ucr_datasets.py` (standard benchmarks)
- Compare with published baselines
- Report average performance

### For Production Deployment:
- Test on your specific domain data
- Use 100+ epochs with early stopping
- Monitor Macro F1 and confusion matrix

---

## ✅ Success Criteria

### Minimum Acceptable:
- ✓ Architecture compliant (TSPulseForClassification)
- ✓ Model learns (loss decreases)
- ✓ Predicts multiple classes (not constant)
- ✓ Macro F1 > 0.60 on hard datasets
- ✓ Macro F1 > 0.90 on easy datasets

### Excellent Performance:
- ✓ All minimum criteria met
- ✓ GunPoint: 95%+ accuracy
- ✓ ItalyPowerDemand: 90%+ accuracy
- ✓ ArrowHead: 75%+ accuracy
- ✓ Early stopping works correctly

---

**Ready to test!** Start with `quick_test.py` and proceed to more comprehensive tests as needed. 🚀
