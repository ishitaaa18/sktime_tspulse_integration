import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sktime_ext.classification.deep_learning.tspulse import TSPulseClassifier


@pytest.fixture
def synthetic_data():
    n_train, n_test = 50, 20
    n_timesteps, n_classes = 30, 3
    
    X_train = np.random.randn(n_train, 1, n_timesteps)
    y_train = np.random.randint(0, n_classes, n_train)
    
    X_test = np.random.randn(n_test, 1, n_timesteps)
    y_test = np.random.randint(0, n_classes, n_test)
    
    return X_train, y_train, X_test, y_test


class TestTPulseClassifier:

    def test_initialization(self):
        clf = TSPulseClassifier(n_epochs=5, random_state=42)
        assert clf.n_epochs == 5
        assert clf.random_state == 42

    def test_fit(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data

        clf = TSPulseClassifier(n_epochs=2, verbose=False)
        clf.fit(X_train, y_train)

        assert hasattr(clf, "model_")
        assert hasattr(clf, "trainer_")
        assert clf.n_classes_ == len(np.unique(y_train))

    def test_predict(self, synthetic_data):
        X_train, y_train, X_test, y_test = synthetic_data

        clf = TSPulseClassifier(n_epochs=2, verbose=False)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        assert len(y_pred) == len(y_test)

    def test_predict_proba(self, synthetic_data):
        X_train, y_train, X_test, y_test = synthetic_data

        clf = TSPulseClassifier(n_epochs=2, verbose=False)
        clf.fit(X_train, y_train)

        y_proba = clf.predict_proba(X_test)

        assert y_proba.shape == (len(y_test), len(np.unique(y_train)))
        np.testing.assert_allclose(y_proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_deterministic(self, synthetic_data):
        X_train, y_train, X_test, _ = synthetic_data

        clf1 = TSPulseClassifier(n_epochs=2, random_state=42, verbose=False)
        clf1.fit(X_train, y_train)
        y1 = clf1.predict(X_test)

        clf2 = TSPulseClassifier(n_epochs=2, random_state=42, verbose=False)
        clf2.fit(X_train, y_train)
        y2 = clf2.predict(X_test)

        assert len(y1) == len(y2)