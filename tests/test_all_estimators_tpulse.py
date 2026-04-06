from sktime.tests.test_all_estimators import run_test_for_class
from sktime_ext.classification.deep_learning.tspulse import TSPulseClassifier


def test_all_estimators():
    run_test_for_class(TSPulseClassifier)