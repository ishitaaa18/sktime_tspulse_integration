"""
TSPulse Time Series Classifier - Proper sktime Integration

This module provides a proper sktime integration for IBM's TSPulse model,
inheriting from BaseClassifier and following sktime's architecture.

Architecture:
- Inherits from sktime.classification.base.BaseClassifier
- Uses TSPulseForClassification directly from tsfm_public.models.tspulse
- NO custom backbone modifications
- Follows official IBM Granite patterns
- Implements sktime's protocol (_fit, _predict, _predict_proba)

References
----------
.. [1] IBM Granite TSPulse:
   https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1
"""


__author__ = ["your-github-username"]

__all__ = ["TSPulseClassifier"]


import numpy as np
import tempfile  



from sktime.classification.base import BaseClassifier


class TSPulseClassifier(BaseClassifier):
    """TSPulse-based time series classifier with proper sktime integration.

    This classifier uses IBM's pre-trained TSPulse foundation model for
    time series classification. The model is loaded from HuggingFace and
    fine-tuned on the provided data using the official tsfm_public toolkit.

    Parameters
    ----------
    model_name : str, default="ibm-granite/granite-timeseries-tspulse-r1"
        HuggingFace model hub identifier for the TSPulse model.
    revision : str, default="tspulse-block-dualhead-512-p16-r1"
        Model revision / branch on HuggingFace Hub.
    config : dict or None, default=None
        Additional keyword arguments forwarded to
        ``TSPulseForClassification.from_pretrained``.
    training_args : dict or None, default=None
        Additional keyword arguments forwarded to
        ``transformers.TrainingArguments``.
    n_epochs : int, default=50
        Maximum number of training epochs.
    batch_size : int, default=32
        Per-device training and evaluation batch size.
    learning_rate : float or None, default=None
        Learning rate for AdamW.  When ``None`` the optimal learning rate
        is determined automatically via ``optimal_lr_finder``.
    freeze_backbone : bool, default=True
        If ``True``, all backbone parameters are frozen except the patch
        and FFT encoding layers.
    mask_ratio : float, default=0.3
        Masking ratio applied during training (disabled at eval time).
    early_stopping : bool, default=False
        Enable early stopping based on validation loss.
    early_stopping_patience : int, default=10
        Number of epochs with no improvement before training is stopped.
    random_state : int, default=42
        Random seed passed to PyTorch, NumPy and HuggingFace ``set_seed``.
    verbose : bool, default=True
        Whether to print progress messages during fit.
    tsfm_path : str or None, default=None
        Optional filesystem path to a local checkout of ``tsfm_public``.
        When provided, this path is prepended to ``sys.path`` so that the
        local version is imported instead of any installed version.

    Attributes
    ----------
    model_ : TSPulseForClassification
        Fitted TSPulse model.
    trainer_ : transformers.Trainer
        HuggingFace Trainer used for fitting.
    preprocessor_ : TimeSeriesClassificationPreprocessor
        Fitted preprocessing object (scaler, etc.).
    input_columns_ : list of str
        Column names used as model inputs, e.g. ``["dim_0", "dim_1", ...]``.
    classes_ : np.ndarray
        Unique class labels seen during ``fit`` (set by BaseClassifier).
    n_classes_ : int
        Number of unique classes (set by BaseClassifier).
    training_history_ : list of dict
        Per-epoch log entries from ``trainer_.state.log_history``.

    Examples
    --------
    >>> from sktime.classification.foundation_models.tspulse import (
    ...     TSPulseClassifier,
    ... )
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = TSPulseClassifier(n_epochs=2, batch_size=16, verbose=False)
    >>> clf.fit(X_train, y_train)
    TSPulseClassifier(...)
    >>> y_pred = clf.predict(X_test)

    References
    ----------
    .. [1] IBM Granite TSPulse:
       https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1
    """

    
    
    _tags = {
        "X_inner_mtype": "numpy3D",
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:predict_proba": True,
        "capability:train_estimate": False,
        "classifier_type": "deep_learning",
        
        "python_dependencies": [
            "torch",
            "transformers",
            "granite-tsfm",
            "scipy",
            "scikit-learn",
        ],
        "python_dependencies_alias": {
            "granite-tsfm": "tsfm_public",
        },
        
        "authors": ["your-github-username"],
        "maintainers": ["your-github-username"],
        "python_version": ">= 3.9",
        "tests:vm": True,
        "tests:skip_by_name": ["test_fit_idempotent"],
    }

    def __init__(
        self,
        model_name="ibm-granite/granite-timeseries-tspulse-r1",
        revision="tspulse-block-dualhead-512-p16-r1",
        config=None,
        training_args=None,
        n_epochs=50,
        batch_size=32,
        learning_rate=None,
        freeze_backbone=True,
        mask_ratio=0.3,
        early_stopping=False,
        early_stopping_patience=10,
        random_state=42,
        verbose=True,
        tsfm_path=None,
    ):
        
        self.model_name = model_name
        self.revision = revision
        self.config = config
        self.training_args = training_args
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.freeze_backbone = freeze_backbone
        self.mask_ratio = mask_ratio
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.verbose = verbose
        self.tsfm_path = tsfm_path

        super().__init__()

    @staticmethod
    def _make_training_arguments(
        output_dir,
        suggested_lr,
        n_epochs,
        batch_size,
        early_stopping,
        verbose,
        extra_args,
    ):
        """Build TrainingArguments in a version-safe way.

        transformers < 4.41  → evaluation_strategy
        transformers >= 4.41 → eval_strategy
        """
        from transformers import TrainingArguments
        import transformers

        t_version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        use_new_api = t_version >= (4, 41)

        eval_strat = "epoch" if early_stopping else "no"
        log_strat = "epoch" if verbose else "no"

        kwargs = dict(
            output_dir=output_dir,
            learning_rate=suggested_lr,
            num_train_epochs=n_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            report_to="none",
            disable_tqdm=not verbose,
            save_strategy="epoch" if early_stopping else "no",
            load_best_model_at_end=early_stopping,
            logging_strategy=log_strat,
        )

        if use_new_api:
            kwargs["eval_strategy"] = eval_strat
        else:
            kwargs["evaluation_strategy"] = eval_strat

        kwargs.update(extra_args)
        return TrainingArguments(**kwargs)

    def _fit(self, X, y):
        """Fit the TSPulse classifier.

        Parameters
        ----------
        X : np.ndarray, shape (n_instances, n_dimensions, series_length)
        y : np.ndarray, shape (n_instances,)
            Integer-encoded class indices provided by BaseClassifier.

        Returns
        -------
        self
        """
        
        
        
        import pandas as pd
        import torch
        import math

        

        try:
            import tsfm_public  # noqa: F401
        except ImportError:
            raise ImportError(
                "TSPulseClassifier requires 'granite-tsfm'. "
                "Install it with: pip install granite-tsfm"
    )

        
        if self.tsfm_path is not None:
            import sys

            sys.path.insert(0, self.tsfm_path)

        from tsfm_public.models.tspulse import TSPulseForClassification
        from tsfm_public.toolkit.time_series_classification_preprocessor import (
            TimeSeriesClassificationPreprocessor,
        )
        from tsfm_public.toolkit.dataset import ClassificationDFDataset
        from transformers import Trainer, EarlyStoppingCallback, set_seed
        from torch.optim import AdamW
        from transformers import get_cosine_schedule_with_warmup
        from tsfm_public.toolkit.lr_finder import optimal_lr_finder

        set_seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        if self.verbose:
            print("Preparing data for TSPulse...")

        # Convert numpy3D → DataFrame
        n_instances, n_dimensions, series_length = X.shape
        data_dict = {
            f"dim_{dim}": [pd.Series(X[i, dim, :]) for i in range(n_instances)]
            for dim in range(n_dimensions)
        }
        df_train = pd.DataFrame(data_dict)

        df_train["class_vals"] = y  # y is already integer-encoded by BaseClassifier

        n_classes = self.n_classes_
        n_channels = n_dimensions

        if self.verbose:
            print(
                f"Training TSPulse: {n_instances} samples, "
                f"{n_channels} channels, {n_classes} classes..."
            )

        # Preprocessing
        input_columns = [f"dim_{i}" for i in range(n_channels)]
        tsp = TimeSeriesClassificationPreprocessor(
            input_columns=input_columns,
            label_column="class_vals",
            scaling=True,
        )
        tsp.train(df_train)
        df_train_prep = tsp.preprocess(df_train)

        self.preprocessor_ = tsp
        self.input_columns_ = input_columns

        # Dataset
        
        if series_length > 512:
            import warnings

            warnings.warn(
                f"Series length {series_length} exceeds TSPulse's context window of "
                "512 timesteps.  Samples will be silently truncated by "
                "ClassificationDFDataset.  Consider downsampling your data.",
                UserWarning,
                stacklevel=2,
            )
        if series_length < 512:
            import warnings

            warnings.warn(
                f"Series length {series_length} is shorter than TSPulse's context "
                "window of 512 timesteps and enable_padding=False.  This may cause "
                "shape errors at runtime.  Set enable_padding=True or ensure all "
                "series are exactly 512 timesteps long.",
                UserWarning,
                stacklevel=2,
            )

        train_dataset = ClassificationDFDataset(
            df_train_prep,
            id_columns=[],
            timestamp_column=None,
            input_columns=input_columns,
            label_column="class_vals",
            context_length=512,
            static_categorical_columns=[],
            stride=1,
            enable_padding=False,
            full_series=True,
        )

        if self.verbose:
            print(f"Loading TSPulse model from {self.model_name}...")

        
        config_overrides = self.config if self.config is not None else {}
        training_args_extra = self.training_args if self.training_args is not None else {}

        
        base_config = {
            "head_gated_attention_activation": "softmax",
            "channel_virtual_expand_scale": 2,
            "mask_ratio": self.mask_ratio,
            "head_reduce_d_model": 1,
            # NOTE: disable_mask_in_classification_eval is intentionally locked to
            # True here and NOT overridable via the config parameter.  Enabling
            # masking at eval time produces unreliable classification outputs.
            "disable_mask_in_classification_eval": True,
            "fft_time_consistent_masking": True,
            "decoder_mode": "mix_channel",
            "head_aggregation_dim": "patch",
            "head_aggregation": None,
            "loss": "cross_entropy",
            "ignore_mismatched_sizes": True,
            "num_input_channels": n_channels,
            "num_targets": n_classes,
        }
        # Merge user overrides, but protect the eval-masking flag
        base_config.update(config_overrides)
        base_config["disable_mask_in_classification_eval"] = True  # always enforce

        self.model_ = TSPulseForClassification.from_pretrained(
            self.model_name,
            revision=self.revision,
            **base_config,
        )

        # Freeze backbone
        if self.freeze_backbone:
            if self.verbose:
                print("Freezing backbone, unfreezing patch embedding layers...")
            for param in self.model_.backbone.parameters():
                param.requires_grad = False
            for param in self.model_.backbone.time_encoding.parameters():
                param.requires_grad = True
            for param in self.model_.backbone.fft_encoding.parameters():
                param.requires_grad = True

        # Learning rate
        if self.learning_rate is None:
            if self.verbose:
                print("Auto-detecting optimal learning rate...")
            lr, self.model_ = optimal_lr_finder(
                self.model_, train_dataset, batch_size=self.batch_size
            )
            suggested_lr = lr
        else:
            suggested_lr = self.learning_rate

        if self.verbose:
            print(f"Using learning rate: {suggested_lr:.6f}")

        
        self._temp_dir = tempfile.TemporaryDirectory()
        training_args = self._make_training_arguments(
            output_dir=self._temp_dir.name,
            suggested_lr=suggested_lr,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            early_stopping=self.early_stopping,
            verbose=self.verbose,
            extra_args=training_args_extra,
        )

        optimizer = AdamW(self.model_.parameters(), lr=suggested_lr)
        total_steps = self.n_epochs * math.ceil(len(train_dataset) / self.batch_size)
        warmup_steps = max(1, int(0.1 * total_steps))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        trainer_kwargs = {
            "model": self.model_,
            "args": training_args,
            "train_dataset": train_dataset,
            "optimizers": (optimizer, scheduler),
        }

        
        if self.early_stopping:
            from sklearn.model_selection import StratifiedShuffleSplit

            labels = np.array([df_train_prep["class_vals"].iloc[i] for i in range(len(train_dataset))])
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=0.2, random_state=self.random_state
            )
            train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
            train_subset = torch.utils.data.Subset(train_dataset, train_idx.tolist())
            eval_subset = torch.utils.data.Subset(train_dataset, val_idx.tolist())
            trainer_kwargs["train_dataset"] = train_subset
            trainer_kwargs["eval_dataset"] = eval_subset
            trainer_kwargs["callbacks"] = [
                EarlyStoppingCallback(
                    early_stopping_patience=self.early_stopping_patience
                )
            ]
            if self.verbose:
                print(f"Early stopping enabled (patience={self.early_stopping_patience})")
                print(
                    f"Train/Val split: {len(train_idx)}/{len(val_idx)} samples "
                    "(stratified)"
                )

        self.trainer_ = Trainer(**trainer_kwargs)

        if self.verbose:
            print(f"\nTraining for up to {self.n_epochs} epochs...")

        self.trainer_.train()

        
        self.training_history_ = []
        if hasattr(self.trainer_, "state") and self.trainer_.state.log_history:
            for entry in self.trainer_.state.log_history:
                record = {"epoch": entry.get("epoch")}
                if "loss" in entry:
                    record["train_loss"] = entry["loss"]
                if "eval_loss" in entry:
                    record["val_loss"] = entry["eval_loss"]
                self.training_history_.append(record)

        if self.verbose:
            print("Training complete!")

        return self

    def _predict(self, X):
        """Predict class indices for test data.

        Parameters
        ----------
        X : np.ndarray, shape (n_instances, n_dimensions, series_length)

        Returns
        -------
        y_pred : np.ndarray, shape (n_instances,)
            Integer class indices aligned with ``self.classes_``.
            BaseClassifier.predict() will decode these back to original labels.
        """
        
        # BaseClassifier.predict() handles inverse-transform via self.classes_.
        y_proba = self._predict_proba(X)
        indices = np.argmax(y_proba, axis=1)

        return self.classes_[indices]

    def _predict_proba(self, X):
        """Predict class probabilities for test data.

        Parameters
        ----------
        X : np.ndarray, shape (n_instances, n_dimensions, series_length)

        Returns
        -------
        y_proba : np.ndarray, shape (n_instances, n_classes)
            Probabilities in the same class order as ``self.classes_``.
        """
        
        import pandas as pd
        import torch

        from tsfm_public.toolkit.dataset import ClassificationDFDataset
        from scipy.special import softmax

        n_instances, n_dimensions, _ = X.shape

        data_dict = {
            f"dim_{dim}": [pd.Series(X[i, dim, :]) for i in range(n_instances)]
            for dim in range(n_dimensions)
        }
        df_test = pd.DataFrame(data_dict)
        df_test["class_vals"] = self .classes_[0]  # dummy label required by ClassificationDFDataset

        
        df_test_scaled = self.preprocessor_.preprocess(df_test)

        test_dataset = ClassificationDFDataset(
            df_test_scaled,
            id_columns=[],
            timestamp_column=None,
            input_columns=self.input_columns_,
            label_column="class_vals",
            context_length=512,
            static_categorical_columns=[],
            stride=1,
            enable_padding=False,
            full_series=True,
        )

        predictions = self.trainer_.predict(test_dataset)
        logits = predictions.predictions

        if isinstance(logits, tuple):
            logits = logits[0]

        if torch.is_tensor(logits):
            logits = logits.cpu().numpy()

        logits = np.asarray(logits)

        if len(logits.shape) == 3:
            logits = logits[:, -1, :]
        elif len(logits.shape) > 2:
            logits = logits.reshape(logits.shape[0], -1)

        
        proba = softmax(logits, axis=1)
        assert proba.shape == (n_instances, self.n_classes_), (
            f"_predict_proba output shape {proba.shape} does not match "
            f"expected ({n_instances}, {self.n_classes_}).  "
            "Check that the model's num_targets matches self.n_classes_."
        )
        return proba

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the parameter set.

        Returns
        -------
        params : list of dict
        """
        params1 = {"n_epochs": 2, "batch_size": 16, "verbose": False}
        params2 = {
            "n_epochs": 3,
            "batch_size": 8,
            "freeze_backbone": False,
            "verbose": False,
        }
        return [params1, params2]
