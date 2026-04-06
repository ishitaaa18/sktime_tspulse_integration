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

Author: IBM Integration Team
Date: 2026-02-12
License: Apache 2.0
"""

import numpy as np
import pandas as pd
import torch
import tempfile

from sktime.classification.base import BaseClassifier

__all__ = ['TSPulseClassifier']


class TSPulseClassifier(BaseClassifier):
    """
    TSPulse-based time series classifier with proper sktime integration.

    This classifier uses IBM's pre-trained TSPulse foundation model for
    time series classification. The model is loaded from HuggingFace and
    fine-tuned on the provided data using the official tsfm_public toolkit.

    Parameters
    ----------
    model_name : str, default="ibm-granite/granite-timeseries-tspulse-r1"
    revision : str, default="tspulse-block-dualhead-512-p16-r1"
    config : dict, optional — additional model config overrides
    training_args : dict, optional — additional TrainingArguments overrides
    n_epochs : int, default=50
    batch_size : int, default=32
    learning_rate : float, optional — if None, uses optimal_lr_finder
    freeze_backbone : bool, default=True
    mask_ratio : float, default=0.3
    early_stopping : bool, default=False
    early_stopping_patience : int, default=10
    random_state : int, default=42
    verbose : bool, default=True
    tsfm_path : str, optional
    """

    _tags = {
        "X_inner_mtype": "numpy3D",
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:predict_proba": True,
        "capability:train_estimate": False,
        "classifier_type": "deep_learning",
        "python_dependencies": ["torch", "transformers"],
        "python_dependencies_alias": None,
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
        self._config = config if config is not None else {}
        self.training_args = training_args
        self._training_args = training_args if training_args is not None else {}
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
    def _make_training_arguments(output_dir, suggested_lr, n_epochs, batch_size,
                                  early_stopping, verbose, extra_args):
        """
        Build TrainingArguments in a version-safe way.

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

        # Pick the right key for the eval strategy depending on transformers version
        if use_new_api:
            kwargs["eval_strategy"] = eval_strat
        else:
            kwargs["evaluation_strategy"] = eval_strat

        kwargs.update(extra_args)
        return TrainingArguments(**kwargs)

    def _fit(self, X, y):
        """
        Fit the TSPulse classifier.

        Parameters
        ----------
        X : np.ndarray, shape (n_instances, n_dimensions, series_length)
        y : np.ndarray, shape (n_instances,)

        Returns
        -------
        self
        """
        try:
            from tsfm_public.models.tspulse import TSPulseForClassification
            from tsfm_public.toolkit.time_series_classification_preprocessor import (
                TimeSeriesClassificationPreprocessor,
            )
            from tsfm_public.toolkit.dataset import ClassificationDFDataset
            from transformers import Trainer, EarlyStoppingCallback, set_seed
            from torch.optim import AdamW
            from torch.optim.lr_scheduler import OneCycleLR
            from tsfm_public.toolkit.lr_finder import optimal_lr_finder
            from sklearn.preprocessing import LabelEncoder
            import math
        except ImportError as e:
            raise ImportError(
                f"Required packages not found: {e}\n"
                "Install with: pip install 'granite-tsfm[notebooks] @ "
                "git+https://github.com/ibm-granite/granite-tsfm.git@v0.3.1'"
            )

        set_seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        if self.verbose:
            print("Preparing data for TSPulse...")

        # Convert numpy3D → DataFrame
        n_instances, n_dimensions, series_length = X.shape
        data_dict = {
            f'dim_{dim}': [pd.Series(X[i, dim, :]) for i in range(n_instances)]
            for dim in range(n_dimensions)
        }
        df_train = pd.DataFrame(data_dict)

        # Label encoding
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        df_train['class_vals'] = y_encoded

        n_classes = self.n_classes_
        n_channels = n_dimensions

        if self.verbose:
            print(f"Training TSPulse: {n_instances} samples, "
                  f"{n_channels} channels, {n_classes} classes...")

        # Preprocessing
        input_columns = [f'dim_{i}' for i in range(n_channels)]
        tsp = TimeSeriesClassificationPreprocessor(
            input_columns=input_columns,
            label_column='class_vals',
            scaling=True,
        )
        tsp.train(df_train)
        df_train_prep = tsp.preprocess(df_train)

        self.preprocessor_ = tsp
        self.input_columns_ = input_columns

        # Dataset
        train_dataset = ClassificationDFDataset(
            df_train_prep,
            id_columns=[],
            timestamp_column=None,
            input_columns=input_columns,
            label_column='class_vals',
            context_length=512,
            static_categorical_columns=[],
            stride=1,
            enable_padding=False,
            full_series=True,
        )

        if self.verbose:
            print(f"Loading TSPulse model from {self.model_name}...")

        # Model config
        base_config = {
            "head_gated_attention_activation": "softmax",
            "channel_virtual_expand_scale": 2,
            "mask_ratio": self.mask_ratio,
            "head_reduce_d_model": 1,
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
        base_config.update(self._config)

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

        # Build version-safe TrainingArguments
        temp_dir = tempfile.mkdtemp()
        training_args = self._make_training_arguments(
            output_dir=temp_dir,
            suggested_lr=suggested_lr,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            early_stopping=self.early_stopping,
            verbose=self.verbose,
            extra_args=self._training_args,
        )

        optimizer = AdamW(self.model_.parameters(), lr=suggested_lr)
        scheduler = OneCycleLR(
            optimizer,
            suggested_lr,
            epochs=self.n_epochs,
            steps_per_epoch=math.ceil(len(train_dataset) / self.batch_size),
        )

        trainer_kwargs = {
            "model": self.model_,
            "args": training_args,
            "train_dataset": train_dataset,
            "optimizers": (optimizer, scheduler),
        }

        # Early stopping split
        if self.early_stopping:
            train_size = int(0.8 * len(train_dataset))
            eval_size = len(train_dataset) - train_size
            train_subset = torch.utils.data.Subset(train_dataset, range(train_size))
            eval_subset = torch.utils.data.Subset(
                train_dataset, range(train_size, train_size + eval_size)
            )
            trainer_kwargs["train_dataset"] = train_subset
            trainer_kwargs["eval_dataset"] = eval_subset
            trainer_kwargs["callbacks"] = [
                EarlyStoppingCallback(
                    early_stopping_patience=self.early_stopping_patience
                )
            ]
            if self.verbose:
                print(f"Early stopping enabled (patience={self.early_stopping_patience})")
                print(f"Train/Val split: {train_size}/{eval_size} samples")

        self.trainer_ = Trainer(**trainer_kwargs)

        if self.verbose:
            print(f"\nTraining for up to {self.n_epochs} epochs...")

        self.trainer_.train()

        # Training history
        self.training_history_ = []
        if hasattr(self.trainer_, "state") and self.trainer_.state.log_history:
            for entry in self.trainer_.state.log_history:
                self.training_history_.append({
                    "epoch": entry.get("epoch"),
                    "train_loss": entry.get("loss"),
                    "val_loss": entry.get("eval_loss"),
                    "train_acc": None,
                    "val_acc": None,
                })

        if self.verbose:
            print("Training complete!")

        return self

    def _predict(self, X):
        """
        Predict class labels for test data.

        Parameters
        ----------
        X : np.ndarray, shape (n_instances, n_dimensions, series_length)

        Returns
        -------
        y_pred : np.ndarray, shape (n_instances,)
        """
        y_proba = self._predict_proba(X)
        y_pred_encoded = np.argmax(y_proba, axis=1)
        return self.label_encoder_.inverse_transform(y_pred_encoded)

    def _predict_proba(self, X):
        """
        Predict class probabilities for test data.

        Parameters
        ----------
        X : np.ndarray, shape (n_instances, n_dimensions, series_length)

        Returns
        -------
        y_proba : np.ndarray, shape (n_instances, n_classes)
        """
        from tsfm_public.toolkit.dataset import ClassificationDFDataset
        from scipy.special import softmax

        n_instances, n_dimensions, series_length = X.shape

        data_dict = {
            f'dim_{dim}': [pd.Series(X[i, dim, :]) for i in range(n_instances)]
            for dim in range(n_dimensions)
        }
        df_test = pd.DataFrame(data_dict)
        df_test['class_vals'] = 0  # dummy label required by ClassificationDFDataset

        # Apply scaling from trained preprocessor
        df_test_scaled = df_test.copy()
        for col in self.input_columns_:
            if col in self.preprocessor_.scaler_dict:
                scaler = self.preprocessor_.scaler_dict[col]
                unnested = df_test_scaled[col].apply(
                    lambda x: x.values if isinstance(x, pd.Series) else x
                )
                scaled_values = scaler.transform(np.vstack(unnested.values))
                df_test_scaled[col] = [
                    pd.Series(scaled_values[i]) for i in range(len(scaled_values))
                ]

        test_dataset = ClassificationDFDataset(
            df_test_scaled,
            id_columns=[],
            timestamp_column=None,
            input_columns=self.input_columns_,
            label_column='class_vals',
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

        return softmax(logits, axis=1)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"n_epochs": 2, "batch_size": 16, "verbose": False}
        params2 = {"n_epochs": 3, "batch_size": 8, "freeze_backbone": False, "verbose": False}
        return [params1, params2]
