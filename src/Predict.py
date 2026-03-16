import pandas as pd
import numpy as np
from pathlib import Path
import joblib


class Predict:
    """
    Predict class for material classification.

    This class loads a trained machine learning model (or pipeline) and
    predicts the material category from extracted sound features.

    Usage:
        p = Predict("trained_pipeline.pkl")
        label, confidence = p.predict(520, 430, 0.32, 0.85)
    """

    def __init__(self, model_path, feature_columns=None):
        # Store model path safely
        self.model_path = Path(model_path)
        # Load the trained pipeline (scaler + classifier)
        self.model = joblib.load(self.model_path)

        # By default, expect these four features in this order.
        self.default_features = [
            "resonant_frequency",
            "spectral_centroid",
            "decay_rate",
            "attack_strength"
        ]

        # If user supplies feature_columns, use that; otherwise try to detect.
        if feature_columns is not None:
            self.feature_columns = feature_columns
        else:
            # Try to infer feature names from the trained estimator (if available)
            # sklearn estimators fitted on pandas DataFrame may have feature_names_in_
            try:
                # If model is a Pipeline, look for the classifier step
                if hasattr(self.model, "named_steps"):
                    clf = self.model.named_steps.get("clf", None)
                    if clf is not None and hasattr(clf, "feature_names_in_"):
                        self.feature_columns = list(clf.feature_names_in_)
                    else:
                        # Some pipelines preserve feature names on the pipeline object
                        if hasattr(self.model, "feature_names_in_"):
                            self.feature_columns = list(self.model.feature_names_in_)
                        else:
                            self.feature_columns = self.default_features
                else:
                    # model is not a pipeline
                    if hasattr(self.model, "feature_names_in_"):
                        self.feature_columns = list(self.model.feature_names_in_)
                    else:
                        self.feature_columns = self.default_features
            except Exception:
                self.feature_columns = self.default_features


    def _build_dataframe(self, values):
        """
        Build a pandas DataFrame with a single row matching the feature order.
        `values` can be a list/tuple or a dict mapping feature->value.
        """
        if isinstance(values, dict):
            # Ensure columns are in the feature_columns order
            data = {f: [values.get(f, np.nan)] for f in self.feature_columns}
        else:
            # assume iterable of values in the correct order
            if len(values) != len(self.feature_columns):
                raise ValueError(f"Expected {len(self.feature_columns)} features, got {len(values)}.")
            data = {f: [v] for f, v in zip(self.feature_columns, values)}

        df = pd.DataFrame(data)
        return df


    def predict(self, resonant_frequency=None, spectral_centroid=None, decay_rate=None, attack_strength=None, features=None):
        """
        Predict material from acoustic features.

        You can either pass the four named feature arguments, or pass a single
        `features` argument which is a list/tuple of values matching the feature order,
        or a dict mapping feature names to values.

        Returns: (prediction_label, confidence_float)
        """

        if features is not None:
            df = self._build_dataframe(features)
        else:
            values = [resonant_frequency, spectral_centroid, decay_rate, attack_strength]
            df = self._build_dataframe(values)

        # Run prediction
        prediction = self.model.predict(df)[0]

        # Get confidence probabilities if available
        confidence = None
        if hasattr(self.model, "predict_proba"):
            try:
                probabilities = self.model.predict_proba(df)[0]
                confidence = float(np.max(probabilities))
            except Exception:
                confidence = None

        return prediction, confidence
