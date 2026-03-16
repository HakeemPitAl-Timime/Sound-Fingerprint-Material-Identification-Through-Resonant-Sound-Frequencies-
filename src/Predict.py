import pandas as pd
import numpy as np
from pathlib import Path
import joblib


class Predict:
    """
    Predict class for material classification.

    This class loads a trained machine learning model and
    predicts the material category from extracted sound features.
    """

    def __init__(self, model_path):
        # Store model path safely
        self.model_path = Path(model_path)

        # Load the trained pipeline (scaler + classifier)
        self.model = joblib.load(self.model_path)

        # Feature order must match training
        self.feature_columns = [
            "resonant_frequency",
            "spectral_centroid",
            "decay_rate",
            "attack_strength"
        ]


    def predict(self, resonant_frequency, spectral_centroid, decay_rate, attack_strength):
        """
        Predict material from acoustic features.
        """

        # Create dictionary representing one feature sample
        data = {
            "resonant_frequency": [resonant_frequency],
            "spectral_centroid": [spectral_centroid],
            "decay_rate": [decay_rate],
            "attack_strength": [attack_strength]
        }

        # Convert to pandas DataFrame (same structure as training data)
        df = pd.DataFrame(data)

        # Run prediction
        prediction = self.model.predict(df)[0]

        # Get confidence probabilities
        probabilities = self.model.predict_proba(df)[0]

        # Find highest confidence score
        confidence = np.max(probabilities)

        return prediction, confidence