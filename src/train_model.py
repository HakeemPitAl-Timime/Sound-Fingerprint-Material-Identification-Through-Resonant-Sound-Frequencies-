import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# RandomForestClassifier is the machine learning algorithm we will use.
# A random forest builds many decision trees and averages their predictions.
from sklearn.ensemble import RandomForestClassifier
# Import tools used for splitting datasets and evaluating models.
from sklearn.model_selection import (
    StratifiedKFold,      # Cross-validation that preserves class proportions
    cross_validate,       # Runs cross-validation and returns multiple metrics
    train_test_split,     # Splits dataset into training and testing portions
    RandomizedSearchCV    # Automatically finds good model parameters
)
# Import evaluation metrics used to measure model performance.
from sklearn.metrics import (
    accuracy_score,           # % of predictions that are correct
    balanced_accuracy_score,  # accuracy adjusted for class imbalance
    f1_score,                 # harmonic mean of precision and recall
    classification_report,    # detailed performance per class
    confusion_matrix          # matrix showing prediction errors
)

# permutation_importance measures how important each feature is
# by randomly shuffling feature values and seeing how model accuracy changes.
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt



class Train_Model:
    """
    Train_Model class for acoustic material classification.

    This class trains a machine learning model that predicts
    the material type based on sound features extracted earlier.

    The dataset must contain these columns:
        resonant_frequency
        spectral_centroid
        decay_rate
        attack_strength
        label

    Each row corresponds to one recorded tap sound.
    """
    def __init__(self, dataset_path,
                 feature_columns=None,
                 label_column="label",
                 random_state=42,
                 cv_splits=4,
                 do_tuning=True):
        self.dataset_path = Path(dataset_path) # Path() ensures it works correctly on any operating system.
        self.label_column = label_column   # Store the name of the column containing the labels

        # random_state ensures reproducible results.
        # Using the same random_state makes experiments repeatable.
        self.random_state = random_state
        self.cv_splits = cv_splits # Number of folds used in cross-validation. Example: cv_splits=5 means the dataset is split into 5 parts.
        self.do_tuning = do_tuning # Boolean flag determining whether hyperparameter tuning

        # Define the four physics-based sound features. These correspond to properties extracted from the audio signal.
        self.default_features = [
            "resonant_frequency",
            "spectral_centroid",
            "decay_rate",
            "attack_strength"
        ]
        # If no custom feature list is provided,
        # use the default features above.
        if feature_columns is None:
            self.feature_columns = self.default_features
        else:
            self.feature_columns = feature_columns

        # Create a pipeline combining preprocessing and the ML model.
        self.pipeline = Pipeline([
            
            # Step 1: StandardScaler
            # This scales features so they have mean 0 and variance 1. Its so feautures are comparable in order of magnitude.
            ("scaler", StandardScaler()),

            # Step 2: RandomForestClassifier
            # The machine learning algorithm used for classification.
            ("clf", RandomForestClassifier(
                n_estimators=100,      # number of trees in the forest
                random_state=self.random_state
            ))
        ])

        # Define a search space of parameters for hyperparameter tuning.
        # RandomizedSearchCV will test combinations of these parameters.
        self.param_distributions = {

            # Number of trees in the forest.
            "clf__n_estimators": [50, 100, 200, 400],

            # Maximum depth of each decision tree.
            "clf__max_depth": [None, 10, 20, 40],

            # Minimum samples required in a leaf node.
            "clf__min_samples_leaf": [1, 2, 4],

            # Number of features considered when splitting.
            "clf__max_features": ["sqrt", "log2", 0.5],

            # Adjusts class weighting if some classes appear more often.
            "clf__class_weight": [None, "balanced"]
        }

        # Variable to store the best trained model after tuning.
        self.best_estimator_ = None

        # Variable that will store evaluation results on the test set.
        self.test_results_ = None

    # Function to load the dataset from CSV.
    def load_dataset(self):
        # Read the dataset file using pandas.
        df = pd.read_csv(self.dataset_path)
        # Determine which expected features are present in the dataset.
        present = [c for c in self.default_features if c in df.columns]
        # Identify any additional columns beyond the default features.
        extras = [c for c in df.columns if c not in (present + [self.label_column])]
        # Combine default features and additional features.
        self.feature_columns = present + extras
        # Extract the feature matrix X (inputs to the model).
        X = df[self.feature_columns].copy()
        # Extract labels (correct material classification).
        y = df[self.label_column].copy()
        # Return features and labels.
        return X, y

    # Baseline cross-validation
    def baseline_cv(self, X, y, scoring=None):
        # Define metrics used during evaluation.
        if scoring is None:
            scoring = ["accuracy", "f1_macro", "balanced_accuracy"]
        # Create a Stratified K-Fold splitter.
        # Stratified means each fold keeps the same class proportions.
        skf = StratifiedKFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state
        )
        # Run cross-validation on the entire dataset.
        scores = cross_validate(
            self.pipeline,
            X,
            y,
            cv=skf,
            scoring=scoring,
            return_train_score=False
        )
        print("=== Baseline cross-validation ===")
        # Print average performance for each metric.
        for metric in scoring:
            mean = scores[f"test_{metric}"].mean()
            std = scores[f"test_{metric}"].std()

            print(f"{metric}: {mean:.4f} ± {std:.4f}")
        return scores



    # Main training function
    def tune_and_train(self, X, y, test_size=0.33, n_iter=40):

        # Split dataset into training and testing sets.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,                    # keep class balance
            random_state=self.random_state
        )

        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        # Perform hyperparameter tuning if enabled.
        if self.do_tuning:
            print("Running RandomizedSearchCV...")
            rs = RandomizedSearchCV(
                estimator=self.pipeline,
                param_distributions=self.param_distributions,
                n_iter=n_iter,
                cv=StratifiedKFold(
                    n_splits=self.cv_splits,
                    shuffle=True,
                    random_state=self.random_state
                ),
                scoring="f1_macro",
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )

            rs.fit(X_train, y_train)

            print("Best params:", rs.best_params_)
            print(f"Best CV f1_macro: {rs.best_score_:.4f}")

            self.best_estimator_ = rs.best_estimator_

        else:

            print("Skipping hyperparameter search.")

            self.best_estimator_ = self.pipeline
            self.best_estimator_.fit(X_train, y_train)

        # Predict on test data.
        y_pred = self.best_estimator_.predict(X_test)

        # Calculate evaluation metrics.
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        print("\n=== Test set evaluation ===")

        print(f"Accuracy: {acc:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print(f"F1 (macro): {f1:.4f}")

        print("\nClassification report:")
        print(classification_report(y_test, y_pred))


        # Compute confusion matrix.
        cm = confusion_matrix(y_test, y_pred)

        self.test_results_ = dict(
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            acc=acc,
            bal_acc=bal_acc,
            f1=f1,
            cm=cm
        )

        return self.test_results_

    # Feature importance analysis
    def print_feature_importance(self):

        clf = self.best_estimator_.named_steps["clf"]

        if hasattr(clf, "feature_importances_"):

            print("\nFeature importances:")

            importances = clf.feature_importances_

            for feat, imp in sorted(
                zip(self.feature_columns, importances),
                key=lambda x: -x[1]
            ):
                print(f"{feat}: {imp:.4f}")
