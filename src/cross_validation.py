from sklearn.model_selection import LeaveOneGroupOut, BaseCrossValidator
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

from src.preprocessor import Preprocessor
from src.metrics import accuracy, sensitivity, specificity, precision, f1_score

class LeaveOneSubjectOut(BaseCrossValidator):
    """
    Leave-One-Subject-Out cross-validator.

    Methods
    -------
    get_n_splits(subjects, targets)
        Returns the number of splits, which is equal to the number of unique subjects.
    split(X, y, groups, **kwargs)
        Generates indices to split data into training and test set.
        The 'groups' parameter should contain the subject identifiers.
    Parameters
    ----------
    subjects : array-like
        List of subject identifiers.
    targets : array-like
        List of target values.
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    groups : array-like
        Group labels for the samples used while splitting the dataset into train/test set.
    **kwargs : dict
        Additional parameters.
    """
    def get_n_splits(self, subjects, targets):
        return len(subjects)

    def split(self, X, y, groups, **kwargs):
        # Create the LeaveOneGroupOut object
        logo = LeaveOneGroupOut()

        # Return the list of train groups and test groups
        for train_idx, test_idx in logo.split(X, y, groups=groups):
            yield train_idx, test_idx

class LOSOCV:
    """
    Leave-One-Subject-Out Cross-Validation (LOSOCV) class.
    This class performs cross-validation where each subject is left out once as the test set,
    and the remaining subjects are used as the training set. It uses a specified model and 
    preprocessor for the cross-validation process.
    Attributes:
        model: The machine learning model to be used for cross-validation.
        preprocessor (Preprocessor): An instance of a Preprocessor class used to preprocess the data.
        cv (LeaveOneSubjectOut): An instance of the LeaveOneSubjectOut cross-validator.
        scoring (dict): A dictionary of scoring metrics to evaluate the model performance.
    Methods:
        cross_validate(subjects, targets):
            Performs cross-validation on the given subjects and targets.
            Args:
                subjects (list): A list of subjects, where each subject is a list of samples.
                targets (list): A list of target values corresponding to each subject.
            Returns:
                dict: A dictionary containing the cross-validation results for each scoring metric.
    """
    def __init__(self, model, preprocessor: Preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.cv = LeaveOneSubjectOut()
        self.scoring = {
            "accuracy": make_scorer(accuracy),
            "sensitivity": make_scorer(sensitivity),
            # "specificity": make_scorer(specificity),
            "precision": make_scorer(precision),
            "f1_score": make_scorer(f1_score),
        }

    def cross_validate(self, subjects, targets):
        # Preprocess the data
        subjects = [self.preprocessor.preprocess(subject) for subject in subjects]

        # Flatten the subjects and targets
        X = [item for subject in subjects for item in subject]
        y = [targets[i] for i, subject in enumerate(subjects) for _ in subject]
        groups = [i for i, subject in enumerate(subjects) for _ in subject]


        cv_results = cross_validate(
            estimator=self.model,
            X=X,
            y=y,
            groups=groups,
            scoring=self.scoring,
            cv=self.cv,
        )
        return cv_results




        
