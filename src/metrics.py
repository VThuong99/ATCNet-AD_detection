
def accuracy(y_true, y_pred, **kwargs):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)

def sensitivity(y_true, y_pred, **kwargs):
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred)

def precision(y_true, y_pred, **kwargs):
    from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred)

def specificity(y_true, y_pred, **kwargs):
    from sklearn.metrics import confusion_matrix
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Check the shape of the confusion matrix and unpack safely
    if cm.shape == (2, 2):  # Standard 2x2 case
        tn, fp, fn, tp = cm.ravel()
    else:  # Handle cases where predictions for a class are missing
        tn = cm[0, 0] if cm.shape[0] > 0 else 0
        fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[1] > 0 else 0
        tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    return tn / (tn + fp)

def f1_score(y_true, y_pred, **kwargs):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred)