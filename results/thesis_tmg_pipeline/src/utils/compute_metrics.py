from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def compute_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    return {"Precision": precision, "Recall": recall, "F1": f1, "Accuracy": accuracy}