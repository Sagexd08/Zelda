"""
Metrics Utilities
Compute FAR, FRR, EER, ROC curves for evaluation
"""

from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt


def compute_far_frr(
    y_true: np.ndarray, 
    y_scores: np.ndarray, 
    threshold: float
) -> Tuple[float, float]:
    """
    Compute False Accept Rate (FAR) and False Reject Rate (FRR).
    
    Args:
        y_true: True labels (1 for genuine, 0 for impostor)
        y_scores: Similarity scores
        threshold: Decision threshold
        
    Returns:
        Tuple[float, float]: (FAR, FRR)
    """
    # Predictions based on threshold
    y_pred = (y_scores >= threshold).astype(int)
    
    # Split genuine and impostor pairs
    genuine_mask = y_true == 1
    impostor_mask = y_true == 0
    
    # False Reject Rate (genuine pairs rejected)
    if np.sum(genuine_mask) > 0:
        frr = np.sum((y_pred[genuine_mask] == 0)) / np.sum(genuine_mask)
    else:
        frr = 0.0
    
    # False Accept Rate (impostor pairs accepted)
    if np.sum(impostor_mask) > 0:
        far = np.sum((y_pred[impostor_mask] == 1)) / np.sum(impostor_mask)
    else:
        far = 0.0
    
    return float(far), float(frr)


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the threshold at which it occurs.
    
    Args:
        y_true: True labels
        y_scores: Similarity scores
        
    Returns:
        Tuple[float, float]: (EER, threshold_at_eer)
    """
    # Compute FAR and FRR at multiple thresholds
    thresholds = np.linspace(y_scores.min(), y_scores.max(), 1000)
    far_list = []
    frr_list = []
    
    for threshold in thresholds:
        far, frr = compute_far_frr(y_true, y_scores, threshold)
        far_list.append(far)
        frr_list.append(frr)
    
    far_array = np.array(far_list)
    frr_array = np.array(frr_list)
    
    # Find EER (where FAR ≈ FRR)
    diff = np.abs(far_array - frr_array)
    eer_idx = np.argmin(diff)
    
    eer = (far_array[eer_idx] + frr_array[eer_idx]) / 2
    threshold_at_eer = thresholds[eer_idx]
    
    return float(eer), float(threshold_at_eer)


def compute_accuracy_at_threshold(
    y_true: np.ndarray, 
    y_scores: np.ndarray, 
    threshold: float
) -> float:
    """
    Compute accuracy at a given threshold.
    
    Args:
        y_true: True labels
        y_scores: Similarity scores
        threshold: Decision threshold
        
    Returns:
        float: Accuracy (0-1)
    """
    y_pred = (y_scores >= threshold).astype(int)
    accuracy = np.mean(y_true == y_pred)
    
    return float(accuracy)


def find_optimal_threshold(
    y_true: np.ndarray, 
    y_scores: np.ndarray,
    criterion: str = 'youden'
) -> float:
    """
    Find optimal threshold using various criteria.
    
    Args:
        y_true: True labels
        y_scores: Similarity scores
        criterion: Optimization criterion ('youden', 'eer', 'f1')
        
    Returns:
        float: Optimal threshold
    """
    if criterion == 'eer':
        _, threshold = compute_eer(y_true, y_scores)
        return threshold
    
    elif criterion == 'youden':
        # Youden's index: maximize (Sensitivity + Specificity - 1)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        return float(thresholds[optimal_idx])
    
    elif criterion == 'f1':
        # Maximize F1 score
        thresholds = np.linspace(y_scores.min(), y_scores.max(), 1000)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        return float(thresholds[optimal_idx])
    
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Area Under ROC Curve (AUC).
    
    Args:
        y_true: True labels
        y_scores: Similarity scores
        
    Returns:
        float: AUC score
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    return float(roc_auc)


def plot_roc_curve(
    y_true: np.ndarray, 
    y_scores: np.ndarray,
    title: str = "ROC Curve",
    save_path: str = None
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_scores: Similarity scores
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_det_curve(
    y_true: np.ndarray, 
    y_scores: np.ndarray,
    title: str = "DET Curve",
    save_path: str = None
) -> plt.Figure:
    """
    Plot Detection Error Tradeoff (DET) curve.
    
    Args:
        y_true: True labels
        y_scores: Similarity scores
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    thresholds = np.linspace(y_scores.min(), y_scores.max(), 1000)
    far_list = []
    frr_list = []
    
    for threshold in thresholds:
        far, frr = compute_far_frr(y_true, y_scores, threshold)
        far_list.append(far)
        frr_list.append(frr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(far_list, frr_list, color='blue', lw=2)
    ax.set_xlabel('False Accept Rate (FAR)')
    ax.set_ylabel('False Reject Rate (FRR)')
    ax.set_title(title)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Mark EER point
    eer, _ = compute_eer(y_true, y_scores)
    ax.plot([eer], [eer], 'ro', markersize=8, label=f'EER = {eer:.3f}')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_far_frr_curve(
    y_true: np.ndarray, 
    y_scores: np.ndarray,
    title: str = "FAR/FRR vs Threshold",
    save_path: str = None
) -> plt.Figure:
    """
    Plot FAR and FRR vs threshold.
    
    Args:
        y_true: True labels
        y_scores: Similarity scores
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    thresholds = np.linspace(y_scores.min(), y_scores.max(), 1000)
    far_list = []
    frr_list = []
    
    for threshold in thresholds:
        far, frr = compute_far_frr(y_true, y_scores, threshold)
        far_list.append(far)
        frr_list.append(frr)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, far_list, 'r-', label='FAR', lw=2)
    ax.plot(thresholds, frr_list, 'b-', label='FRR', lw=2)
    
    # Mark EER
    eer, threshold_at_eer = compute_eer(y_true, y_scores)
    ax.axvline(threshold_at_eer, color='g', linestyle='--', 
               label=f'EER = {eer:.3f} @ threshold = {threshold_at_eer:.3f}')
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Error Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compute_confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute detailed metrics from confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dict: Dictionary of metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
        'precision': float(tp / (tp + fp + 1e-8)),
        'recall': float(tp / (tp + fn + 1e-8)),
        'specificity': float(tn / (tn + fp + 1e-8)),
        'f1_score': float(2 * tp / (2 * tp + fp + fn + 1e-8)),
        'false_accept_rate': float(fp / (fp + tn + 1e-8)),
        'false_reject_rate': float(fn / (fn + tp + 1e-8))
    }
    
    return metrics


def cosine_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine distance (1 - cosine_similarity).
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        float: Cosine distance
    """
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 1.0
    
    cosine_sim = dot_product / (norm1 * norm2)
    
    return float(1.0 - cosine_sim)


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute Euclidean distance.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        float: Euclidean distance
    """
    return float(np.linalg.norm(embedding1 - embedding2))

