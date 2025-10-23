
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime

from app.core.database import BiasMetric, Session

@dataclass
class DemographicGroup:
    category: str
    group_name: str

@dataclass
class PerformanceMetrics:
    accuracy: float
    far: float
    frr: float
    eer: float
    sample_count: int

class BiasMonitor:

    def __init__(self):
        self.demographics_cache = {}

    def compute_fairness_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        ground_truth: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        group_performances = {}

        for group_name in predictions.keys():
            pred = predictions[group_name]
            gt = ground_truth[group_name]

            accuracy = np.mean(pred == gt)
            tp = np.sum((gt == 1) & (pred == 1))
            fp = np.sum((gt == 0) & (pred == 1))
            tn = np.sum((gt == 0) & (pred == 0))
            fn = np.sum((gt == 1) & (pred == 0))

            far = fp / (fp + tn + 1e-8)
            frr = fn / (fn + tp + 1e-8)

            group_performances[group_name] = {
                'accuracy': accuracy,
                'far': far,
                'frr': frr
            }

        fairness_metrics = {}

        accuracies = [p['accuracy'] for p in group_performances.values()]
        fairness_metrics['demographic_parity_diff'] = max(accuracies) - min(accuracies)

        fars = [p['far'] for p in group_performances.values()]
        frrs = [p['frr'] for p in group_performances.values()]
        fairness_metrics['equalized_odds_far_diff'] = max(fars) - min(fars)
        fairness_metrics['equalized_odds_frr_diff'] = max(frrs) - min(frrs)
        fairness_metrics['equalized_odds_diff'] = (
            fairness_metrics['equalized_odds_far_diff'] +
            fairness_metrics['equalized_odds_frr_diff']
        ) / 2

        fairness_metrics['mean_accuracy'] = np.mean(accuracies)
        fairness_metrics['min_accuracy'] = min(accuracies)
        fairness_metrics['max_accuracy'] = max(accuracies)

        return fairness_metrics

    def check_bias_threshold(
        self,
        fairness_metrics: Dict[str, float],
        threshold: float = 0.05
    ) -> Tuple[bool, List[str]]:
        violations = []

        if fairness_metrics['demographic_parity_diff'] > threshold:
            violations.append(
                f"Demographic parity violation: {fairness_metrics['demographic_parity_diff']:.3f}"
            )

        if fairness_metrics['equalized_odds_diff'] > threshold:
            violations.append(
                f"Equalized odds violation: {fairness_metrics['equalized_odds_diff']:.3f}"
            )

        has_bias = len(violations) > 0

        return has_bias, violations

    def store_bias_metrics(
        self,
        db: Session,
        category: str,
        group_metrics: Dict[str, Dict]
    ):
        for group_name, metrics in group_metrics.items():
            bias_metric = BiasMetric(
                category=category,
                group_name=group_name,
                total_samples=metrics.get('total_samples', 0),
                true_positives=metrics.get('tp', 0),
                false_positives=metrics.get('fp', 0),
                true_negatives=metrics.get('tn', 0),
                false_negatives=metrics.get('fn', 0),
                accuracy=metrics.get('accuracy', 0.0),
                far=metrics.get('far', 0.0),
                frr=metrics.get('frr', 0.0),
                demographic_parity_diff=metrics.get('demographic_parity_diff'),
                equalized_odds_diff=metrics.get('equalized_odds_diff'),
                computed_at=datetime.utcnow()
            )
            db.add(bias_metric)

        db.commit()

    def analyze_embedding_bias(
        self,
        embeddings: np.ndarray,
        demographic_labels: np.ndarray
    ) -> Dict[str, float]:
        from sklearn.metrics.pairwise import cosine_similarity

        unique_groups = np.unique(demographic_labels)

        intra_group_sims = []
        inter_group_sims = []

        for group in unique_groups:
            group_mask = demographic_labels == group
            group_embeddings = embeddings[group_mask]

            if len(group_embeddings) > 1:
                group_sim_matrix = cosine_similarity(group_embeddings)
                triu_indices = np.triu_indices_from(group_sim_matrix, k=1)
                intra_sims = group_sim_matrix[triu_indices]
                intra_group_sims.extend(intra_sims)

            other_mask = demographic_labels != group
            other_embeddings = embeddings[other_mask]
            if len(other_embeddings) > 0:
                inter_sim_matrix = cosine_similarity(group_embeddings, other_embeddings)
                inter_group_sims.extend(inter_sim_matrix.flatten())

        metrics = {
            'intra_group_mean': float(np.mean(intra_group_sims)),
            'intra_group_std': float(np.std(intra_group_sims)),
            'inter_group_mean': float(np.mean(inter_group_sims)),
            'inter_group_std': float(np.std(inter_group_sims)),
            'separation_score': float(np.mean(intra_group_sims) - np.mean(inter_group_sims))
        }

        return metrics

    def generate_bias_report(
        self,
        db: Session,
        category: str,
        recent_days: int = 30
    ) -> Dict:
        from datetime import timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=recent_days)

        metrics = db.query(BiasMetric).filter(
            BiasMetric.category == category,
            BiasMetric.computed_at >= cutoff_date
        ).all()

        if not metrics:
            return {'error': 'No data available'}

        group_data = {}
        for metric in metrics:
            if metric.group_name not in group_data:
                group_data[metric.group_name] = []
            group_data[metric.group_name].append(metric)

        report = {
            'category': category,
            'period_days': recent_days,
            'groups': {}
        }

        for group_name, group_metrics in group_data.items():
            accuracies = [m.accuracy for m in group_metrics if m.accuracy is not None]
            fars = [m.far for m in group_metrics if m.far is not None]
            frrs = [m.frr for m in group_metrics if m.frr is not None]

            report['groups'][group_name] = {
                'sample_count': sum(m.total_samples for m in group_metrics),
                'mean_accuracy': float(np.mean(accuracies)) if accuracies else 0.0,
                'mean_far': float(np.mean(fars)) if fars else 0.0,
                'mean_frr': float(np.mean(frrs)) if frrs else 0.0
            }

        accuracies = [g['mean_accuracy'] for g in report['groups'].values()]
        report['overall'] = {
            'min_accuracy': min(accuracies),
            'max_accuracy': max(accuracies),
            'accuracy_gap': max(accuracies) - min(accuracies)
        }

        return report

    def recommend_mitigation(
        self,
        fairness_metrics: Dict[str, float]
    ) -> List[str]:
        recommendations = []

        if fairness_metrics['demographic_parity_diff'] > 0.1:
            recommendations.append(
                "High demographic parity difference detected. "
                "Consider: (1) Collecting more balanced training data, "
                "(2) Applying fairness-aware loss functions, "
                "(3) Post-processing calibration per group"
            )

        if fairness_metrics['equalized_odds_diff'] > 0.1:
            recommendations.append(
                "High equalized odds difference detected. "
                "Consider: (1) Threshold optimization per group, "
                "(2) Adversarial debiasing during training, "
                "(3) Ensemble methods with diverse models"
            )

        accuracy_gap = fairness_metrics['max_accuracy'] - fairness_metrics['min_accuracy']
        if accuracy_gap > 0.15:
            recommendations.append(
                "Large accuracy gap between groups. "
                "Consider: (1) Augmenting underperforming group's training data, "
                "(2) Domain adaptation techniques, "
                "(3) Group-specific fine-tuning"
            )

        if not recommendations:
            recommendations.append("No significant bias detected. Continue monitoring.")

        return recommendations

_bias_monitor_instance: Optional[BiasMonitor] = None

def get_bias_monitor() -> BiasMonitor:
    global _bias_monitor_instance
    if _bias_monitor_instance is None:
        _bias_monitor_instance = BiasMonitor()
    return _bias_monitor_instance
