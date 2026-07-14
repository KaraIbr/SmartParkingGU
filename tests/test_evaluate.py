"""Tests para el modulo evaluate.py."""

import numpy as np
import pytest

from src.evaluate import (
    compare_models,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_training_curves,
    print_metrics,
)


@pytest.fixture
def sample_metrics():
    """Metricas de ejemplo para testing."""
    return {
        'accuracy': 0.85,
        'precision_macro': 0.83,
        'recall_macro': 0.82,
        'f1_macro': 0.825,
        'precision_per_class': [0.85, 0.81],
        'recall_per_class': [0.80, 0.84],
        'f1_per_class': [0.82, 0.82],
        'classification_report': "              precision    recall  f1-score   support\n\n     Ocupado       0.85      0.80      0.82       100\n       Libre       0.81      0.84      0.82       100\n\n    accuracy                           0.82       200\n   macro avg       0.83      0.82      0.82       200\nweighted avg       0.83      0.82      0.82       200\n",
    }


@pytest.fixture
def sample_predictions():
    """Predicciones de ejemplo."""
    np.random.seed(42)
    n = 100
    labels = np.random.randint(0, 2, n)
    preds = np.random.randint(0, 2, n)
    probs = np.random.rand(n, 2)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return labels, preds, probs


class TestPlotConfusionMatrix:
    def test_returns_array(self, sample_predictions):
        labels, preds, _ = sample_predictions
        cm = plot_confusion_matrix(labels, preds)
        assert cm.shape == (2, 2)

    def test_save_path(self, sample_predictions, tmp_path):
        labels, preds, _ = sample_predictions
        save_path = str(tmp_path / "cm.png")
        cm = plot_confusion_matrix(labels, preds, save_path=save_path)
        import os
        assert os.path.exists(save_path)


class TestPlotROCCurve:
    def test_returns_tuple(self, sample_predictions):
        labels, _, probs = sample_predictions
        result = plot_roc_curve(labels, probs)
        assert len(result) == 3  # fpr, tpr, auc


class TestPlotPRCurve:
    def test_returns_tuple(self, sample_predictions):
        labels, _, probs = sample_predictions
        result = plot_pr_curve(labels, probs)
        assert len(result) == 3


class TestPlotTrainingCurves:
    def test_runs_without_error(self):
        history = {
            'train_loss': [0.9, 0.7, 0.5],
            'val_loss': [0.95, 0.75, 0.55],
            'train_acc': [50, 65, 80],
            'val_acc': [48, 63, 78],
        }
        plot_training_curves(history)


class TestPrintMetrics:
    def test_runs_without_error(self, sample_metrics, capsys):
        print_metrics(sample_metrics)
        captured = capsys.readouterr()
        assert "Accuracy" in captured.out


class TestCompareModels:
    def test_runs_without_error(self, sample_metrics, capsys):
        base = dict(sample_metrics)
        tuned = dict(sample_metrics)
        tuned['accuracy'] = 0.95
        tuned['f1_macro'] = 0.94
        compare_models(base, tuned)
        captured = capsys.readouterr()
        assert "MEJORA" in captured.out
