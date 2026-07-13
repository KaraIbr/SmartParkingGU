"""
Scripts para el proyecto SmartParkingGU.
"""

from .models import SmartParkingCNN, get_model_info
from .train import train_model, load_checkpoint, EarlyStopping
from .evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_training_curves,
    print_metrics,
    compare_models
)

__all__ = [
    'SmartParkingCNN',
    'get_model_info',
    'train_model',
    'load_checkpoint',
    'EarlyStopping',
    'evaluate_model',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_pr_curve',
    'plot_training_curves',
    'print_metrics',
    'compare_models'
]
