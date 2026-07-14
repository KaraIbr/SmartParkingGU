"""
SmartParkingGU - Sistema de Deteccion de Ocupacion de Estacionamiento.

Paquete principal que expone los modulos del pipeline ML.
"""

from .data import ParkingDataset, get_default_transforms
from .evaluate import (
    compare_models,
    evaluate_model,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_training_curves,
    print_metrics,
)
from .models import SmartParkingCNN, get_model_info
from .train import EarlyStopping, load_checkpoint, train_model

__all__ = [
    # Data
    'ParkingDataset',
    'get_default_transforms',
    # Models
    'SmartParkingCNN',
    'get_model_info',
    # Training
    'train_model',
    'load_checkpoint',
    'EarlyStopping',
    # Evaluation
    'evaluate_model',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_pr_curve',
    'plot_training_curves',
    'print_metrics',
    'compare_models',
]

__version__ = '2.0.0'
