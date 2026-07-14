"""
Script de evaluacion para SmartParkingCNN.

Incluye:
- Metricas: Accuracy, Precision, Recall, F1-Score
- Matriz de confusion
- ROC-AUC curve
- PR curve
- Curvas de loss y accuracy
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda',
    class_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evalua el modelo en el conjunto de prueba.

    Args:
        model: Modelo PyTorch entrenado
        test_loader: DataLoader de prueba
        device: 'cuda' o 'cpu'
        class_names: Nombres de las clases

    Returns:
        Tupla de (metrics dict, all_preds, all_labels, all_probs)
    """
    if class_names is None:
        class_names = ['Ocupado', 'Libre']

    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    probs_arr = np.array(all_probs)

    metrics: Dict[str, Any] = {
        'accuracy': accuracy_score(labels_arr, preds_arr),
        'precision_macro': precision_score(labels_arr, preds_arr, average='macro'),
        'recall_macro': recall_score(labels_arr, preds_arr, average='macro'),
        'f1_macro': f1_score(labels_arr, preds_arr, average='macro'),
        'precision_per_class': precision_score(labels_arr, preds_arr, average=None).tolist(),
        'recall_per_class': recall_score(labels_arr, preds_arr, average=None).tolist(),
        'f1_per_class': f1_score(labels_arr, preds_arr, average=None).tolist(),
        'classification_report': classification_report(
            labels_arr, preds_arr, target_names=class_names
        ),
    }

    return metrics, preds_arr, labels_arr, probs_arr


def plot_confusion_matrix(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = 'Matriz de Confusion',
) -> np.ndarray:
    """Genera y muestra la matriz de confusion."""
    if class_names is None:
        class_names = ['Ocupado', 'Libre']

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Prediccion', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return cm


def plot_roc_curve(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = 'ROC Curve',
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Genera la curva ROC para clasificacion binaria."""
    y_scores = all_probs[:, 1] if all_probs.ndim == 2 else all_probs
    fpr, tpr, _ = roc_curve(all_labels, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return fpr, tpr, roc_auc


def plot_pr_curve(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = 'Precision-Recall Curve',
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Genera la curva Precision-Recall."""
    y_scores = all_probs[:, 1] if all_probs.ndim == 2 else all_probs
    precision, recall, _ = precision_recall_curve(all_labels, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return precision, recall, pr_auc


def plot_training_curves(
    history: Dict[str, Any],
    save_path: Optional[str] = None,
    title: str = 'Curvas de Entrenamiento',
) -> None:
    """Genera graficos de loss y accuracy durante el entrenamiento."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', color='red', linewidth=2)
    axes[0].set_title('Loss durante Entrenamiento', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoca', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train Accuracy', color='blue', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Accuracy', color='red', linewidth=2)
    axes[1].set_title('Accuracy durante Entrenamiento', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoca', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_metrics(metrics: Dict[str, Any], title: str = "METRICAS DE EVALUACION") -> None:
    """Imprime las metricas de evaluacion de forma formateada."""
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    print(f"Accuracy:     {metrics['accuracy']*100:.2f}%")
    print(f"Precision:    {metrics['precision_macro']*100:.2f}%")
    print(f"Recall:       {metrics['recall_macro']*100:.2f}%")
    print(f"F1-Score:     {metrics['f1_macro']*100:.2f}%")
    print(f"\nPor clase:")
    print(
        f"  Ocupado  - Precision: {metrics['precision_per_class'][0]*100:.2f}%, "
        f"Recall: {metrics['recall_per_class'][0]*100:.2f}%, "
        f"F1: {metrics['f1_per_class'][0]*100:.2f}%"
    )
    print(
        f"  Libre    - Precision: {metrics['precision_per_class'][1]*100:.2f}%, "
        f"Recall: {metrics['recall_per_class'][1]*100:.2f}%, "
        f"F1: {metrics['f1_per_class'][1]*100:.2f}%"
    )
    print(f"{'='*60}")
    print(f"\nReporte de clasificacion completo:")
    print(metrics['classification_report'])


def compare_models(
    base_metrics: Dict[str, Any],
    tuned_metrics: Dict[str, Any],
    model_names: Optional[List[str]] = None,
) -> None:
    """Compara metricas de dos modelos y muestra la mejora."""
    if model_names is None:
        model_names = ['Base Model', 'Tuned Model']

    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    base_values = [
        base_metrics['accuracy'],
        base_metrics['precision_macro'],
        base_metrics['recall_macro'],
        base_metrics['f1_macro'],
    ]
    tuned_values = [
        tuned_metrics['accuracy'],
        tuned_metrics['precision_macro'],
        tuned_metrics['recall_macro'],
        tuned_metrics['f1_macro'],
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, base_values, width, label=model_names[0], color='steelblue')
    bars2 = ax.bar(x + width/2, tuned_values, width, label=model_names[1], color='coral')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comparacion de Modelos', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f'{height:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom', fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f'{height:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom', fontsize=9,
        )

    plt.tight_layout()
    plt.close()

    print(f"\n{'='*60}")
    print("MEJORA ENTRE MODELOS")
    print(f"{'='*60}")
    for name, base, tuned in zip(metrics_names, base_values, tuned_values):
        improvement = (tuned - base) * 100
        print(f"{name}: {base*100:.2f}% -> {tuned*100:.2f}% ({improvement:+.2f}%)")
    print(f"{'='*60}")
