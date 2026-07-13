"""
Script de evaluacion para SmartParkingCNN.

Incluye:
- Metricas: Accuracy, Precision, Recall, F1-Score
- Matriz de confusion
- ROC-AUC curve
- PR curve
- Curvas de loss y accuracy
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)


def evaluate_model(model, test_loader, device='cuda', class_names=['Ocupado', 'Libre']):
    """
    Evalua el modelo en el conjunto de prueba.
    
    Args:
        model: Modelo PyTorch entrenado
        test_loader: DataLoader de prueba
        device: 'cuda' o 'cpu'
        class_names: Nombres de las clases
    
    Returns:
        metrics: Diccionario con todas las metricas
        all_preds: Predicciones del modelo
        all_labels: Etiquetas reales
        all_probs: Probabilidades predichas
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calcular metricas
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro'),
        'recall_macro': recall_score(all_labels, all_preds, average='macro'),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'precision_per_class': precision_score(all_labels, all_preds, average=None),
        'recall_per_class': recall_score(all_labels, all_preds, average=None),
        'f1_per_class': f1_score(all_labels, all_preds, average=None),
        'classification_report': classification_report(
            all_labels, all_preds, target_names=class_names
        )
    }
    
    return metrics, all_preds, all_labels, all_probs


def plot_confusion_matrix(all_labels, all_preds, class_names=['Ocupado', 'Libre'], 
                         save_path=None, title='Matriz de Confusion'):
    """
    Genera y muestra la matriz de confusion.
    
    Args:
        all_labels: Etiquetas reales
        all_preds: Predicciones del modelo
        class_names: Nombres de las clases
        save_path: Ruta para guardar la imagen (opcional)
        title: Titulo del grafico
    """
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Prediccion', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return cm


def plot_roc_curve(all_labels, all_probs, class_names=['Ocupado', 'Libre'],
                   save_path=None, title='ROC Curve'):
    """
    Genera la curva ROC para clasificacion binaria.
    
    Args:
        all_labels: Etiquetas reales
        all_probs: Probabilidades predichas
        class_names: Nombres de las clases
        save_path: Ruta para guardar la imagen (opcional)
        title: Titulo del grafico
    """
    # Para clasificacion binaria, usar la probabilidad de la clase positiva
    if all_probs.ndim == 2:
        y_scores = all_probs[:, 1]
    else:
        y_scores = all_probs
    
    fpr, tpr, thresholds = roc_curve(all_labels, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
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
    plt.show()
    
    return fpr, tpr, roc_auc


def plot_pr_curve(all_labels, all_probs, class_names=['Ocupado', 'Libre'],
                  save_path=None, title='Precision-Recall Curve'):
    """
    Genera la curva Precision-Recall.
    
    Args:
        all_labels: Etiquetas reales
        all_probs: Probabilidades predichas
        class_names: Nombres de las clases
        save_path: Ruta para guardar la imagen (opcional)
        title: Titulo del grafico
    """
    if all_probs.ndim == 2:
        y_scores = all_probs[:, 1]
    else:
        y_scores = all_probs
    
    precision, recall, thresholds = precision_recall_curve(all_labels, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return precision, recall, pr_auc


def plot_training_curves(history, save_path=None, title='Curvas de Entrenamiento'):
    """
    Genera graficos de loss y accuracy durante el entrenamiento.
    
    Args:
        history: Diccionario con historial de entrenamiento
        save_path: Ruta para guardar la imagen (opcional)
        title: Titulo del grafico
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Grafico de Loss
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', color='red', linewidth=2)
    axes[0].set_title('Loss durante Entrenamiento', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoca', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Grafico de Accuracy
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
    plt.show()


def print_metrics(metrics, title="METRICAS DE EVALUACION"):
    """
    Imprime las metricas de evaluacion de forma formateada.
    
    Args:
        metrics: Diccionario con metricas
        title: Titulo a imprimir
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Accuracy:     {metrics['accuracy']*100:.2f}%")
    print(f"Precision:    {metrics['precision_macro']*100:.2f}%")
    print(f"Recall:       {metrics['recall_macro']*100:.2f}%")
    print(f"F1-Score:     {metrics['f1_macro']*100:.2f}%")
    print(f"\nPor clase:")
    print(f"  Ocupado  - Precision: {metrics['precision_per_class'][0]*100:.2f}%, "
          f"Recall: {metrics['recall_per_class'][0]*100:.2f}%, "
          f"F1: {metrics['f1_per_class'][0]*100:.2f}%")
    print(f"  Libre    - Precision: {metrics['precision_per_class'][1]*100:.2f}%, "
          f"Recall: {metrics['recall_per_class'][1]*100:.2f}%, "
          f"F1: {metrics['f1_per_class'][1]*100:.2f}%")
    print(f"{'='*60}")
    print(f"\nReporte de clasificacion completo:")
    print(metrics['classification_report'])


def compare_models(base_metrics, tuned_metrics, model_names=['Base Model', 'Tuned Model']):
    """
    Compara metricas de dos modelos y muestra la mejora.
    
    Args:
        base_metrics: Metricas del modelo base
        tuned_metrics: Metricas del modelo optimizado
        model_names: Nombres de los modelos
    """
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    base_values = [
        base_metrics['accuracy'],
        base_metrics['precision_macro'],
        base_metrics['recall_macro'],
        base_metrics['f1_macro']
    ]
    tuned_values = [
        tuned_metrics['accuracy'],
        tuned_metrics['precision_macro'],
        tuned_metrics['recall_macro'],
        tuned_metrics['f1_macro']
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
    
    # Agregar valores sobre las barras
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir mejoras
    print(f"\n{'='*60}")
    print("MEJORA ENTRE MODELOS")
    print(f"{'='*60}")
    for name, base, tuned in zip(metrics_names, base_values, tuned_values):
        improvement = (tuned - base) * 100
        print(f"{name}: {base*100:.2f}% -> {tuned*100:.2f}% ({improvement:+.2f}%)")
    print(f"{'='*60}")
