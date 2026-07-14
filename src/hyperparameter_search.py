"""
Script de busqueda de hiperparametros para SmartParkingCNN.

Incluye:
- Grid Search sistematico
- Random Search
- Registro de experimentos
- Analisis de importancia de parametros
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import SmartParkingCNN
from train import train_model, EarlyStopping
from evaluate import evaluate_model


def set_seed(seed=42):
    """
    Establece semillas para reproducibilidad.
    
    Args:
        seed: Valor de semilla (default: 42)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def grid_search(
    train_loader,
    val_loader,
    test_loader,
    param_grid=None,
    num_epochs=30,
    device='cuda',
    save_dir='experiments/cnrpark'
):
    """
    Realiza Grid Search sobre combinaciones de hiperparametros.
    
    Args:
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validacion
        test_loader: DataLoader de prueba
        param_grid: Diccionario con rangos de parametros
        num_epochs: Numero de epocas por experimento
        device: 'cuda' o 'cpu'
        save_dir: Directorio para guardar resultados
    
    Returns:
        results: Lista de diccionarios con resultados
    """
    if param_grid is None:
        param_grid = {
            'learning_rate': [1e-3, 5e-4, 1e-4, 5e-5],
            'weight_decay': [0, 1e-4, 1e-3, 5e-3],
            'dropout_rate': [0.3, 0.5],
            'batch_size': [32]
        }
    
    # Generar todas las combinaciones
    import itertools
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))
    
    print(f"{'='*60}")
    print(f"GRID SEARCH")
    print(f"{'='*60}")
    print(f"Total de combinaciones: {len(combinations)}")
    print(f"Epocas por experimento: {num_epochs}")
    print(f"{'='*60}\n")
    
    results = []
    
    for idx, values in enumerate(combinations, 1):
        config = dict(zip(keys, values))
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENTO {idx}/{len(combinations)}")
        print(f"Config: {config}")
        print(f"{'='*60}")
        
        # Establecer semilla para reproducibilidad
        set_seed(42)
        
        # Crear modelo con dropout configurado
        model = SmartParkingCNN(
            num_classes=2,
            dropout_rate=config.get('dropout_rate', 0.5)
        )
        
        # Entrenar modelo
        save_dir_exp = os.path.join(save_dir, f"grid_experiment_{idx}")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0),
            device=device,
            save_dir=save_dir_exp,
            patience=10
        )
        
        # Evaluar en test
        model, _ = load_checkpoint_from_dir(model, save_dir_exp, 'best_model.pt')
        metrics, _, _, _ = evaluate_model(model, test_loader, device)
        
        # Guardar resultado
        result = {
            'experiment_id': idx,
            'learning_rate': config['learning_rate'],
            'weight_decay': config.get('weight_decay', 0),
            'dropout_rate': config.get('dropout_rate', 0.5),
            'batch_size': config.get('batch_size', 32),
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision_macro'],
            'recall': metrics['recall_macro'],
            'f1_score': metrics['f1_macro'],
            'best_val_loss': history['val_loss'][-1] if history['val_loss'] else float('inf'),
            'epochs_trained': len(history['train_loss'])
        }
        results.append(result)
        
        print(f"\nResultado: Acc={metrics['accuracy']*100:.2f}%, "
              f"F1={metrics['f1_macro']*100:.2f}%")
    
    return results


def random_search(
    train_loader,
    val_loader,
    test_loader,
    param_distributions=None,
    n_iter=15,
    num_epochs=30,
    device='cuda',
    save_dir='experiments/cnrpark'
):
    """
    Realiza Random Search sobre distribuciones de parametros.
    
    Args:
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validacion
        test_loader: DataLoader de prueba
        param_distributions: Diccionario con distribuciones de parametros
        n_iter: Numero de iteraciones aleatorias
        num_epochs: Numero de epocas por experimento
        device: 'cuda' o 'cpu'
        save_dir: Directorio para guardar resultados
    
    Returns:
        results: Lista de diccionarios con resultados
    """
    if param_distributions is None:
        param_distributions = {
            'learning_rate': [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            'weight_decay': [0, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2],
            'dropout_rate': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            'batch_size': [16, 32, 64]
        }
    
    print(f"\n{'='*60}")
    print(f"RANDOM SEARCH")
    print(f"{'='*60}")
    print(f"Numero de iteraciones: {n_iter}")
    print(f"Epocas por experimento: {num_epochs}")
    print(f"{'='*60}\n")
    
    results = []
    
    for idx in range(1, n_iter + 1):
        # Muestrear parametros aleatoriamente
        config = {
            'learning_rate': random.choice(param_distributions['learning_rate']),
            'weight_decay': random.choice(param_distributions['weight_decay']),
            'dropout_rate': random.choice(param_distributions['dropout_rate']),
            'batch_size': random.choice(param_distributions['batch_size'])
        }
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENTO RANDOM {idx}/{n_iter}")
        print(f"Config: {config}")
        print(f"{'='*60}")
        
        # Establecer semilla
        set_seed(42 + idx)
        
        # Crear modelo
        model = SmartParkingCNN(
            num_classes=2,
            dropout_rate=config['dropout_rate']
        )
        
        # Entrenar
        save_dir_exp = os.path.join(save_dir, f"random_experiment_{idx}")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            device=device,
            save_dir=save_dir_exp,
            patience=10
        )
        
        # Evaluar
        model, _ = load_checkpoint_from_dir(model, save_dir_exp, 'best_model.pt')
        metrics, _, _, _ = evaluate_model(model, test_loader, device)
        
        # Guardar resultado
        result = {
            'experiment_id': f"random_{idx}",
            'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'dropout_rate': config['dropout_rate'],
            'batch_size': config['batch_size'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision_macro'],
            'recall': metrics['recall_macro'],
            'f1_score': metrics['f1_macro'],
            'best_val_loss': history['val_loss'][-1] if history['val_loss'] else float('inf'),
            'epochs_trained': len(history['train_loss'])
        }
        results.append(result)
        
        print(f"\nResultado: Acc={metrics['accuracy']*100:.2f}%, "
              f"F1={metrics['f1_macro']*100:.2f}%")
    
    return results


def save_results_to_csv(results, save_path):
    """
    Guarda los resultados de experimentos en un archivo CSV.
    
    Args:
        results: Lista de diccionarios con resultados
        save_path: Ruta del archivo CSV
    """
    if not results:
        print("No hay resultados para guardar.")
        return
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    keys = results[0].keys()
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResultados guardados en: {save_path}")


def load_results_from_csv(csv_path):
    """
    Carga resultados de experimentos desde un archivo CSV.
    
    Args:
        csv_path: Ruta del archivo CSV
    
    Returns:
        results: Lista de diccionarios con resultados
    """
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convertir strings a numeros
            for key in ['learning_rate', 'weight_decay', 'dropout_rate', 
                       'accuracy', 'precision', 'recall', 'f1_score', 'best_val_loss']:
                if key in row:
                    row[key] = float(row[key])
            if 'batch_size' in row:
                row['batch_size'] = int(row['batch_size'])
            if 'epochs_trained' in row:
                row['epochs_trained'] = int(row['epochs_trained'])
            results.append(row)
    
    return results


def find_best_config(results, metric='f1_score'):
    """
    Encuentra la mejor configuracion basada en una metrica.
    
    Args:
        results: Lista de diccionarios con resultados
        metric: Metrica para ordenar (default: 'f1_score')
    
    Returns:
        best_result: Mejor resultado encontrado
    """
    if not results:
        return None
    
    best_result = max(results, key=lambda x: x.get(metric, 0))
    return best_result


def load_checkpoint_from_dir(model, checkpoint_dir, checkpoint_name):
    """
    Carga un modelo desde un directorio de checkpoint.
    
    Args:
        model: Modelo PyTorch
        checkpoint_dir: Directorio del checkpoint
        checkpoint_name: Nombre del archivo de checkpoint
    
    Returns:
        model: Modelo con pesos cargados
        checkpoint: Checkpoint completo
    """
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


if __name__ == "__main__":
    print("Script de busqueda de hiperparametros")
    print("Este script se ejecuta desde los notebooks.")
    print("Ver notebooks/03_finetune_cnrpark.ipynb para uso completo.")
