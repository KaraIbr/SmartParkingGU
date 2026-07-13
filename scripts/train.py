"""
Script de entrenamiento para SmartParkingCNN.

Incluye:
- Funcion de entrenamiento con validacion
- Early Stopping
- Guardado de checkpoints
- Historial de metricas
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


class EarlyStopping:
    """
    Detiene el entrenamiento cuando la metrica de validacion deja de mejorar.
    
    Args:
        patience: Epocas sin mejora antes de detener (default: 10)
        min_delta: Minima mejora considerada (default: 0)
        mode: 'min' para loss, 'max' para accuracy (default: 'min')
    """
    
    def __init__(self, patience=10, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
        elif self._is_better(score):
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def _is_better(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta
    
    def get_best_model(self, model):
        """Retorna el modelo con mejor rendimiento."""
        model.load_state_dict(self.best_model_state)
        return model


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=60,
    learning_rate=1e-3,
    weight_decay=0,
    device='cuda',
    save_dir='experiments/cnrpark/base_model',
    patience=10,
    class_weights=None
):
    """
    Entrena el modelo con validacion y early stopping.
    
    Args:
        model: Modelo PyTorch a entrenar
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validacion
        num_epochs: Numero maximo de epocas
        learning_rate: Tasa de aprendizaje
        weight_decay: Regularizacion L2
        device: 'cuda' o 'cpu'
        save_dir: Directorio para guardar checkpoints
        patience: Paciencia para early stopping
        class_weights: Pesos para las clases (para imbalance)
    
    Returns:
        history: Diccionario con historial de entrenamiento
    """
    # Configurar dispositivo
    model = model.to(device)
    
    # Configurar funcion de perdida
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Configurar optimizador
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mode='min')
    
    # Historial de entrenamiento
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rate': learning_rate
    }
    
    # Crear directorio de guardado
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"INICIO DE ENTRENAMIENTO")
    print(f"{'='*60}")
    print(f"Epocas maximas: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # ==================== FASE DE ENTRENAMIENTO ====================
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Estadisticas
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Actualizar barra de progreso
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })
        
        train_loss = running_loss / total
        train_acc = 100 * correct / total
        
        # ==================== FASE DE VALIDACION ====================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]  ")
            
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
        
        # Guardar historial
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Imprimir resumen de epoca
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'train_acc': train_acc
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
            print(f"  -> Modelo guardado (mejor val_loss: {val_loss:.4f})")
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\n{'='*60}")
            print(f"EARLY STOPPING: Epoch {epoch+1}")
            print(f"{'='*60}")
            break
    
    # Cargar mejor modelo
    model = early_stopping.get_best_model(model)
    
    # Guardar modelo final
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': 'SmartParkingCNN',
        'num_classes': 2,
        'final_metrics': {
            'best_val_loss': best_val_loss,
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1]
        }
    }, os.path.join(save_dir, 'final_model.pt'))
    
    print(f"\n{'='*60}")
    print(f"ENTRENAMIENTO COMPLETADO")
    print(f"{'='*60}")
    print(f"Mejor val_loss: {best_val_loss:.4f}")
    print(f"Modelo guardado en: {save_dir}")
    print(f"{'='*60}")
    
    return history


def load_checkpoint(model, checkpoint_path):
    """
    Carga un checkpoint del modelo.
    
    Args:
        model: Modelo PyTorch
        checkpoint_path: Ruta al archivo .pt
    
    Returns:
        model: Modelo con pesos cargados
        checkpoint_info: Informacion del checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint
