"""
Script de inferencia para SmartParkingCNN.

Permite cargar el modelo serializado y hacer predicciones
en nuevas imagenes de estacionamiento.
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np

# Agregar scripts al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import SmartParkingCNN


class SmartParkingPredictor:
    """
    Clase para realizar predicciones con el modelo entrenado.
    
    Ejemplo de uso:
        predictor = SmartParkingPredictor('models/final_model_cnrpark.pt')
        resultado = predictor.predict('imagen_estacionamiento.jpg')
        print(resultado)
    """
    
    def __init__(self, model_path, device=None):
        """
        Inicializa el predictor.
        
        Args:
            model_path: Ruta al archivo .pt del modelo
            device: 'cuda' o 'cpu' (auto-detect si es None)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.class_names = ['Ocupado', 'Libre']
        self.img_size = (150, 150)
        
        # Transformaciones para inferencia
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Cargar modelo
        self.model = self._load_model(model_path)
        print(f"Modelo cargado desde: {model_path}")
        print(f"Dispositivo: {self.device}")
    
    def _load_model(self, model_path):
        """
        Carga el modelo desde un archivo .pt.
        
        Args:
            model_path: Ruta al archivo .pt
        
        Returns:
            model: Modelo cargado
        """
        # Cargar checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Obtener informacion del checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Checkpoint completo con metadatos
            state_dict = checkpoint['model_state_dict']
            self.metadata = {
                'architecture': checkpoint.get('architecture', 'SmartParkingCNN'),
                'num_classes': checkpoint.get('num_classes', 2),
                'class_names': checkpoint.get('class_names', ['Ocupado', 'Libre']),
                'best_hyperparams': checkpoint.get('best_hyperparams', {}),
                'final_metrics': checkpoint.get('final_metrics', {})
            }
        else:
            # Solo state_dict
            state_dict = checkpoint
            self.metadata = {}
        
        # Crear modelo
        model = SmartParkingCNN(num_classes=2, dropout_rate=0.5)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, image_path, return_probs=False):
        """
        Predice la clase de una imagen de estacionamiento.
        
        Args:
            image_path: Ruta a la imagen
            return_probs: Si True, retorna probabilidades tambien
        
        Returns:
            prediction: Diccionario con la prediccion
        """
        # Cargar y preprocesar imagen
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Realizar prediccion
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        result = {
            'class': self.class_names[predicted.item()],
            'class_idx': predicted.item(),
            'confidence': confidence.item(),
            'image_path': image_path
        }
        
        if return_probs:
            result['probabilities'] = {
                self.class_names[i]: probs[0][i].item()
                for i in range(len(self.class_names))
            }
        
        return result
    
    def predict_batch(self, image_paths):
        """
        Predice multiples imagenes.
        
        Args:
            image_paths: Lista de rutas a imagenes
        
        Returns:
            predictions: Lista de diccionarios con predicciones
        """
        predictions = []
        for path in image_paths:
            pred = self.predict(path)
            predictions.append(pred)
        return predictions
    
    def get_metadata(self):
        """
        Retorna los metadatos del modelo.
        
        Returns:
            metadata: Diccionario con metadatos
        """
        return self.metadata


def main():
    """
    Ejemplo de uso del predictor.
    """
    # Verificar que exista el modelo
    model_path = 'models/final_model_cnrpark.pt'
    
    if not os.path.exists(model_path):
        print(f"Error: No se encontro el modelo en {model_path}")
        print("Primero ejecuta el notebook 03_finetune_cnrpark.ipynb para entrenar y exportar el modelo.")
        return
    
    # Crear predictor
    predictor = SmartParkingPredictor(model_path)
    
    # Mostrar metadatos
    print("\nMetadatos del modelo:")
    metadata = predictor.get_metadata()
    if metadata:
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    print("\nModelo listo para inferencia.")
    print("Uso:")
    print("  predictor = SmartParkingPredictor('models/final_model_cnrpark.pt')")
    print("  resultado = predictor.predict('imagen.jpg')")
    print("  print(resultado)")


if __name__ == "__main__":
    main()
