"""
Script de inferencia para SmartParkingCNN.

Permite cargar el modelo serializado y hacer predicciones
en nuevas imagenes de estacionamiento.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .models import SmartParkingCNN


class SmartParkingPredictor:
    """
    Clase para realizar predicciones con el modelo entrenado.

    Ejemplo de uso:
        predictor = SmartParkingPredictor('models/final_model.pt')
        resultado = predictor.predict('imagen_estacionamiento.jpg')
        print(resultado)
    """

    CLASS_NAMES: List[str] = ['Ocupado', 'Libre']
    IMG_SIZE: Tuple[int, int] = (150, 150)

    def __init__(
        self, model_path: str, device: Optional[str] = None
    ) -> None:
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.transform = transforms.Compose([
            transforms.Resize(self.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.metadata: Dict[str, Any] = {}
        self.model = self._load_model(model_path)

        print(f"Modelo cargado desde: {model_path}")
        print(f"Dispositivo: {self.device}")

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Carga el modelo desde un archivo .pt."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            self.metadata = {
                'architecture': checkpoint.get('architecture', 'SmartParkingCNN'),
                'num_classes': checkpoint.get('num_classes', 2),
                'class_names': checkpoint.get('class_names', self.CLASS_NAMES),
                'best_hyperparams': checkpoint.get('best_hyperparams', {}),
                'final_metrics': checkpoint.get('final_metrics', {}),
            }
        else:
            state_dict = checkpoint

        model = SmartParkingCNN(num_classes=2, dropout_rate=0.5)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model

    def predict(
        self, image_path: str, return_probs: bool = False
    ) -> Dict[str, Any]:
        """
        Predice la clase de una imagen de estacionamiento.

        Args:
            image_path: Ruta a la imagen
            return_probs: Si True, retorna probabilidades tambien

        Returns:
            Diccionario con la prediccion
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            probs = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

        result: Dict[str, Any] = {
            'class': self.CLASS_NAMES[predicted.item()],
            'class_idx': predicted.item(),
            'confidence': confidence.item(),
            'image_path': image_path,
        }

        if return_probs:
            result['probabilities'] = {
                self.CLASS_NAMES[i]: probs[0][i].item()
                for i in range(len(self.CLASS_NAMES))
            }

        return result

    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Predice multiples imagenes."""
        return [self.predict(path) for path in image_paths]

    def get_metadata(self) -> Dict[str, Any]:
        """Retorna los metadatos del modelo."""
        return self.metadata


def main() -> None:
    """Ejemplo de uso del predictor."""
    model_path = 'models/final_model_cnrpark.pt'

    if not os.path.exists(model_path):
        print(f"Error: No se encontro el modelo en {model_path}")
        print("Primero ejecuta el notebook 03_finetune_cnrpark.ipynb para entrenar.")
        return

    predictor = SmartParkingPredictor(model_path)

    print("\nMetadatos del modelo:")
    metadata = predictor.get_metadata()
    if metadata:
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    print("\nModelo listo para inferencia.")
    print("Uso:")
    print("  predictor = SmartParkingPredictor('models/final_model.pt')")
    print("  resultado = predictor.predict('imagen.jpg')")
    print("  print(resultado)")


if __name__ == "__main__":
    main()
