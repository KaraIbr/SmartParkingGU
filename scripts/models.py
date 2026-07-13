"""
SmartParkingCNN - Arquitectura de red neuronal convolucional
para deteccion de ocupacion de espacios de estacionamiento.

Dataset: CNRPark-Patches-150x150
Entrada: Imagenes 150x150x3 (RGB)
Salida: Clasificacion binaria (Ocupado/Libre)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmartParkingCNN(nn.Module):
    """
    Red neuronal convolucional construida desde cero.
    
    Arquitectura:
        - 4 bloques convolucionales (Conv2d + BatchNorm + ReLU + MaxPool)
        - 2 capas totalmente conectadas
        - Dropout para regularizacion
    
    Dimensiones (imagenes 150x150):
        - Bloque 1: 150x150 -> 75x75 (32 filtros)
        - Bloque 2: 75x75 -> 37x37 (64 filtros)
        - Bloque 3: 37x37 -> 18x18 (128 filtros)
        - Bloque 4: 18x18 -> 9x9 (256 filtros)
        - Flatten: 256 * 9 * 9 = 20736
        - FC1: 20736 -> 512
        - FC2: 512 -> 2 (clases)
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        """
        Inicializa la arquitectura de la red.
        
        Args:
            num_classes: Numero de clases de salida (default: 2)
            dropout_rate: Tasa de dropout para regularizacion (default: 0.5)
        """
        super(SmartParkingCNN, self).__init__()
        
        # Bloque convolucional 1: 3 canales -> 32 filtros
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloque convolucional 2: 32 -> 64 filtros
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloque convolucional 3: 64 -> 128 filtros
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloque convolucional 4: 128 -> 256 filtros
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Capas totalmente conectadas
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 9 * 9, 512)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """
        Forward pass de la red.
        
        Args:
            x: Tensor de entrada [batch_size, 3, 150, 150]
        
        Returns:
            Logits de salida [batch_size, num_classes]
        """
        # Bloque 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Bloque 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Bloque 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Bloque 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Capas completamente conectadas
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """Cuenta el numero total de parametros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model_info(model):
    """
    Retorna informacion del modelo.
    
    Args:
        model: Instancia de SmartParkingCNN
    
    Returns:
        Diccionario con informacion del modelo
    """
    total_params = model.count_parameters()
    return {
        "arquitectura": "SmartParkingCNN",
        "capas_convolucionales": 4,
        "capas_fc": 2,
        "parametros_totales": total_params,
        "parametros_miles": f"{total_params / 1000:.1f}K"
    }


if __name__ == "__main__":
    # Prueba rapida de la arquitectura
    model = SmartParkingCNN(num_classes=2, dropout_rate=0.5)
    info = get_model_info(model)
    
    print("=" * 50)
    print("SmartParkingCNN - Informacion del Modelo")
    print("=" * 50)
    for key, value in info.items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    # Probar con tensor de prueba
    dummy_input = torch.randn(1, 3, 150, 150)
    output = model(dummy_input)
    print(f"Entrada: {dummy_input.shape}")
    print(f"Salida: {output.shape}")
