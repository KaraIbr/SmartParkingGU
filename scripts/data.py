"""
Dataset personalizado para imagenes de estacionamiento.

Soporta la estructura de carpetas del dataset CNRPark y datasets
generados a partir de video (frames extraidos).
"""

import os
from typing import Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ParkingDataset(Dataset):
    """
    Dataset personalizado para imagenes de estacionamiento.

    Espera la siguiente estructura de carpetas:
        dataset/
            entrenamiento/
                ocupado/
                no_ocupado/
            validacion/
                ocupado/
                no_ocupado/
            test/
                ocupado/
                no_ocupado/

    Tambien soporta la variante en ingles:
        dataset/
            train/
                occupied/
                empty/
            val/
                occupied/
                empty/
            test/
                occupied/
                empty/
    """

    # Mapeo de nombres de clase (espanol e ingles)
    CLASS_ALIASES: Dict[str, str] = {
        'ocupado': 'occupied',
        'no_ocupado': 'empty',
        'occupied': 'occupied',
        'empty': 'empty',
    }

    LABEL_MAP: Dict[str, int] = {
        'occupied': 0,
        'empty': 1,
    }

    def __init__(
        self,
        root_dir: str,
        split: str = 'entrenamiento',
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """
        Args:
            root_dir: Directorio raiz del dataset.
            split: 'entrenamiento'/'train', 'validacion'/'val', o 'test'.
            transform: Transformaciones a aplicar a cada imagen.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples: List[Dict[str, object]] = []
        self.class_names: List[str] = ['occupied', 'empty']
        self.class_to_idx: Dict[str, int] = dict(self.LABEL_MAP)

        self._load_samples()

    def _resolve_class_dir(self, class_name: str) -> Optional[str]:
        """Resuelve el directorio de clase con aliases espanol/ingles."""
        # Intentar nombre directo
        direct = os.path.join(self.root_dir, self.split, class_name)
        if os.path.isdir(direct):
            return direct

        # Intentar alias
        alias = self.CLASS_ALIASES.get(class_name)
        if alias:
            aliased = os.path.join(self.root_dir, self.split, alias)
            if os.path.isdir(aliased):
                return aliased

        return None

    def _load_samples(self) -> None:
        """Carga las rutas de todas las imagenes validas."""
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        for class_name in self.class_to_idx:
            class_dir = self._resolve_class_dir(class_name)
            if class_dir is None:
                continue

            for img_name in sorted(os.listdir(class_dir)):
                if img_name.lower().endswith(valid_ext):
                    self.samples.append({
                        'path': os.path.join(class_dir, img_name),
                        'label': self.class_to_idx[class_name],
                        'class': class_name,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[object, int]:
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        label = sample['label']

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> List[float]:
        """Calcula pesos de clase inversamente proporcionales a la frecuencia."""
        counts = {c: 0 for c in self.class_to_idx}
        for s in self.samples:
            counts[s['class']] += 1

        total = len(self.samples)
        weights = []
        for cls in self.class_names:
            w = total / (len(self.class_names) * max(counts[cls], 1))
            weights.append(w)
        return weights


def get_default_transforms(
    img_size: Tuple[int, int] = (150, 150),
    augment: bool = False,
) -> Dict[str, transforms.Compose]:
    """
    Retorna transformaciones por defecto para entrenamiento, validacion y test.

    Args:
        img_size: Tamano目标 de la imagen (ancho, alto).
        augment: Si True, aplica augmentation al set de entrenamiento.

    Returns:
        Dict con claves 'train', 'val', 'test'.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tfms = [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]

    if augment:
        train_tfms.insert(1, transforms.RandomHorizontalFlip(p=0.5))
        train_tfms.insert(2, transforms.RandomRotation(10))

    val_tfms = [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]

    return {
        'train': transforms.Compose(train_tfms),
        'val': transforms.Compose(val_tfms),
        'test': transforms.Compose(val_tfms),
    }
