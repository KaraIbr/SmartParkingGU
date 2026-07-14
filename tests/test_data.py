"""Tests para el modulo data.py."""

import os
import tempfile

import pytest
import torch
from PIL import Image

from src.data import ParkingDataset, get_default_transforms


def _create_fake_dataset(tmp_dir: str, split: str = 'train', n_images: int = 5) -> str:
    """Crea un dataset fake para testing."""
    for cls in ['occupied', 'empty']:
        cls_dir = os.path.join(tmp_dir, split, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n_images):
            img = Image.new('RGB', (200, 200), color=(i * 50, i * 30, i * 20))
            img.save(os.path.join(cls_dir, f'img_{i}.jpg'))
    return tmp_dir


class TestParkingDataset:
    """Tests para la clase ParkingDataset."""

    def test_loads_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            _create_fake_dataset(tmp, n_images=3)
            ds = ParkingDataset(tmp, split='train')
            assert len(ds) == 6  # 3 occupied + 3 empty

    def test_getitem_returns_tuple(self):
        with tempfile.TemporaryDirectory() as tmp:
            _create_fake_dataset(tmp, n_images=2)
            ds = ParkingDataset(tmp, split='train')
            img, label = ds[0]
            assert isinstance(img, Image.Image)
            assert label in [0, 1]

    def test_class_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            _create_fake_dataset(tmp, n_images=5)
            ds = ParkingDataset(tmp, split='train')
            weights = ds.get_class_weights()
            assert len(weights) == 2
            assert all(w > 0 for w in weights)

    def test_with_transform(self):
        from torchvision import transforms
        tfm = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            _create_fake_dataset(tmp, n_images=2)
            ds = ParkingDataset(tmp, split='train', transform=tfm)
            img, label = ds[0]
            assert isinstance(img, torch.Tensor)
            assert img.shape == (3, 150, 150)

    def test_spanish_aliases(self):
        """Verifica que funciona con nombres de carpetas en espanol."""
        with tempfile.TemporaryDirectory() as tmp:
            for cls in ['ocupado', 'no_ocupado']:
                cls_dir = os.path.join(tmp, 'entrenamiento', cls)
                os.makedirs(cls_dir)
                Image.new('RGB', (100, 100)).save(os.path.join(cls_dir, 'img.jpg'))
            ds = ParkingDataset(tmp, split='entrenamiento')
            assert len(ds) == 2


class TestGetDefaultTransforms:
    """Tests para get_default_transforms."""

    def test_returns_dict(self):
        tfms = get_default_transforms()
        assert isinstance(tfms, dict)
        assert 'train' in tfms
        assert 'val' in tfms
        assert 'test' in tfms

    def test_augment_flag(self):
        tfms_no_aug = get_default_transforms(augment=False)
        tfms_aug = get_default_transforms(augment=True)
        # Train transform with augmentation should be different (more transforms)
        assert tfms_no_aug['train'] is not tfms_aug['train']
