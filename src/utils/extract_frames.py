"""
Script de extraccion de frames de video para SmartParkingGU.

Extrae frames a 1 FPS de videos de estacionamiento y genera
un dataset listo para etiquetar y entrenar.

Uso:
    python scripts/extract_frames.py --video_dir Video/ --output_dir data/raw/frames_video/
"""

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def get_video_info(video_path: str) -> Dict[str, object]:
    """Obtiene informacion basica de un video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    info = {
        'path': video_path,
        'filename': os.path.basename(video_path),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration_seconds': 0.0,
    }
    info['duration_seconds'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0
    cap.release()
    return info


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: int = 1,
    resize: Optional[Tuple[int, int]] = None,
    prefix: str = '',
) -> List[Dict[str, object]]:
    """
    Extrae frames de un video a una frecuencia especifica.

    Args:
        video_path: Ruta al video.
        output_dir: Directorio de salida para los frames.
        fps: Frames por segundo a extraer (default: 1).
        resize: Tupla (ancho, alto) para redimensionar (None = original).
        prefix: Prefijo para los nombres de archivo.

    Returns:
        Lista de diccionarios con metadata de cada frame extraido.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if fps > 0 else 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_dir, exist_ok=True)

    metadata: List[Dict[str, object]] = []
    frame_idx = 0
    extracted = 0

    print(f"Extrayendo frames de: {os.path.basename(video_path)}")
    print(f"  FPS original: {video_fps}, Extrayendo a: {fps} FPS")
    print(f"  Intervalo: cada {frame_interval} frames")
    print(f"  Total frames en video: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if resize:
                frame = cv2.resize(frame, resize)

            timestamp = frame_idx / video_fps
            filename = f"{prefix}frame_{extracted:04d}.jpg"
            filepath = os.path.join(output_dir, filename)

            cv2.imwrite(filepath, frame)

            metadata.append({
                'video': os.path.basename(video_path),
                'frame_idx': frame_idx,
                'extracted_idx': extracted,
                'timestamp_seconds': round(timestamp, 2),
                'filename': filename,
                'width': frame.shape[1],
                'height': frame.shape[0],
            })

            extracted += 1

        frame_idx += 1

    cap.release()
    print(f"  Extraidos: {extracted} frames")
    return metadata


def save_metadata_csv(metadata: List[Dict[str, object]], output_path: str) -> None:
    """Guarda la metadata de frames extraidos en un CSV."""
    if not metadata:
        print("No hay metadata para guardar.")
        return

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    fieldnames = list(metadata[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)

    print(f"Metadata guardada en: {output_path}")


def process_video_dir(
    video_dir: str,
    output_dir: str,
    fps: int = 1,
    resize: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, object]]:
    """
    Procesa todos los videos de un directorio.

    Args:
        video_dir: Directorio con videos (.mp4, .avi, .mov).
        output_dir: Directorio de salida para frames.
        fps: Frames por segundo a extraer.
        resize: Tupla (ancho, alto) para redimensionar.

    Returns:
        Lista completa de metadata de todos los frames.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    all_metadata: List[Dict[str, object]] = []

    if not os.path.isdir(video_dir):
        raise ValueError(f"Directorio no encontrado: {video_dir}")

    videos = sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith(video_extensions)
    ])

    if not videos:
        print(f"No se encontraron videos en: {video_dir}")
        return all_metadata

    print(f"Encontrados {len(videos)} videos en {video_dir}\n")

    for i, video_name in enumerate(videos, 1):
        video_path = os.path.join(video_dir, video_name)
        prefix = f"v{i}_"  # v1_, v2_, v3_ para diferenciar videos

        frames_dir = os.path.join(output_dir)
        metadata = extract_frames(video_path, frames_dir, fps, resize, prefix)
        all_metadata.extend(metadata)
        print()

    return all_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Extrae frames de videos de estacionamiento.'
    )
    parser.add_argument(
        '--video_dir', type=str, default='Video/',
        help='Directorio con los videos (default: Video/)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data/raw/frames_video/',
        help='Directorio de salida para frames (default: data/raw/frames_video/)'
    )
    parser.add_argument(
        '--fps', type=int, default=1,
        help='Frames por segundo a extraer (default: 1)'
    )
    parser.add_argument(
        '--resize', type=int, nargs=2, default=None, metavar=('WIDTH', 'HEIGHT'),
        help='Redimensionar frames a WIDTHxHEIGHT (ej: 224 224)'
    )
    parser.add_argument(
        '--csv_output', type=str, default=None,
        help='Ruta para guardar metadata CSV (default: output_dir/frames_metadata.csv)'
    )
    args = parser.parse_args()

    resize = tuple(args.resize) if args.resize else None

    all_metadata = process_video_dir(
        args.video_dir, args.output_dir, args.fps, resize
    )

    csv_path = args.csv_output or os.path.join(args.output_dir, 'frames_metadata.csv')
    save_metadata_csv(all_metadata, csv_path)

    print(f"\n{'='*60}")
    print(f"EXTRACCION COMPLETADA")
    print(f"{'='*60}")
    print(f"Total frames extraidos: {len(all_metadata)}")
    print(f"Directorio de salida: {args.output_dir}")
    print(f"Metadata CSV: {csv_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
