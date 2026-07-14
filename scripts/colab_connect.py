"""
SmartParkingGU - Script de verificacion de conexion Colab-VS Code

Ejecutar este script en Colab para verificar que la conexion
con VS Code esta funcionando correctamente.

Uso:
    python scripts/colab_connect.py
"""

import subprocess
import sys
import os


def check_gpu():
    """Verificar disponibilidad de GPU"""
    try:
        import torch
        print("=" * 50)
        print("GPU STATUS")
        print("=" * 50)
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"GPU: {gpu_name}")
            print(f"VRAM: {gpu_mem:.1f} GB total, {allocated:.2f} GB asignada")
            return True
        else:
            print("No hay GPU disponible")
            return False
    except ImportError:
        print("PyTorch no instalado")
        return False


def check_jupyter():
    """Verificar servidor Jupyter"""
    print("\n" + "=" * 50)
    print("JUPYTER STATUS")
    print("=" * 50)

    result = subprocess.run(
        ["pgrep", "-f", "jupyter"],
        capture_output=True,
        text=True
    )

    if result.stdout.strip():
        print("Servidor Jupyter: ACTIVO")
        return True
    else:
        print("Servidor Jupyter: INACTIVO")
        print("Ejecuta: !jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --NotebookApp.token=smartparking --allow-root")
        return False


def check_tunnel():
    """Verificar tunnel"""
    print("\n" + "=" * 50)
    print("TUNNEL STATUS")
    print("=" * 50)

    result = subprocess.run(
        ["pgrep", "-f", "localtunnel"],
        capture_output=True,
        text=True
    )

    if result.stdout.strip():
        print("Tunnel: ACTIVO")
        return True
    else:
        print("Tunnel: INACTIVO")
        print("Ejecuta: !lt --port 8888")
        return False


def check_project():
    """Verificar estructura del proyecto"""
    print("\n" + "=" * 50)
    print("PROJECT STATUS")
    print("=" * 50)

    required_files = [
        "scripts/models.py",
        "scripts/train.py",
        "scripts/evaluate.py",
        "requirements.txt",
        "notebooks/02_base_model_cnrpark.ipynb"
    ]

    all_ok = True
    for f in required_files:
        if os.path.exists(f):
            print(f"  OK: {f}")
        else:
            print(f"  MISSING: {f}")
            all_ok = False

    return all_ok


def main():
    print("SmartParkingGU - Verificacion de Conexion Colab-VS Code\n")

    gpu_ok = check_gpu()
    jupyter_ok = check_jupyter()
    tunnel_ok = check_tunnel()
    project_ok = check_project()

    print("\n" + "=" * 50)
    print("RESUMEN")
    print("=" * 50)
    print(f"GPU:       {'OK' if gpu_ok else 'ERROR'}")
    print(f"Jupyter:   {'OK' if jupyter_ok else 'ERROR'}")
    print(f"Tunnel:    {'OK' if tunnel_ok else 'ERROR'}")
    print(f"Project:   {'OK' if project_ok else 'ERROR'}")

    if all([gpu_ok, jupyter_ok, tunnel_ok, project_ok]):
        print("\nTodo listo. Copia la URL del tunnel y conecta desde VS Code.")
        print("  Ctrl+Shift+P > Jupyter: Specify Jupyter Server URL")
    else:
        print("\nHay problemas. Revisa los errores arriba.")

    return 0 if all([gpu_ok, jupyter_ok, tunnel_ok, project_ok]) else 1


if __name__ == "__main__":
    sys.exit(main())