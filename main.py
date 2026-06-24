"""
main.py

Punto de entrada principal del proyecto. Permite ejecutar el
entrenamiento o la predicción on-demand desde la raíz del repositorio,
de forma sencilla para revisión del profesor.

Uso:
    python main.py train      # entrena el modelo y registra en MLflow
    python main.py predict    # corre el workflow on-demand de predicción
"""

import sys

sys.path.insert(0, "src")

from train import main as train_main          # noqa: E402
from predict import main as predict_main       # noqa: E402


def usage():
    print("Uso: python main.py [train|predict]")
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage()

    command = sys.argv[1]

    if command == "train":
        train_main()
    elif command == "predict":
        predict_main()
    else:
        usage()
