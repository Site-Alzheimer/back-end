"""
modelos.py
----------
Carrega as duas CNNs na memória e aquece os grafos. Usado pela API, quando ela mesma roda as
análises, e pelo worker da fila (worker.py). Os modelos que faltarem no disco são baixados do
Google Drive (model_loader.py).
"""

import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow_addons.metrics import F1Score

from engine import LOTE_CNN1, explicar_gradcam
from model_loader import ensure_models

logger = logging.getLogger("alzheimer-api")

MODEL_ROI_PATH = os.getenv("MODEL_ROI_PATH", "models/modelo_cnn3_L20_3A5.h5")
MODEL_CLF_PATH = os.getenv("MODEL_CLF_PATH", "models/CNN-4-2023-1.h5")


def carregar_modelos() -> dict:
    """Devolve {"roi": CNN1, "clf": CNN2}, já aquecidas."""
    ensure_models(MODEL_ROI_PATH, MODEL_CLF_PATH)
    modelos = {}

    logger.info("Carregando modelo de ROI: %s", MODEL_ROI_PATH)
    try:
        modelos["roi"] = tf.keras.models.load_model(MODEL_ROI_PATH)
        logger.info("✓ Modelo ROI carregado com sucesso")
    except Exception as e:
        logger.error("✗ Erro ao carregar modelo ROI: %s", e)
        raise

    logger.info("Carregando modelo classificador: %s", MODEL_CLF_PATH)
    try:
        clf = tf.keras.models.load_model(MODEL_CLF_PATH, compile=False)
        clf.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[F1Score(num_classes=2, average="macro")],
        )
        modelos["clf"] = clf
        logger.info("✓ Modelo classificador carregado com sucesso")
    except Exception as e:
        logger.error("✗ Erro ao carregar modelo classificador: %s", e)
        raise

    # Aquecimento: compila os grafos com as mesmas formas de lote das requisições,
    # para a primeira análise não pagar esse custo.
    modelos["roi"].predict_on_batch(np.zeros((LOTE_CNN1, 176, 176, 3), dtype=np.float32))
    modelos["clf"].predict_on_batch(np.zeros((19, 176, 176, 3), dtype=np.float32))
    explicar_gradcam([np.zeros((176, 176), dtype=np.uint8)] * 19, modelos["clf"], 0)
    logger.info("✓ Modelos aquecidos (lotes de %d e 19, Grad-CAM)", LOTE_CNN1)
    return modelos
