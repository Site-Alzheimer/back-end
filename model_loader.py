"""
model_loader.py
---------------
Garante que os modelos CNN existam no disco antes do startup da API.

Se o arquivo ja existir, nada e baixado. Se estiver faltando, o download e feito
do Google Drive usando gdown. Isso permite manter os .h5 fora da imagem Docker.
"""

import logging
import os
import tempfile
import zipfile
from pathlib import Path

import gdown

logger = logging.getLogger("alzheimer-model-loader")


def _download_if_missing(path: str, url: str | None, label: str, env_name: str) -> None:
    target = Path(path)

    if target.exists():
        logger.info("Modelo %s ja existe em %s. Download ignorado.", label, target)
        return

    if not url:
        raise RuntimeError(
            f"Modelo {label} nao encontrado em {target}. "
            f"Defina a variavel de ambiente {env_name} com a URL do Google Drive "
            "ou coloque o arquivo manualmente nesse caminho."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Modelo %s nao encontrado em %s. Baixando do Google Drive...", label, target)

    try:
        gdown.download(url=url, output=str(target), quiet=False, fuzzy=True)
    except Exception:
        if target.exists():
            target.unlink()
        raise

    if not target.exists() or target.stat().st_size == 0:
        raise RuntimeError(f"Falha ao baixar o modelo {label} para {target}")

    logger.info("Modelo %s baixado com sucesso em %s", label, target)


def ensure_models(roi_path: str, clf_path: str) -> None:
    """Baixa os modelos configurados caso eles ainda nao existam no disco."""
    roi_url = os.getenv("MODEL_ROI_URL")
    clf_url = os.getenv("MODEL_CLF_URL")

    _download_if_missing(roi_path, roi_url, "ROI", "MODEL_ROI_URL")
    _download_if_missing(clf_path, clf_url, "classificador", "MODEL_CLF_URL")


def ensure_amostras(pasta: str) -> None:
    """Baixa os exames de exemplo (zip com amostras.json, exames e miniaturas) de AMOSTRAS_URL.

    Serve para hospedagens sem disco próprio, como o Space do Hugging Face: o zip fica num link
    guardado como segredo, e os exames não aparecem nos arquivos públicos. Sem a variável, ou com a
    pasta já pronta, nada muda. Uma falha aqui não derruba a API: o carrossel só fica vazio.
    """
    url = os.getenv("AMOSTRAS_URL")
    destino = Path(pasta)
    if not url or (destino / "amostras.json").exists():
        return

    logger.info("Exames de exemplo nao encontrados em %s. Baixando de AMOSTRAS_URL...", destino)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "amostras.zip"
            gdown.download(url=url, output=str(zip_path), quiet=True, fuzzy=True)
            destino.mkdir(parents=True, exist_ok=True)
            base = destino.resolve()
            with zipfile.ZipFile(zip_path) as z:
                for nome in z.namelist():
                    # nenhum caminho do zip pode sair da pasta de destino
                    if not (base / nome).resolve().is_relative_to(base):
                        raise RuntimeError(f"Caminho invalido no zip: {nome}")
                z.extractall(base)
        logger.info("Exames de exemplo prontos em %s", destino)
    except Exception as exc:
        logger.warning("Exames de exemplo indisponiveis (%s: %s). O carrossel fica vazio.",
                       type(exc).__name__, exc)
