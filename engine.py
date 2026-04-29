"""
engine.py
---------
Motor de inferência para classificação de Alzheimer via MRI.

Extrai TODOS os parâmetros necessários para reconstrução de gráficos no frontend:
  - ROI scores da 1ª CNN (varredura)
  - Predições bruas dos 19 slices (2ª CNN)
  - Estatísticas de precisão (média, desvio, CV)
  - Metadados do NIfTI (dimensões, spacing)
  - Metadados de extração (ini_cranio)

Arquitetura headless — ZERO bibliotecas visuais.
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2
import nibabel as nib

logger = logging.getLogger("alzheimer-engine")

# Offsets dos 19 slices ao redor do corte central do hipocampo
OFFSETS = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9]


# ═════════════════════════════════════════════════════════════════════════════
# BLOCO 1 — Funções auxiliares de pré-processamento
# ═════════════════════════════════════════════════════════════════════════════

def robust_normalize(arr: np.ndarray, 
                     pct_low: float = 2.0, 
                     pct_high: float = 98.0) -> np.ndarray:
    """Normaliza pelo percentil para suprimir pixels espúrios de ruído.

    Em vez de dividir pelo valor máximo (que pode ser um pixel isolado de ruído),
    clipa os valores entre o 2º e o 98º percentil dos pixels não-nulos e então
    reescala para [0, 1].

    Args:
        arr: Array de intensidades (qualquer dimensão).
        pct_low: Percentil inferior (padrão 2.0).
        pct_high: Percentil superior (padrão 98.0).

    Returns:
        Array normalizado em [0.0, 1.0], dtype float32.
    """
    nonzero = arr[arr > 0]
    if nonzero.size == 0:
        return arr.astype(np.float32)
    
    p_low, p_high = np.percentile(nonzero, [pct_low, pct_high])
    clipped = np.clip(arr, p_low, p_high)
    span = p_high - p_low + 1e-8
    
    return ((clipped - p_low) / span).astype(np.float32)


def find_skull_start(img_uint8: np.ndarray) -> int:
    """Detecta a primeira linha vertical com sinal cerebral relevante.

    Percorre a imagem de cima para baixo em blocos de largura proporcional.
    Retorna o índice de linha (0-based) onde o crânio começa.

    Args:
        img_uint8: Imagem 2D em uint8 (valores 0-255).

    Returns:
        Índice inteiro da linha onde o crânio é detectado (fallback: 0).
    """
    h, w = img_uint8.shape
    block_w = max(1, w // 5)       # blocos proporcionais
    threshold_sum = 250            # limiar de soma por bloco

    for row in range(h):
        for col in range(0, w - block_w + 1, block_w):
            if np.sum(img_uint8[row, col: col + block_w]) > threshold_sum:
                return row

    return 0  # fallback: crânio não detectado


def prepare_for_roi_model(raw_slice: np.ndarray, 
                          target: Tuple[int, int] = (176, 176)) -> np.ndarray:
    """Pré-processa um slice 2D para o modelo detector de ROI (modelo_cnn1).

    Estratégia sem cortes fixos:
      1. Redimensiona para 176×256 (proporção clínica padrão).
      2. Corta os 39% superiores (região supra-hipocampal em vista coronal).
      3. Redimensiona para target, rotaciona 180° (orientação esperada).
      4. Normaliza por percentil e empilha em 3 canais RGB.

    Args:
        raw_slice: Slice 2D do arquivo NIfTI.
        target: Resolução alvo (padrão 176×176).

    Returns:
        Tensor (1, H, W, 3) float32 pronto para model.predict().
    """
    H_REF, W_REF = 256, 176

    resized = cv2.resize(raw_slice, (W_REF, H_REF))
    norm = robust_normalize(resized)
    img_u8 = (norm * 255).astype(np.uint8)

    # Corte proporcional: 39% do topo corresponde à região supra-hipocampal
    cut_top = int(H_REF * 0.39)
    cropped = img_u8[cut_top:, :]

    final = cv2.resize(cropped, target)
    final = cv2.rotate(final, cv2.ROTATE_180)

    rgb = np.stack([final, final, final], axis=-1).astype(np.float32) / 255.0
    return rgb.reshape(1, *rgb.shape)  # (1, 176, 176, 3)


def prepare_for_clf_model(raw_slice: np.ndarray,
                          ini_cranio: int,
                          int_max: float,
                          target: Tuple[int, int] = (176, 176)) -> np.ndarray:
    """Pré-processa um slice 2D para o modelo classificador (CNN-4).

    Usa os parâmetros extraídos do slice central (ini_cranio, int_max) para
    garantir consistência em toda a janela de 19 slices.

    Args:
        raw_slice: Slice 2D do arquivo NIfTI.
        ini_cranio: Linha de início do crânio (do slice central).
        int_max: Intensidade máxima da ROI (do slice central).
        target: Resolução alvo (padrão 176×176).

    Returns:
        Tensor (1, H, W, 3) float32.
    """
    H_REF, W_REF = 256, 176

    resized = cv2.resize(raw_slice, (W_REF, H_REF))
    margin = max(0, ini_cranio - 15)

    safe_max = float(int_max) if int_max > 0 else 1.0
    norm = np.clip(resized / safe_max, 0.0, 1.0)

    cropped = norm[margin: margin + target[1], :]
    img_u8 = (cropped * 255).astype(np.uint8)
    final = cv2.resize(img_u8, target)

    rgb = np.stack([final, final, final], axis=-1).astype(np.float32) / 255.0
    return rgb.reshape(1, *rgb.shape)  # (1, 176, 176, 3)


# ═════════════════════════════════════════════════════════════════════════════
# BLOCO 2 — Pipeline de inferência (com captura completa de metadados)
# ═════════════════════════════════════════════════════════════════════════════

def scan_hippocampus_roi(img_data: np.ndarray, model_roi) -> Dict[str, Any]:
    """Varre todos os cortes coronais para localizar o hipocampo.

    Para cada corte, alimenta o modelo_cnn1 e coleta o score de ROI.
    Determina a faixa de slices onde o score >= 80% do máximo.

    Args:
        img_data: Array 3D do NIfTI (SafeImage).
        model_roi: Modelo Keras para detecção de ROI.

    Returns:
        Dict com:
          - roi_scores: List[float] — scores de todos os cortes
          - limite_inferior: int
          - limite_superior: int
          - slice_central: int
    """
    n_coronal = img_data.shape[1]
    scores = np.zeros(n_coronal, dtype=np.float32)

    # Loop 1: Calcula scores para todos os coronais
    for ii in range(n_coronal):
        inp = prepare_for_roi_model(img_data[:, ii, :])
        scores[ii] = float(model_roi.predict(inp, verbose=0)[0][0])

    logger.debug("ROI scores calculados: min=%.4f, max=%.4f, mean=%.4f",
                 scores.min(), scores.max(), scores.mean())

    threshold = scores.max() * 0.80

    # Loop 2: Define limites da ROI usando o threshold
    candidates = np.where(scores >= threshold)[0]
    if len(candidates) == 0:
        lim_inf, lim_sup = n_coronal // 3, 2 * n_coronal // 3
        logger.warning("Nenhum score >= threshold. Usando fallback: [%d, %d]",
                       lim_inf, lim_sup)
    else:
        lim_inf, lim_sup = int(candidates[0]), int(candidates[-1])

    # Loop 3: Remove vales internos abaixo do limiar
    interval = scores[lim_inf: lim_sup + 1]
    while len(interval) > 0 and interval.min() < threshold:
        min_pos = lim_inf + int(np.argmin(interval))
        if (min_pos - lim_inf) < (lim_sup - min_pos):
            lim_inf = min_pos + 1
        else:
            lim_sup = min_pos - 1
        interval = scores[lim_inf: lim_sup + 1]
        if len(interval) == 0:
            break

    sl_central = int(round((lim_inf + lim_sup) / 2))

    logger.info("ROI detectada: [%d, %d] → slice_central=%d (threshold=%.4f)",
                lim_inf, lim_sup, sl_central, threshold)

    return {
        "roi_scores": scores.tolist(),
        "limite_inferior": int(lim_inf),
        "limite_superior": int(lim_sup),
        "slice_central": int(sl_central),
    }


def build_roi_dataset(img_data: np.ndarray, sl_central: int) -> Tuple[List, int, float]:
    """Constrói o dataset de 19 slices ao redor do corte central.

    Extrai dois parâmetros do slice central como referência:
      - ini_cranio: linha de início do crânio
      - int_max: 90% da intensidade máxima da ROI central

    Args:
        img_data: Array 3D do NIfTI.
        sl_central: Índice do slice central a ser usado como referência.

    Returns:
        Tupla (dataset, ini_cranio, int_max) onde:
          - dataset: List[np.ndarray] — 19 slices uint8 (176×176)
          - ini_cranio: int
          - int_max: float
    """
    H_REF, W_REF = 256, 176
    n_coronal = img_data.shape[1]

    # Extrai parâmetros do slice central
    central_raw = cv2.resize(img_data[:, sl_central, :], (W_REF, H_REF))
    central_u8 = (robust_normalize(central_raw) * 255).astype(np.uint8)
    ini_cranio = find_skull_start(central_u8)

    margin = max(0, ini_cranio - 15)
    roi_central = central_raw[margin: margin + 176, :]
    int_max = float(roi_central.max()) * 0.90 if roi_central.size > 0 else 1.0

    logger.debug("Parâmetros extraídos: ini_cranio=%d, int_max=%.2f",
                 ini_cranio, int_max)

    # Constrói dataset de 19 slices
    dataset = []
    for offset in OFFSETS:
        idx = max(0, min(sl_central + offset, n_coronal - 1))
        inp = prepare_for_clf_model(img_data[:, idx, :], ini_cranio, int_max)
        # Armazena como uint8 (176×176)
        dataset.append((inp[0, :, :, 0] * 255).astype(np.uint8))

    return dataset, ini_cranio, int_max


def classify_alzheimer(dataset: List[np.ndarray], model_clf) -> Dict[str, Any]:
    """Classifica cada slice do dataset e agrega o resultado com estatísticas.

    Retorna:
      - predicoes_19_slices: List[float] — scores brutos dos 19 slices
      - probabilidade_media: float — média das probabilidades
      - desvio_padrao: float — desvio padrão
      - coeficiente_variacao: float — CV (desvio/média ou desvio/(1-média))
      - classificacao_cv: str — classificação qualitativa do CV
      - diagnostico: str — "ALZHEIMER" ou "NORMAL"

    Args:
        dataset: List[np.ndarray] — 19 slices uint8.
        model_clf: Modelo Keras classificador.

    Returns:
        Dict com todos os parâmetros acima.
    """
    predicoes = []

    for img_u8 in dataset:
        rgb = np.stack([img_u8, img_u8, img_u8], axis=-1).astype(np.float32) / 255.0
        inp = rgb.reshape(1, *rgb.shape)
        pred = model_clf.predict(inp, verbose=0)  # shape (1, 2)
        predicoes.append(float(pred[0][0]))

    predicoes_array = np.array(predicoes)
    media = float(np.mean(predicoes_array))
    desvio = float(np.std(predicoes_array))

    logger.debug("Classificação: média=%.4f, desvio=%.4f",
                 media, desvio)

    # Calcula coeficiente de variação
    if media > 0.5:
        diagnostico = "ALZHEIMER"
        # CV = desvio / média
        coef_variacao = desvio / (media + 1e-8)
    else:
        diagnostico = "NORMAL"
        # CV = desvio / (1 - média)
        coef_variacao = desvio / ((1.0 - media) + 1e-8)

    # Classifica CV qualitativo
    classificacao_cv = _classificar_cv(coef_variacao)

    logger.info("Diagnóstico: %s (CV=%.4f, classificação=%s)",
                diagnostico, coef_variacao, classificacao_cv)

    return {
        "predicoes_19_slices": predicoes,
        "probabilidade_media": media,
        "desvio_padrao": desvio,
        "coeficiente_variacao": coef_variacao,
        "classificacao_cv": classificacao_cv,
        "diagnostico": diagnostico,
    }


def _classificar_cv(valor: float) -> str:
    """Classifica o coeficiente de variação em categorias qualitativas.

    Args:
        valor: Valor do coeficiente de variação (float).

    Returns:
        String descritiva da classificação.
    """
    if valor < 0.10:
        return "High precision"
    elif 0.10 <= valor < 0.15:
        return "Acceptable precision"
    elif 0.15 <= valor < 0.20:
        return "Moderate variability"
    else:
        return "High variability"


def run_pipeline(nifti_path: str, model_roi, model_clf) -> Dict[str, Any]:
    """Orquestra o pipeline completo com extração de TODOS os metadados.

    Passos:
    1. Carrega NIfTI e extrai metadados (dimensões, spacing).
    2. Varre slices coronais → localiza hipocampo (com scores).
    3. Monta dataset de 19 slices (extrai ini_cranio).
    4. Classifica com estatísticas completas.

    Args:
        nifti_path: Caminho do arquivo NIfTI temporário.
        model_roi: Modelo Keras para ROI.
        model_clf: Modelo Keras para classificação.

    Returns:
        Dict contendo TODOS os dados necessários para frontend:
          - diagnostico
          - probabilidade_media
          - estatisticas_predicao (com predicoes_19_slices, desvio, CV, etc)
          - metadados_nifti (dimensões, spacing)
          - metadados_varredura_cnn1 (roi_scores, limites, slice_central)
          - metadados_extracao_cnn2 (inicio_cranio_y)
    """
    logger.info("Iniciando pipeline para: %s", nifti_path)

    # ──────────────────────────────────────────────────────────────────────
    # Passo 1: Carrega NIfTI e extrai metadados
    # ──────────────────────────────────────────────────────────────────────
    img_ni = nib.load(nifti_path)
    img_ni.header.check_fix()
    img_data = nib.as_closest_canonical(img_ni).get_fdata()

    # Extrai dimensões e spacing (zooms)
    dimensoes = list(img_ni.shape)
    espacamento = list(img_ni.header.get_zooms()[:3])

    logger.info("Metadados NIfTI: dimensões=%s, spacing=%s",
                dimensoes, espacamento)

    # ──────────────────────────────────────────────────────────────────────
    # Passo 2: Varre ROI (retorna scores e limites)
    # ──────────────────────────────────────────────────────────────────────
    roi_result = scan_hippocampus_roi(img_data, model_roi)

    # ──────────────────────────────────────────────────────────────────────
    # Passo 3: Monta dataset de 19 slices (retorna ini_cranio)
    # ──────────────────────────────────────────────────────────────────────
    dataset, ini_cranio, int_max = build_roi_dataset(
        img_data, roi_result["slice_central"]
    )

    logger.debug("Dataset construído: 19 slices, ini_cranio=%d, int_max=%.2f",
                 ini_cranio, int_max)

    # ──────────────────────────────────────────────────────────────────────
    # Passo 4: Classifica com estatísticas
    # ──────────────────────────────────────────────────────────────────────
    clf_result = classify_alzheimer(dataset, model_clf)

    # ──────────────────────────────────────────────────────────────────────
    # Retorna resultado COMPLETO com TODOS os metadados
    # ──────────────────────────────────────────────────────────────────────
    return {
        "diagnostico": clf_result["diagnostico"],
        "probabilidade_media": clf_result["probabilidade_media"],
        "estatisticas_predicao": {
            "desvio_padrao": clf_result["desvio_padrao"],
            "coeficiente_variacao": clf_result["coeficiente_variacao"],
            "classificacao_cv": clf_result["classificacao_cv"],
            "predicoes_19_slices": clf_result["predicoes_19_slices"],
        },
        "metadados_nifti": {
            "dimensoes": dimensoes,
            "espacamento": espacamento,
        },
        "metadados_varredura_cnn1": {
            "roi_scores": roi_result["roi_scores"],
            "limite_inferior": roi_result["limite_inferior"],
            "limite_superior": roi_result["limite_superior"],
            "slice_central": roi_result["slice_central"],
        },
        "metadados_extracao_cnn2": {
            "inicio_cranio_y": ini_cranio,
        },
    }