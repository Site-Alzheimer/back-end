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

Além dos resultados acima, devolve dados extras para o painel (tempos por etapa, recortes,
as imagens que as redes receberam e Grad-CAM), sem alterar o que as redes calculam:
scripts/paridade.py compara com as respostas gravadas antes dessas mudanças.
"""

import base64
import logging
import os
import time
from typing import Dict, List, Tuple, Any, Callable, Optional
import numpy as np
import cv2
import nibabel as nib
from nibabel import orientations

logger = logging.getLogger("alzheimer-engine")

# Offsets dos 19 slices ao redor do corte central do hipocampo
OFFSETS = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9]

# Tamanho fixo dos lotes da CNN1: o último lote é completado para não recompilar o grafo
LOTE_CNN1 = 32

# Pesos de cada etapa no progresso geral (0–100), para o progresso nunca voltar
PESOS_ETAPAS = {"load": (0, 5), "cnn1": (5, 70), "extract": (75, 3), "cnn2": (78, 20), "done": (100, 0)}


class ExameInvalido(ValueError):
    """O arquivo não é um volume NIfTI 3D utilizável (vira HTTP 400 na API)."""


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

def scan_hippocampus_roi(img_data: np.ndarray,
                         model_roi,
                         progress_callback: Optional[Callable[..., None]] = None,
                         lote: int = LOTE_CNN1) -> Dict[str, Any]:
    """Varre todos os cortes coronais para localizar o hipocampo.

    Para cada corte, alimenta o modelo_cnn1 e coleta o score de ROI.
    Determina a faixa de slices onde o score >= 80% do máximo.

    Os cortes são avaliados em lotes de `lote` (o último é completado com zeros e
    descartado), com as mesmas entradas de antes; só a chamada ao modelo muda.

    Args:
        img_data: Array 3D do NIfTI (SafeImage).
        model_roi: Modelo Keras para detecção de ROI.
        progress_callback: callback(percentual_da_etapa, mensagem, **extra); o extra traz
            stage="cnn1", done, total, start e os scores do lote recém-calculado.
        lote: tamanho fixo do lote.

    Returns:
        Dict com:
          - roi_scores: List[float] — scores de todos os cortes
          - limite_inferior: int
          - limite_superior: int
          - slice_central: int
          - limiar, score_max, indice_max, fallback
    """
    n_coronal = img_data.shape[1]
    scores = np.zeros(n_coronal, dtype=np.float32)

    # Loop 1: Calcula scores para todos os coronais, em lotes
    for inicio in range(0, n_coronal, lote):
        fim = min(inicio + lote, n_coronal)
        entradas = np.concatenate([prepare_for_roi_model(img_data[:, ii, :]) for ii in range(inicio, fim)])
        if fim - inicio < lote:
            preenchimento = np.zeros((lote - (fim - inicio),) + entradas.shape[1:], dtype=entradas.dtype)
            entradas = np.concatenate([entradas, preenchimento])
        saida = np.asarray(model_roi.predict_on_batch(entradas))[: fim - inicio, 0]
        scores[inicio:fim] = saida

        if progress_callback:
            progress_callback(fim / n_coronal * 100, f"Varredura ROI: {fim}/{n_coronal} cortes",
                              stage="cnn1", done=fim, total=n_coronal, start=inicio,
                              scores=[round(float(s), 4) for s in scores[inicio:fim]])

    logger.debug("ROI scores calculados: min=%.4f, max=%.4f, mean=%.4f",
                 scores.min(), scores.max(), scores.mean())

    threshold = scores.max() * 0.80

    # Loop 2: Define limites da ROI usando o threshold
    candidates = np.where(scores >= threshold)[0]
    fallback = len(candidates) == 0
    if fallback:
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

    sl_central = int(round((lim_inf + lim_sup) / 2)) #valor sl_central de cada varredura

    logger.info("ROI detectada: [%d, %d] → slice_central=%d (threshold=%.4f)",
                lim_inf, lim_sup, sl_central, threshold)

    return {
        "roi_scores": scores.tolist(),
        "limite_inferior": int(lim_inf),
        "limite_superior": int(lim_sup),
        "slice_central": int(sl_central),
        "limiar": float(threshold),
        "score_max": float(scores.max()),
        "indice_max": int(np.argmax(scores)),
        "fallback": bool(fallback),
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


def classify_alzheimer(dataset: List[np.ndarray],
                       model_clf,
                       progress_callback: Optional[Callable[..., None]] = None,
                       ) -> Dict[str, Any]:
    """Classifica cada slice do dataset e agrega o resultado com estatísticas.

    Retorna:
      - predicoes_19_slices: List[float] — scores brutos dos 19 slices
      - probabilidade_media: float — média das probabilidades
      - desvio_padrao: float — desvio padrão
      - coeficiente_variacao: float — CV (desvio/média ou desvio/(1-média))
      - classificacao_cv: str — classificação qualitativa do CV
      - diagnostico: str — "ALZHEIMER" ou "NORMAL"
      - votos_alzheimer: int — fatias com probabilidade > 0,5 (informativo)

    Args:
        dataset: List[np.ndarray] — 19 slices uint8.
        model_clf: Modelo Keras classificador.
        progress_callback: Função de callback para atualizar o progresso da classificação.
    Returns:
        Dict com todos os parâmetros acima.
    """
    total = len(dataset)
    lote = np.stack([np.stack([u, u, u], axis=-1) for u in dataset]).astype(np.float32) / 255.0
    probs = np.asarray(model_clf.predict_on_batch(lote))  # shape (19, 2)
    predicoes = [float(p) for p in probs[:, 0]]  # coluna 0 — probabilidade de Alzheimer

    if progress_callback:
        progress_callback(100, f"Classificando slice: {total}/{total}", stage="cnn2", done=total, total=total)

    predicoes_array = np.array(predicoes)
    media = float(np.mean(predicoes_array))
    desvio = float(np.std(predicoes_array))
    votos = int((predicoes_array > 0.5).sum())

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
        "votos_alzheimer": votos,
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


# ═════════════════════════════════════════════════════════════════════════════
# BLOCO 3 — Leitura e validação do volume e dados para o painel
# ═════════════════════════════════════════════════════════════════════════════

def carregar_volume(nifti_path: str) -> "nib.Nifti1Image":
    """Carrega e valida o NIfTI. Levanta ExameInvalido se não for um volume 3D utilizável."""
    try:
        img = nib.load(nifti_path)
    except Exception as exc:  # arquivo corrompido ou que não é NIfTI
        raise ExameInvalido(f"Não foi possível ler o arquivo como NIfTI ({type(exc).__name__}).") from exc
    img.header.check_fix()
    if len(img.shape) == 4 and img.shape[3] == 1:
        img = nib.funcs.squeeze_image(img)
    if len(img.shape) != 3:
        raise ExameInvalido(f"O volume precisa ser 3D; o arquivo tem dimensões {list(img.shape)}.")
    if min(img.shape) < 32 or max(img.shape) > 1024:
        raise ExameInvalido(f"Dimensões fora do esperado para uma RM de crânio: {list(img.shape)}.")
    zooms = [float(z) for z in img.header.get_zooms()[:3]]
    if not all(np.isfinite(z) and 0 < z < 10 for z in zooms):
        raise ExameInvalido(f"Espaçamento de voxel inválido: {zooms}.")
    return img


def _fatias_contiguas(vol: np.ndarray) -> np.ndarray:
    """Mesmos valores e mesma forma, com cada fatia coronal vol[:, j, :] num bloco contíguo.

    Num array em ordem Fortran, as colunas de cada fatia podem ficar a 512 kB umas das outras,
    e copiar para a ordem C leva ~2 s por conflito de cache. Copiar na ordem em que os dados
    já estão na memória leva ~30 ms.
    """
    if vol.flags.c_contiguous:
        return vol
    if vol.flags.f_contiguous:
        return np.ascontiguousarray(vol.transpose(1, 2, 0)).transpose(2, 0, 1)
    return np.ascontiguousarray(vol)


def _forma_canonica(img) -> Tuple[List[int], List[float], List[List[float]]]:
    """Dimensões e espaçamento na orientação RAS sem reordenar os dados."""
    ornt = orientations.io_orientation(img.affine)
    dims, zooms = [0, 0, 0], [0.0, 0.0, 0.0]
    for eixo, (saida, _) in enumerate(ornt):
        dims[int(saida)] = int(img.shape[eixo])
        zooms[int(saida)] = float(img.header.get_zooms()[eixo])
    return dims, zooms, ornt.tolist()


def _webp(img_u8: np.ndarray) -> str:
    """PNG/WebP sem perda em data URL (cv2 usa WebP sem perda quando não se passa qualidade)."""
    ok, buf = cv2.imencode(".webp", img_u8)
    formato = "webp"
    if not ok:
        ok, buf = cv2.imencode(".png", img_u8)
        formato = "png"
    return f"data:image/{formato};base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _sprite(dataset: List[np.ndarray], colunas: int = 5) -> Tuple[str, Dict[str, int]]:
    """As 19 imagens da CNN2 numa grade, na mesma ordem das predições."""
    lado = dataset[0].shape[0]
    linhas = int(np.ceil(len(dataset) / colunas))
    grade = np.zeros((linhas * lado, colunas * lado), dtype=np.uint8)
    for k, img in enumerate(dataset):
        r, c = divmod(k, colunas)
        grade[r * lado:(r + 1) * lado, c * lado:(c + 1) * lado] = img
    return _webp(grade), {"colunas": colunas, "linhas": linhas, "tamanho": int(lado)}


def run_pipeline(nifti_path: str,
                 model_roi,
                 model_clf,
                 progress_callback: Optional[Callable[..., None]] = None,
                 visualizacao: bool = True,
                 explicacao: bool = False,
                 ) -> Dict[str, Any]:
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
        progress_callback: callback(percentual_da_etapa, mensagem, **extra) com stage em
            "load" | "cnn1" | "extract" | "cnn2" | "done".
        visualizacao: inclui as imagens que as CNNs receberam (sprite em data URL).
        explicacao: inclui os mapas Grad-CAM da CNN2.

    Returns:
        Dict contendo TODOS os dados necessários para frontend:
          - diagnostico
          - probabilidade_media
          - estatisticas_predicao (com predicoes_19_slices, desvio, CV, etc)
          - metadados_nifti (dimensões, spacing)
          - metadados_varredura_cnn1 (roi_scores, limites, slice_central)
          - metadados_extracao_cnn2 (inicio_cranio_y)
          - tempos_ms, imagens (opcional), gradcam (opcional)
    """
    cb = progress_callback or (lambda *a, **k: None)
    t_inicio = time.perf_counter()
    logger.info("Iniciando pipeline para: %s", nifti_path)
    cb(0, "Carregando NIfTI", stage="load")
    # ──────────────────────────────────────────────────────────────────────
    # Passo 1: Carrega NIfTI e extrai metadados
    # ──────────────────────────────────────────────────────────────────────
    img_ni = carregar_volume(nifti_path)
    img_data = _fatias_contiguas(nib.as_closest_canonical(img_ni).get_fdata())

    # Extrai dimensões e spacing (zooms)
    dimensoes = list(img_ni.shape)
    espacamento = list(img_ni.header.get_zooms()[:3])
    dims_canon, zooms_canon, ornt = _forma_canonica(img_ni)

    logger.info("Metadados NIfTI: dimensões=%s, spacing=%s",
                dimensoes, espacamento)
    cb(100, "Volume carregado", stage="load")
    t_carga = time.perf_counter()

    # ──────────────────────────────────────────────────────────────────────
    # Passo 2: Varre ROI (retorna scores e limites)
    # ──────────────────────────────────────────────────────────────────────
    roi_result = scan_hippocampus_roi(img_data, model_roi, progress_callback)
    t_cnn1 = time.perf_counter()

    # ──────────────────────────────────────────────────────────────────────
    # Passo 3: Monta dataset de 19 slices (retorna ini_cranio)
    # ──────────────────────────────────────────────────────────────────────
    central = roi_result["slice_central"]
    n_coronal = img_data.shape[1]
    indices = [max(0, min(central + o, n_coronal - 1)) for o in OFFSETS]
    dataset, ini_cranio, int_max = build_roi_dataset(img_data, central)
    margem = max(0, ini_cranio - 15)
    # Faixa de cada fatia que as redes recebem, em voxels do espaço canônico. As linhas da
    # fatia img_data[:, j, :] são o eixo x (esquerda → direita), redimensionado para 256.
    n_linhas = img_data.shape[0]
    recorte_cnn1 = {"eixo": "x", "inicio": 99 / 256 * n_linhas, "fim": float(n_linhas)}
    recorte_cnn2 = {"eixo": "x", "inicio": margem / 256 * n_linhas,
                    "fim": min(256, margem + 176) / 256 * n_linhas}
    cb(100, "19 fatias extraídas", stage="extract", total=len(dataset))

    logger.debug("Dataset construído: 19 slices, ini_cranio=%d", ini_cranio)
    t_extracao = time.perf_counter()

    # ──────────────────────────────────────────────────────────────────────
    # Passo 4: Classifica com estatísticas
    # ──────────────────────────────────────────────────────────────────────
    clf_result = classify_alzheimer(dataset, model_clf, progress_callback)
    t_cnn2 = time.perf_counter()

    imagens = None
    if visualizacao:
        sprite, grade = _sprite(dataset)
        entrada_cnn1 = prepare_for_roi_model(img_data[:, central, :])
        imagens = {
            "cnn2_sprite": sprite,
            "grade": grade,
            "cnn1_central": _webp((entrada_cnn1[0, :, :, 0] * 255).astype(np.uint8)),
            # para onde apontam as linhas e colunas de cada imagem (letras RAS/LPI)
            "orientacao_cnn2": {"linhas": "R", "colunas": "S"},
            "orientacao_cnn1": {"linhas": "L", "colunas": "I"},
        }
    t_visual = time.perf_counter()

    gradcam = None
    if explicacao:
        # classe prevista: coluna 0 = Alzheimer, coluna 1 = normal
        classe = 0 if clf_result["diagnostico"] == "ALZHEIMER" else 1
        gradcam = explicar_gradcam(dataset, model_clf, classe)
    t_fim = time.perf_counter()

    cb(100, "Concluído", stage="done")
    ms = lambda a, b: round((b - a) * 1000)  # noqa: E731

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
            "votos_alzheimer": clf_result["votos_alzheimer"],
        },
        "metadados_nifti": {
            "dimensoes": dimensoes,
            "espacamento": espacamento,
            "dimensoes_canonicas": dims_canon,
            "espacamento_canonico": zooms_canon,
            "orientacao_original": "".join(nib.aff2axcodes(img_ni.affine)),
            "ornt": ornt,
            "tipo_dado": str(img_ni.header.get_data_dtype()),
        },
        "metadados_varredura_cnn1": {
            "roi_scores": roi_result["roi_scores"],
            "limite_inferior": roi_result["limite_inferior"],
            "limite_superior": roi_result["limite_superior"],
            "slice_central": roi_result["slice_central"],
            "limiar": roi_result["limiar"],
            "score_max": roi_result["score_max"],
            "indice_max": roi_result["indice_max"],
            "fallback": roi_result["fallback"],
            "recorte": recorte_cnn1,
        },
        "metadados_extracao_cnn2": {
            "inicio_cranio_y": ini_cranio,
            "offsets": OFFSETS,
            "indices": indices,
            "recorte": recorte_cnn2,
            "margem": margem,
            "int_max": float(int_max),
        },
        "tempos_ms": {
            "carga": ms(t_inicio, t_carga), "cnn1": ms(t_carga, t_cnn1), "extracao": ms(t_cnn1, t_extracao),
            "cnn2": ms(t_extracao, t_cnn2), "visualizacao": ms(t_cnn2, t_visual),
            **({"gradcam": ms(t_visual, t_fim)} if explicacao else {}), "total": ms(t_inicio, t_fim),
        },
        "imagens": imagens,
        "gradcam": gradcam,
    }


# modelo de gradiente compilado uma vez por classificador carregado
_GRADCAM: Dict[int, Any] = {}


def explicar_gradcam(dataset: List[np.ndarray], model_clf, classe: int) -> Optional[Dict[str, Any]]:
    """Grad-CAM (Selvaraju et al., 2017) da CNN2 sobre as 19 entradas. Só lê o modelo.

    Usa a última Conv2D (ou GRADCAM_CAMADA) e o gradiente do logit da classe, isto é, a
    última Dense aplicada sem o softmax, que satura em predições confiantes. Os mapas
    passam por ReLU e são normalizados pelo máximo do exame, para as fatias serem
    comparáveis entre si. Na CNN-4 a última convolução trabalha numa grade de 5x5.

    Returns:
        {camada, classe_alvo, h, w, mapas: 19 listas h*w em 0–1}, ou None se a arquitetura
        não permitir (o resto da resposta não é afetado).
    """
    import tensorflow as tf

    try:
        if id(model_clf) not in _GRADCAM:
            nome = os.getenv("GRADCAM_CAMADA")
            convs = [l for l in model_clf.layers if l.__class__.__name__ == "Conv2D"]
            camada = model_clf.get_layer(nome) if nome else convs[-1]
            saida = model_clf.layers[-1]
            modelo = tf.keras.Model(model_clf.inputs, [camada.output, saida.input])

            @tf.function
            def mapas_de(lote, classe_alvo):
                with tf.GradientTape() as tape:
                    ativacoes, h = modelo(lote, training=False)
                    alvo = (tf.matmul(h, saida.kernel) + saida.bias)[:, classe_alvo]
                grads = tape.gradient(alvo, ativacoes)          # (19, h, w, canais)
                pesos = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
                return tf.nn.relu(tf.reduce_sum(pesos * ativacoes, axis=-1))

            _GRADCAM[id(model_clf)] = (camada.name, mapas_de)
        nome_camada, mapas_de = _GRADCAM[id(model_clf)]

        lote = np.stack([np.stack([u, u, u], axis=-1) for u in dataset]).astype(np.float32) / 255.0
        mapas = mapas_de(tf.constant(lote), tf.constant(classe)).numpy()
        mapas = mapas / (mapas.max() + 1e-8)
    except Exception as exc:  # arquitetura inesperada: segue sem explicação
        logger.warning("Grad-CAM indisponível: %s", exc)
        return None

    return {
        "camada": nome_camada,
        "classe_alvo": int(classe),
        "h": int(mapas.shape[1]),
        "w": int(mapas.shape[2]),
        "mapas": [[round(float(v), 4) for v in m.ravel()] for m in mapas],
    }