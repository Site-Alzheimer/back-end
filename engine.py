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

O pré-processamento, a janela e a decisão seguem o notebook do Dr. Raimundo
(Pipeline_Alzheimer.ipynb, 28/03/2026, células 6, 13, 16, 18 e 20). O volume é lido na orientação
IPL, a dos exames do ADNI em que o notebook roda, e os índices das fatias são os do notebook
(0 = anterior). scripts/paridade.py compara com respostas geradas rodando o próprio notebook.

Além disso, devolve dados extras para o painel (tempos por etapa, recortes, as imagens que as
redes receberam e Grad-CAM), sem alterar o que as redes calculam.
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

# Offsets dos 19 slices ao redor do corte central do hipocampo (`desl` da célula 13).
# No índice IPL, +9 é a fatia mais posterior.
OFFSETS = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9]

# Orientação em que as redes recebem o volume: a dos exames do ADNI, em que o notebook roda
ORIENTACAO_ANALISE = ("I", "P", "L")

# Tamanho fixo dos lotes da CNN1: o último lote é completado para não recompilar o grafo
LOTE_CNN1 = 32

# Pesos de cada etapa no progresso geral (0–100), para o progresso nunca voltar
PESOS_ETAPAS = {"load": (0, 5), "cnn1": (5, 70), "extract": (75, 3), "cnn2": (78, 20), "done": (100, 0)}


class ExameInvalido(ValueError):
    """O arquivo não é um volume NIfTI 3D utilizável (vira HTTP 400 na API)."""


# ═════════════════════════════════════════════════════════════════════════════
# BLOCO 1 — Pré-processamento das fatias (notebook, células 6 e 13)
# ═════════════════════════════════════════════════════════════════════════════

def volume_ipl(img) -> np.ndarray:
    """Dados do volume na orientação IPL, como o notebook os recebe de `img_ni.get_fdata()`.

    Nos exames do ADNI (já em IPL) é exatamente o `get_fdata()` do notebook. Arquivos em outra
    orientação são reordenados para IPL, para as fatias chegarem às redes em pé, como no treino.
    """
    dados = img.get_fdata()
    transf = orientations.ornt_transform(orientations.io_orientation(img.affine),
                                         orientations.axcodes2ornt(ORIENTACAO_ANALISE))
    if np.array_equal(transf, [[0, 1], [1, 1], [2, 1]]):
        return dados
    return orientations.apply_orientation(dados, transf)


def prepare_for_roi_model(raw_slice: np.ndarray) -> np.ndarray:
    """Pré-processa uma fatia coronal para a CNN1, como a célula 6 do notebook.

    A fatia chega em pé (linhas de cima para baixo). Passos:
      1. gira 180° e, se tiver menos de 176 colunas, redimensiona para 256 × 176;
      2. divide pelo máximo da fatia e converte para 8 bits;
      3. guarda as linhas 65 a 219 (k = 40): tira o pescoço e o topo do crânio;
      4. redimensiona para 176 × 176 e desvira;
      5. repete nos 3 canais e divide por 255.

    Returns:
        Tensor (1, 176, 176, 3) float32 pronto para model.predict().
    """
    img_salva = cv2.rotate(raw_slice, cv2.ROTATE_180)
    if raw_slice.shape[1] < 176:
        img_salva = cv2.resize(img_salva, (176, 256))
    img2 = img_salva
    if img2.max() != 0:
        img2 = img2 / img2.max()
    img2 = 255 * img2
    img2 = img2.astype(np.uint8)
    k = 40  # início do crânio fixo
    img_corta2 = img2[100 - k + 5:255 - k + 5, :]
    img_salva = cv2.resize(img_corta2, (176, 176))
    img_salva = cv2.rotate(img_salva, cv2.ROTATE_180)

    rgb = np.stack([img_salva, img_salva, img_salva], axis=-1).astype(np.float32) / 255.0
    return rgb.reshape(1, *rgb.shape)  # (1, 176, 176, 3)


def find_skull_start(img8: np.ndarray) -> Optional[int]:
    """Início do crânio (célula 13): de cima para baixo, a primeira linha com um bloco de 5 px
    cuja soma passa de 250. Devolve None se nenhuma linha passar (o notebook pararia com erro)."""
    for k in range(256):  # varre vertical
        j = 0
        while j < (176 / 5):
            if np.sum(img8[k, 5 * j:5 * j + 5]) > 250:
                return k
            j = j + 1
    return None


def _recorte_cnn2(img: np.ndarray, ini_cranio: int) -> np.ndarray:
    """176 linhas a partir de 15 linhas acima do crânio (célula 13)."""
    if ini_cranio >= 15:
        return img[ini_cranio - 15:ini_cranio - 15 + 176, :]
    return img[0:176, :]


def prepare_for_clf_model(raw_slice: np.ndarray, ini_cranio: int, int_max: float) -> np.ndarray:
    """Uma das 19 entradas da CNN2, como no laço da célula 13: imagem uint8 de 176 × 176.

    Como no notebook, o brilho não é limitado: o que passa de int_max dá a volta na conversão
    para 8 bits (256 vira 0). Foi assim que o modelo foi treinado e testado.
    """
    new_image2 = cv2.resize(raw_slice, (176, 256))
    img_corta1 = _recorte_cnn2(new_image2, ini_cranio)
    slice_x = img_corta1 / int_max
    slice_x = 255 * slice_x
    with np.errstate(invalid="ignore", over="ignore"):  # só silencia o aviso da volta; o valor é o mesmo
        img8 = slice_x.astype(np.uint8)
    if img8.shape != (176, 176):
        # crânio abaixo da linha 95: o recorte tem menos de 176 linhas e o notebook pararia com erro
        img8 = cv2.resize(img8, (176, 176))
    return img8


# ═════════════════════════════════════════════════════════════════════════════
# BLOCO 2 — Pipeline de inferência (com captura completa de metadados)
# ═════════════════════════════════════════════════════════════════════════════

def scan_hippocampus_roi(img_data: np.ndarray,
                         model_roi,
                         progress_callback: Optional[Callable[..., None]] = None,
                         lote: int = LOTE_CNN1) -> Dict[str, Any]:
    """Varre todos os cortes coronais para localizar o hipocampo (célula 6 do notebook).

    Para cada corte, alimenta a CNN1 e guarda o score. A janela são as fatias com score acima
    de 80% do máximo, sem vales; a fatia central é o meio da janela.

    Os cortes são avaliados em lotes de `lote` (o último é completado com zeros e descartado):
    as entradas são as mesmas do notebook, só a chamada ao modelo muda.

    Args:
        img_data: volume IPL (índice 1 = coronal, 0 = anterior).
        model_roi: Modelo Keras para detecção de ROI.
        progress_callback: callback(percentual_da_etapa, mensagem, **extra); o extra traz
            stage="cnn1", done, total, start e os scores do lote recém-calculado.
        lote: tamanho fixo do lote.

    Returns:
        Dict com roi_scores, limite_inferior, limite_superior, slice_central, limiar,
        score_max, indice_max e fallback (sempre False, mantido por compatibilidade).

    Raises:
        ExameInvalido: nenhuma fatia passou do limiar (no notebook, `lim_inf` não existiria).
    """
    n_coronal = img_data.shape[1]
    slice_pred = np.zeros(n_coronal)  # float64, como no notebook

    # Loop 1: score de todos os cortes coronais, em lotes
    for inicio in range(0, n_coronal, lote):
        fim = min(inicio + lote, n_coronal)
        entradas = np.concatenate([prepare_for_roi_model(img_data[:, ii, :]) for ii in range(inicio, fim)])
        if fim - inicio < lote:
            preenchimento = np.zeros((lote - (fim - inicio),) + entradas.shape[1:], dtype=entradas.dtype)
            entradas = np.concatenate([entradas, preenchimento])
        saida = np.asarray(model_roi.predict_on_batch(entradas))[: fim - inicio, 0]
        slice_pred[inicio:fim] = saida

        if progress_callback:
            progress_callback(fim / n_coronal * 100, f"Varredura ROI: {fim}/{n_coronal} cortes",
                              stage="cnn1", done=fim, total=n_coronal, start=inicio,
                              scores=[round(float(s), 4) for s in slice_pred[inicio:fim]])

    logger.debug("ROI scores calculados: min=%.4f, max=%.4f, mean=%.4f",
                 slice_pred.min(), slice_pred.max(), slice_pred.mean())

    # Limiar = 80% do valor máximo
    limiar_aj = slice_pred.max() * 0.8

    # Limites: a primeira e a última fatia acima do limiar
    acima = np.flatnonzero(slice_pred > limiar_aj)
    if len(acima) == 0:
        raise ExameInvalido("A CNN1 não encontrou a região do hipocampo neste exame "
                            "(nenhuma fatia passou do limiar).")
    lim_inf, lim_sup = int(acima[0]), int(acima[-1])

    # Remove os vales abaixo do limiar, encolhendo a janela pelo lado mais perto do vale
    intervalo = slice_pred[lim_inf:lim_sup]
    if len(intervalo) != 0:
        while min(intervalo) < limiar_aj:
            minimo = lim_inf + np.where(intervalo == intervalo.min())[0][0]
            if minimo - lim_inf < lim_sup - minimo:
                lim_inf = minimo + 1
            else:
                lim_sup = minimo - 1
            intervalo = slice_pred[lim_inf:lim_sup]
            if len(intervalo) == 0:
                break

    sl_central = round((lim_inf + lim_sup) / 2)

    logger.info("ROI detectada: [%d, %d] → slice_central=%d (limiar=%.4f)",
                lim_inf, lim_sup, sl_central, limiar_aj)

    return {
        "roi_scores": slice_pred.tolist(),
        "limite_inferior": int(lim_inf),
        "limite_superior": int(lim_sup),
        "slice_central": int(sl_central),
        "limiar": float(limiar_aj),
        "score_max": float(slice_pred.max()),
        "indice_max": int(np.argmax(slice_pred)),
        "fallback": False,
    }


def build_roi_dataset(img_data: np.ndarray, sl_central: int) -> Tuple[List, int, float]:
    """Monta as 19 fatias em torno do corte central (célula 13 do notebook).

    Da fatia central saem dois valores que valem para as 19:
      - ini_cranio: linha onde o crânio começa (blocos de 5 px);
      - int_max: 90% da intensidade máxima do recorte central.

    Args:
        img_data: volume IPL.
        sl_central: índice da fatia central.

    Returns:
        Tupla (dataset, ini_cranio, int_max), com dataset = 19 imagens uint8 (176×176)
        na ordem de OFFSETS.
    """
    n_coronal = img_data.shape[1]

    # Início do crânio, na fatia central redimensionada para 256 × 176 e em 8 bits
    new_image2 = cv2.resize(img_data[:, sl_central, :], (176, 256))
    img_corta2 = new_image2
    slice_x = img_corta2 / img_corta2.max()
    slice_x = 255 * slice_x
    img8 = slice_x.astype(np.uint8)
    ini_cranio = find_skull_start(img8)
    if ini_cranio is None:  # fatia praticamente vazia; o notebook pararia com erro
        ini_cranio = 0

    # Máxima intensidade do recorte central
    img_corta1 = _recorte_cnn2(img_corta2, ini_cranio)
    int_max = img_corta1.max() * 0.9

    logger.debug("Parâmetros extraídos: ini_cranio=%d, int_max=%.2f", ini_cranio, int_max)

    # As 19 fatias. O índice fica dentro do volume (o notebook não limita, mas com a fatia
    # central longe das bordas isso nunca muda nada)
    dataset = []
    for offset in OFFSETS:
        idx = max(0, min(sl_central + offset, n_coronal - 1))
        dataset.append(prepare_for_clf_model(img_data[:, idx, :], ini_cranio, int_max))

    return dataset, int(ini_cranio), float(int_max)


def classify_alzheimer(dataset: List[np.ndarray],
                       model_clf,
                       progress_callback: Optional[Callable[..., None]] = None,
                       ) -> Dict[str, Any]:
    """Classifica as 19 fatias e agrega o resultado (células 16, 18 e 20 do notebook).

    Retorna:
      - predicoes_19_slices: List[float] — P(DA) de cada fatia (coluna 0)
      - probabilidade_media: float — média das 19
      - desvio_padrao: float — desvio padrão
      - coeficiente_variacao: float — CV (desvio/média ou desvio/(1-média))
      - classificacao_cv: str — classificação qualitativa do CV
      - diagnostico: str — "ALZHEIMER" (média > 0,5) ou "NORMAL"
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
    # Volume IPL, como no notebook: img_data[:, ii, :] é a fatia coronal ii em pé (0 = anterior)
    img_data = _fatias_contiguas(volume_ipl(img_ni))

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
    margem = ini_cranio - 15 if ini_cranio >= 15 else 0
    # Faixa de cada fatia que as redes recebem, em voxels do eixo z canônico (inferior → superior).
    # As linhas da fatia IPL vão de cima para baixo: a linha r da imagem de altura h é z = h − r.
    n_linhas = img_data.shape[0]
    altura_cnn1 = 256 if img_data.shape[2] < 176 else n_linhas  # célula 6: só redimensiona abaixo de 176
    # a CNN1 corta as linhas 65–219 da fatia girada 180°, que já estão no sentido de z
    recorte_cnn1 = {"eixo": "z", "inicio": 65 / altura_cnn1 * n_linhas, "fim": 220 / altura_cnn1 * n_linhas}
    recorte_cnn2 = {"eixo": "z", "inicio": (256 - min(256, margem + 176)) / 256 * n_linhas,
                    "fim": (256 - margem) / 256 * n_linhas}
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
            # para onde apontam as linhas e colunas de cada imagem (letras RAS/LPI): em pé, como no IPL
            "orientacao_cnn2": {"linhas": "I", "colunas": "L"},
            "orientacao_cnn1": {"linhas": "I", "colunas": "L"},
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
            # orientação em que as redes recebem o volume; os índices das fatias são nela (0 = anterior)
            "orientacao_analise": "".join(ORIENTACAO_ANALISE),
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