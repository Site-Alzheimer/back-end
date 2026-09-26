"""
main.py
-------
API FastAPI de inferência médica para classificação de Alzheimer via MRI.

Endpoints:
  GET  /healthz                    — Liveness probe
  POST /v1/predict/alzheimer?       — Classificação com metadados completos
       ?job_id=&visualizacao=true&explicacao=false
  POST /v1/predict/alzheimer/amostra/{id} — Mesma resposta, com um exame de exemplo do servidor
  GET  /v1/amostras                 — Exames de exemplo (pasta AMOSTRAS_DIR)
  GET  /v1/amostras/{id}/arquivo    — NIfTI da amostra
  GET  /v1/amostras/{id}/miniatura  — Miniatura WebP da amostra
  WS   /ws/progress/{job_id}        — Progresso: {"stage":"ready"} ao conectar, depois
                                      load → cnn1 (lotes com scores) → extract → cnn2 → done

Uso:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Response JSON (Paridade Total com Script Base):
    {
        "status": "success",
        "diagnostico": "ALZHEIMER",
        "probabilidade_media": 94.25,
        "estatisticas_predicao": {
            "desvio_padrao": 0.045,
            "coeficiente_variacao": 0.047,
            "classificacao_cv": "High precision",
            "predicoes_19_slices": [0.92, 0.95, 0.94, ...]
        },
        "metadados_nifti": {
            "dimensoes": [256, 256, 176],
            "espacamento": [1.0, 1.0, 1.0]
        },
        "metadados_varredura_cnn1": {
            "roi_scores": [0.1, 0.2, ...],
            "limite_inferior": 143,
            "limite_superior": 147,
            "slice_central": 145
        },
        "metadados_extracao_cnn2": {
            "inicio_cranio_y": 42
        }
    }
"""

import json
import os
import re
import tempfile
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import tensorflow as tf
from tensorflow_addons.metrics import F1Score

import asyncio
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from functools import partial

# Importa o motor de inferência
from engine import LOTE_CNN1, PESOS_ETAPAS, ExameInvalido, explicar_gradcam, run_pipeline
from model_loader import ensure_models

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("alzheimer-api")

# ─────────────────────���───────────────────────────────────────────────────────
# Caminhos dos modelos
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ROI_PATH = os.getenv("MODEL_ROI_PATH", "models/modelo_cnn1_08_maio.h5")
MODEL_CLF_PATH = os.getenv("MODEL_CLF_PATH", "models/CNN-4-2023-1.h5")

# Dicionário global — preenchido no startup
MODELS: dict = {}

# Exames de exemplo servidos pela API (pasta local, ignorada pelo git)
AMOSTRAS_DIR = os.getenv("AMOSTRAS_DIR", "amostras")

# job_id vem do cliente e vira chave do WebSocket: aceita UUID e identificadores simples
JOB_ID_VALIDO = re.compile(r"^[A-Za-z0-9-]{8,64}$")


# ═════════════════════════════════════════════════════════════════════════════
# Pydantic Schemas (Paridade Total)
# ═════════════════════════════════════════════════════════════════════════════

class EstatisticasPredicao(BaseModel):
    """Estatísticas da segunda CNN (classificador)."""
    
    desvio_padrao: float = Field(
        ...,
        ge=0.0,
        description="Desvio padrão das 19 predições",
        example=0.045
    )
    
    coeficiente_variacao: float = Field(
        ...,
        ge=0.0,
        description="Coeficiente de variação (desvio/média)",
        example=0.047
    )
    
    classificacao_cv: str = Field(
        ...,
        description="Classificação qualitativa: 'High precision' | 'Acceptable precision' | 'Moderate variability' | 'High variability'",
        example="High precision"
    )
    
    predicoes_19_slices: List[float] = Field(
        ...,
        description="Predições bruas da CNN-4 para os 19 slices (coluna 0 — probabilidade de Alzheimer)",
        example=[0.92, 0.95, 0.94, 0.88, 0.91, 0.93, 0.96, 0.89, 0.90, 0.94, 0.92, 0.91, 0.88, 0.95, 0.93, 0.90, 0.92, 0.94, 0.91]
    )

    votos_alzheimer: Optional[int] = Field(None, description="Fatias com probabilidade > 0,5 (informativo)")


class MetadadosNifti(BaseModel):
    """Metadados do arquivo NIfTI."""
    
    dimensoes: List[int] = Field(
        ...,
        description="Dimensões da imagem 3D em voxels (I, J, K)",
        example=[256, 256, 176]
    )
    
    espacamento: List[float] = Field(
        ...,
        description="Espaçamento dos voxels em mm (dI, dJ, dK)",
        example=[1.0, 1.0, 1.0]
    )

    dimensoes_canonicas: Optional[List[int]] = Field(None, description="Dimensões na orientação RAS", example=[166, 256, 256])
    espacamento_canonico: Optional[List[float]] = Field(None, description="Espaçamento na orientação RAS (mm)")
    orientacao_original: Optional[str] = Field(None, description="Códigos de eixo do arquivo (nibabel)", example="IPL")
    ornt: Optional[List[List[float]]] = Field(None, description="nib.io_orientation do arquivo")
    tipo_dado: Optional[str] = Field(None, description="Tipo dos voxels no arquivo", example=">f4")


class Recorte(BaseModel):
    """Faixa do volume que uma CNN recebe, em voxels do espaço canônico (RAS)."""

    eixo: Literal["x", "z"] = Field(..., description="Eixo canônico da faixa (x: esquerda→direita, z: inferior→superior)")
    inicio: float
    fim: float


class MetadadosVarreduraCNN1(BaseModel):
    """Metadados da varredura de ROI (1ª CNN — modelo_cnn1)."""
    
    roi_scores: List[float] = Field(
        ...,
        description="Scores de detecção de ROI para cada slice coronal (lista completa)",
        example=[0.05, 0.08, 0.12, 0.25, 0.45, 0.65, 0.85, 0.92, 0.88, 0.75, 0.60, 0.45, 0.30, 0.15, 0.10]
    )
    
    limite_inferior: int = Field(
        ...,
        ge=0,
        description="Índice do primeiro slice onde ROI >= 80% do máximo",
        example=143
    )
    
    limite_superior: int = Field(
        ...,
        ge=0,
        description="Índice do último slice onde ROI >= 80% do máximo",
        example=147
    )
    
    slice_central: int = Field(
        ...,
        ge=0,
        description="Índice do slice central (hipocampo) — midpoint dos limites",
        example=145
    )

    limiar: Optional[float] = Field(None, description="80% do score máximo")
    score_max: Optional[float] = None
    indice_max: Optional[int] = None
    fallback: Optional[bool] = Field(None, description="True quando nenhum score atingiu o limiar (todos ≤ 0)")
    recorte: Optional[Recorte] = None


class MetadadosExtracaoCNN2(BaseModel):
    """Metadados de extração para a 2ª CNN (classificador)."""
    
    inicio_cranio_y: int = Field(
        ...,
        ge=0,
        description="Índice Y (linha) onde o crânio é detectado no slice central redimensionado (256 altura)",
        example=42
    )

    offsets: Optional[List[int]] = None
    indices: Optional[List[int]] = Field(None, description="Índices coronais reais das 19 fatias, na ordem das predições")
    recorte: Optional[Recorte] = None
    margem: Optional[int] = None
    int_max: Optional[float] = None


class TemposMs(BaseModel):
    """Tempo de cada etapa no servidor, em milissegundos."""

    carga: int
    cnn1: int
    extracao: int
    cnn2: int
    visualizacao: int
    gradcam: Optional[int] = None
    total: int


class OrientacaoImagem(BaseModel):
    linhas: str = Field(..., description="Letra RAS/LPI para onde cresce o índice das linhas")
    colunas: str = Field(..., description="Letra RAS/LPI para onde cresce o índice das colunas")


class GradeSprite(BaseModel):
    colunas: int
    linhas: int
    tamanho: int


class Imagens(BaseModel):
    """O que as CNNs receberam, sem perda (data URL)."""

    cnn2_sprite: str = Field(..., description="As 19 entradas da CNN2 numa grade, na ordem das predições")
    grade: GradeSprite
    cnn1_central: str = Field(..., description="Entrada da CNN1 na fatia central")
    orientacao_cnn2: OrientacaoImagem
    orientacao_cnn1: OrientacaoImagem


class GradCam(BaseModel):
    camada: str
    classe_alvo: int
    h: int
    w: int
    mapas: List[List[float]] = Field(..., description="19 mapas h×w (linha a linha), 0–1, normalizados por exame")


class PredictionResponse(BaseModel):
    """Response completo da classificação de Alzheimer — PARIDADE TOTAL."""
    
    status: str = Field(
        ...,
        description="Status da operação",
        example="success"
    )
    
    diagnostico: str = Field(
        ...,
        description="Diagnóstico: 'ALZHEIMER' ou 'NORMAL'",
        example="ALZHEIMER"
    )
    
    probabilidade_media: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade média das 19 predições (0.0 = NORMAL, 1.0 = ALZHEIMER)",
        example=0.9425
    )
    
    estatisticas_predicao: EstatisticasPredicao = Field(
        ...,
        description="Estatísticas das 19 predições"
    )
    
    metadados_nifti: MetadadosNifti = Field(
        ...,
        description="Metadados do arquivo NIfTI"
    )
    
    metadados_varredura_cnn1: MetadadosVarreduraCNN1 = Field(
        ...,
        description="Metadados da varredura de ROI (1ª CNN)"
    )
    
    metadados_extracao_cnn2: MetadadosExtracaoCNN2 = Field(
        ...,
        description="Metadados de extração (2ª CNN)"
    )

    tempos_ms: Optional[TemposMs] = None
    imagens: Optional[Imagens] = None
    gradcam: Optional[GradCam] = None


# ═════════════════════════════════════════════════════════════════════════════
# Lifespan Management
# ═════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega os dois modelos na memória global no startup.
    
    Os modelos ficam disponíveis durante toda a vida da aplicação.
    """
    ensure_models(MODEL_ROI_PATH, MODEL_CLF_PATH)

    logger.info("Carregando modelo de ROI: %s", MODEL_ROI_PATH)
    try:
        MODELS["roi"] = tf.keras.models.load_model(MODEL_ROI_PATH)
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
        MODELS["clf"] = clf
        logger.info("✓ Modelo classificador carregado com sucesso")
    except Exception as e:
        logger.error("✗ Erro ao carregar modelo classificador: %s", e)
        raise

    # Aquecimento: compila os grafos com as mesmas formas de lote das requisições,
    # para a primeira análise não pagar esse custo.
    MODELS["roi"].predict_on_batch(np.zeros((LOTE_CNN1, 176, 176, 3), dtype=np.float32))
    MODELS["clf"].predict_on_batch(np.zeros((19, 176, 176, 3), dtype=np.float32))
    explicar_gradcam([np.zeros((176, 176), dtype=np.uint8)] * 19, MODELS["clf"], 0)
    logger.info("✓ Modelos aquecidos (lotes de %d e 19, Grad-CAM)", LOTE_CNN1)

    logger.info("═" * 70)
    logger.info("SERVIDOR PRONTO | Modelos carregados | Aguardando requisições")
    logger.info("═" * 70)

    yield  # ← servidor ativo

    MODELS.clear()
    logger.info("Modelos liberados. Servidor encerrado.")


# ═════════════════════════════════════════════════════════════════════════════
# FastAPI App
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Alzheimer MRI Classifier",
    description="Classifica arquivos NIfTI de MRI cerebral — Retorna paridade total com pipeline base.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────
# Gerenciador de conexões WebSocket para progresso
# ──────────────────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: dict[str, WebSocket] = {}

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active[job_id] = websocket

    def disconnect(self, job_id: str):
        self.active.pop(job_id, None)

    async def send_progress(self, job_id: str, data: dict):
        ws = self.active.get(job_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(job_id)

manager = ConnectionManager()



# ═════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═════════════════════════════════════════════════════════════════════════════
@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    if not JOB_ID_VALIDO.match(job_id):
        await websocket.close(code=1008)
        return
    await manager.connect(job_id, websocket)
    # O cliente espera esta mensagem antes de enviar o arquivo, para não perder o início do progresso
    await websocket.send_json({"progress": 0, "status": "Conectado", "stage": "ready"})
    try:
        while True:
            await websocket.receive_text()  # só mantém viva a conexão
    except WebSocketDisconnect:
        manager.disconnect(job_id)


@app.get("/healthz", tags=["infra"])
async def health():
    """Liveness probe."""
    ok = "roi" in MODELS and "clf" in MODELS
    return {
        "status": "ok" if ok else "degraded",
        "models_loaded": ok
    }


def _validar_job_id(job_id: Optional[str]) -> None:
    if job_id is not None and not JOB_ID_VALIDO.match(job_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="job_id inválido (use 8 a 64 caracteres: letras, números e hífen).")


def _exigir_modelos() -> None:
    if "roi" not in MODELS or "clf" not in MODELS:
        logger.error("Modelos não carregados")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelos ainda não carregados. Tente novamente em instantes.",
        )


async def _analisar(caminho: str, job_id: Optional[str],
                    visualizacao: bool, explicacao: bool) -> PredictionResponse:
    """Roda o pipeline num thread (sem travar o event loop) e monta a resposta."""

    # Executa pipeline (retorna TODOS os metadados)
    loop = asyncio.get_running_loop()
    def progress_callback(percent: float, message: str, **extra):
        if not job_id:
            return
        # o engine informa o percentual da etapa; aqui vira progresso geral, sempre crescente
        etapa = extra.get("stage")
        base, peso = PESOS_ETAPAS.get(etapa, (0, 100))
        asyncio.run_coroutine_threadsafe(
            manager.send_progress(job_id, {"progress": round(base + peso * percent / 100, 2),
                                           "status": message, **extra}),
            loop,
        )

    pipeline_fn = partial(
        run_pipeline, caminho, MODELS["roi"], MODELS["clf"],
        progress_callback=progress_callback,
        visualizacao=visualizacao, explicacao=explicacao,
    )
    result = await loop.run_in_executor(None, pipeline_fn)

    logger.info("Pipeline concluído: %s (prob_media=%.4f, CV=%.4f, %d ms)",
                result["diagnostico"],
                result["probabilidade_media"],
                result["estatisticas_predicao"]["coeficiente_variacao"],
                result["tempos_ms"]["total"])

    # Monta response com PARIDADE TOTAL (os campos novos são opcionais)
    return PredictionResponse(status="success", **result)


def _erro_http(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, ExameInvalido):
        logger.warning("Exame rejeitado: %s", exc)
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    logger.exception("Erro no pipeline: %s", exc)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Erro interno: {type(exc).__name__}: {exc}",
    )


@app.post(
    "/v1/predict/alzheimer",
    response_model=PredictionResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
    tags=["inference"],
    summary="Classifica MRI cerebral — Paridade Total com Script Base",
)
async def predict(
    job_id: str | None = None,          
    file: UploadFile = File(
        ...,
        description="Arquivo NIfTI (.nii ou .nii.gz)"
    ),
    visualizacao: bool = Query(True, description="Inclui as imagens que as CNNs receberam"),
    explicacao: bool = Query(False, description="Inclui os mapas Grad-CAM da CNN2"),
):
    """Classifica MRI e retorna TODOS os metadados para reconstrução frontend.

    **Retorna (Paridade Total):**
    - Diagnóstico + probabilidade média
    - Estatísticas das 19 predições (desvio, CV, classificação)
    - Predições brutas dos 19 slices
    - Dimensões e spacing do NIfTI
    - ROI scores completos da 1ª CNN
    - Parâmetro de extração da 2ª CNN (ini_cranio)
    - Opcionais: tempos por etapa, recortes, imagens e Grad-CAM

    **Raises:**
    - 400: Formato inválido ou NIfTI que não é um volume 3D utilizável
    - 503: Modelos não carregados
    - 500: Erro no pipeline
    """
    fname = file.filename or ""
    _validar_job_id(job_id)
    
    # Validação 1: Extensão
    if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
        logger.warning("Arquivo rejeitado: extensão inválida (%s)", fname)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formato inválido. Envie um arquivo .nii ou .nii.gz.",
        )

    # Validação 2: Modelos
    _exigir_modelos()

    suffix = ".nii.gz" if fname.endswith(".nii.gz") else ".nii"
    tmp_path: str | None = None

    try:
        # Salva upload em temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        logger.info("Arquivo recebido: %s (%d bytes) → temp: %s",
                    fname, len(content), tmp_path)

        return await _analisar(tmp_path, job_id, visualizacao, explicacao)

    except Exception as exc:
        raise _erro_http(exc)

    finally:
        # Limpeza garantida
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.debug("Arquivo temporário removido: %s", tmp_path)
            except Exception as e:
                logger.warning("Erro ao remover temporário %s: %s", tmp_path, e)


# ─────────────────────────────────────────────────────────────────────────────
# Exames de exemplo (pasta AMOSTRAS_DIR com amostras.json)
# ─────────────────────────────────────────────────────────────────────────────

class Amostra(BaseModel):
    id: str
    rotulo: str
    fonte: str
    diagnostico_clinico: Optional[str] = None
    licenca: Optional[str] = None
    arquivo: str
    tamanho_bytes: int
    tem_miniatura: bool


def _amostras() -> List[Dict[str, Any]]:
    manifesto = os.path.join(AMOSTRAS_DIR, "amostras.json")
    if not os.path.exists(manifesto):
        return []
    with open(manifesto, encoding="utf-8") as f:
        itens = json.load(f).get("amostras", [])
    return [a for a in itens if os.path.isfile(_caminho_amostra(a["arquivo"]))]


def _caminho_amostra(relativo: str) -> str:
    """Resolve um caminho do manifesto sem deixar sair da pasta de amostras."""
    base = os.path.realpath(AMOSTRAS_DIR)
    alvo = os.path.realpath(os.path.join(base, relativo))
    if os.path.commonpath([base, alvo]) != base:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Amostra não encontrada.")
    return alvo


def _amostra(amostra_id: str) -> Dict[str, Any]:
    for a in _amostras():
        if a["id"] == amostra_id:
            return a
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Amostra não encontrada.")


@app.get("/v1/amostras", response_model=List[Amostra], response_model_exclude_none=True, tags=["amostras"])
async def listar_amostras():
    """Exames de exemplo disponíveis no servidor."""
    return [
        Amostra(id=a["id"], rotulo=a["rotulo"], fonte=a["fonte"],
                diagnostico_clinico=a.get("diagnostico_clinico"), licenca=a.get("licenca"),
                arquivo=os.path.basename(a["arquivo"]),
                tamanho_bytes=os.path.getsize(_caminho_amostra(a["arquivo"])),
                tem_miniatura=bool(a.get("miniatura")) and os.path.isfile(_caminho_amostra(a["miniatura"])))
        for a in _amostras()
    ]


@app.get("/v1/amostras/{amostra_id}/arquivo", tags=["amostras"])
async def arquivo_amostra(amostra_id: str):
    """O arquivo NIfTI da amostra (o navegador usa para mostrar a ressonância)."""
    a = _amostra(amostra_id)
    caminho = _caminho_amostra(a["arquivo"])
    tipo = "application/gzip" if caminho.endswith(".gz") else "application/octet-stream"
    return FileResponse(caminho, media_type=tipo, filename=os.path.basename(caminho))


@app.get("/v1/amostras/{amostra_id}/miniatura", tags=["amostras"])
async def miniatura_amostra(amostra_id: str):
    a = _amostra(amostra_id)
    if not a.get("miniatura"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Amostra sem miniatura.")
    return FileResponse(_caminho_amostra(a["miniatura"]), media_type="image/webp")


@app.post(
    "/v1/predict/alzheimer/amostra/{amostra_id}",
    response_model=PredictionResponse,
    response_model_exclude_none=True,
    tags=["inference"],
    summary="Classifica um exame de exemplo do servidor (sem upload)",
)
async def predict_amostra(
    amostra_id: str,
    job_id: str | None = None,
    visualizacao: bool = True,
    explicacao: bool = False,
):
    _validar_job_id(job_id)
    _exigir_modelos()
    a = _amostra(amostra_id)
    try:
        return await _analisar(_caminho_amostra(a["arquivo"]), job_id, visualizacao, explicacao)
    except Exception as exc:
        raise _erro_http(exc)
