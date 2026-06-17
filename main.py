"""
main.py
-------
API FastAPI de inferência médica para classificação de Alzheimer via MRI.

Endpoints:
  GET  /healthz                    — Liveness probe
  POST /v1/predict/alzheimer?       — Classificação com metadados completos

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

import os
import tempfile
import logging
from contextlib import asynccontextmanager
from typing import List

import tensorflow as tf
from tensorflow_addons.metrics import F1Score

import asyncio
from fastapi import FastAPI, File, HTTPException, UploadFile, status, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from functools import partial

# Importa o motor de inferência
from engine import run_pipeline
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


class MetadadosExtracaoCNN2(BaseModel):
    """Metadados de extração para a 2ª CNN (classificador)."""
    
    inicio_cranio_y: int = Field(
        ...,
        ge=0,
        description="Índice Y (linha) onde o crânio é detectado no slice central redimensionado (256 altura)",
        example=42
    )


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
    await manager.connect(job_id, websocket)
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


@app.post(
    "/v1/predict/alzheimer",
    response_model=PredictionResponse,
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
):
    """Classifica MRI e retorna TODOS os metadados para reconstrução frontend.

    **Retorna (Paridade Total):**
    - Diagnóstico + probabilidade média
    - Estatísticas das 19 predições (desvio, CV, classificação)
    - Predições brutas dos 19 slices
    - Dimensões e spacing do NIfTI
    - ROI scores completos da 1ª CNN
    - Parâmetro de extração da 2ª CNN (ini_cranio)

    **Raises:**
    - 400: Formato inválido
    - 503: Modelos não carregados
    - 500: Erro no pipeline
    """
    fname = file.filename or ""
    
    # Validação 1: Extensão
    if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
        logger.warning("Arquivo rejeitado: extensão inválida (%s)", fname)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formato inválido. Envie um arquivo .nii ou .nii.gz.",
        )

    # Validação 2: Modelos
    if "roi" not in MODELS or "clf" not in MODELS:
        logger.error("Modelos não carregados")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelos ainda não carregados. Tente novamente em instantes.",
        )

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
        

        # Executa pipeline (retorna TODOS os metadados)
        loop = asyncio.get_running_loop()
        def progress_callback(percent: float, message: str):
            if job_id:
                asyncio.run_coroutine_threadsafe(
                    manager.send_progress(job_id, {"progress": round(percent, 2), "status": message}),
                    loop,
                )

        pipeline_fn = partial(
            run_pipeline, tmp_path, MODELS["roi"], MODELS["clf"],
            progress_callback=progress_callback,
        )
        result = await loop.run_in_executor(None, pipeline_fn)

        logger.info("Pipeline concluído: %s (prob_media=%.4f, CV=%.4f)",
                    result["diagnostico"],
                    result["probabilidade_media"],
                    result["estatisticas_predicao"]["coeficiente_variacao"])

        # Monta response com PARIDADE TOTAL
        return PredictionResponse(
            status="success",
            diagnostico=result["diagnostico"],
            probabilidade_media=result["probabilidade_media"],
            estatisticas_predicao=EstatisticasPredicao(
                desvio_padrao=result["estatisticas_predicao"]["desvio_padrao"],
                coeficiente_variacao=result["estatisticas_predicao"]["coeficiente_variacao"],
                classificacao_cv=result["estatisticas_predicao"]["classificacao_cv"],
                predicoes_19_slices=result["estatisticas_predicao"]["predicoes_19_slices"],
            ),
            metadados_nifti=MetadadosNifti(
                dimensoes=result["metadados_nifti"]["dimensoes"],
                espacamento=result["metadados_nifti"]["espacamento"],
            ),
            metadados_varredura_cnn1=MetadadosVarreduraCNN1(
                roi_scores=result["metadados_varredura_cnn1"]["roi_scores"],
                limite_inferior=result["metadados_varredura_cnn1"]["limite_inferior"],
                limite_superior=result["metadados_varredura_cnn1"]["limite_superior"],
                slice_central=result["metadados_varredura_cnn1"]["slice_central"],
            ),
            metadados_extracao_cnn2=MetadadosExtracaoCNN2(
                inicio_cranio_y=result["metadados_extracao_cnn2"]["inicio_cranio_y"],
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Erro no pipeline: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno: {type(exc).__name__}: {exc}",
        )

    finally:
        # Limpeza garantida
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.debug("Arquivo temporário removido: %s", tmp_path)
            except Exception as e:
                logger.warning("Erro ao remover temporário %s: %s", tmp_path, e)

                #uvicorn main:app --reload