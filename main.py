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
  GET  /v1/amostras/{id}/miniatura  — Miniatura WebP da amostra
  GET  /v1/progress/{job_id}        — Progresso por SSE: {"stage":"ready"} ao conectar, depois
                                      load → cnn1 (lotes com scores) → extract → cnn2 → done
  WS   /ws/progress/{job_id}        — O mesmo progresso por WebSocket (versões antigas do front)

Fila de análises (RabbitMQ, só com RABBITMQ_URL; ver fila.py e worker.py):
  POST /v1/analises?job_id=         — Põe um exame enviado na fila: 202 com job_id e posição
  POST /v1/analises/amostra/{id}    — O mesmo com um exame de exemplo
  GET  /v1/analises/{job_id}        — 200 com o resultado (guardado por 1 h), 202 enquanto espera
  Com a fila ativa, as rotas /v1/predict também passam por ela (mesma resposta de antes), e a API
  não carrega as CNNs: quem roda as análises é o worker.

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
import shutil
import tempfile
import time
import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Literal, Optional

import asyncio
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from functools import partial

# Motor de inferência e fila. O TensorFlow só entra com as CNNs (modelos.py), e só sem a fila
import fila
from engine import ExameInvalido, run_pipeline
from model_loader import ensure_amostras

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("alzheimer-api")

# ─────────────────────────────────────────────────────────────────────────────
# Modelos (caminhos em modelos.py) e fila
# ─────────────────────────────────────────────────────────────────────────────

# Dicionário global — preenchido no startup, quando a API mesma roda as análises (sem fila)
MODELS: dict = {}

# Conexão com o RabbitMQ — só com RABBITMQ_URL (fila.ATIVA)
FILA: Optional[fila.FilaAnalises] = None

# Exames de exemplo servidos pela API (pasta local, ignorada pelo git)
AMOSTRAS_DIR = os.getenv("AMOSTRAS_DIR", "amostras")

# job_id vem do cliente e vira chave do progresso: aceita UUID e identificadores simples
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
    orientacao_analise: Optional[str] = Field(
        None, description="Orientação em que as redes recebem o volume; os índices das fatias são nela "
                          "(IPL: 0 = anterior, como no notebook do Dr. Raimundo)", example="IPL")
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
        description="Índice do primeiro slice onde ROI > 80% do máximo (índice IPL, 0 = anterior)",
        example=143
    )
    
    limite_superior: int = Field(
        ...,
        ge=0,
        description="Índice do último slice onde ROI > 80% do máximo, depois da remoção de vales",
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
    fallback: Optional[bool] = Field(None, description="Sempre False: sem fatia acima do limiar, a API responde 400")
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
    """No startup: exames de exemplo e, conforme o modo, a fila ou as duas CNNs.

    - Com RABBITMQ_URL: conecta na fila. As CNNs ficam com o worker (worker.py).
    - Sem: carrega as CNNs aqui, e a API roda as análises direto, como antes da fila.
    """
    global FILA
    ensure_amostras(AMOSTRAS_DIR)

    limpeza = None
    if fila.ATIVA:
        FILA = fila.FilaAnalises(manager.send_progress)
        await FILA.iniciar()
        limpeza = asyncio.create_task(fila.limpar_periodicamente())
        pronto = "Fila de análises conectada (as CNNs rodam no worker)"
    else:
        # Import aqui: com a fila, o processo da API nem carrega o TensorFlow
        from modelos import carregar_modelos
        MODELS.update(carregar_modelos())
        pronto = "Modelos carregados"

    logger.info("═" * 70)
    logger.info("SERVIDOR PRONTO | %s | Aguardando requisições", pronto)
    logger.info("═" * 70)

    yield  # ← servidor ativo

    if limpeza is not None:
        limpeza.cancel()
    if FILA is not None:
        await FILA.fechar()
        FILA = None
    MODELS.clear()
    logger.info("Servidor encerrado.")


# ═════════════════════════════════════════════════════════════════════════════
# FastAPI App
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Alzheimer MRI Classifier",
    description="Classifica arquivos NIfTI de MRI cerebral — Retorna paridade total com pipeline base.",
    version="2.0.0",
    lifespan=lifespan,
)

# Origens que podem chamar a API de outro site (lista separada por vírgula). Em produção o front
# fica no mesmo domínio e a lista é só ele; sem a variável, qualquer origem (uso local e scripts).
# Sem cookies nem login, então sem credenciais.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────
# Progresso das análises: SSE (GET /v1/progress/{job_id}) e o WebSocket antigo
# ──────────────────────────────────────────────────────────────────────

# Comentário SSE a cada 15 s sem progresso: o nginx fecha a conexão depois de 60 s parada
SSE_PING_S = 15
# Tempo máximo de um stream (a análise mais lenta, na VM, leva ~40 s)
SSE_MAX_S = 600


class ProgressHub:
    """Entrega o progresso de cada job_id a quem acompanha a análise: um stream SSE ou, nas
    versões antigas do front, um WebSocket."""

    def __init__(self):
        self.filas: dict[str, asyncio.Queue] = {}
        self.sockets: dict[str, WebSocket] = {}

    def abrir_fila(self, job_id: str) -> asyncio.Queue:
        fila: asyncio.Queue = asyncio.Queue()
        self.filas[job_id] = fila  # se o EventSource reconectar, a fila nova substitui a antiga
        return fila

    def fechar_fila(self, job_id: str, fila: asyncio.Queue):
        if self.filas.get(job_id) is fila:
            del self.filas[job_id]

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        self.sockets[job_id] = websocket

    def disconnect(self, job_id: str):
        self.sockets.pop(job_id, None)

    async def send_progress(self, job_id: str, data: dict):
        fila = self.filas.get(job_id)
        if fila is not None:
            fila.put_nowait(data)
        ws = self.sockets.get(job_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(job_id)

manager = ProgressHub()


def _evento_sse(dados: dict) -> str:
    return f"data: {json.dumps(dados, separators=(',', ':'), ensure_ascii=False)}\n\n"



# ═════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/v1/progress/{job_id}", tags=["inference"], summary="Progresso da análise (Server-Sent Events)")
async def progresso_sse(job_id: str, request: Request):
    """Stream `text/event-stream` com o progresso da análise de mesmo `job_id`.

    O primeiro evento é `{"stage": "ready"}`: o cliente só envia o exame depois dele. Os seguintes
    têm o formato das mensagens do WebSocket (load → cnn1 → extract → cnn2 → done). O stream fica
    aberto até o cliente fechar (ou `SSE_MAX_S`)."""
    _validar_job_id(job_id)

    async def eventos():
        fila = manager.abrir_fila(job_id)
        relogio = asyncio.get_running_loop().time
        fim = relogio() + SSE_MAX_S
        try:
            # O cliente espera este evento antes de enviar o arquivo, para não perder o início do progresso
            yield _evento_sse({"progress": 0, "status": "Conectado", "stage": "ready"})
            while (resta := fim - relogio()) > 0:
                try:
                    dados = await asyncio.wait_for(fila.get(), timeout=min(SSE_PING_S, resta))
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        break
                    yield ": ping\n\n"
                    continue
                yield _evento_sse(dados)
        finally:
            manager.fechar_fila(job_id, fila)

    return StreamingResponse(eventos(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # o nginx repassa cada evento na hora, sem juntar num buffer
    })


# Versões antigas do front (antes do SSE) ainda abertas nos navegadores
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
    """Liveness probe.

    Com a fila, `models_loaded` diz se há worker conectado (o worker só começa a consumir depois de
    carregar as CNNs), e `fila` traz os workers e os exames esperando."""
    if FILA is not None:
        situacao = await FILA.situacao()
        ok = situacao["workers"] > 0
        return {"status": "ok" if ok else "degraded", "models_loaded": ok, "fila": situacao}
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
        asyncio.run_coroutine_threadsafe(
            manager.send_progress(job_id, fila.evento_progresso(percent, message, **extra)),
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


def _sufixo_nifti(nome: Optional[str]) -> str:
    """.nii ou .nii.gz; outro formato vira 400."""
    nome = nome or ""
    if nome.endswith(".nii.gz"):
        return ".nii.gz"
    if nome.endswith(".nii"):
        return ".nii"
    # O nome do arquivo pode identificar o paciente: o log guarda só a extensão
    logger.warning("Arquivo rejeitado: extensão inválida (%s)", os.path.splitext(nome)[1] or "sem extensão")
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Formato inválido. Envie um arquivo .nii ou .nii.gz.",
    )


def _gravar_upload(origem: BinaryIO, destino: Path) -> int:
    """Copia o upload em blocos de 1 MB, sem carregar o arquivo inteiro na memória."""
    with open(destino, "wb") as f:
        shutil.copyfileobj(origem, f, length=1024 * 1024)
    return destino.stat().st_size


# ─────────────────────────────────────────────────────────────────────────────
# Fila de análises (RabbitMQ): helpers das rotas
# ─────────────────────────────────────────────────────────────────────────────

def _exigir_fila() -> fila.FilaAnalises:
    if FILA is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="A fila de análises não está ativa neste servidor. Use /v1/predict/alzheimer.",
        )
    return FILA


def _novo_job(job_id: Optional[str]) -> str:
    """O job_id do cliente (que abre o SSE antes do envio) ou um novo. Repetido vira 409."""
    _validar_job_id(job_id)
    job = job_id or str(uuid.uuid4())
    if _exigir_fila().conhecido(job):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Já existe uma análise com este job_id.")
    return job


async def _enfileirar(job_id: str, arquivo: Path, apagar: bool,
                      visualizacao: bool, explicacao: bool) -> None:
    pedido = {
        "job_id": job_id,
        "arquivo": str(arquivo),
        "apagar_arquivo": apagar,  # exame enviado: sai depois da análise; amostra: fica
        "visualizacao": visualizacao,
        "explicacao": explicacao,
        "enviado_em": time.time(),
    }
    try:
        await _exigir_fila().enviar(pedido)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Falha ao publicar na fila: %s", exc)
        if apagar:
            arquivo.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="A fila de análises está indisponível. Tente de novo em instantes.")


def _resposta_salva(salvo: Dict[str, Any]) -> PredictionResponse:
    """Resultado gravado pelo worker: a resposta, ou o mesmo erro HTTP que a análise direta daria."""
    corpo = salvo["corpo"]
    if salvo["http_status"] != status.HTTP_200_OK:
        raise HTTPException(status_code=salvo["http_status"], detail=corpo.get("detail", "Erro na análise."))
    return PredictionResponse(**corpo)


async def _pela_fila(job_id: str, arquivo: Path, apagar: bool,
                     visualizacao: bool, explicacao: bool) -> PredictionResponse:
    """Rotas /v1/predict com a fila ativa: põe o exame na fila e espera o worker terminar."""
    fila_api = _exigir_fila()
    fim = fila_api.esperar(job_id)
    try:
        await _enfileirar(job_id, arquivo, apagar, visualizacao, explicacao)
        await fim
    finally:
        fila_api.parar_de_esperar(job_id)
    salvo = await asyncio.to_thread(fila.ler_resultado, job_id)
    if salvo is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="O resultado da análise não foi encontrado.")
    return _resposta_salva(salvo)


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
    _validar_job_id(job_id)
    suffix = _sufixo_nifti(file.filename)

    # Com a fila: o worker analisa, e esta rota espera o resultado (mesma resposta de antes)
    if FILA is not None:
        job = _novo_job(job_id)
        destino = fila.caminho_entrada(job, suffix)
        tamanho = await asyncio.to_thread(_gravar_upload, file.file, destino)
        logger.info("Arquivo recebido: %s, %d bytes (fila)", suffix, tamanho)
        return await _pela_fila(job, destino, True, visualizacao, explicacao)

    _exigir_modelos()
    tmp_path: str | None = None

    try:
        # Salva upload em temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        logger.info("Arquivo recebido: %s, %d bytes", suffix, len(content))

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
    idade: Optional[int] = Field(None, description="Idade do participante no exame (anos)")
    sexo: Optional[Literal["F", "M"]] = Field(None, description="Sexo do participante")
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
                idade=a.get("idade"), sexo=a.get("sexo"),
                arquivo=os.path.basename(a["arquivo"]),
                tamanho_bytes=os.path.getsize(_caminho_amostra(a["arquivo"])),
                tem_miniatura=bool(a.get("miniatura")) and os.path.isfile(_caminho_amostra(a["miniatura"])))
        for a in _amostras()
    ]


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
    a = _amostra(amostra_id)
    if FILA is not None:
        return await _pela_fila(_novo_job(job_id), Path(_caminho_amostra(a["arquivo"])), False,
                                visualizacao, explicacao)
    _exigir_modelos()
    try:
        return await _analisar(_caminho_amostra(a["arquivo"]), job_id, visualizacao, explicacao)
    except Exception as exc:
        raise _erro_http(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Fila de análises (RabbitMQ): envio assíncrono e resultado guardado
# ─────────────────────────────────────────────────────────────────────────────

class AnaliseNaFila(BaseModel):
    """Resposta 202: o exame está na fila ou em análise."""

    job_id: str
    estado: Literal["na_fila", "processando"]
    posicao: Optional[int] = Field(None, description="Lugar na fila, contando os exames em análise")
    resultado: str = Field(..., description="Onde buscar o resultado (GET)")


def _aceita(job_id: str) -> JSONResponse:
    """202 Accepted com o endereço do resultado (Location) e o intervalo sugerido para consultar."""
    url = f"/v1/analises/{job_id}"
    fila_api = _exigir_fila()
    corpo = AnaliseNaFila(job_id=job_id, estado=fila_api.estado(job_id) or "processando",
                          posicao=fila_api.posicao(job_id), resultado=url)
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=corpo.model_dump(exclude_none=True),
                        headers={"Location": url, "Retry-After": "3"})


_DESCRICAO_FILA = (
    "Responde 202 na hora, com a posição na fila. O progresso sai no SSE de mesmo `job_id` "
    "(`queued` com a posição enquanto espera, `start` quando o worker pega o exame e `result` no "
    "fim), e o resultado em `GET /v1/analises/{job_id}`. Só com a fila ativa (senão, 503)."
)


@app.post("/v1/analises", status_code=status.HTTP_202_ACCEPTED, response_model=AnaliseNaFila,
          response_model_exclude_none=True, tags=["fila"], summary="Põe um exame enviado na fila",
          description=_DESCRICAO_FILA)
async def enfileirar_analise(
    file: UploadFile = File(..., description="Arquivo NIfTI (.nii ou .nii.gz)"),
    job_id: Optional[str] = Query(None, description="Escolhido pelo cliente para abrir o SSE antes do envio"),
    visualizacao: bool = Query(True, description="Inclui as imagens que as CNNs receberam"),
    explicacao: bool = Query(False, description="Inclui os mapas Grad-CAM da CNN2"),
):
    suffix = _sufixo_nifti(file.filename)
    job = _novo_job(job_id)
    destino = fila.caminho_entrada(job, suffix)
    tamanho = await asyncio.to_thread(_gravar_upload, file.file, destino)
    logger.info("Arquivo recebido para a fila: %s, %d bytes", suffix, tamanho)
    await _enfileirar(job, destino, True, visualizacao, explicacao)
    return _aceita(job)


@app.post("/v1/analises/amostra/{amostra_id}", status_code=status.HTTP_202_ACCEPTED,
          response_model=AnaliseNaFila, response_model_exclude_none=True, tags=["fila"],
          summary="Põe um exame de exemplo na fila", description=_DESCRICAO_FILA)
async def enfileirar_amostra(
    amostra_id: str,
    job_id: Optional[str] = Query(None, description="Escolhido pelo cliente para abrir o SSE antes do envio"),
    visualizacao: bool = Query(True, description="Inclui as imagens que as CNNs receberam"),
    explicacao: bool = Query(False, description="Inclui os mapas Grad-CAM da CNN2"),
):
    a = _amostra(amostra_id)
    job = _novo_job(job_id)
    await _enfileirar(job, Path(_caminho_amostra(a["arquivo"])), False, visualizacao, explicacao)
    return _aceita(job)


@app.get(
    "/v1/analises/{job_id}",
    response_model=PredictionResponse,
    response_model_exclude_none=True,
    responses={
        202: {"model": AnaliseNaFila, "description": "Ainda na fila ou em análise"},
        404: {"description": "Análise desconhecida ou resultado vencido"},
    },
    tags=["fila"],
    summary="Resultado de uma análise da fila",
)
async def resultado_analise(job_id: str):
    """200 com a mesma resposta de /v1/predict, guardada por `RESULTADO_TTL_S` (1 h). Se a análise
    falhou, o mesmo erro HTTP que a análise direta daria (400 para exame inválido)."""
    _validar_job_id(job_id)
    fila_api = _exigir_fila()
    salvo = await asyncio.to_thread(fila.ler_resultado, job_id)
    if salvo is not None:
        return _resposta_salva(salvo)
    if fila_api.estado(job_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Análise não encontrada. O resultado fica guardado por {fila.RESULTADO_TTL_S // 60} "
                   "minutos; envie o exame de novo.",
        )
    return _aceita(job_id)
