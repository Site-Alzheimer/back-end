"""
fila.py
-------
Fila de análises com RabbitMQ (mensageria). Fica ativa quando RABBITMQ_URL está definida; sem ela,
a API roda as análises direto, como antes.

    Navegador ─POST─> API ─publica─> fila "neuroia.analises" ─1 por vez─> worker.py (as CNNs)
        ▲              │                                                      │
        └──── SSE ─────┴──<── exchange "neuroia.eventos" (progresso e fim) <──┘

- A mensagem leva só o pedido (job_id, caminho do exame e opções). O arquivo fica em FILA_DIR,
  pasta que a API e o worker enxergam (no Space, o mesmo contêiner; na VM, um volume do compose).
- O worker pega um exame por vez (prefetch 1) e só confirma a mensagem (ack) depois de salvar o
  resultado: se ele cair no meio, o RabbitMQ devolve o exame para a fila.
- A fila é do tipo quorum, a recomendada pelo RabbitMQ para mensagens que não podem se perder. Ela
  conta as entregas de cada mensagem (x-delivery-count): o exame que derrubou o worker
  MAX_TENTATIVAS vezes vira erro, e um que passe de x-delivery-limit vai para a fila de falhas.
- O resultado fica em FILA_DIR/resultados por RESULTADO_TTL_S (1 h) e sai em GET /v1/analises/{job_id}.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

import aio_pika

from engine import PESOS_ETAPAS

logger = logging.getLogger("alzheimer-api")

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "").strip()
ATIVA = bool(RABBITMQ_URL)
FILA_DIR = Path(os.getenv("FILA_DIR", "/tmp/neuroia-fila"))
RESULTADO_TTL_S = int(os.getenv("RESULTADO_TTL_S", "3600"))
# Exame enviado que nunca foi processado (fila apagada, por exemplo) sai depois disso
ENTRADA_TTL_S = 6 * 3600

FILA_ANALISES = "neuroia.analises"
FILA_FALHAS = "neuroia.analises.falhas"
EXCHANGE_EVENTOS = "neuroia.eventos"
# Um exame que derrubou o worker duas vezes vira erro, em vez de voltar para a fila para sempre
MAX_TENTATIVAS = 2

# A API e o worker declaram a fila com os mesmos argumentos (o RabbitMQ recusa se mudarem)
ARGUMENTOS_FILA = {
    "x-queue-type": "quorum",
    # rede de segurança, caso o worker caia antes até de gravar o erro: vai para a fila de falhas
    "x-delivery-limit": 5,
    "x-dead-letter-exchange": "",
    "x-dead-letter-routing-key": FILA_FALHAS,
}

Evento = Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────
# Arquivos compartilhados entre a API e o worker
# ──────────────────────────────────────────────────────────────────────

def _pasta(nome: str) -> Path:
    p = FILA_DIR / nome
    p.mkdir(parents=True, exist_ok=True)
    return p


def caminho_entrada(job_id: str, sufixo: str) -> Path:
    return _pasta("entradas") / f"{job_id}{sufixo}"


def _caminho_resultado(job_id: str) -> Path:
    return _pasta("resultados") / f"{job_id}.json"


def _json_padrao(o: Any) -> Any:
    # tipos do numpy no resultado do engine (np.float32, np.int64, arrays)
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} não é serializável")


def salvar_resultado(job_id: str, http_status: int, corpo: Dict[str, Any]) -> None:
    """Grava o resultado (ou o erro) de uma análise. Escreve num temporário e renomeia: a API nunca
    lê um arquivo pela metade."""
    destino = _caminho_resultado(job_id)
    tmp = destino.with_suffix(".tmp")
    tmp.write_text(json.dumps({"http_status": http_status, "corpo": corpo}, default=_json_padrao),
                   encoding="utf-8")
    os.replace(tmp, destino)


def ler_resultado(job_id: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(_caminho_resultado(job_id).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def limpar_antigos() -> int:
    """Apaga resultados vencidos e exames esquecidos. Devolve quantos arquivos saíram."""
    agora = time.time()
    apagados = 0
    for nome, ttl in (("resultados", RESULTADO_TTL_S), ("entradas", ENTRADA_TTL_S)):
        for arq in _pasta(nome).iterdir():
            try:
                if agora - arq.stat().st_mtime > ttl:
                    arq.unlink()
                    apagados += 1
            except FileNotFoundError:
                pass
    return apagados


def evento_progresso(percent: float, message: str, **extra) -> Evento:
    """O engine informa o percentual da etapa; aqui vira progresso geral, sempre crescente."""
    base, peso = PESOS_ETAPAS.get(extra.get("stage"), (0, 100))
    return {"progress": round(base + peso * percent / 100, 2), "status": message, **extra}


# ──────────────────────────────────────────────────────────────────────
# Conexão e topologia (filas e exchange), iguais para a API e o worker
# ──────────────────────────────────────────────────────────────────────

async def _porta_aberta(host: str, porta: int) -> bool:
    try:
        _, escrita = await asyncio.wait_for(asyncio.open_connection(host, porta), timeout=2)
    except (OSError, asyncio.TimeoutError):
        return False
    escrita.close()
    return True


async def conectar(quem: str, tentativas: int = 60):
    """Conexão robusta (reconecta sozinha). Na partida, espera o RabbitMQ aceitar conexões."""
    url = urlsplit(RABBITMQ_URL)
    host, porta = url.hostname or "localhost", url.port or 5672
    for _ in range(tentativas):
        if await _porta_aberta(host, porta):
            break
        logger.info("Esperando o RabbitMQ em %s:%s...", host, porta)
        await asyncio.sleep(2)
    return await aio_pika.connect_robust(RABBITMQ_URL, client_properties={"connection_name": quem})


async def declarar_topologia(canal):
    """Declara (idempotente) a fila de análises, a de falhas e o exchange de eventos."""
    await canal.declare_queue(FILA_FALHAS, durable=True)
    fila = await canal.declare_queue(FILA_ANALISES, durable=True, arguments=ARGUMENTOS_FILA)
    eventos = await canal.declare_exchange(EXCHANGE_EVENTOS, aio_pika.ExchangeType.FANOUT, durable=True)
    return fila, eventos


# ──────────────────────────────────────────────────────────────────────
# Lado da API: publica os pedidos e acompanha os eventos do worker
# ──────────────────────────────────────────────────────────────────────

class FilaAnalises:
    """Conexão da API com o RabbitMQ.

    Guarda em memória a ordem dos exames que esta API pôs na fila, para dizer a cada um a sua
    posição. Repassa os eventos do worker (progresso, início e fim) a `ao_evento`, que os entrega
    ao SSE de cada análise."""

    def __init__(self, ao_evento: Callable[[str, Evento], Awaitable[None]]):
        self.ao_evento = ao_evento
        self.pendentes: List[str] = []
        self.processando: Set[str] = set()
        self.esperas: Dict[str, asyncio.Future] = {}
        self.anunciadas: Dict[str, int] = {}  # última posição avisada a cada exame na fila
        self.conexao = None
        self.canal = None
        self.fila = None

    async def iniciar(self) -> None:
        self.conexao = await conectar("neuroia-api")
        # publisher confirms (padrão do aio-pika): o publish só volta depois de o RabbitMQ guardar
        self.canal = await self.conexao.channel()
        self.fila, eventos = await declarar_topologia(self.canal)
        # Cada API tem a sua fila de eventos, que some quando ela desconecta
        minha = await self.canal.declare_queue(
            f"{EXCHANGE_EVENTOS}.api-{uuid.uuid4().hex[:8]}", exclusive=True, auto_delete=True)
        await minha.bind(eventos)
        await minha.consume(self._ao_receber, no_ack=True)
        logger.info("Fila de análises conectada (RabbitMQ, fila %s)", FILA_ANALISES)

    async def fechar(self) -> None:
        if self.conexao is not None:
            await self.conexao.close()

    async def situacao(self) -> Dict[str, Any]:
        """Workers conectados e exames esperando, direto do RabbitMQ (para o /healthz)."""
        try:
            ok = await self.fila.declare()
            return {"conectada": True, "workers": ok.consumer_count, "aguardando": ok.message_count}
        except Exception as exc:
            logger.warning("Fila indisponível: %s", exc)
            return {"conectada": False, "workers": 0, "aguardando": 0}

    def conhecido(self, job_id: str) -> bool:
        return (job_id in self.processando or job_id in self.pendentes
                or ler_resultado(job_id) is not None)

    def estado(self, job_id: str) -> Optional[str]:
        if job_id in self.pendentes:
            return "na_fila"
        if job_id in self.processando:
            return "processando"
        return None

    def posicao(self, job_id: str) -> Optional[int]:
        """Lugar na fila contando os exames em análise: com um em análise, o primeiro que espera é o 2º."""
        if job_id not in self.pendentes:
            return None
        return len(self.processando) + self.pendentes.index(job_id) + 1

    def esperar(self, job_id: str) -> asyncio.Future:
        """Futuro que termina com o http_status quando o worker acabar este exame."""
        futuro = asyncio.get_running_loop().create_future()
        self.esperas[job_id] = futuro
        return futuro

    def parar_de_esperar(self, job_id: str) -> None:
        self.esperas.pop(job_id, None)

    async def enviar(self, pedido: Dict[str, Any]) -> int:
        """Põe o pedido na fila. Devolve a posição dele."""
        job_id = pedido["job_id"]
        # Antes de publicar: um worker livre pode começar (e avisar) antes de o publish voltar
        self.pendentes.append(job_id)
        try:
            await self.canal.default_exchange.publish(
                aio_pika.Message(
                    json.dumps(pedido).encode(),
                    content_type="application/json",
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,  # sobrevive a um reinício do RabbitMQ
                    message_id=job_id,
                ),
                routing_key=FILA_ANALISES,
            )
        except Exception:
            self.pendentes.remove(job_id)
            raise
        posicao = self.posicao(job_id) or 1
        await self._anunciar_posicoes()
        return posicao

    async def _ao_receber(self, mensagem) -> None:
        try:
            dados = json.loads(mensagem.body)
            job_id, evento = dados["job_id"], dados["evento"]
        except Exception:
            logger.warning("Evento da fila ignorado (formato inválido)")
            return
        try:
            etapa = evento.get("stage")
            if etapa in ("start", "result") and job_id in self.pendentes:
                self.pendentes.remove(job_id)
                self.anunciadas.pop(job_id, None)
            if etapa == "start":
                self.processando.add(job_id)
            elif etapa == "result":
                self.processando.discard(job_id)
                futuro = self.esperas.pop(job_id, None)
                if futuro is not None and not futuro.done():
                    futuro.set_result(evento.get("http_status", 500))
            await self.ao_evento(job_id, evento)
            if etapa in ("start", "result"):
                await self._anunciar_posicoes()
        except Exception:
            logger.exception("Erro ao tratar evento da fila (job %s)", job_id)

    async def _anunciar_posicoes(self) -> None:
        """Avisa cada exame na fila da sua posição, só quando ela muda."""
        for job_id in list(self.pendentes):
            posicao = self.posicao(job_id)
            if posicao is not None and self.anunciadas.get(job_id) != posicao:
                self.anunciadas[job_id] = posicao
                await self.ao_evento(job_id, {
                    "progress": 0, "stage": "queued", "posicao": posicao,
                    "status": f"Aguardando na fila (posição {posicao})",
                })


async def limpar_periodicamente(intervalo_s: int = 600) -> None:
    while True:
        try:
            n = await asyncio.to_thread(limpar_antigos)
            if n:
                logger.info("Fila: %d arquivo(s) vencido(s) apagado(s)", n)
        except Exception as exc:
            logger.warning("Fila: falha ao limpar arquivos vencidos: %s", exc)
        await asyncio.sleep(intervalo_s)
