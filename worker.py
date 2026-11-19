"""
worker.py
---------
Worker da fila de análises (fila.py): carrega as duas CNNs uma vez e processa os exames da fila do
RabbitMQ, um por vez. O progresso e o fim de cada análise vão para o exchange de eventos, e a API
repassa ao navegador.

Mesma imagem da API, outro comando:
    python worker.py

Garantias:
- prefetch 1: o RabbitMQ só entrega o próximo exame depois do ack do atual;
- ack só depois de gravar o resultado: se o worker cair, o exame volta para a fila;
- consumidor idempotente: um exame que já tem resultado não é analisado de novo;
- SIGTERM: para de pegar exames, termina o atual e só então sai.
"""

import asyncio
import json
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import aio_pika
from aio_pika.abc import AbstractExchange, AbstractIncomingMessage

import fila
from engine import ExameInvalido, run_pipeline
from modelos import carregar_modelos

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | worker | %(message)s")
logger = logging.getLogger("alzheimer-worker")

# Sinal de vida para o healthcheck do contêiner: atualizado enquanto a conexão com o RabbitMQ está de pé
ARQUIVO_VIVO = Path(os.getenv("WORKER_VIVO", "/tmp/neuroia-worker-vivo"))
VIVO_INTERVALO_S = 15


class Worker:
    def __init__(self, modelos: dict, eventos: AbstractExchange):
        self.modelos = modelos
        self.eventos = eventos
        # As CNNs rodam numa thread só, fora do event loop: os heartbeats do RabbitMQ continuam
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cnn")
        # Enquanto um exame está em análise; o desligamento espera por ele
        self.ocupado = asyncio.Lock()

    async def publicar(self, job_id: str, evento: Dict[str, Any]) -> None:
        corpo = json.dumps({"job_id": job_id, "evento": evento}, default=fila._json_padrao)
        await self.eventos.publish(
            aio_pika.Message(corpo.encode(), content_type="application/json"), routing_key="")

    async def processar(self, mensagem: AbstractIncomingMessage) -> None:
        try:
            pedido = json.loads(mensagem.body)
            job_id = pedido["job_id"]
            arquivo = pedido["arquivo"]
        except Exception:
            logger.error("Mensagem inválida: vai para a fila de falhas")
            await mensagem.reject(requeue=False)
            return

        async with self.ocupado:
            try:
                await self._processar(mensagem, pedido, job_id, arquivo)
            except Exception:
                # Falha fora da análise (disco, conexão): devolve para a fila e conta mais uma entrega.
                # Sem isso, com prefetch 1, o worker ficaria parado esperando um ack que nunca vem.
                logger.exception("Job %s: falha ao processar; a mensagem volta para a fila", job_id)
                await asyncio.sleep(5)
                await mensagem.nack(requeue=True)

    async def _processar(self, mensagem: AbstractIncomingMessage, pedido: dict, job_id: str,
                         arquivo: str) -> None:
        # Já analisado (o worker caiu entre gravar o resultado e o ack): só avisa de novo
        salvo = await asyncio.to_thread(fila.ler_resultado, job_id)
        if salvo is not None:
            logger.info("Job %s já tinha resultado; mensagem repetida confirmada", job_id)
            await self._concluir(mensagem, job_id, salvo["http_status"])
            return

        entregas = int((mensagem.headers or {}).get("x-delivery-count") or 0)
        await self.publicar(job_id, {"progress": 0, "stage": "start", "status": "Análise iniciada"})
        if entregas >= fila.MAX_TENTATIVAS:
            logger.error("Job %s interrompeu o worker %d vezes: cancelado", job_id, entregas)
            codigo, corpo = 500, {"detail": "A análise deste exame foi interrompida duas vezes "
                                            "e foi cancelada. Tente de novo mais tarde."}
        else:
            codigo, corpo = await self._analisar(job_id, arquivo, pedido)

        await asyncio.to_thread(fila.salvar_resultado, job_id, codigo, corpo)
        # O exame enviado só sai depois do resultado gravado (se o worker cair antes, ele volta)
        if pedido.get("apagar_arquivo"):
            Path(arquivo).unlink(missing_ok=True)
        await self._concluir(mensagem, job_id, codigo)

    async def _concluir(self, mensagem: AbstractIncomingMessage, job_id: str, codigo: int) -> None:
        await self.publicar(job_id, {"progress": 100, "stage": "result", "http_status": codigo,
                                     "status": "Resultado pronto" if codigo == 200 else "Análise falhou"})
        await mensagem.ack()

    async def _analisar(self, job_id: str, arquivo: str, pedido: dict) -> Tuple[int, Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        enviados = []

        def progresso(percent: float, message: str, **extra) -> None:
            # chamado na thread das CNNs: o publish acontece no event loop
            enviados.append(asyncio.run_coroutine_threadsafe(
                self.publicar(job_id, fila.evento_progresso(percent, message, **extra)), loop))

        pipeline = partial(
            run_pipeline, arquivo, self.modelos["roi"], self.modelos["clf"],
            progress_callback=progresso,
            visualizacao=pedido.get("visualizacao", True),
            explicacao=pedido.get("explicacao", False),
        )
        try:
            resultado = await loop.run_in_executor(self.executor, pipeline)
        except ExameInvalido as exc:
            logger.warning("Job %s: exame rejeitado: %s", job_id, exc)
            return 400, {"detail": str(exc)}
        except Exception as exc:
            logger.exception("Job %s: erro no pipeline", job_id)
            return 500, {"detail": f"Erro interno: {type(exc).__name__}: {exc}"}
        finally:
            # o evento de fim só sai depois de todos os de progresso
            await asyncio.gather(*(asyncio.wrap_future(f) for f in enviados), return_exceptions=True)

        logger.info("Job %s concluído: %s (prob_media=%.4f, CV=%.4f, %d ms)", job_id,
                    resultado["diagnostico"], resultado["probabilidade_media"],
                    resultado["estatisticas_predicao"]["coeficiente_variacao"],
                    resultado["tempos_ms"]["total"])
        return 200, {"status": "success", **resultado}


async def sinal_de_vida(conexao) -> None:
    while True:
        if not conexao.is_closed:
            ARQUIVO_VIVO.touch()
        await asyncio.sleep(VIVO_INTERVALO_S)


async def rodar(modelos: dict) -> None:
    conexao = await fila.conectar("neuroia-worker")
    async with conexao:
        canal = await conexao.channel()
        await canal.set_qos(prefetch_count=1)
        analises, eventos = await fila.declarar_topologia(canal)
        worker = Worker(modelos, eventos)
        etiqueta = await analises.consume(worker.processar)

        parar = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sinal in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sinal, parar.set)
        vivo = asyncio.create_task(sinal_de_vida(conexao))

        logger.info("═" * 70)
        logger.info("WORKER PRONTO | Modelos carregados | Aguardando exames na fila %s", fila.FILA_ANALISES)
        logger.info("═" * 70)

        await parar.wait()
        logger.info("Desligando: não pega exames novos e termina o que estiver em análise")
        await analises.cancel(etiqueta)
        async with worker.ocupado:
            pass
        vivo.cancel()
        ARQUIVO_VIVO.unlink(missing_ok=True)
        worker.executor.shutdown(wait=True)
    logger.info("Worker encerrado")


def main() -> None:
    if not fila.ATIVA:
        raise SystemExit("Defina RABBITMQ_URL (por exemplo amqp://usuario:senha@rabbitmq:5672/)")
    inicio = time.perf_counter()
    modelos = carregar_modelos()
    logger.info("Modelos prontos em %.1f s", time.perf_counter() - inicio)
    asyncio.run(rodar(modelos))


if __name__ == "__main__":
    main()
