#!/usr/bin/env bash
# RabbitMQ, worker e API num contêiner só, para hospedagens que rodam um contêiner por aplicação
# (o Space do Hugging Face). Na VM, cada um tem o seu contêiner no deploy/compose.yml; aqui este
# script faz o papel do compose.
#
# - O RabbitMQ só escuta dentro do contêiner (127.0.0.1) e roda sem root: dados, logs e o cookie do
#   Erlang ficam em RABBITMQ_BASE_DIR.
# - Se qualquer um dos três terminar, o script encerra os outros e sai com erro, e a hospedagem
#   liga o contêiner de novo.
# - Ao receber SIGTERM: para a API e o worker (que termina o exame em análise) e só depois o RabbitMQ.
set -uo pipefail

export RABBITMQ_URL="${RABBITMQ_URL:-amqp://guest:guest@127.0.0.1:5672/}"
base="${RABBITMQ_BASE_DIR:-/tmp/rabbitmq}"
mkdir -p "$base"

# O rabbitmq-server do Debian (/usr/sbin) exige root ou o usuário rabbitmq; o de /usr/lib/rabbitmq/bin
# roda com o usuário do contêiner. RABBITMQ_LOGS=- manda o log para a saída padrão.
env HOME="$base" \
    RABBITMQ_MNESIA_BASE="$base/mnesia" \
    RABBITMQ_LOG_BASE="$base/log" \
    RABBITMQ_LOGS=- \
    RABBITMQ_NODENAME=rabbit@localhost \
    RABBITMQ_NODE_IP_ADDRESS=127.0.0.1 \
    ERL_EPMD_ADDRESS=127.0.0.1 \
    /usr/lib/rabbitmq/bin/rabbitmq-server &
rabbit=$!

# A API e o worker esperam o RabbitMQ subir (fila.conectar tenta de novo a cada 2 s)
python worker.py &
worker=$!

uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}" &
api=$!

encerrar() {
  kill -TERM "$api" "$worker" 2>/dev/null
  wait "$api" "$worker" 2>/dev/null
  kill -TERM "$rabbit" 2>/dev/null
  wait "$rabbit" 2>/dev/null
}

trap 'echo "Desligando: API e worker, depois o RabbitMQ"; encerrar; exit 0' TERM INT

wait -n "$rabbit" "$worker" "$api"
echo "Um dos processos (RabbitMQ, worker ou API) terminou; encerrando o contêiner" >&2
encerrar
exit 1
