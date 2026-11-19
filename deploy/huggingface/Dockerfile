# Space do Hugging Face (deploy temporário, nov/2026, com o servidor do IFCE fora do ar).
# Usa a mesma imagem da VM, publicada pelo GitHub Actions no GHCR. Os modelos não vão na imagem:
# o worker baixa do Google Drive no startup (MODEL_ROI_URL e MODEL_CLF_URL, segredos do Space).
ARG BASE=ghcr.io/site-alzheimer/back-end:main
FROM ${BASE}

# Fila de análises (RabbitMQ). O Space roda um contêiner só: o RabbitMQ, o worker e a API sobem
# juntos pelo scripts/conteiner-unico.sh. Na VM, cada um tem o seu contêiner (deploy/compose.yml).
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends rabbitmq-server \
    && rm -rf /var/lib/apt/lists/*

# O Space roda o contêiner com o usuário 1000, que só escreve em /tmp e na própria pasta.
# Os exames de exemplo vêm de AMOSTRAS_URL (outro segredo): um Space público mostra os arquivos dele
ENV MODEL_ROI_PATH=/tmp/models/modelo_cnn3_L20_3A5.h5 \
    MODEL_CLF_PATH=/tmp/models/CNN-4-Site.h5 \
    AMOSTRAS_DIR=/tmp/amostras \
    RABBITMQ_URL=amqp://guest:guest@127.0.0.1:5672/ \
    RABBITMQ_BASE_DIR=/tmp/rabbitmq \
    FILA_DIR=/tmp/neuroia-fila \
    PORT=8000

CMD ["bash", "/app/scripts/conteiner-unico.sh"]
