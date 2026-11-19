# Space do Hugging Face (deploy temporário, nov/2026, com o servidor do IFCE fora do ar).
# Usa a mesma imagem da VM, publicada pelo GitHub Actions no GHCR. Os modelos não vão na imagem:
# a API baixa do Google Drive no startup (MODEL_ROI_URL e MODEL_CLF_URL, segredos do Space).
FROM ghcr.io/site-alzheimer/back-end:main

# O Space roda o contêiner com o usuário 1000, que só escreve em /tmp e na própria pasta
ENV MODEL_ROI_PATH=/tmp/models/modelo_cnn3_L20_3A5.h5 \
    MODEL_CLF_PATH=/tmp/models/CNN-4-2023-1.h5 \
    PORT=8000
