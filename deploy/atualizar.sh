#!/usr/bin/env bash
# Deploy automático do NeuroIA: confere se o GitHub Actions publicou imagem nova (tag main) no GHCR
# e troca só o container que mudou. Se a versão nova não ficar saudável, volta para a anterior e
# não tenta de novo aquela mesma versão. Roda pelo timer do systemd a cada 5 minutos.
#
# Só imagens: mudanças em deploy/ (compose, nginx) continuam manuais.
# Manual, na VM:  sudo /usr/local/sbin/neuroia-atualizar
# Histórico:      journalctl -u neuroia-atualizar

set -euo pipefail

cd "${NEUROIA_DIR:-/opt/neuroia}"

# Nunca duas atualizações ao mesmo tempo
exec 9> "${NEUROIA_LOCK:-/run/neuroia-atualizar.lock}"
flock -n 9 || exit 0

id_da() { docker image inspect -f '{{.Id}}' "$1" 2>/dev/null || true; }

for servico in api web; do
  imagem=$(docker compose config --images "$servico")
  repo=${imagem%:*}
  atual=$(id_da "$imagem")

  docker pull -q "$imagem" >/dev/null
  nova=$(id_da "$imagem")
  [[ "$nova" == "$atual" ]] && continue

  # Esta versão já falhou antes: mantém a que está no ar
  if [[ -f ".ruim-$servico" && "$(cat ".ruim-$servico")" == "$nova" ]]; then
    docker tag "$nova" "$repo:ruim"
    [[ -n "$atual" ]] && docker tag "$atual" "$imagem"
    continue
  fi

  echo "$servico: versão nova ${nova:7:12} (antes ${atual:7:12})"
  [[ -n "$atual" ]] && docker tag "$atual" "$repo:anterior"

  if docker compose up -d --wait --wait-timeout "${NEUROIA_ESPERA:-300}" "$servico"; then
    echo "$servico: no ar e saudável"
    rm -f ".ruim-$servico"
  else
    echo "$servico: a versão nova não ficou saudável"
    echo "$nova" > ".ruim-$servico"
    docker tag "$nova" "$repo:ruim"
    if [[ -n "$atual" ]]; then
      echo "$servico: voltando para a anterior"
      docker tag "$atual" "$imagem"
      docker compose up -d --wait --wait-timeout "${NEUROIA_ESPERA:-300}" "$servico" || echo "$servico: ATENÇÃO, a anterior também não subiu"
    fi
  fi
done

# Imagens sem nome (versões substituídas); main, anterior e ruim ficam
docker image prune -f >/dev/null
