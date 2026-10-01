#!/usr/bin/env bash
# Publica o site em https://neuroia.ifce.edu.br: liga a porta 443 no nginx.
# Rodar só depois do teste local pelo túnel SSH (porta 8088), como pede a CTI.
#
# Uso, na VM:  sudo bash /opt/neuroia/back-end/deploy/publicar.sh

set -euo pipefail

DOMINIO="neuroia.ifce.edu.br"
DEPLOY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP="/root/backup-neuroia/$(date +%Y%m%d-%H%M%S)"

[[ $EUID -eq 0 ]] || { echo "Rode com sudo: sudo bash $0"; exit 1; }
[[ -f "/etc/letsencrypt/live/$DOMINIO/fullchain.pem" ]] || {
  echo "Ainda não há certificado para $DOMINIO. Rode o base.sh de novo (passo 8) antes."; exit 1; }

# Os containers precisam estar no ar: senão o site público mostraria erro 502
curl -fsS --max-time 5 http://127.0.0.1:8000/healthz | grep -q '"models_loaded":true' || {
  echo "A API não está pronta em 127.0.0.1:8000 (docker compose ps). Publicação cancelada."; exit 1; }
curl -fsS --max-time 5 -o /dev/null http://127.0.0.1:8080/ || {
  echo "O front não responde em 127.0.0.1:8080 (docker compose ps). Publicação cancelada."; exit 1; }

if [[ -e /etc/nginx/sites-available/neuroia-https ]]; then
  mkdir -p "$BACKUP/etc/nginx/sites-available"
  cp -a /etc/nginx/sites-available/neuroia-https "$BACKUP/etc/nginx/sites-available/"
fi
install -m 644 "$DEPLOY/nginx/neuroia-https.conf" /etc/nginx/sites-available/neuroia-https
ln -sf /etc/nginx/sites-available/neuroia-https /etc/nginx/sites-enabled/neuroia-https
nginx -t
systemctl reload nginx

echo
echo "Site no ar. Conferência pela própria VM:"
curl -s -o /dev/null -w "  https://$DOMINIO/         -> %{http_code}\n" --resolve "$DOMINIO:443:127.0.0.1" "https://$DOMINIO/"
curl -s -w "  https://$DOMINIO/healthz  -> %{http_code} " --resolve "$DOMINIO:443:127.0.0.1" "https://$DOMINIO/healthz"; echo
echo "Agora abra https://$DOMINIO fora da VPN (por exemplo no celular, no 4G)."
