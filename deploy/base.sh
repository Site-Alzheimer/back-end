#!/usr/bin/env bash
# Base da VM do NeuroIA (IFCE Campus Limoeiro do Norte): Docker, nginx, certbot e o certificado
# HTTPS. A configuração do nginx vem deste repositório. Pode rodar de novo sem estragar nada.
#
# Uso, na VM (o repositório clonado em /opt/neuroia/back-end):
#   sudo bash /opt/neuroia/back-end/deploy/base.sh
#
# Regras da CTI respeitadas: não mexe no root, no usuário administrador, no sshd, no sudoers
# nem na política de senhas; não liga nem desliga o ufw; guarda cópia de todo arquivo que altera.

set -euo pipefail

DOMINIO="neuroia.ifce.edu.br"
USUARIO="${SUDO_USER:-}"
DEPLOY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP="/root/backup-neuroia/$(date +%Y%m%d-%H%M%S)"
LOG="/var/log/neuroia-base.log"

[[ $EUID -eq 0 ]] || { echo "Rode com sudo: sudo bash $0"; exit 1; }
[[ -n "$USUARIO" && "$USUARIO" != root ]] || { echo "Rode com sudo a partir do seu usuário, não como root."; exit 1; }
[[ -f "$DEPLOY/nginx/neuroia.conf" ]] || { echo "Não achei $DEPLOY/nginx/neuroia.conf: rode a partir do clone do repositório."; exit 1; }

exec > >(tee -a "$LOG") 2>&1

passo() { echo; echo "==> $*"; }

# Guarda uma cópia antes de alterar um arquivo que já existe (pedido da CTI no banner da VM)
copia() {
  [[ -e "$1" ]] || return 0
  mkdir -p "$BACKUP$(dirname "$1")"
  cp -a "$1" "$BACKUP$1"
  echo "    cópia de $1 em $BACKUP$1"
}

# Sem perguntas na tela; arquivos de configuração que já existem ficam como estão;
# espera o unattended-upgrades soltar o apt, se estiver rodando
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a
APT=(apt-get -y -o DPkg::Lock::Timeout=600 -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold)

passo "1/9 Atualizando o sistema"
"${APT[@]}" update
"${APT[@]}" upgrade

passo "2/9 Docker Engine e Compose, do repositório oficial do Docker"
"${APT[@]}" install ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
CODINOME=$(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
copia /etc/apt/sources.list.d/docker.sources
cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $CODINOME
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF
"${APT[@]}" update
"${APT[@]}" install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

passo "3/9 Docker: logs limitados, containers vivos durante atualizações e faixa de rede própria"
# As redes do Docker ficam numa faixa pequena (172.31.240.0/21 e 172.31.255.0/24) para não
# esconder redes internas do IFCE que usem as faixas padrão (172.17 a 172.30)
copia /etc/docker/daemon.json
mkdir -p /etc/docker
cat > /etc/docker/daemon.json <<'EOF'
{
  "log-driver": "json-file",
  "log-opts": { "max-size": "10m", "max-file": "3" },
  "live-restore": true,
  "bip": "172.31.255.1/24",
  "default-address-pools": [{ "base": "172.31.240.0/21", "size": 24 }]
}
EOF
systemctl enable docker
systemctl restart docker
usermod -aG docker "$USUARIO"
echo "    $USUARIO está no grupo docker (vale no próximo login)"

passo "4/9 nginx e certbot"
"${APT[@]}" install nginx certbot

passo "5/9 nginx: porta 80 (Let's Encrypt e redirecionamento) e teste local em 127.0.0.1:8088"
install -d -m 755 /var/www/letsencrypt
# A versão do nginx fica escondida dentro dos nossos blocos server (server_tokens off). O Ubuntu já
# traz "server_tokens build;" no nginx.conf; repetir a diretiva em conf.d daria "duplicate".
rm -f /etc/nginx/conf.d/neuroia-seguranca.conf
for f in neuroia-proxy.conf neuroia-rotas.conf; do
  copia "/etc/nginx/snippets/$f"
  install -m 644 "$DEPLOY/nginx/$f" "/etc/nginx/snippets/$f"
done
copia /etc/nginx/sites-available/neuroia
install -m 644 "$DEPLOY/nginx/neuroia.conf" /etc/nginx/sites-available/neuroia
ln -sf /etc/nginx/sites-available/neuroia /etc/nginx/sites-enabled/neuroia
if [[ -L /etc/nginx/sites-enabled/default ]]; then
  rm /etc/nginx/sites-enabled/default
  echo "    site padrão desativado (o original continua em sites-available/default)"
fi
nginx -t
systemctl enable nginx
systemctl reload nginx

passo "6/9 Firewall (ufw): garante 80 e 443; a regra do SSH não é tocada"
if ufw status | grep -q "Status: active"; then
  ufw allow 80/tcp comment "nginx: redireciona para HTTPS e Let's Encrypt"
  ufw allow 443/tcp comment "nginx: HTTPS"
fi
ufw status verbose | sed 's/^/    /'

passo "7/9 Pastas da aplicação em /opt/neuroia"
install -d -o "$USUARIO" -g "$USUARIO" -m 750 /opt/neuroia /opt/neuroia/models /opt/neuroia/amostras /opt/neuroia/dados-teste
sudo -u "$USUARIO" ln -sfn back-end/deploy/compose.yml /opt/neuroia/compose.yml
ls -la /opt/neuroia | sed 's/^/    /'

passo "8/9 Deploy automático: instalado, mas desligado (liga na última etapa)"
install -m 755 "$DEPLOY/atualizar.sh" /usr/local/sbin/neuroia-atualizar
for f in neuroia-atualizar.service neuroia-atualizar.timer; do
  copia "/etc/systemd/system/$f"
  install -m 644 "$DEPLOY/systemd/$f" "/etc/systemd/system/$f"
done
systemctl daemon-reload
echo "    timer: $(systemctl is-enabled neuroia-atualizar.timer 2>/dev/null || true)"

passo "9/9 Certificado HTTPS (Let's Encrypt), renovado sozinho pelo certbot"
if [[ -d "/etc/letsencrypt/live/$DOMINIO" ]]; then
  echo "    já existe: $(openssl x509 -enddate -noout -in "/etc/letsencrypt/live/$DOMINIO/fullchain.pem")"
elif certbot certonly --webroot -w /var/www/letsencrypt -d "$DOMINIO" --non-interactive --agree-tos \
    --register-unsafely-without-email --deploy-hook "systemctl reload nginx"; then
  echo "    certificado emitido"
else
  echo "    AVISO: o certificado não saiu. O mais provável é a porta 80 não chegar da internet até a"
  echo "    VM. Confira com a CTI e rode este script de novo (o resto já está pronto)."
fi
systemctl is-enabled certbot.timer >/dev/null 2>&1 && echo "    renovação automática: certbot.timer ativo"

passo "Relógio (NTP), só diagnóstico"
timedatectl timesync-status 2>&1 | sed 's/^/    /' || true

passo "Resumo"
echo "    $(docker --version)"
echo "    $(docker compose version)"
echo "    $(nginx -v 2>&1)"
echo "    $(certbot --version 2>&1)"
echo "    serviços: docker $(systemctl is-active docker), nginx $(systemctl is-active nginx)"
docker run --rm hello-world >/dev/null && docker rmi -f hello-world >/dev/null && echo "    docker run hello-world: ok"
curl -s -o /dev/null -w "    http://$DOMINIO (local) -> %{http_code} %{redirect_url}\n" -H "Host: $DOMINIO" http://127.0.0.1/
df -h / /var | sed 's/^/    /'
if [[ -f /var/run/reboot-required ]]; then
  echo "    ATENÇÃO: as atualizações pedem reinício. Combine antes: a VM sai do ar ~1 min."
fi
echo "    Saia e entre de novo no SSH para usar o docker sem sudo."
echo "    Log completo: $LOG"
