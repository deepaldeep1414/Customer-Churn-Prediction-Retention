#!/usr/bin/env bash
# ==============================================================
# infra/aws/ec2_setup.sh
# Bootstrap script for a fresh Ubuntu 22.04 EC2 instance.
# Run once as root: sudo bash ec2_setup.sh
#
# Installs: Docker, Docker Compose plugin, AWS CLI v2,
#           Python 3.11, git, nginx (for TLS cert init only),
#           certbot, and sets up the /opt/churn deploy directory.
# ==============================================================

set -euo pipefail
LOG="/var/log/churn_bootstrap.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Churn Platform EC2 Bootstrap $(date) ==="

# ── System updates ────────────────────────────────────────────
apt-get update -y && apt-get upgrade -y
apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    jq \
    unzip \
    python3.11 \
    python3.11-venv \
    python3-pip \
    nginx \
    certbot \
    python3-certbot-nginx

# ── Docker ───────────────────────────────────────────────────
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add ubuntu user to docker group
usermod -aG docker ubuntu
systemctl enable docker
systemctl start docker

echo "Docker version: $(docker --version)"

# ── AWS CLI v2 ────────────────────────────────────────────────
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install
rm -rf /tmp/aws /tmp/awscliv2.zip

echo "AWS CLI version: $(aws --version)"

# ── Deploy directory ──────────────────────────────────────────
mkdir -p /opt/churn
chown ubuntu:ubuntu /opt/churn

# ── Systemd service: auto-start Docker Compose on reboot ─────
cat > /etc/systemd/system/churn-platform.service << 'EOF'
[Unit]
Description=Customer Churn Platform
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/churn
EnvironmentFile=/opt/churn/.env
ExecStart=/usr/bin/docker compose -f docker-compose.prod.yml up -d
ExecStop=/usr/bin/docker compose -f docker-compose.prod.yml down
TimeoutStartSec=300
User=ubuntu

[Install]
WantedBy=multi-user.target
EOF

systemctl enable churn-platform

# ── Swap (for single-instance deployments with limited RAM) ──
if [ ! -f /swapfile ]; then
    fallocate -l 4G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "Swap enabled: 4GB"
fi

# ── SSH hardening ─────────────────────────────────────────────
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/'   /etc/ssh/sshd_config
systemctl reload sshd

# ── UFW firewall ──────────────────────────────────────────────
ufw --force enable
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
# Block direct access to API/MLflow ports (nginx only)
ufw deny 8000/tcp
ufw deny 5000/tcp
ufw deny 8088/tcp

echo "UFW status:"
ufw status

echo ""
echo "=== Bootstrap complete ==="
echo "Next steps:"
echo "  1. Clone repo: cd /opt/churn && git clone <repo_url> ."
echo "  2. Populate /opt/churn/.env (pull from AWS Secrets Manager)"
echo "  3. docker compose -f docker-compose.prod.yml up -d"
echo "  4. certbot --nginx -d api.churn.example.com"
