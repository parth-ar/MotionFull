#!/usr/bin/env bash
set -e

# ==============================================================================
# SWSTP Edge Service Automated Installer for Raspberry Pi
# ==============================================================================

SERVICE_NAME="swstp-edge.service"
CURRENT_USER=$(whoami)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"

echo "=================================================================="
echo " SWSTP EDGE — SYSTEMD SERVICE INSTALLER"
echo "=================================================================="
echo "  User:             $CURRENT_USER"
echo "  Working Directory: $SCRIPT_DIR"
echo "  Python Binary:    $PYTHON_BIN"
echo "=================================================================="

# 1. Verify virtual environment exists
if [ ! -f "$PYTHON_BIN" ]; then
    echo "❌ ERROR: Virtual environment not found at $PYTHON_BIN"
    echo "👉 Please create the venv first:"
    echo "   cd $SCRIPT_DIR"
    echo "   python3 -m venv venv --system-site-packages"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# 2. Ensure user belongs to hardware groups (gpio, i2c, video, render)
echo "[1/4] Ensuring user groups (gpio, i2c, video, render)..."
sudo usermod -aG gpio,i2c,video,render "$CURRENT_USER" || true

# 3. Generate systemd service file
echo "[2/4] Writing /etc/systemd/system/$SERVICE_NAME..."
sudo tee /etc/systemd/system/$SERVICE_NAME > /dev/null <<EOF
[Unit]
Description=SWSTP Edge Gateway — Motion & Litter Detection (Raspberry Pi 4B)
After=network-online.target time-sync.target
Wants=network-online.target

[Service]
Type=simple
User=$CURRENT_USER
Group=$CURRENT_USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$PYTHON_BIN $SCRIPT_DIR/main.py --headless
Environment=PYTHONUNBUFFERED=1
Environment=QT_QPA_PLATFORM=offscreen
Restart=always
RestartSec=5s
TimeoutStartSec=90
StandardOutput=journal
StandardError=journal
SyslogIdentifier=swstp-edge
SupplementaryGroups=gpio i2c video render

[Install]
WantedBy=multi-user.target
EOF

# 4. Reload systemd daemon
echo "[3/4] Reloading systemd daemon..."
sudo systemctl daemon-reload

# 5. Enable and start service
echo "[4/4] Enabling and starting $SERVICE_NAME..."
sudo systemctl enable $SERVICE_NAME
sudo systemctl restart $SERVICE_NAME

echo ""
echo "=================================================================="
echo " ✔ SUCCESS! $SERVICE_NAME is installed and active."
echo "=================================================================="
echo "  Status check:  sudo systemctl status $SERVICE_NAME"
echo "  Live logs:     sudo journalctl -u $SERVICE_NAME -f"
echo "  Stop service:  sudo systemctl stop $SERVICE_NAME"
echo "  Start service: sudo systemctl start $SERVICE_NAME"
echo "=================================================================="
