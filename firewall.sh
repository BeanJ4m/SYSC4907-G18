#!/bin/bash

set -e

echo "[+] Setting up IDS Firewall Access Bridge..."

# ----------------------------
# Create helper script
# ----------------------------
echo "[+] Creating privileged firewall reader..."

sudo tee /usr/local/bin/readfw > /dev/null << 'EOF'
#!/bin/bash
/usr/sbin/nft list ruleset
EOF

sudo chmod 755 /usr/local/bin/readfw

# ----------------------------
# Create group
# ----------------------------
echo "[+] Creating access group..."

sudo groupadd -f ids-analyst

# Add current user to group
echo "[+] Adding current user to ids-analyst group..."
sudo usermod -aG ids-analyst $USER

# ----------------------------
# Allow group to run readfw as root
# ----------------------------
echo "[+] Configuring sudo permissions..."

echo "%ids-analyst ALL=(ALL) NOPASSWD: /usr/local/bin/readfw" | sudo tee /etc/sudoers.d/ids-analyst > /dev/null

sudo chmod 440 /etc/sudoers.d/ids-analyst

# ----------------------------
# Done
# ----------------------------
echo ""
echo "Setup complete."
echo ""
echo "IMPORTANT:"
echo "You must log out and log back in for group changes to apply."
echo ""
echo "After that, your Streamlit app can securely read firewall rules."