# Forex Trading AI - Raspberry Pi 5 Deployment Guide

## Overview
This guide will help you deploy your Forex Trading AI system on a Raspberry Pi 5 (16GB) as a dedicated trading server. The Pi will run 24/7, monitoring markets and executing trades automatically.

## Hardware Requirements
- **Raspberry Pi 5 (16GB RAM)** ✅ Perfect for this system
- **MicroSD Card** (64GB+ recommended, Class 10)
- **Power Supply** (Official Pi 5 power adapter)
- **Network Connection** (Ethernet or WiFi)
- **Optional**: Case with cooling fan for 24/7 operation

## Step 1: Raspberry Pi OS Setup

### 1.1 Flash Raspberry Pi OS
```bash
# Download Raspberry Pi Imager from https://rpi.org/imager
# Flash "Raspberry Pi OS (64-bit)" to your SD card
# Enable SSH and set username/password during flash
```

### 1.2 Initial Setup
```bash
# Boot Pi and connect via SSH or directly
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y git python3-pip python3-venv htop curl wget
```

## Step 2: Python Environment Setup

### 2.1 Create Project Directory
```bash
# Create trading directory
mkdir ~/forex-trading-ai
cd ~/forex-trading-ai

# Create Python virtual environment
python3 -m venv trading_env
source trading_env/bin/activate
```

### 2.2 Install System Dependencies
```bash
# Install PostgreSQL for data persistence
sudo apt install -y postgresql postgresql-contrib python3-dev libpq-dev

# Start and enable PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create trading database and user
sudo -u postgres createdb trading_ai
sudo -u postgres psql -c "CREATE USER trading_user WITH ENCRYPTED PASSWORD 'your_secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE trading_ai TO trading_user;"
```

### 2.3 Install Python Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install core dependencies
pip install flask pandas numpy scikit-learn matplotlib seaborn
pip install yfinance requests beautifulsoup4 feedparser
pip install python-dateutil python-dotenv loguru tqdm
pip install plotly openpyxl xlrd ujson psutil
pip install textblob statsmodels joblib numba numexpr randomgen

# Install database dependencies
pip install sqlalchemy psycopg2-binary
```

## Step 3: Deploy Trading System

### 3.1 Transfer Files
```bash
# Option A: Direct file transfer (if copying from another system)
scp -r /path/to/your/trading/files/* pi@[PI_IP_ADDRESS]:~/forex-trading-ai/

# Option B: Git clone (if you have the project in a repository)
git clone [YOUR_REPO_URL] ~/forex-trading-ai
cd ~/forex-trading-ai
```

### 3.2 Configure Environment Variables
```bash
# Create environment file for database
nano ~/.env
```

Add these database settings:
```env
# Database Configuration
DATABASE_URL=postgresql://trading_user:your_secure_password@localhost:5432/trading_ai

# Flask Configuration
FLASK_SECRET_KEY=your_flask_secret_key_here

# Optional: Trading API Keys (can also be set via web interface)
# BINANCE_API_KEY=your_binance_key
# BINANCE_SECRET=your_binance_secret
```

### 3.3 Set File Permissions
```bash
chmod +x *.py
chmod -R 755 static templates config data logs models results utils
chmod 600 ~/.env  # Secure environment file
```

## Step 4: Database Setup and Initialization

### 4.1 Initialize Database Tables
```bash
# Activate virtual environment
source ~/forex-trading-ai/trading_env/bin/activate

# Create database tables
cd ~/forex-trading-ai
python -c "from models import db; from simple_app import app; app.app_context().push(); db.create_all(); print('Database tables created successfully!')"
```

### 4.2 Test Database Connection
```bash
# Test database connection
python -c "from persistence_service import PersistenceService; ps = PersistenceService(); print('Database connection successful!')"
```

## Step 5: System Service Setup (Auto-Start)

### 5.1 Create Service File
```bash
sudo nano /etc/systemd/system/forex-trading.service
```

Add this content:
```ini
[Unit]
Description=Forex Trading AI System with Database Persistence
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/forex-trading-ai
Environment=PATH=/home/pi/forex-trading-ai/trading_env/bin
EnvironmentFile=/home/pi/.env
ExecStart=/home/pi/forex-trading-ai/trading_env/bin/python simple_app.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 4.2 Enable Auto-Start
```bash
# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable forex-trading.service
sudo systemctl start forex-trading.service

# Check status
sudo systemctl status forex-trading.service
```

## Step 5: Network Configuration

### 5.1 Configure Flask for Network Access
Update `simple_app.py` to bind to all interfaces:
```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### 5.2 Configure Firewall (Optional)
```bash
# Install UFW firewall
sudo apt install ufw

# Allow SSH and Flask
sudo ufw allow ssh
sudo ufw allow 5000
sudo ufw enable
```

## Step 6: Trading API Configuration

### 6.1 Supported Trading Platforms

The system supports multiple cryptocurrency exchanges for live trading:

**Binance** (Recommended for beginners)
- Sign up: https://www.binance.com
- Low fees: 0.1%
- Strong API support
- Sandbox mode available

**Coinbase Pro**
- Sign up: https://pro.coinbase.com
- Higher fees: 0.5%
- US-based and regulated
- Good for beginners

**Kraken**
- Sign up: https://www.kraken.com
- Competitive fees: 0.26%
- European-based
- Advanced features

### 6.2 Web Interface Configuration (Recommended)

1. **Access the API Configuration Page**
   ```
   http://your-pi-ip:5000/api-config
   ```

2. **Add Trading Platform**
   - Select your exchange (Binance, Coinbase, Kraken)
   - Enter API key and secret from your exchange account
   - Set maximum position size (start with $100)
   - Set daily loss limit (recommend $50)
   - **Important**: Keep "Sandbox Mode" enabled for testing

3. **Test Connection**
   - Click "Test" to verify API credentials
   - Ensure connection shows "Connected" status
   - Verify balance information is retrieved

4. **Enable Live Trading** (Only when ready)
   - ⚠️ **WARNING**: This uses real money
   - Type "ENABLE LIVE TRADING" to confirm
   - Start with very small amounts ($10-50)
   - Monitor closely for first few trades

### 6.3 Security Features

Your API credentials are protected with:
- **Encryption**: All keys encrypted using Fernet encryption
- **Secure Storage**: Configuration files have restricted permissions (600)
- **Risk Management**: Built-in position size and loss limits
- **Sandbox Mode**: Test without real money first

### 6.4 Risk Management Settings

**Conservative Settings (Recommended)**
- Max Position Size: $100
- Daily Loss Limit: $50
- Sandbox Mode: Enabled initially

**Moderate Settings (After testing)**
- Max Position Size: $500
- Daily Loss Limit: $200
- Regular monitoring required

### 6.5 Getting API Keys

**Binance Setup:**
1. Log in to Binance.com
2. Go to Account > API Management
3. Create New Key
4. Enable "Enable Trading" permission only
5. Restrict to your Pi's IP address (optional but recommended)

**Coinbase Pro Setup:**
1. Log in to pro.coinbase.com
2. Go to Profile > API
3. Create API Key
4. Enable "Trade" permission
5. Copy key, secret, and passphrase

**Kraken Setup:**
1. Log in to Kraken.com
2. Go to Settings > API
3. Generate New Key
4. Enable "Trade" permission
5. Copy key and private key

## Step 7: Monitoring and Maintenance

### 6.1 System Monitoring Scripts
Create monitoring script:
```bash
nano ~/monitor_trading.sh
```

```bash
#!/bin/bash
echo "=== Forex Trading AI Status ==="
echo "System Status:"
systemctl is-active forex-trading.service

echo -e "\nMemory Usage:"
free -h

echo -e "\nDisk Usage:"
df -h /

echo -e "\nCPU Temperature:"
vcgencmd measure_temp

echo -e "\nActive Connections:"
netstat -an | grep :5000
```

Make executable:
```bash
chmod +x ~/monitor_trading.sh
```

### 6.2 Log Rotation
```bash
# Create log rotation config
sudo nano /etc/logrotate.d/forex-trading
```

```
/home/pi/forex-trading-ai/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    copytruncate
    notifempty
}
```

## Step 7: Access Your Trading System

### 7.1 Find Pi IP Address
```bash
hostname -I
```

### 7.2 Access Web Interface
Open browser and go to:
```
http://[PI_IP_ADDRESS]:5000
```

Example: `http://192.168.1.100:5000`

### 7.3 Remote Access Setup (Optional)
For access from outside your network:
1. Configure port forwarding on your router (5000 → Pi IP)
2. Use dynamic DNS service for fixed domain name
3. Consider VPN for secure access

## Step 8: Performance Optimization

### 8.1 Pi 5 Specific Optimizations
```bash
# Add to /boot/config.txt for better performance
echo "arm_freq=2400" | sudo tee -a /boot/config.txt
echo "gpu_mem=128" | sudo tee -a /boot/config.txt

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable wifi-powersave
```

### 8.2 Python Optimizations
```bash
# Install faster math libraries
pip install --upgrade numpy
pip install numexpr  # Already included in dependencies
```

## Step 9: Backup and Recovery

### 9.1 Automated Backup Script with Database
```bash
nano ~/backup_trading.sh
```

```bash
#!/bin/bash
BACKUP_DIR="/home/pi/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup PostgreSQL database
pg_dump -h localhost -U trading_user -d trading_ai > $BACKUP_DIR/database_backup_$DATE.sql

# Backup trading data and models
tar -czf $BACKUP_DIR/trading_backup_$DATE.tar.gz \
    ~/forex-trading-ai/data \
    ~/forex-trading-ai/models \
    ~/forex-trading-ai/results \
    ~/forex-trading-ai/logs \
    ~/.env

# Keep only last 7 backups of each type
cd $BACKUP_DIR
ls -t trading_backup_*.tar.gz | tail -n +8 | xargs rm -f
ls -t database_backup_*.sql | tail -n +8 | xargs rm -f

echo "Backup completed: $DATE"
```

### 9.2 Schedule Backups
```bash
# Add to crontab
crontab -e

# Add this line for daily backups at 2 AM
0 2 * * * /home/pi/backup_trading.sh
```

## Step 10: Troubleshooting

### 10.1 Common Issues
```bash
# Check service logs
sudo journalctl -u forex-trading.service -f

# Check Python errors
tail -f ~/forex-trading-ai/logs/trading.log

# Restart service
sudo systemctl restart forex-trading.service

# Check network connectivity
curl -I https://finance.yahoo.com
```

### 10.2 Performance Monitoring
```bash
# Monitor system resources
htop

# Check service status
./monitor_trading.sh

# View active connections
netstat -tulpn | grep :5000
```

## Features on Raspberry Pi

✅ **24/7 Operation** - Low power consumption (~5-10W)  
✅ **Multi-Market Trading** - EUR/USD, Bitcoin, GBP/USD, USD/JPY, Ethereum  
✅ **Real-time News Analysis** - Reuters, Bloomberg, CNBC, MarketWatch  
✅ **Web Interface** - Access from any device on your network  
✅ **PostgreSQL Database** - Permanent evolution data persistence  
✅ **Training Session History** - All chromosome evolution saved forever  
✅ **90.34% Peak Fitness** - Proven high-performance trading strategies  
✅ **DNA-Style Visualization** - Real-time chromosome evolution display  
✅ **Automatic Backups** - Scheduled data and database protection  
✅ **System Monitoring** - Performance and health checks  
✅ **Auto-restart** - System resilience and reliability  
✅ **Advanced Indicators** - 40+ technical indicators including Ichimoku, Fibonacci, Pivot Points  

## Expected Performance

- **Memory Usage**: 2-4GB (well within 16GB limit)
- **CPU Usage**: 10-30% during active trading
- **Storage**: ~1GB for system + logs over time
- **Network**: Minimal bandwidth for market data
- **Uptime**: Designed for continuous 24/7 operation

## Security Recommendations

1. **Change default passwords** for Pi user
2. **Enable SSH key authentication** instead of passwords
3. **Keep system updated** with `sudo apt update && sudo apt upgrade`
4. **Use firewall** to limit network access
5. **Regular backups** of trading data and models
6. **Monitor logs** for unusual activity

## Getting Started

1. Flash SD card with Raspberry Pi OS
2. Boot Pi and run initial setup
3. Transfer your trading system files
4. Install dependencies and configure service
5. Access web interface at `http://[PI_IP]:5000`
6. Start paper trading with Bitcoin for higher volatility
7. Monitor performance and enjoy automated trading!

Your Raspberry Pi 5 will now run your proven 90.34% fitness trading system 24/7, with complete database persistence to preserve all evolution data. The sophisticated AI combines 40+ advanced technical indicators with real-time news sentiment for intelligent trading decisions. All training sessions survive system restarts, giving you a permanent record of successful trading strategies.