# EvoSphere API Documentation

Complete REST API documentation for the EvoSphere AI Trading System.

## Base URL
```
http://localhost:5000
```

## Authentication
Currently no authentication required for local development. Production deployments should implement proper authentication and rate limiting.

## Core Endpoints

### Training & Evolution

#### Get Current Training Status
```http
GET /api/evolution/current
```

**Response:**
```json
{
  "status": "training|idle|completed",
  "current_generation": 15,
  "total_generations": 30,
  "best_fitness": 0.898,
  "population": [
    {
      "id": "chr_001",
      "genes": [
        {
          "indicator_type": "SMA",
          "parameters": {"period": 20},
          "feature_name": "SMA_20"
        }
      ],
      "fitness": 0.898
    }
  ]
}
```

#### Start Training
```http
POST /api/training/start
Content-Type: application/json
```

**Request Body:**
```json
{
  "config": {
    "population_size": 20,
    "num_generations": 30,
    "mutation_rate": 0.2,
    "data_file": "EURUSD_data.csv"
  }
}
```

#### Stop Training
```http
POST /api/training/stop
```

#### Reset Training State
```http
POST /api/training/reset
```

### Paper Trading

#### Get Trading Status
```http
GET /api/paper-trading/status
```

**Response:**
```json
{
  "balance": 10000.0,
  "profit_loss": 2500.0,
  "current_symbol": "BTC-USD",
  "position": {
    "size": 1000,
    "entry_price": 45000.0,
    "current_price": 47500.0,
    "unrealized_pnl": 2500.0
  },
  "market_analysis": {
    "rsi": 67.5,
    "momentum": 0.025,
    "volatility": 0.08,
    "price_position": 0.75
  }
}
```

#### Get Trading History
```http
GET /api/paper-trading/trades
```

**Response:**
```json
{
  "trades": [
    {
      "id": "trade_001",
      "symbol": "BTC-USD",
      "action": "BUY",
      "size": 1000,
      "price": 45000.0,
      "timestamp": "2025-07-02T10:30:00Z",
      "profit_loss": 0.0
    }
  ],
  "total_trades": 5,
  "winning_trades": 5,
  "win_rate": 1.0
}
```

#### Reset Paper Trading
```http
POST /api/paper-trading/reset
Content-Type: application/json
```

**Request Body:**
```json
{
  "balance": 10000.0
}
```

#### Switch Trading Symbol
```http
POST /api/paper-trading/switch-symbol
Content-Type: application/json
```

**Request Body:**
```json
{
  "symbol": "BTC-USD"
}
```

### Data Management

#### Upload Market Data
```http
POST /api/upload
Content-Type: multipart/form-data
```

**Form Data:**
- `file`: CSV file with OHLCV data

#### Fetch Yahoo Finance Data
```http
POST /api/fetch-data
Content-Type: application/json
```

**Request Body:**
```json
{
  "symbol": "EURUSD=X",
  "period": "1y",
  "interval": "1d"
}
```

#### Get Available Data Files
```http
GET /api/data/files
```

### Network & Consensus

#### Get Network Status
```http
GET /api/network/status
```

**Response:**
```json
{
  "node_count": 5,
  "consensus_fitness": 0.923,
  "network_health": "healthy",
  "connected_peers": [
    {
      "sphere_id": "evosphere_vietnam_001",
      "location": "Ho Chi Minh City",
      "fitness": 0.918,
      "last_seen": "2025-07-02T10:45:00Z"
    }
  ]
}
```

#### Join Network
```http
POST /api/network/join
Content-Type: application/json
```

**Request Body:**
```json
{
  "sphere_id": "evosphere_new_001",
  "location": "Your City",
  "region": "Your Region"
}
```

#### Get Collective Intelligence
```http
GET /api/network/collective-intelligence
```

### Performance Monitoring

#### Get Live Market Data
```http
GET /api/live-data
```

**Response:**
```json
{
  "symbol": "BTC-USD",
  "price": 47500.0,
  "change_percent": 5.56,
  "volume": 28500000000,
  "technical_analysis": {
    "rsi": 67.5,
    "sma_20": 46000.0,
    "momentum": 0.025,
    "volatility": 0.08
  },
  "news_sentiment": {
    "score": 0.65,
    "articles_analyzed": 25,
    "sources": ["Reuters", "Bloomberg", "CNBC"]
  }
}
```

#### Get System Performance
```http
GET /api/system/performance
```

**Response:**
```json
{
  "cpu_usage": 45.2,
  "memory_usage": 2.1,
  "disk_usage": 15.8,
  "network_latency": 12,
  "uptime": 86400,
  "active_processes": 8
}
```

### Configuration

#### Get System Configuration
```http
GET /api/config
```

#### Update Configuration
```http
POST /api/config
Content-Type: application/json
```

**Request Body:**
```json
{
  "trading": {
    "max_position_size": 10000,
    "risk_per_trade": 0.02,
    "stop_loss_pct": 0.05
  },
  "evolution": {
    "population_size": 20,
    "mutation_rate": 0.2,
    "elite_ratio": 0.1
  }
}
```

### Results & Analytics

#### Get Latest Results
```http
GET /api/results/latest
```

#### List All Result Files
```http
GET /api/results/files
```

#### Download Result File
```http
GET /api/results/download/{filename}
```

#### Get Trading Analytics
```http
GET /api/analytics/trading
```

**Response:**
```json
{
  "performance_metrics": {
    "total_return_pct": 13162.0,
    "win_rate": 1.0,
    "max_drawdown": 0.0,
    "sharpe_ratio": 8.5,
    "profit_factor": "Infinity"
  },
  "trade_statistics": {
    "total_trades": 1,
    "winning_trades": 1,
    "losing_trades": 0,
    "average_win": 51585.94,
    "average_loss": 0.0
  }
}
```

## WebSocket Endpoints

### Real-time Evolution Updates
```javascript
const ws = new WebSocket('ws://localhost:5000/ws');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  if (data.type === 'evolution_update') {
    // Handle chromosome evolution data
    console.log('Generation:', data.generation);
    console.log('Population:', data.population);
  }
};
```

### Live Trading Updates
```javascript
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  if (data.type === 'trading_update') {
    // Handle live trading data
    console.log('Action:', data.action);
    console.log('Price:', data.price);
    console.log('Profit:', data.profit);
  }
};
```

## Error Responses

All endpoints return consistent error responses:

```json
{
  "error": true,
  "message": "Detailed error description",
  "code": "ERROR_CODE",
  "timestamp": "2025-07-02T10:30:00Z"
}
```

### Common Error Codes
- `TRAINING_IN_PROGRESS`: Training already running
- `INVALID_DATA_FORMAT`: Uploaded data format invalid
- `INSUFFICIENT_DATA`: Not enough data points for analysis
- `NETWORK_UNAVAILABLE`: Network consensus not available
- `CONFIGURATION_ERROR`: Invalid configuration parameters

## Rate Limiting

- **Training Operations**: 1 request per minute
- **Data Uploads**: 10 requests per hour
- **Status Queries**: 60 requests per minute
- **Live Data**: 30 requests per minute

## Data Formats

### Market Data CSV Format
```csv
Date,Open,High,Low,Close,Volume
2025-01-01,1.0500,1.0550,1.0480,1.0520,1000000
2025-01-02,1.0520,1.0580,1.0510,1.0565,1200000
```

### Configuration Schema
```json
{
  "trading": {
    "max_position_size": "number",
    "risk_per_trade": "number (0-1)",
    "stop_loss_pct": "number (0-1)"
  },
  "evolution": {
    "population_size": "integer (10-100)",
    "num_generations": "integer (10-1000)",
    "mutation_rate": "number (0-1)",
    "elite_ratio": "number (0-1)"
  },
  "network": {
    "enable_consensus": "boolean",
    "consensus_threshold": "number (0-1)",
    "peer_timeout": "integer (seconds)"
  }
}
```

## SDK Examples

### Python SDK Usage
```python
import requests

class EvoSphereAPI:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def get_trading_status(self):
        response = requests.get(f"{self.base_url}/api/paper-trading/status")
        return response.json()
    
    def start_training(self, config):
        response = requests.post(
            f"{self.base_url}/api/training/start",
            json={"config": config}
        )
        return response.json()

# Usage
api = EvoSphereAPI()
status = api.get_trading_status()
print(f"Current balance: ${status['balance']}")
```

### JavaScript SDK Usage
```javascript
class EvoSphereAPI {
  constructor(baseUrl = 'http://localhost:5000') {
    this.baseUrl = baseUrl;
  }
  
  async getTradingStatus() {
    const response = await fetch(`${this.baseUrl}/api/paper-trading/status`);
    return response.json();
  }
  
  async startTraining(config) {
    const response = await fetch(`${this.baseUrl}/api/training/start`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({config})
    });
    return response.json();
  }
}

// Usage
const api = new EvoSphereAPI();
const status = await api.getTradingStatus();
console.log(`Current balance: $${status.balance}`);
```

## Integration Examples

### Raspberry Pi Hardware Integration
```python
# Monitor hardware performance during trading
import psutil
import requests

def monitor_system():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    # Send metrics to EvoSphere
    requests.post('http://localhost:5000/api/system/metrics', json={
        'cpu_usage': cpu_usage,
        'memory_usage': memory.percent,
        'available_memory_gb': memory.available / (1024**3)
    })
```

### Network Consensus Integration
```python
# Join EvoSphere network and participate in consensus
def join_evosphere_network():
    node_config = {
        'sphere_id': 'my_evosphere_001',
        'location': 'My Location',
        'capabilities': ['trading', 'data_collection', 'consensus']
    }
    
    response = requests.post(
        'http://localhost:5000/api/network/join',
        json=node_config
    )
    
    if response.json()['success']:
        print("Successfully joined EvoSphere network!")
```

---

## Support & Documentation

- **GitHub Issues**: Report bugs and request features
- **Community Discord**: Real-time developer support
- **Documentation Wiki**: Comprehensive guides and tutorials
- **Video Tutorials**: Step-by-step implementation guides

For additional API questions or custom integrations, contact the development team or open an issue on GitHub.