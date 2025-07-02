"""
Trading API Configuration and Real Trading Integration
Supports multiple trading platforms with secure API key management
"""

import os
import json
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
import hashlib
import base64
from cryptography.fernet import Fernet
import ccxt  # CryptoCurrency eXchange Trading Library
import yfinance as yf
from datetime import datetime, timedelta

@dataclass
class TradingAPIConfig:
    """Configuration for trading API connections"""
    platform: str
    api_key: str
    api_secret: str
    sandbox_mode: bool = True
    max_position_size: float = 100.0  # Maximum USD per trade
    daily_loss_limit: float = 50.0    # Maximum daily loss limit
    enabled: bool = False

class TradingAPIManager:
    """Manages multiple trading API connections with security and risk controls"""
    
    def __init__(self, config_file: str = "trading_config.json"):
        self.config_file = config_file
        self.apis: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        self.encryption_key = self._get_or_create_encryption_key()
        self.daily_pnl = {}
        self.load_config()
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for API credentials"""
        key_file = ".trading_key"
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Secure file permissions
            return key
    
    def _encrypt_credentials(self, data: str) -> str:
        """Encrypt sensitive data"""
        f = Fernet(self.encryption_key)
        return f.encrypt(data.encode()).decode()
    
    def _decrypt_credentials(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        f = Fernet(self.encryption_key)
        return f.decrypt(encrypted_data.encode()).decode()
    
    def add_api_config(self, platform: str, api_key: str, api_secret: str, 
                      sandbox_mode: bool = True, max_position_size: float = 100.0,
                      daily_loss_limit: float = 50.0) -> bool:
        """Add new API configuration with encryption"""
        try:
            # Encrypt sensitive data
            encrypted_key = self._encrypt_credentials(api_key)
            encrypted_secret = self._encrypt_credentials(api_secret)
            
            config = {
                'platform': platform,
                'api_key': encrypted_key,
                'api_secret': encrypted_secret,
                'sandbox_mode': sandbox_mode,
                'max_position_size': max_position_size,
                'daily_loss_limit': daily_loss_limit,
                'enabled': False,
                'created_at': datetime.now().isoformat()
            }
            
            self.save_config_entry(platform, config)
            self.logger.info(f"Added API configuration for {platform}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add API config for {platform}: {e}")
            return False
    
    def save_config_entry(self, platform: str, config: Dict) -> None:
        """Save individual API configuration"""
        configs = self.load_all_configs()
        configs[platform] = config
        
        with open(self.config_file, 'w') as f:
            json.dump(configs, f, indent=2)
        os.chmod(self.config_file, 0o600)  # Secure file permissions
    
    def load_all_configs(self) -> Dict:
        """Load all API configurations"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load config: {e}")
                return {}
        return {}
    
    def load_config(self) -> None:
        """Load and initialize API connections"""
        configs = self.load_all_configs()
        
        for platform, config in configs.items():
            if config.get('enabled', False):
                try:
                    self.initialize_api(platform, config)
                except Exception as e:
                    self.logger.error(f"Failed to initialize {platform}: {e}")
    
    def initialize_api(self, platform: str, config: Dict) -> bool:
        """Initialize trading API connection"""
        try:
            # Decrypt credentials
            api_key = self._decrypt_credentials(config['api_key'])
            api_secret = self._decrypt_credentials(config['api_secret'])
            
            # Initialize based on platform
            if platform.lower() == 'binance':
                exchange = ccxt.binance({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'sandbox': config.get('sandbox_mode', True),
                    'enableRateLimit': True,
                })
            elif platform.lower() == 'coinbase':
                exchange = ccxt.coinbasepro({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'passphrase': config.get('passphrase', ''),
                    'sandbox': config.get('sandbox_mode', True),
                    'enableRateLimit': True,
                })
            elif platform.lower() == 'kraken':
                exchange = ccxt.kraken({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                })
            else:
                self.logger.warning(f"Unsupported platform: {platform}")
                return False
            
            # Test connection
            balance = exchange.fetch_balance()
            self.apis[platform] = {
                'exchange': exchange,
                'config': config,
                'last_test': datetime.now(),
                'status': 'connected'
            }
            
            self.logger.info(f"Successfully connected to {platform}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {platform}: {e}")
            return False
    
    def test_api_connection(self, platform: str) -> Dict[str, Any]:
        """Test API connection and return status"""
        if platform not in self.apis:
            return {'status': 'error', 'message': 'API not configured'}
        
        try:
            exchange = self.apis[platform]['exchange']
            balance = exchange.fetch_balance()
            
            return {
                'status': 'success',
                'platform': platform,
                'balance': balance.get('total', {}),
                'sandbox_mode': self.apis[platform]['config'].get('sandbox_mode', True),
                'last_test': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'platform': platform,
                'message': str(e)
            }
    
    def execute_trade(self, platform: str, symbol: str, side: str, 
                     amount: float, order_type: str = 'market') -> Dict[str, Any]:
        """Execute trade with risk management"""
        if platform not in self.apis:
            return {'status': 'error', 'message': 'API not configured'}
        
        # Risk management checks
        config = self.apis[platform]['config']
        max_position = config.get('max_position_size', 100.0)
        daily_limit = config.get('daily_loss_limit', 50.0)
        
        if amount > max_position:
            return {'status': 'error', 'message': f'Amount exceeds max position size: ${max_position}'}
        
        # Check daily loss limit
        today = datetime.now().date().isoformat()
        daily_loss = self.daily_pnl.get(f"{platform}_{today}", 0)
        
        if daily_loss <= -daily_limit:
            return {'status': 'error', 'message': f'Daily loss limit reached: ${daily_limit}'}
        
        try:
            exchange = self.apis[platform]['exchange']
            
            if side.lower() == 'buy':
                order = exchange.create_market_buy_order(symbol, amount)
            elif side.lower() == 'sell':
                order = exchange.create_market_sell_order(symbol, amount)
            else:
                return {'status': 'error', 'message': 'Invalid side. Use "buy" or "sell"'}
            
            # Log trade
            self.logger.info(f"Executed {side} order: {symbol} {amount} on {platform}")
            
            return {
                'status': 'success',
                'platform': platform,
                'order': order,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Trade execution failed on {platform}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_account_balance(self, platform: str) -> Dict[str, Any]:
        """Get current account balance"""
        if platform not in self.apis:
            return {'status': 'error', 'message': 'API not configured'}
        
        try:
            exchange = self.apis[platform]['exchange']
            balance = exchange.fetch_balance()
            
            return {
                'status': 'success',
                'platform': platform,
                'balance': balance,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_price(self, platform: str, symbol: str) -> Dict[str, Any]:
        """Get current price for symbol"""
        if platform not in self.apis:
            return {'status': 'error', 'message': 'API not configured'}
        
        try:
            exchange = self.apis[platform]['exchange']
            ticker = exchange.fetch_ticker(symbol)
            
            return {
                'status': 'success',
                'platform': platform,
                'symbol': symbol,
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def enable_platform(self, platform: str) -> bool:
        """Enable trading for a platform"""
        configs = self.load_all_configs()
        if platform in configs:
            configs[platform]['enabled'] = True
            with open(self.config_file, 'w') as f:
                json.dump(configs, f, indent=2)
            return self.initialize_api(platform, configs[platform])
        return False
    
    def disable_platform(self, platform: str) -> bool:
        """Disable trading for a platform"""
        configs = self.load_all_configs()
        if platform in configs:
            configs[platform]['enabled'] = False
            with open(self.config_file, 'w') as f:
                json.dump(configs, f, indent=2)
            if platform in self.apis:
                del self.apis[platform]
            return True
        return False
    
    def get_supported_platforms(self) -> Dict[str, Dict]:
        """Get list of supported trading platforms"""
        return {
            'binance': {
                'name': 'Binance',
                'type': 'cryptocurrency',
                'fees': '0.1%',
                'min_deposit': '$10',
                'signup_url': 'https://www.binance.com'
            },
            'coinbase': {
                'name': 'Coinbase Pro',
                'type': 'cryptocurrency',
                'fees': '0.5%',
                'min_deposit': '$25',
                'signup_url': 'https://pro.coinbase.com'
            },
            'kraken': {
                'name': 'Kraken',
                'type': 'cryptocurrency',
                'fees': '0.26%',
                'min_deposit': '$10',
                'signup_url': 'https://www.kraken.com'
            }
        }
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get status of all configured APIs"""
        status = {}
        configs = self.load_all_configs()
        
        for platform, config in configs.items():
            status[platform] = {
                'enabled': config.get('enabled', False),
                'sandbox_mode': config.get('sandbox_mode', True),
                'max_position_size': config.get('max_position_size', 100.0),
                'daily_loss_limit': config.get('daily_loss_limit', 50.0),
                'connected': platform in self.apis,
                'last_test': self.apis.get(platform, {}).get('last_test'),
                'created_at': config.get('created_at')
            }
        
        return status

# Global instance
trading_api_manager = TradingAPIManager()