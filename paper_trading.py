"""
Paper Trading System - Live market data with simulated trades
No registration required - uses Yahoo Finance API
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import joblib
from threading import Thread, Event
import threading

# Import our trained components
from forex_simulation_env import ForexSimulationEnv
from news_sentiment_analyzer import NewsSentimentAnalyzer
from trading_pairs_config import trading_pairs_manager, MarketType
from dqn_agent_sklearn import DQNAgentSklearn
from main_trading_system import ForexTradingSystem

class PaperTradingSystem:
    """Paper trading system using live market data and trained AI model"""
    
    def __init__(self, initial_balance: float = 10000.0, symbol: str = 'EURUSD=X'):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}
        self.trade_history = []
        self.is_running = False
        self.stop_event = Event()
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        
        # AI model
        self.model = None
        self.trading_env = None
        self.last_features = None
        
        # Live data
        self.current_data = None
        self.data_buffer = []
        self.lookback_periods = 100  # Keep last 100 periods for technical indicators
        
        # Initialize news sentiment analyzer
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_trained_model(self, model_path: str = None):
        """Load the trained DQN model"""
        if model_path is None:
            # Find the latest model file
            results_dir = 'results'
            if os.path.exists(results_dir):
                model_files = [f for f in os.listdir(results_dir) if f.startswith('dqn_model_') and f.endswith('.pkl')]
                if model_files:
                    model_path = os.path.join(results_dir, sorted(model_files)[-1])
        
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.logger.info(f"Loaded trained model from {model_path}")
                return True
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                return False
        else:
            self.logger.error("No trained model found")
            return False
    
    def fetch_live_data(self) -> Optional[pd.DataFrame]:
        """Fetch live market data from Yahoo Finance"""
        try:
            # Get the last 2 days of data to ensure we have recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=2)
            
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(start=start_date, end=end_date, interval='1m')
            
            if data.empty:
                self.logger.warning("No live data received")
                return None
            
            # Rename columns to match our system - handle different column counts
            if len(data.columns) == 7:
                # Yahoo returns: Open, High, Low, Close, Adj Close, Volume, Dividends, Stock Splits
                data.columns = ['open', 'high', 'low', 'close', 'adjclose', 'volume', 'dividends']
                data = data[['open', 'high', 'low', 'close', 'volume', 'adjclose']]  # Keep only needed columns
            elif len(data.columns) == 6:
                # Standard format: Open, High, Low, Close, Adj Close, Volume
                data.columns = ['open', 'high', 'low', 'close', 'adjclose', 'volume']
            elif len(data.columns) == 5:
                # Basic format: Open, High, Low, Close, Volume
                data.columns = ['open', 'high', 'low', 'close', 'volume']
                data['adjclose'] = data['close']  # Add adjclose for compatibility
            
            # Get the most recent data point
            return data.tail(1)
            
        except Exception as e:
            self.logger.error(f"Error fetching live data: {e}")
            return None
    
    def update_data_buffer(self, new_data: pd.DataFrame):
        """Update the rolling data buffer with new market data"""
        if new_data is not None and not new_data.empty:
            self.data_buffer.append(new_data.iloc[-1])
            
            # Keep only the last N periods
            if len(self.data_buffer) > self.lookback_periods:
                self.data_buffer = self.data_buffer[-self.lookback_periods:]
            
            # Convert to DataFrame for technical indicator calculation
            # Reduce minimum data requirement to get trades faster
            if len(self.data_buffer) >= 5:  # Only need 5 data points minimum
                self.current_data = pd.DataFrame(self.data_buffer)
                self.current_data.index = pd.date_range(end=datetime.now(), periods=len(self.data_buffer), freq='1min')
                return True
        return False
    
    def calculate_features(self) -> Optional[np.ndarray]:
        """Calculate technical indicators from current data"""
        if self.current_data is None or len(self.current_data) < 5:
            return None
        
        try:
            # Calculate the same 5 features used in training
            features = self._calculate_trading_features(self.current_data)
            
            if features is not None:
                # Create state vector (flatten to match training format)
                state = np.array(features, dtype=np.float32)
                
                # Pad to match training state size if needed
                if len(state) < 150:  # Training used 150 features
                    padded_state = np.zeros(150)
                    padded_state[:len(state)] = state
                    state = padded_state
                
                return state
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
        
        return None
    
    def _calculate_trading_features(self, data: pd.DataFrame) -> Optional[list]:
        """Calculate the same 5 trading features used in training"""
        try:
            if len(data) < 5:
                return None
            
            close_prices = data['close']
            
            # 1. SMA_20 (or SMA of available data if less than 20)
            window = min(20, len(data))
            sma = close_prices.rolling(window=window).mean().iloc[-1]
            
            # 2. RSI_14 (or RSI of available data if less than 14)
            rsi_window = min(14, len(data) - 1)
            if rsi_window > 1:
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1]
            else:
                rsi = 50  # Neutral RSI
            
            # 3. MOMENTUM (price change over period)
            momentum_window = min(10, len(data) - 1)
            if momentum_window > 0:
                momentum = (close_prices.iloc[-1] - close_prices.iloc[-momentum_window-1]) / close_prices.iloc[-momentum_window-1] * 100
            else:
                momentum = 0
            
            # 4. VOLATILITY (standard deviation of returns)
            vol_window = min(10, len(data) - 1)
            if vol_window > 1:
                returns = close_prices.pct_change().dropna()
                volatility = returns.rolling(window=vol_window).std().iloc[-1] * 100
            else:
                volatility = 0
            
            # 5. PRICE_POSITION (where current price is relative to recent range)
            range_window = min(20, len(data))
            high_val = data['high'].rolling(window=range_window).max().iloc[-1]
            low_val = data['low'].rolling(window=range_window).min().iloc[-1]
            current_price = close_prices.iloc[-1]
            
            if high_val != low_val:
                price_position = (current_price - low_val) / (high_val - low_val) * 100
            else:
                price_position = 50  # Middle position if no range
            
            return [sma, rsi, momentum, volatility, price_position]
            
        except Exception as e:
            self.logger.error(f"Error in feature calculation: {e}")
            return None
    
    def make_trading_decision(self, state: np.ndarray) -> int:
        """Make trading decision using trained AI model"""
        if self.model is None:
            return 2  # Hold if no model
        
        try:
            # Handle different model formats
            if hasattr(self.model, 'act'):
                # Direct model object
                action = self.model.act(state.reshape(1, -1))
            elif hasattr(self.model, 'predict'):
                # Sklearn-style model
                action = self.model.predict(state.reshape(1, -1))[0]
            elif isinstance(self.model, dict) and 'model' in self.model:
                # Dictionary with model inside
                inner_model = self.model['model']
                if hasattr(inner_model, 'predict'):
                    action = inner_model.predict(state.reshape(1, -1))[0]
                else:
                    action = 2  # Hold if can't predict
            else:
                # Enhanced prediction combining technical analysis with news sentiment
                sma, rsi, momentum, volatility, price_position = state[:5]
                
                # Get current news sentiment
                try:
                    sentiment_strength, sentiment_signal = self.sentiment_analyzer.get_sentiment_signal(6)
                    self.logger.info(f"News sentiment: {sentiment_signal} (strength: {sentiment_strength:.2f})")
                    
                    # Combine technical and sentiment analysis
                    technical_signal = 0  # Neutral by default
                    
                    # Technical analysis with logging
                    if rsi < 30 and momentum > 0:  # Oversold and gaining momentum
                        technical_signal = 1  # Buy signal
                        self.logger.info(f"Technical BUY signal: RSI={rsi:.2f}, momentum={momentum:.4f}")
                    elif rsi > 70 and momentum < 0:  # Overbought and losing momentum
                        technical_signal = -1  # Sell signal
                        self.logger.info(f"Technical SELL signal: RSI={rsi:.2f}, momentum={momentum:.4f}")
                    else:
                        self.logger.info(f"Technical NEUTRAL: RSI={rsi:.2f}, momentum={momentum:.4f}, volatility={volatility:.4f}")
                    
                    # Combine with sentiment (sentiment gets 40% weight, technical gets 60%)
                    combined_signal = (0.6 * technical_signal) + (0.4 * sentiment_strength)
                    
                    # Make final decision (more sensitive thresholds for testing)
                    if combined_signal > 0.1:  # Lowered from 0.2
                        action = 0  # Buy
                    elif combined_signal < -0.1:  # Lowered from -0.2
                        action = 1  # Sell
                    else:
                        action = 2  # Hold
                        
                except Exception as e:
                    self.logger.warning(f"Could not get sentiment data: {e}")
                    # Fall back to technical analysis only
                    if rsi < 30 and momentum > 0:
                        action = 0  # Buy
                    elif rsi > 70 and momentum < 0:
                        action = 1  # Sell
                    else:
                        action = 2  # Hold
            
            return int(action)
        except Exception as e:
            self.logger.error(f"Error making trading decision: {e}")
            return 2  # Hold on error
    
    def execute_paper_trade(self, action: int, current_price: float, timestamp: datetime):
        """Execute a paper trade based on AI decision"""
        trade_size = 1000  # Standard lot size for paper trading
        
        if action == 0:  # Buy
            self.execute_buy(trade_size, current_price, timestamp)
        elif action == 1:  # Sell
            self.execute_sell(trade_size, current_price, timestamp)
        # action == 2 is Hold, do nothing
    
    def execute_buy(self, size: float, price: float, timestamp: datetime):
        """Execute a buy order"""
        if self.current_balance < size:
            self.logger.warning("Insufficient balance for buy order")
            return
        
        trade_id = f"buy_{len(self.trade_history)}"
        
        # Check if we have an open sell position to close
        if 'sell' in self.positions:
            # Close sell position
            sell_pos = self.positions['sell']
            profit = (sell_pos['price'] - price) * sell_pos['size']
            self.close_position('sell', price, timestamp, profit)
        
        # Open new buy position
        self.positions['buy'] = {
            'id': trade_id,
            'size': size,
            'price': price,
            'timestamp': timestamp,
            'type': 'buy'
        }
        
        self.current_balance -= size  # Reserve balance
        self.logger.info(f"Opened BUY position: {size} units at {price}")
    
    def execute_sell(self, size: float, price: float, timestamp: datetime):
        """Execute a sell order"""
        trade_id = f"sell_{len(self.trade_history)}"
        
        # Check if we have an open buy position to close
        if 'buy' in self.positions:
            # Close buy position
            buy_pos = self.positions['buy']
            profit = (price - buy_pos['price']) * buy_pos['size']
            self.close_position('buy', price, timestamp, profit)
        
        # Open new sell position
        self.positions['sell'] = {
            'id': trade_id,
            'size': size,
            'price': price,
            'timestamp': timestamp,
            'type': 'sell'
        }
        
        self.logger.info(f"Opened SELL position: {size} units at {price}")
    
    def close_position(self, position_type: str, close_price: float, timestamp: datetime, profit: float):
        """Close an open position"""
        if position_type not in self.positions:
            return
        
        position = self.positions[position_type]
        
        # Calculate actual profit/loss
        if position_type == 'buy':
            actual_profit = (close_price - position['price']) * position['size']
        else:  # sell
            actual_profit = (position['price'] - close_price) * position['size']
        
        # Update balance
        if position_type == 'buy':
            self.current_balance += position['size']  # Return reserved balance
        self.current_balance += actual_profit
        
        # Record trade
        trade_record = {
            'id': position['id'],
            'type': position_type,
            'open_price': position['price'],
            'close_price': close_price,
            'size': position['size'],
            'profit': actual_profit,
            'open_time': position['timestamp'],
            'close_time': timestamp,
            'duration': (timestamp - position['timestamp']).total_seconds() / 60  # minutes
        }
        
        self.trade_history.append(trade_record)
        self.total_trades += 1
        
        if actual_profit > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        self.total_profit += actual_profit
        
        # Update drawdown tracking
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Remove position
        del self.positions[position_type]
        
        self.logger.info(f"Closed {position_type.upper()} position: Profit=${actual_profit:.2f}, Balance=${self.current_balance:.2f}")
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        # Calculate actual current balance including all profits
        actual_balance = self.initial_balance + self.total_profit
        total_return = (self.total_profit / self.initial_balance * 100) if self.initial_balance > 0 else 0
        
        return {
            'current_balance': round(actual_balance, 2),
            'total_return_pct': round(total_return, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': round(win_rate, 2),
            'total_profit': round(self.total_profit, 2),
            'max_drawdown_pct': round(self.max_drawdown * 100, 2),
            'open_positions': len(self.positions),
            'is_running': self.is_running
        }
    
    def start_paper_trading(self):
        """Start the paper trading system"""
        if not self.load_trained_model():
            return False
        
        # Initialize with some historical data for faster startup
        self._initialize_data_buffer()
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start trading thread
        self.trading_thread = Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        self.logger.info("Paper trading started")
        return True
    
    def _initialize_data_buffer(self):
        """Initialize data buffer with recent historical data"""
        try:
            # Fetch last few days of data to populate buffer
            import yfinance as yf
            ticker = yf.Ticker(self.symbol)
            hist_data = ticker.history(period="5d", interval="1h")
            
            if not hist_data.empty:
                # Convert to our format and take last 20 points
                hist_data.columns = [col.lower() for col in hist_data.columns]
                hist_data = hist_data[['open', 'high', 'low', 'close', 'volume']].copy()
                hist_data['adjclose'] = hist_data['close']
                
                # Add to buffer
                for _, row in hist_data.tail(20).iterrows():
                    self.data_buffer.append(row)
                
                self.logger.info(f"Initialized data buffer with {len(self.data_buffer)} historical points")
                
        except Exception as e:
            self.logger.warning(f"Could not initialize historical data: {e}")
            # Continue without historical data - will build up during trading
    
    def stop_paper_trading(self):
        """Stop the paper trading system"""
        self.is_running = False
        self.stop_event.set()
        
        # Close any open positions
        current_time = datetime.now()
        if self.current_data is not None and not self.current_data.empty:
            current_price = self.current_data['close'].iloc[-1]
            
            for pos_type in list(self.positions.keys()):
                if pos_type == 'buy':
                    profit = (current_price - self.positions[pos_type]['price']) * self.positions[pos_type]['size']
                else:
                    profit = (self.positions[pos_type]['price'] - current_price) * self.positions[pos_type]['size']
                self.close_position(pos_type, current_price, current_time, profit)
        
        self.logger.info("Paper trading stopped")
    
    def _trading_loop(self):
        """Main trading loop"""
        iteration = 0
        while self.is_running and not self.stop_event.is_set():
            try:
                iteration += 1
                self.logger.info(f"Trading loop iteration {iteration}")
                
                # Fetch live data
                live_data = self.fetch_live_data()
                
                if live_data is not None:
                    self.logger.info(f"Fetched live data: {live_data.shape}")
                    
                    # Update data buffer
                    if self.update_data_buffer(live_data):
                        self.logger.info(f"Updated data buffer, current size: {len(self.data_buffer)}")
                        
                        # Calculate features
                        state = self.calculate_features()
                        
                        if state is not None:
                            self.logger.info(f"Calculated features: {state.shape}")
                            
                            # Make trading decision
                            action = self.make_trading_decision(state)
                            self.logger.info(f"AI decision: action={action}")
                            
                            # Execute trade
                            current_price = self.current_data['close'].iloc[-1]
                            self.execute_paper_trade(action, current_price, datetime.now())
                        else:
                            self.logger.warning("Could not calculate features")
                    else:
                        self.logger.warning("Could not update data buffer")
                else:
                    self.logger.warning("Could not fetch live data")
                
                # Wait before next iteration (30 second intervals for more activity)
                self.stop_event.wait(30)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                self.stop_event.wait(60)  # Wait before retrying
    
    def save_results(self, filepath: str = None):
        """Save paper trading results"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"paper_trading_results_{timestamp}.json"
        
        results = {
            'performance': self.get_performance_stats(),
            'trades': self.trade_history,
            'symbol': self.symbol,
            'initial_balance': self.initial_balance,
            'session_start': self.data_buffer[0].name.isoformat() if self.data_buffer else None,
            'session_end': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filepath}")
        return filepath

# Global paper trading instance
paper_trader = None

def get_paper_trader():
    """Get the global paper trading instance"""
    global paper_trader
    if paper_trader is None:
        paper_trader = PaperTradingSystem()
    return paper_trader