import pandas as pd
import numpy as np
import logging
from typing import Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use fallback implementations for technical indicators
# This avoids compatibility issues with pandas_ta
ta = None
logger.info("Using native technical indicator implementations")

class TechnicalIndicatorCalculatorPandasTa:
    """Technical indicator calculator using pandas_ta library."""
    
    def __init__(self, data_df: pd.DataFrame):
        """
        Initialize with market data DataFrame.
        
        Args:
            data_df: DataFrame with OHLCV columns (open, high, low, close, volume)
        """
        self.df = data_df.copy()
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}. Some indicators may not work.")
        
        # Ensure data types are numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Add pandas_ta indicators to DataFrame if available
        if ta is not None:
            try:
                self.df.ta.add_all_indicators()
            except Exception as e:
                logger.warning(f"Failed to add all indicators: {e}")
                pass
        
        logger.info(f"TechnicalIndicatorCalculator initialized with {len(self.df)} rows")
    
    def _validate_length(self, length: int) -> int:
        """Validate and adjust length parameter."""
        if length <= 0:
            logger.warning(f"Invalid length {length}, using default 14")
            return 14
        if length >= len(self.df):
            logger.warning(f"Length {length} too large for data size {len(self.df)}, using {len(self.df)//2}")
            return max(1, len(self.df)//2)
        return length
    
    def sma(self, length: int = 20) -> Optional[pd.Series]:
        """Simple Moving Average."""
        try:
            length = self._validate_length(length)
            if ta is not None:
                result = ta.sma(self.df['close'], length=length)
                if result is not None:
                    result.name = f"SMA_{length}"
                return result
            else:
                # Fallback implementation
                result = self.df['close'].rolling(window=length).mean()
                result.name = f"SMA_{length}"
                return result
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return None
    
    def ema(self, length: int = 20) -> Optional[pd.Series]:
        """Exponential Moving Average."""
        try:
            length = self._validate_length(length)
            result = ta.ema(self.df['close'], length=length)
            if result is not None:
                result.name = f"EMA_{length}"
            return result
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return None
    
    def rsi(self, length: int = 14) -> Optional[pd.Series]:
        """Relative Strength Index."""
        try:
            length = self._validate_length(length)
            result = ta.rsi(self.df['close'], length=length)
            if result is not None:
                result.name = f"RSI_{length}"
            return result
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return None
    
    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[pd.DataFrame]:
        """MACD (Moving Average Convergence Divergence)."""
        try:
            fast = self._validate_length(fast)
            slow = self._validate_length(slow)
            signal = self._validate_length(signal)
            
            if fast >= slow:
                logger.warning(f"Fast period {fast} should be less than slow period {slow}")
                fast = max(1, slow - 1)
            
            result = ta.macd(self.df['close'], fast=fast, slow=slow, signal=signal)
            return result
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return None
    
    def bbands(self, length: int = 20, std: Union[int, float] = 2) -> Optional[pd.DataFrame]:
        """Bollinger Bands."""
        try:
            length = self._validate_length(length)
            if std <= 0:
                std = 2
            
            result = ta.bbands(self.df['close'], length=length, std=std)
            return result
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return None
    
    def atr(self, length: int = 14) -> Optional[pd.Series]:
        """Average True Range."""
        try:
            length = self._validate_length(length)
            result = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=length)
            if result is not None:
                result.name = f"ATR_{length}"
            return result
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return None
    
    def stoch(self, k: int = 14, d: int = 3, smooth_k: int = 3) -> Optional[pd.DataFrame]:
        """Stochastic Oscillator."""
        try:
            k = self._validate_length(k)
            d = self._validate_length(d)
            smooth_k = self._validate_length(smooth_k)
            
            result = ta.stoch(self.df['high'], self.df['low'], self.df['close'], 
                            k=k, d=d, smooth_k=smooth_k)
            return result
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return None
    
    def cci(self, length: int = 20) -> Optional[pd.Series]:
        """Commodity Channel Index."""
        try:
            length = self._validate_length(length)
            result = ta.cci(self.df['high'], self.df['low'], self.df['close'], length=length)
            if result is not None:
                result.name = f"CCI_{length}"
            return result
        except Exception as e:
            logger.error(f"Error calculating CCI: {e}")
            return None
    
    def willr(self, length: int = 14) -> Optional[pd.Series]:
        """Williams %R."""
        try:
            length = self._validate_length(length)
            result = ta.willr(self.df['high'], self.df['low'], self.df['close'], length=length)
            if result is not None:
                result.name = f"WILLR_{length}"
            return result
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {e}")
            return None
    
    def adx(self, length: int = 14) -> Optional[pd.Series]:
        """Average Directional Index."""
        try:
            length = self._validate_length(length)
            result = ta.adx(self.df['high'], self.df['low'], self.df['close'], length=length)
            if result is not None and hasattr(result, 'iloc'):
                # ADX returns a DataFrame, get the ADX column
                adx_cols = [col for col in result.columns if 'ADX_' in col]
                if adx_cols:
                    adx_series = result[adx_cols[0]]
                    adx_series.name = f"ADX_{length}"
                    return adx_series
            return None
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return None
    
    def obv(self) -> Optional[pd.Series]:
        """On-Balance Volume."""
        try:
            result = ta.obv(self.df['close'], self.df['volume'])
            if result is not None:
                result.name = "OBV"
            return result
        except Exception as e:
            logger.error(f"Error calculating OBV: {e}")
            return None
    
    def vwap(self) -> Optional[pd.Series]:
        """Volume Weighted Average Price."""
        try:
            result = ta.vwap(self.df['high'], self.df['low'], self.df['close'], self.df['volume'])
            if result is not None:
                result.name = "VWAP"
            return result
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return None
    
    def momentum(self, length: int = 10) -> Optional[pd.Series]:
        """Momentum."""
        try:
            length = self._validate_length(length)
            result = ta.mom(self.df['close'], length=length)
            if result is not None:
                result.name = f"MOM_{length}"
            return result
        except Exception as e:
            logger.error(f"Error calculating Momentum: {e}")
            return None
    
    def roc(self, length: int = 10) -> Optional[pd.Series]:
        """Rate of Change."""
        try:
            length = self._validate_length(length)
            result = ta.roc(self.df['close'], length=length)
            if result is not None:
                result.name = f"ROC_{length}"
            return result
        except Exception as e:
            logger.error(f"Error calculating ROC: {e}")
            return None
    
    def get_all_indicators(self) -> pd.DataFrame:
        """Get all calculated indicators."""
        return self.df
    
    def get_available_indicators(self) -> list:
        """Get list of available indicator methods."""
        indicators = []
        for attr in dir(self):
            if not attr.startswith('_') and callable(getattr(self, attr)) and attr not in ['get_all_indicators', 'get_available_indicators']:
                indicators.append(attr)
        return indicators
