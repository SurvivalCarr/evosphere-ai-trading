"""
Native Technical Indicator Calculator
Pure Python implementation without external TA libraries
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicatorCalculatorNative:
    """Technical indicator calculator using native pandas operations."""
    
    def __init__(self, data_df: pd.DataFrame):
        """
        Initialize with market data DataFrame.
        
        Args:
            data_df: DataFrame with OHLCV columns (open, high, low, close, volume)
        """
        self.df = data_df.copy()
        
        # Validate required columns
        required_columns = ['close']
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            # Use close as fallback for missing OHLCV columns
            if 'close' in self.df.columns:
                for col in ['open', 'high', 'low']:
                    if col not in self.df.columns:
                        self.df[col] = self.df['close']
                if 'volume' not in self.df.columns:
                    self.df['volume'] = 0
        
        # Ensure data types are numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
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
            result = self.df['close'].ewm(span=length, adjust=False).mean()
            result.name = f"EMA_{length}"
            return result
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return None
    
    def rsi(self, length: int = 14) -> Optional[pd.Series]:
        """Relative Strength Index."""
        try:
            length = self._validate_length(length)
            
            # Calculate price changes
            delta = self.df['close'].diff()
            
            # Calculate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=length).mean()
            avg_losses = losses.rolling(window=length).mean()
            
            # Calculate RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            rsi.name = f"RSI_{length}"
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return None
    
    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[pd.DataFrame]:
        """MACD (Moving Average Convergence Divergence)."""
        try:
            fast = self._validate_length(fast)
            slow = self._validate_length(slow)
            signal = self._validate_length(signal)
            
            # Calculate EMAs
            ema_fast = self.df['close'].ewm(span=fast).mean()
            ema_slow = self.df['close'].ewm(span=slow).mean()
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            # Create result DataFrame
            result = pd.DataFrame({
                f'MACD_line_{fast}_{slow}_{signal}': macd_line,
                f'MACD_signal_{fast}_{slow}_{signal}': signal_line,
                f'MACD_hist_{fast}_{slow}_{signal}': histogram
            }, index=self.df.index)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return None
    
    def bbands(self, length: int = 20, std: Union[int, float] = 2) -> Optional[pd.DataFrame]:
        """Bollinger Bands."""
        try:
            length = self._validate_length(length)
            
            # Calculate middle band (SMA)
            middle_band = self.df['close'].rolling(window=length).mean()
            
            # Calculate standard deviation
            std_dev = self.df['close'].rolling(window=length).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std_dev * std)
            lower_band = middle_band - (std_dev * std)
            
            # Create result DataFrame
            result = pd.DataFrame({
                f'BB_upper_{length}_{std}': upper_band,
                f'BB_middle_{length}_{std}': middle_band,
                f'BB_lower_{length}_{std}': lower_band
            }, index=self.df.index)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return None
    
    def atr(self, length: int = 14) -> Optional[pd.Series]:
        """Average True Range."""
        try:
            length = self._validate_length(length)
            
            # Calculate True Range components
            high_low = self.df['high'] - self.df['low']
            high_close_prev = np.abs(self.df['high'] - self.df['close'].shift(1))
            low_close_prev = np.abs(self.df['low'] - self.df['close'].shift(1))
            
            # True Range is the maximum of the three
            true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            
            # Average True Range
            atr_values = true_range.rolling(window=length).mean()
            atr_values.name = f"ATR_{length}"
            
            return atr_values
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return None
    
    def stoch(self, k: int = 14, d: int = 3, smooth_k: int = 3) -> Optional[pd.DataFrame]:
        """Stochastic Oscillator."""
        try:
            k = self._validate_length(k)
            d = self._validate_length(d)
            smooth_k = self._validate_length(smooth_k)
            
            # Calculate %K
            lowest_low = self.df['low'].rolling(window=k).min()
            highest_high = self.df['high'].rolling(window=k).max()
            
            k_percent = ((self.df['close'] - lowest_low) / (highest_high - lowest_low)) * 100
            
            # Smooth %K
            k_percent_smooth = k_percent.rolling(window=smooth_k).mean()
            
            # Calculate %D
            d_percent = k_percent_smooth.rolling(window=d).mean()
            
            # Create result DataFrame
            result = pd.DataFrame({
                f'STOCH_K_{k}_{d}': k_percent_smooth,
                f'STOCH_D_{k}_{d}': d_percent
            }, index=self.df.index)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return None
    
    def cci(self, length: int = 20) -> Optional[pd.Series]:
        """Commodity Channel Index."""
        try:
            length = self._validate_length(length)
            
            # Calculate Typical Price
            typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
            
            # Calculate SMA of Typical Price
            sma_tp = typical_price.rolling(window=length).mean()
            
            # Calculate Mean Absolute Deviation
            mad = typical_price.rolling(window=length).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True
            )
            
            # Calculate CCI
            cci_values = (typical_price - sma_tp) / (0.015 * mad)
            cci_values.name = f"CCI_{length}"
            
            return cci_values
            
        except Exception as e:
            logger.error(f"Error calculating CCI: {e}")
            return None
    
    def willr(self, length: int = 14) -> Optional[pd.Series]:
        """Williams %R."""
        try:
            length = self._validate_length(length)
            
            # Calculate highest high and lowest low over the period
            highest_high = self.df['high'].rolling(window=length).max()
            lowest_low = self.df['low'].rolling(window=length).min()
            
            # Calculate Williams %R
            willr_values = ((highest_high - self.df['close']) / (highest_high - lowest_low)) * -100
            willr_values.name = f"WILLIAMS_R_{length}"
            
            return willr_values
            
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {e}")
            return None
    
    def adx(self, length: int = 14) -> Optional[pd.Series]:
        """Average Directional Index (simplified version)."""
        try:
            length = self._validate_length(length)
            
            # Calculate directional movements
            high_diff = self.df['high'].diff()
            low_diff = self.df['low'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            # Calculate True Range
            tr1 = self.df['high'] - self.df['low']
            tr2 = np.abs(self.df['high'] - self.df['close'].shift(1))
            tr3 = np.abs(self.df['low'] - self.df['close'].shift(1))
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calculate smoothed values
            plus_dm_smooth = pd.Series(plus_dm).rolling(window=length).mean()
            minus_dm_smooth = pd.Series(minus_dm).rolling(window=length).mean()
            tr_smooth = pd.Series(true_range).rolling(window=length).mean()
            
            # Calculate DI+ and DI-
            plus_di = (plus_dm_smooth / tr_smooth) * 100
            minus_di = (minus_dm_smooth / tr_smooth) * 100
            
            # Calculate DX
            dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            
            # Calculate ADX
            adx_values = dx.rolling(window=length).mean()
            adx_values.name = f"ADX_{length}"
            
            return adx_values
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return None
    
    def obv(self) -> Optional[pd.Series]:
        """On-Balance Volume."""
        try:
            # Calculate price changes
            price_change = self.df['close'].diff()
            
            # Calculate OBV
            obv_values = np.where(price_change > 0, self.df['volume'],
                                np.where(price_change < 0, -self.df['volume'], 0))
            
            obv_cumulative = pd.Series(obv_values, index=self.df.index).cumsum()
            obv_cumulative.name = "OBV"
            
            return obv_cumulative
            
        except Exception as e:
            logger.error(f"Error calculating OBV: {e}")
            return None
    
    def vwap(self) -> Optional[pd.Series]:
        """Volume Weighted Average Price."""
        try:
            # Calculate typical price
            typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
            
            # Calculate VWAP
            vwap_values = (typical_price * self.df['volume']).cumsum() / self.df['volume'].cumsum()
            vwap_values.name = "VWAP"
            
            return vwap_values
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return None
    
    def momentum(self, length: int = 10) -> Optional[pd.Series]:
        """Momentum."""
        try:
            length = self._validate_length(length)
            
            momentum_values = self.df['close'] - self.df['close'].shift(length)
            momentum_values.name = f"MOMENTUM_{length}"
            
            return momentum_values
            
        except Exception as e:
            logger.error(f"Error calculating Momentum: {e}")
            return None
    
    def roc(self, length: int = 10) -> Optional[pd.Series]:
        """Rate of Change."""
        try:
            length = self._validate_length(length)
            
            roc_values = ((self.df['close'] - self.df['close'].shift(length)) / 
                         self.df['close'].shift(length)) * 100
            roc_values.name = f"ROC_{length}"
            
            return roc_values
            
        except Exception as e:
            logger.error(f"Error calculating ROC: {e}")
            return None
    
    def get_all_indicators(self) -> pd.DataFrame:
        """Get all calculated indicators."""
        try:
            all_indicators = pd.DataFrame(index=self.df.index)
            
            # Add basic indicators
            indicators_to_calculate = [
                ('SMA_20', lambda: self.sma(20)),
                ('EMA_20', lambda: self.ema(20)),
                ('RSI_14', lambda: self.rsi(14)),
                ('ATR_14', lambda: self.atr(14)),
                ('CCI_20', lambda: self.cci(20)),
                ('WILLIAMS_R_14', lambda: self.willr(14)),
                ('ADX_14', lambda: self.adx(14)),
                ('OBV', lambda: self.obv()),
                ('VWAP', lambda: self.vwap()),
                ('MOMENTUM_10', lambda: self.momentum(10)),
                ('ROC_10', lambda: self.roc(10))
            ]
            
            for name, calc_func in indicators_to_calculate:
                try:
                    result = calc_func()
                    if result is not None:
                        all_indicators[name] = result
                except Exception as e:
                    logger.warning(f"Failed to calculate {name}: {e}")
            
            # Add MACD indicators
            try:
                macd_df = self.macd()
                if macd_df is not None:
                    for col in macd_df.columns:
                        all_indicators[col] = macd_df[col]
            except Exception as e:
                logger.warning(f"Failed to calculate MACD: {e}")
            
            # Add Bollinger Bands
            try:
                bb_df = self.bbands()
                if bb_df is not None:
                    for col in bb_df.columns:
                        all_indicators[col] = bb_df[col]
            except Exception as e:
                logger.warning(f"Failed to calculate Bollinger Bands: {e}")
            
            # Add Stochastic
            try:
                stoch_df = self.stoch()
                if stoch_df is not None:
                    for col in stoch_df.columns:
                        all_indicators[col] = stoch_df[col]
            except Exception as e:
                logger.warning(f"Failed to calculate Stochastic: {e}")
            
            logger.info(f"Calculated {len(all_indicators.columns)} technical indicators")
            return all_indicators
            
        except Exception as e:
            logger.error(f"Error calculating all indicators: {e}")
            return pd.DataFrame(index=self.df.index)
    
    def bb_position(self, length: int = 20, std: Union[int, float] = 2) -> Optional[pd.Series]:
        """Bollinger Band Position - where price sits within bands (0=lower, 0.5=middle, 1=upper)."""
        try:
            bb_data = self.bbands(length, std)
            if bb_data is None or bb_data.empty:
                return None
            
            # Calculate position relative to bands
            upper = bb_data['BB_upper']
            lower = bb_data['BB_lower']
            close = self.df['close']
            
            position = (close - lower) / (upper - lower)
            position = position.fillna(0.5).clip(0, 1)
            position.name = f'BB_POSITION_{length}_{std}'
            return position
            
        except Exception as e:
            logger.error(f"Error calculating BB position: {e}")
            return None
    
    def momentum_composite(self, rsi_length: int = 14, mom_length: int = 10) -> Optional[pd.Series]:
        """Multi-indicator momentum signal combining RSI and Momentum."""
        try:
            rsi = self.rsi(rsi_length)
            momentum = self.momentum(mom_length)
            
            if rsi is None or momentum is None:
                return None
            
            # Normalize RSI to -1 to 1 range
            rsi_norm = (rsi - 50) / 50
            
            # Normalize momentum using rolling standard deviation
            momentum_std = momentum.rolling(20).std()
            momentum_norm = momentum / momentum_std.where(momentum_std > 0, 1)
            momentum_norm = momentum_norm.fillna(0).clip(-3, 3) / 3  # Clip and normalize
            
            # Combine signals
            composite = (rsi_norm + momentum_norm) / 2
            composite.name = f'MOMENTUM_COMPOSITE_{rsi_length}_{mom_length}'
            return composite
            
        except Exception as e:
            logger.error(f"Error calculating momentum composite: {e}")
            return None
    
    def volatility_breakout(self, atr_length: int = 14, threshold: float = 1.5) -> Optional[pd.Series]:
        """Volatility spike detection using ATR."""
        try:
            atr = self.atr(atr_length)
            if atr is None:
                return None
            
            # Calculate ATR moving average
            atr_ma = atr.rolling(20).mean()
            
            # Detect breakouts (when ATR > threshold * average ATR)
            breakout = (atr > (atr_ma * threshold)).astype(int)
            breakout.name = f'VOLATILITY_BREAKOUT_{atr_length}_{threshold}'
            return breakout
            
        except Exception as e:
            logger.error(f"Error calculating volatility breakout: {e}")
            return None
    
    def trend_alignment_short(self, short_ema: int = 5, long_ema: int = 20) -> Optional[pd.Series]:
        """Short-term trend confirmation using EMA alignment."""
        try:
            ema_short = self.ema(short_ema)
            ema_long = self.ema(long_ema)
            
            if ema_short is None or ema_long is None:
                return None
            
            # Calculate trend alignment: 1 = uptrend, -1 = downtrend, 0 = sideways
            alignment = pd.Series(0, index=ema_short.index)
            alignment[ema_short > ema_long] = 1
            alignment[ema_short < ema_long] = -1
            
            alignment.name = f'TREND_ALIGNMENT_SHORT_{short_ema}_{long_ema}'
            return alignment
            
        except Exception as e:
            logger.error(f"Error calculating trend alignment: {e}")
            return None
    
    def get_optimal_8_features(self) -> pd.DataFrame:
        """Get the 8 optimal trading markers that achieved 65% returns in your original system."""
        try:
            features_dict = {}
            
            # 1. SMA_20 - Trend direction
            sma_20 = self.sma(20)
            if sma_20 is not None:
                features_dict['SMA_20'] = sma_20
            
            # 2. RSI_14 - Momentum strength
            rsi_14 = self.rsi(14)
            if rsi_14 is not None:
                features_dict['RSI_14'] = rsi_14
            
            # 3. MACD_signal - Trend momentum confirmation
            macd_data = self.macd(12, 26, 9)
            if macd_data is not None and 'MACD_signal' in macd_data.columns:
                features_dict['MACD_signal'] = macd_data['MACD_signal']
            
            # 4. ATR_14 - Volatility measurement
            atr_14 = self.atr(14)
            if atr_14 is not None:
                features_dict['ATR_14'] = atr_14
            
            # 5. BB_POSITION - Price position within Bollinger Bands
            bb_pos = self.bb_position(20, 2)
            if bb_pos is not None:
                features_dict['BB_POSITION'] = bb_pos
            
            # 6. MOMENTUM_COMPOSITE - Multi-indicator momentum signal
            mom_comp = self.momentum_composite(14, 10)
            if mom_comp is not None:
                features_dict['MOMENTUM_COMPOSITE'] = mom_comp
            
            # 7. VOLATILITY_BREAKOUT - Volatility spike detection
            vol_break = self.volatility_breakout(14, 1.5)
            if vol_break is not None:
                features_dict['VOLATILITY_BREAKOUT'] = vol_break
            
            # 8. TREND_ALIGNMENT_SHORT - Short-term trend confirmation
            trend_align = self.trend_alignment_short(5, 20)
            if trend_align is not None:
                features_dict['TREND_ALIGNMENT_SHORT'] = trend_align
            
            if not features_dict:
                logger.warning("No optimal features could be calculated")
                return pd.DataFrame()
            
            # Combine all features and handle missing values
            features_df = pd.DataFrame(features_dict)
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            logger.info(f"Generated {len(features_df.columns)} optimal features: {list(features_df.columns)}")
            return features_df
            
        except Exception as e:
            logger.error(f"Error generating optimal 8 features: {e}")
            return pd.DataFrame()

    def ichimoku(self, conversion_periods: int = 9, base_periods: int = 26, 
                 lagging_span_periods: int = 52, displacement: int = 26) -> Optional[pd.DataFrame]:
        """
        Ichimoku Kinko Hyo (Ichimoku Cloud) - Advanced trend and momentum indicator.
        
        Args:
            conversion_periods: Periods for Tenkan-sen (Conversion Line)
            base_periods: Periods for Kijun-sen (Base Line)
            lagging_span_periods: Periods for Chikou Span (Lagging Span)
            displacement: Displacement for Senkou Span (Leading Span)
            
        Returns:
            DataFrame with Ichimoku components
        """
        try:
            high = self.df['high']
            low = self.df['low']
            close = self.df['close']
            
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
            tenkan_sen = (high.rolling(window=conversion_periods).max() + 
                         low.rolling(window=conversion_periods).min()) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
            kijun_sen = (high.rolling(window=base_periods).max() + 
                        low.rolling(window=base_periods).min()) / 2
            
            # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
            senkou_span_b = ((high.rolling(window=lagging_span_periods).max() + 
                             low.rolling(window=lagging_span_periods).min()) / 2).shift(displacement)
            
            # Chikou Span (Lagging Span): Close price shifted back
            chikou_span = close.shift(-displacement)
            
            # Cloud thickness (Kumo): Distance between Senkou Spans
            cloud_thickness = senkou_span_a - senkou_span_b
            
            # Price position relative to cloud
            price_above_cloud = (close > senkou_span_a) & (close > senkou_span_b)
            price_below_cloud = (close < senkou_span_a) & (close < senkou_span_b)
            price_in_cloud = ~(price_above_cloud | price_below_cloud)
            
            # Bullish/Bearish cloud (Senkou A vs Senkou B)
            bullish_cloud = senkou_span_a > senkou_span_b
            
            # Create results DataFrame
            ichimoku_df = pd.DataFrame({
                'ICHIMOKU_TENKAN': tenkan_sen,
                'ICHIMOKU_KIJUN': kijun_sen,
                'ICHIMOKU_SENKOU_A': senkou_span_a,
                'ICHIMOKU_SENKOU_B': senkou_span_b,
                'ICHIMOKU_CHIKOU': chikou_span,
                'ICHIMOKU_CLOUD_THICKNESS': cloud_thickness,
                'ICHIMOKU_PRICE_ABOVE_CLOUD': price_above_cloud.astype(int),
                'ICHIMOKU_PRICE_IN_CLOUD': price_in_cloud.astype(int),
                'ICHIMOKU_PRICE_BELOW_CLOUD': price_below_cloud.astype(int),
                'ICHIMOKU_BULLISH_CLOUD': bullish_cloud.astype(int)
            }, index=self.df.index)
            
            logger.info("Calculated Ichimoku Cloud indicators")
            return ichimoku_df
            
        except Exception as e:
            logger.error(f"Error calculating Ichimoku: {e}")
            return None
    
    def fibonacci_retracement(self, lookback_periods: int = 100, 
                             levels: list = [0.236, 0.382, 0.5, 0.618, 0.786]) -> Optional[pd.DataFrame]:
        """
        Fibonacci Retracement Levels - Key support and resistance levels.
        
        Args:
            lookback_periods: Periods to look back for swing high/low
            levels: Fibonacci ratios to calculate
            
        Returns:
            DataFrame with Fibonacci levels and price positions
        """
        try:
            high = self.df['high']
            low = self.df['low']
            close = self.df['close']
            
            # Calculate rolling swing highs and lows
            swing_high = high.rolling(window=lookback_periods).max()
            swing_low = low.rolling(window=lookback_periods).min()
            
            # Calculate Fibonacci levels
            fib_range = swing_high - swing_low
            
            fib_data = {}
            
            # Calculate each Fibonacci level
            for level in levels:
                # Uptrend retracement (from swing low)
                fib_level_up = swing_low + (fib_range * level)
                # Downtrend retracement (from swing high)
                fib_level_down = swing_high - (fib_range * level)
                
                fib_data[f'FIB_{int(level*1000)}_UP'] = fib_level_up
                fib_data[f'FIB_{int(level*1000)}_DOWN'] = fib_level_down
                
                # Price position relative to Fibonacci levels
                above_fib_up = close > fib_level_up
                below_fib_down = close < fib_level_down
                
                fib_data[f'FIB_{int(level*1000)}_ABOVE_UP'] = above_fib_up.astype(int)
                fib_data[f'FIB_{int(level*1000)}_BELOW_DOWN'] = below_fib_down.astype(int)
            
            # Calculate price position in Fibonacci zones
            # Zone 1: 0-23.6%, Zone 2: 23.6-38.2%, Zone 3: 38.2-61.8%, Zone 4: 61.8-100%
            price_pct_retracement = (close - swing_low) / fib_range
            
            fib_zone_1 = (price_pct_retracement >= 0) & (price_pct_retracement < 0.236)
            fib_zone_2 = (price_pct_retracement >= 0.236) & (price_pct_retracement < 0.382)
            fib_zone_3 = (price_pct_retracement >= 0.382) & (price_pct_retracement < 0.618)
            fib_zone_4 = (price_pct_retracement >= 0.618) & (price_pct_retracement <= 1.0)
            
            fib_data.update({
                'FIB_SWING_HIGH': swing_high,
                'FIB_SWING_LOW': swing_low,
                'FIB_RANGE': fib_range,
                'FIB_PCT_RETRACEMENT': price_pct_retracement,
                'FIB_ZONE_1': fib_zone_1.astype(int),
                'FIB_ZONE_2': fib_zone_2.astype(int),
                'FIB_ZONE_3': fib_zone_3.astype(int),
                'FIB_ZONE_4': fib_zone_4.astype(int)
            })
            
            fib_df = pd.DataFrame(fib_data, index=self.df.index)
            
            logger.info(f"Calculated Fibonacci retracement levels: {levels}")
            return fib_df
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci retracement: {e}")
            return None
    
    def fibonacci_extensions(self, lookback_periods: int = 100,
                           extension_levels: list = [1.272, 1.414, 1.618, 2.0, 2.618]) -> Optional[pd.DataFrame]:
        """
        Fibonacci Extension Levels - Price targets beyond 100% retracement.
        
        Args:
            lookback_periods: Periods to look back for swing points
            extension_levels: Fibonacci extension ratios
            
        Returns:
            DataFrame with Fibonacci extension levels
        """
        try:
            high = self.df['high']
            low = self.df['low']
            close = self.df['close']
            
            # Calculate swing points
            swing_high = high.rolling(window=lookback_periods).max()
            swing_low = low.rolling(window=lookback_periods).min()
            fib_range = swing_high - swing_low
            
            ext_data = {}
            
            # Calculate extension levels
            for level in extension_levels:
                # Uptrend extensions (beyond swing high)
                ext_level_up = swing_high + (fib_range * (level - 1))
                # Downtrend extensions (beyond swing low)
                ext_level_down = swing_low - (fib_range * (level - 1))
                
                ext_data[f'FIB_EXT_{int(level*1000)}_UP'] = ext_level_up
                ext_data[f'FIB_EXT_{int(level*1000)}_DOWN'] = ext_level_down
                
                # Price reaching extension levels
                reached_ext_up = close >= ext_level_up
                reached_ext_down = close <= ext_level_down
                
                ext_data[f'FIB_EXT_{int(level*1000)}_REACHED_UP'] = reached_ext_up.astype(int)
                ext_data[f'FIB_EXT_{int(level*1000)}_REACHED_DOWN'] = reached_ext_down.astype(int)
            
            ext_df = pd.DataFrame(ext_data, index=self.df.index)
            
            logger.info(f"Calculated Fibonacci extension levels: {extension_levels}")
            return ext_df
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci extensions: {e}")
            return None
    
    def pivot_points(self, method: str = 'standard') -> Optional[pd.DataFrame]:
        """
        Pivot Points - Support and resistance levels based on previous period's OHLC.
        
        Args:
            method: Calculation method ('standard', 'woodie', 'camarilla', 'fibonacci')
            
        Returns:
            DataFrame with pivot point levels
        """
        try:
            high = self.df['high']
            low = self.df['low']
            close = self.df['close']
            
            # Calculate previous period's values (shift by 1)
            prev_high = high.shift(1)
            prev_low = low.shift(1)
            prev_close = close.shift(1)
            
            if method == 'standard':
                # Standard Pivot Points
                pivot = (prev_high + prev_low + prev_close) / 3
                
                support_1 = (2 * pivot) - prev_high
                support_2 = pivot - (prev_high - prev_low)
                support_3 = prev_low - 2 * (prev_high - pivot)
                
                resistance_1 = (2 * pivot) - prev_low
                resistance_2 = pivot + (prev_high - prev_low)
                resistance_3 = prev_high + 2 * (pivot - prev_low)
                
            elif method == 'woodie':
                # Woodie's Pivot Points
                pivot = (prev_high + prev_low + 2 * prev_close) / 4
                
                support_1 = (2 * pivot) - prev_high
                support_2 = pivot - (prev_high - prev_low)
                
                resistance_1 = (2 * pivot) - prev_low
                resistance_2 = pivot + (prev_high - prev_low)
                
                support_3 = support_1 - (prev_high - prev_low)
                resistance_3 = resistance_1 + (prev_high - prev_low)
                
            elif method == 'camarilla':
                # Camarilla Pivot Points
                pivot = (prev_high + prev_low + prev_close) / 3
                
                support_1 = prev_close - 1.1 * (prev_high - prev_low) / 12
                support_2 = prev_close - 1.1 * (prev_high - prev_low) / 6
                support_3 = prev_close - 1.1 * (prev_high - prev_low) / 4
                
                resistance_1 = prev_close + 1.1 * (prev_high - prev_low) / 12
                resistance_2 = prev_close + 1.1 * (prev_high - prev_low) / 6
                resistance_3 = prev_close + 1.1 * (prev_high - prev_low) / 4
                
            elif method == 'fibonacci':
                # Fibonacci Pivot Points
                pivot = (prev_high + prev_low + prev_close) / 3
                
                support_1 = pivot - 0.382 * (prev_high - prev_low)
                support_2 = pivot - 0.618 * (prev_high - prev_low)
                support_3 = pivot - 1.000 * (prev_high - prev_low)
                
                resistance_1 = pivot + 0.382 * (prev_high - prev_low)
                resistance_2 = pivot + 0.618 * (prev_high - prev_low)
                resistance_3 = pivot + 1.000 * (prev_high - prev_low)
                
            else:
                raise ValueError(f"Unknown pivot point method: {method}")
            
            # Create DataFrame
            pivot_data = {
                f'PIVOT_{method.upper()}_PP': pivot,
                f'PIVOT_{method.upper()}_S1': support_1,
                f'PIVOT_{method.upper()}_S2': support_2,
                f'PIVOT_{method.upper()}_S3': support_3,
                f'PIVOT_{method.upper()}_R1': resistance_1,
                f'PIVOT_{method.upper()}_R2': resistance_2,
                f'PIVOT_{method.upper()}_R3': resistance_3,
            }
            
            # Add price position relative to levels
            current_close = close
            pivot_data[f'PIVOT_{method.upper()}_ABOVE_PP'] = (current_close > pivot).astype(int)
            pivot_data[f'PIVOT_{method.upper()}_ABOVE_R1'] = (current_close > resistance_1).astype(int)
            pivot_data[f'PIVOT_{method.upper()}_BELOW_S1'] = (current_close < support_1).astype(int)
            
            pivot_df = pd.DataFrame(pivot_data, index=self.df.index)
            
            logger.info(f"Calculated {method} pivot points")
            return pivot_df
            
        except Exception as e:
            logger.error(f"Error calculating pivot points: {e}")
            return None

    def get_available_indicators(self) -> list:
        """Get list of available indicator methods."""
        return [
            'sma', 'ema', 'rsi', 'macd', 'bbands', 'atr', 'stoch',
            'cci', 'willr', 'adx', 'obv', 'vwap', 'momentum', 'roc',
            'bb_position', 'momentum_composite', 'volatility_breakout', 
            'trend_alignment_short', 'ichimoku', 'fibonacci_retracement',
            'fibonacci_extensions', 'pivot_points', 'get_optimal_8_features'
        ]