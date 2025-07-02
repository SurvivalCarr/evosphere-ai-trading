"""
Trading Pairs Configuration
Comprehensive support for multiple markets: Forex, Crypto, Stocks, Commodities
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MarketType(Enum):
    FOREX = "forex"
    CRYPTO = "crypto"
    STOCKS = "stocks"
    COMMODITIES = "commodities"
    INDICES = "indices"

@dataclass
class TradingPair:
    symbol: str
    yahoo_symbol: str
    market_type: MarketType
    base_currency: str
    quote_currency: str
    min_trade_size: float
    pip_size: float
    spread_estimate: float  # Typical spread in pips/points
    volatility_factor: float  # Relative volatility (1.0 = normal)
    trading_hours: str
    description: str
    binance_symbol: str = None
    coinbase_symbol: str = None
    kraken_symbol: str = None

class TradingPairsManager:
    """Manages all available trading pairs across different markets"""
    
    def __init__(self):
        self.pairs = self._initialize_trading_pairs()
        
    def _initialize_trading_pairs(self) -> Dict[str, TradingPair]:
        """Initialize comprehensive list of trading pairs"""
        pairs = {}
        
        # Major Forex Pairs
        forex_pairs = [
            TradingPair("EURUSD", "EURUSD=X", MarketType.FOREX, "EUR", "USD", 0.01, 0.0001, 1.2, 0.8, "24/5", "Euro/US Dollar"),
            TradingPair("GBPUSD", "GBPUSD=X", MarketType.FOREX, "GBP", "USD", 0.01, 0.0001, 1.5, 1.0, "24/5", "British Pound/US Dollar"),
            TradingPair("USDJPY", "USDJPY=X", MarketType.FOREX, "USD", "JPY", 0.01, 0.01, 1.0, 0.9, "24/5", "US Dollar/Japanese Yen"),
            TradingPair("USDCHF", "USDCHF=X", MarketType.FOREX, "USD", "CHF", 0.01, 0.0001, 1.3, 0.9, "24/5", "US Dollar/Swiss Franc"),
            TradingPair("AUDUSD", "AUDUSD=X", MarketType.FOREX, "AUD", "USD", 0.01, 0.0001, 1.4, 1.1, "24/5", "Australian Dollar/US Dollar"),
            TradingPair("NZDUSD", "NZDUSD=X", MarketType.FOREX, "NZD", "USD", 0.01, 0.0001, 1.8, 1.2, "24/5", "New Zealand Dollar/US Dollar"),
            TradingPair("USDCAD", "USDCAD=X", MarketType.FOREX, "USD", "CAD", 0.01, 0.0001, 1.5, 0.8, "24/5", "US Dollar/Canadian Dollar"),
            TradingPair("EURGBP", "EURGBP=X", MarketType.FOREX, "EUR", "GBP", 0.01, 0.0001, 1.2, 0.7, "24/5", "Euro/British Pound"),
            TradingPair("EURJPY", "EURJPY=X", MarketType.FOREX, "EUR", "JPY", 0.01, 0.01, 1.8, 1.3, "24/5", "Euro/Japanese Yen"),
            TradingPair("GBPJPY", "GBPJPY=X", MarketType.FOREX, "GBP", "JPY", 0.01, 0.01, 2.5, 1.5, "24/5", "British Pound/Japanese Yen"),
            TradingPair("CHFJPY", "CHFJPY=X", MarketType.FOREX, "CHF", "JPY", 0.01, 0.01, 2.0, 1.2, "24/5", "Swiss Franc/Japanese Yen"),
            TradingPair("AUDNZD", "AUDNZD=X", MarketType.FOREX, "AUD", "NZD", 0.01, 0.0001, 2.5, 1.3, "24/5", "Australian Dollar/New Zealand Dollar"),
            TradingPair("AUDCAD", "AUDCAD=X", MarketType.FOREX, "AUD", "CAD", 0.01, 0.0001, 2.0, 1.2, "24/5", "Australian Dollar/Canadian Dollar"),
            TradingPair("AUDCHF", "AUDCHF=X", MarketType.FOREX, "AUD", "CHF", 0.01, 0.0001, 2.3, 1.4, "24/5", "Australian Dollar/Swiss Franc"),
            TradingPair("AUDJPY", "AUDJPY=X", MarketType.FOREX, "AUD", "JPY", 0.01, 0.01, 2.2, 1.4, "24/5", "Australian Dollar/Japanese Yen"),
            TradingPair("CADJPY", "CADJPY=X", MarketType.FOREX, "CAD", "JPY", 0.01, 0.01, 2.5, 1.3, "24/5", "Canadian Dollar/Japanese Yen"),
            TradingPair("EURCHF", "EURCHF=X", MarketType.FOREX, "EUR", "CHF", 0.01, 0.0001, 1.5, 0.6, "24/5", "Euro/Swiss Franc"),
            TradingPair("EURAUD", "EURAUD=X", MarketType.FOREX, "EUR", "AUD", 0.01, 0.0001, 2.2, 1.3, "24/5", "Euro/Australian Dollar"),
            TradingPair("EURCAD", "EURCAD=X", MarketType.FOREX, "EUR", "CAD", 0.01, 0.0001, 2.0, 1.1, "24/5", "Euro/Canadian Dollar"),
            TradingPair("EURNZD", "EURNZD=X", MarketType.FOREX, "EUR", "NZD", 0.01, 0.0001, 3.0, 1.5, "24/5", "Euro/New Zealand Dollar"),
            TradingPair("GBPAUD", "GBPAUD=X", MarketType.FOREX, "GBP", "AUD", 0.01, 0.0001, 3.5, 1.6, "24/5", "British Pound/Australian Dollar"),
            TradingPair("GBPCAD", "GBPCAD=X", MarketType.FOREX, "GBP", "CAD", 0.01, 0.0001, 3.2, 1.5, "24/5", "British Pound/Canadian Dollar"),
            TradingPair("GBPCHF", "GBPCHF=X", MarketType.FOREX, "GBP", "CHF", 0.01, 0.0001, 3.0, 1.4, "24/5", "British Pound/Swiss Franc"),
            TradingPair("GBPNZD", "GBPNZD=X", MarketType.FOREX, "GBP", "NZD", 0.01, 0.0001, 4.0, 1.8, "24/5", "British Pound/New Zealand Dollar"),
        ]
        
        # Major Cryptocurrencies
        crypto_pairs = [
            TradingPair("BTCUSD", "BTC-USD", MarketType.CRYPTO, "BTC", "USD", 0.001, 1.0, 50.0, 4.0, "24/7", "Bitcoin/US Dollar", "BTCUSDT", "BTC-USD", "BTCUSD"),
            TradingPair("ETHUSD", "ETH-USD", MarketType.CRYPTO, "ETH", "USD", 0.01, 0.1, 20.0, 3.5, "24/7", "Ethereum/US Dollar", "ETHUSDT", "ETH-USD", "ETHUSD"),
            TradingPair("ADAUSD", "ADA-USD", MarketType.CRYPTO, "ADA", "USD", 1.0, 0.001, 5.0, 5.0, "24/7", "Cardano/US Dollar", "ADAUSDT", "ADA-USD", "ADAUSD"),
            TradingPair("BNBUSD", "BNB-USD", MarketType.CRYPTO, "BNB", "USD", 0.01, 0.1, 15.0, 3.0, "24/7", "Binance Coin/US Dollar", "BNBUSDT", "BNB-USD", "BNBUSD"),
            TradingPair("SOLUSD", "SOL-USD", MarketType.CRYPTO, "SOL", "USD", 0.01, 0.01, 10.0, 4.5, "24/7", "Solana/US Dollar", "SOLUSDT", "SOL-USD", "SOLUSD"),
            TradingPair("XRPUSD", "XRP-USD", MarketType.CRYPTO, "XRP", "USD", 1.0, 0.001, 3.0, 4.0, "24/7", "Ripple/US Dollar", "XRPUSDT", "XRP-USD", "XRPUSD"),
            TradingPair("DOTUSD", "DOT-USD", MarketType.CRYPTO, "DOT", "USD", 0.1, 0.01, 8.0, 4.2, "24/7", "Polkadot/US Dollar", "DOTUSDT", "DOT-USD", "DOTUSD"),
            TradingPair("AVAXUSD", "AVAX-USD", MarketType.CRYPTO, "AVAX", "USD", 0.1, 0.01, 12.0, 5.0, "24/7", "Avalanche/US Dollar", "AVAXUSDT", "AVAX-USD", "AVAXUSD"),
            TradingPair("LINKUSD", "LINK-USD", MarketType.CRYPTO, "LINK", "USD", 0.1, 0.01, 8.0, 3.8, "24/7", "Chainlink/US Dollar", "LINKUSDT", "LINK-USD", "LINKUSD"),
            TradingPair("MATICUSD", "MATIC-USD", MarketType.CRYPTO, "MATIC", "USD", 1.0, 0.001, 5.0, 4.5, "24/7", "Polygon/US Dollar", "MATICUSDT", "MATIC-USD", "MATICUSD"),
            TradingPair("UNIUSD", "UNI-USD", MarketType.CRYPTO, "UNI", "USD", 0.1, 0.01, 10.0, 4.0, "24/7", "Uniswap/US Dollar", "UNIUSDT", "UNI-USD", "UNIUSD"),
            TradingPair("LTCUSD", "LTC-USD", MarketType.CRYPTO, "LTC", "USD", 0.01, 0.1, 15.0, 3.2, "24/7", "Litecoin/US Dollar", "LTCUSDT", "LTC-USD", "LTCUSD"),
            TradingPair("BCHUSD", "BCH-USD", MarketType.CRYPTO, "BCH", "USD", 0.01, 0.1, 20.0, 3.5, "24/7", "Bitcoin Cash/US Dollar", "BCHUSDT", "BCH-USD", "BCHUSD"),
            TradingPair("XLMUSD", "XLM-USD", MarketType.CRYPTO, "XLM", "USD", 1.0, 0.0001, 2.0, 4.8, "24/7", "Stellar/US Dollar", "XLMUSDT", "XLM-USD", "XLMUSD"),
            TradingPair("VETUSD", "VET-USD", MarketType.CRYPTO, "VET", "USD", 10.0, 0.00001, 1.0, 5.5, "24/7", "VeChain/US Dollar", "VETUSDT", "VET-USD", "VETUSD"),
            TradingPair("FILUSD", "FIL-USD", MarketType.CRYPTO, "FIL", "USD", 0.1, 0.01, 12.0, 4.8, "24/7", "Filecoin/US Dollar", "FILUSDT", "FIL-USD", "FILUSD"),
            TradingPair("ATOMUSD", "ATOM-USD", MarketType.CRYPTO, "ATOM", "USD", 0.1, 0.01, 8.0, 4.3, "24/7", "Cosmos/US Dollar", "ATOMUSDT", "ATOM-USD", "ATOMUSD"),
            TradingPair("ALGOUSD", "ALGO-USD", MarketType.CRYPTO, "ALGO", "USD", 1.0, 0.001, 3.0, 5.0, "24/7", "Algorand/US Dollar", "ALGOUSDT", "ALGO-USD", "ALGOUSD"),
        ]
        
        # Major Stock Indices
        indices_pairs = [
            TradingPair("SPY", "SPY", MarketType.INDICES, "SPY", "USD", 1.0, 0.01, 2.0, 1.2, "9:30-16:00 EST", "S&P 500 ETF"),
            TradingPair("QQQ", "QQQ", MarketType.INDICES, "QQQ", "USD", 1.0, 0.01, 3.0, 1.5, "9:30-16:00 EST", "Nasdaq 100 ETF"),
            TradingPair("IWM", "IWM", MarketType.INDICES, "IWM", "USD", 1.0, 0.01, 4.0, 1.8, "9:30-16:00 EST", "Russell 2000 ETF"),
            TradingPair("DIA", "DIA", MarketType.INDICES, "DIA", "USD", 1.0, 0.01, 3.0, 1.3, "9:30-16:00 EST", "Dow Jones ETF"),
            TradingPair("VTI", "VTI", MarketType.INDICES, "VTI", "USD", 1.0, 0.01, 2.0, 1.1, "9:30-16:00 EST", "Total Stock Market ETF"),
            TradingPair("EFA", "EFA", MarketType.INDICES, "EFA", "USD", 1.0, 0.01, 3.0, 1.4, "9:30-16:00 EST", "EAFE Index ETF"),
            TradingPair("EEM", "EEM", MarketType.INDICES, "EEM", "USD", 1.0, 0.01, 5.0, 2.0, "9:30-16:00 EST", "Emerging Markets ETF"),
        ]
        
        # Major Individual Stocks
        stock_pairs = [
            TradingPair("AAPL", "AAPL", MarketType.STOCKS, "AAPL", "USD", 1.0, 0.01, 2.0, 1.8, "9:30-16:00 EST", "Apple Inc."),
            TradingPair("MSFT", "MSFT", MarketType.STOCKS, "MSFT", "USD", 1.0, 0.01, 2.0, 1.6, "9:30-16:00 EST", "Microsoft Corporation"),
            TradingPair("GOOGL", "GOOGL", MarketType.STOCKS, "GOOGL", "USD", 1.0, 0.01, 3.0, 1.9, "9:30-16:00 EST", "Alphabet Inc."),
            TradingPair("AMZN", "AMZN", MarketType.STOCKS, "AMZN", "USD", 1.0, 0.01, 5.0, 2.2, "9:30-16:00 EST", "Amazon.com Inc."),
            TradingPair("TSLA", "TSLA", MarketType.STOCKS, "TSLA", "USD", 1.0, 0.01, 8.0, 3.5, "9:30-16:00 EST", "Tesla Inc."),
            TradingPair("META", "META", MarketType.STOCKS, "META", "USD", 1.0, 0.01, 4.0, 2.5, "9:30-16:00 EST", "Meta Platforms Inc."),
            TradingPair("NVDA", "NVDA", MarketType.STOCKS, "NVDA", "USD", 1.0, 0.01, 6.0, 3.0, "9:30-16:00 EST", "NVIDIA Corporation"),
            TradingPair("NFLX", "NFLX", MarketType.STOCKS, "NFLX", "USD", 1.0, 0.01, 5.0, 2.8, "9:30-16:00 EST", "Netflix Inc."),
            TradingPair("JPM", "JPM", MarketType.STOCKS, "JPM", "USD", 1.0, 0.01, 3.0, 1.7, "9:30-16:00 EST", "JPMorgan Chase & Co."),
            TradingPair("BAC", "BAC", MarketType.STOCKS, "BAC", "USD", 1.0, 0.01, 2.0, 1.9, "9:30-16:00 EST", "Bank of America Corp."),
            TradingPair("V", "V", MarketType.STOCKS, "V", "USD", 1.0, 0.01, 3.0, 1.5, "9:30-16:00 EST", "Visa Inc."),
            TradingPair("JNJ", "JNJ", MarketType.STOCKS, "JNJ", "USD", 1.0, 0.01, 2.0, 1.2, "9:30-16:00 EST", "Johnson & Johnson"),
            TradingPair("WMT", "WMT", MarketType.STOCKS, "WMT", "USD", 1.0, 0.01, 2.0, 1.1, "9:30-16:00 EST", "Walmart Inc."),
            TradingPair("PG", "PG", MarketType.STOCKS, "PG", "USD", 1.0, 0.01, 2.0, 1.0, "9:30-16:00 EST", "Procter & Gamble Co."),
            TradingPair("KO", "KO", MarketType.STOCKS, "KO", "USD", 1.0, 0.01, 1.5, 0.9, "9:30-16:00 EST", "The Coca-Cola Company"),
        ]
        
        # Commodities
        commodity_pairs = [
            TradingPair("GOLD", "GC=F", MarketType.COMMODITIES, "XAU", "USD", 0.1, 0.1, 3.0, 1.5, "24/5", "Gold Futures"),
            TradingPair("SILVER", "SI=F", MarketType.COMMODITIES, "XAG", "USD", 1.0, 0.01, 5.0, 2.5, "24/5", "Silver Futures"),
            TradingPair("OIL", "CL=F", MarketType.COMMODITIES, "CL", "USD", 0.1, 0.01, 8.0, 2.8, "24/5", "Crude Oil Futures"),
            TradingPair("NATGAS", "NG=F", MarketType.COMMODITIES, "NG", "USD", 1.0, 0.001, 15.0, 4.0, "24/5", "Natural Gas Futures"),
            TradingPair("COPPER", "HG=F", MarketType.COMMODITIES, "HG", "USD", 0.1, 0.0001, 10.0, 2.2, "24/5", "Copper Futures"),
            TradingPair("WHEAT", "ZW=F", MarketType.COMMODITIES, "ZW", "USD", 1.0, 0.25, 12.0, 2.0, "24/5", "Wheat Futures"),
            TradingPair("CORN", "ZC=F", MarketType.COMMODITIES, "ZC", "USD", 1.0, 0.25, 10.0, 1.8, "24/5", "Corn Futures"),
            TradingPair("SOYBEAN", "ZS=F", MarketType.COMMODITIES, "ZS", "USD", 1.0, 0.25, 15.0, 2.2, "24/5", "Soybean Futures"),
        ]
        
        # Combine all pairs
        all_pairs = forex_pairs + crypto_pairs + indices_pairs + stock_pairs + commodity_pairs
        
        for pair in all_pairs:
            pairs[pair.symbol] = pair
            
        return pairs
    
    def get_all_pairs(self) -> Dict[str, TradingPair]:
        """Get all trading pairs"""
        return self.pairs
    
    def get_pairs_by_market(self, market_type: MarketType) -> Dict[str, TradingPair]:
        """Get pairs filtered by market type"""
        return {symbol: pair for symbol, pair in self.pairs.items() if pair.market_type == market_type}
    
    def get_pair(self, symbol: str) -> TradingPair:
        """Get specific trading pair"""
        return self.pairs.get(symbol)
    
    def get_yahoo_symbols(self) -> List[str]:
        """Get all Yahoo Finance symbols"""
        return [pair.yahoo_symbol for pair in self.pairs.values()]
    
    def get_binance_symbols(self) -> List[str]:
        """Get all Binance symbols"""
        return [pair.binance_symbol for pair in self.pairs.values() if pair.binance_symbol]
    
    def get_high_volatility_pairs(self, min_volatility: float = 3.0) -> Dict[str, TradingPair]:
        """Get pairs with high volatility for aggressive trading"""
        return {symbol: pair for symbol, pair in self.pairs.items() if pair.volatility_factor >= min_volatility}
    
    def get_low_volatility_pairs(self, max_volatility: float = 1.5) -> Dict[str, TradingPair]:
        """Get pairs with low volatility for conservative trading"""
        return {symbol: pair for symbol, pair in self.pairs.items() if pair.volatility_factor <= max_volatility}
    
    def get_24_7_pairs(self) -> Dict[str, TradingPair]:
        """Get pairs that trade 24/7 (mainly crypto)"""
        return {symbol: pair for symbol, pair in self.pairs.items() if "24/7" in pair.trading_hours}
    
    def search_pairs(self, query: str) -> Dict[str, TradingPair]:
        """Search pairs by symbol or description"""
        query = query.lower()
        results = {}
        for symbol, pair in self.pairs.items():
            if (query in symbol.lower() or 
                query in pair.description.lower() or 
                query in pair.base_currency.lower() or 
                query in pair.quote_currency.lower()):
                results[symbol] = pair
        return results
    
    def get_market_summary(self) -> Dict[str, int]:
        """Get summary of pairs by market type"""
        summary = {}
        for market_type in MarketType:
            summary[market_type.value] = len(self.get_pairs_by_market(market_type))
        return summary

# Global instance
trading_pairs_manager = TradingPairsManager()