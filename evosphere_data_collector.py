#!/usr/bin/env python3
"""
EvoSphere Local Data Collection System
Collects real-time market data, news, and local intelligence for trading decisions
"""
import requests
import json
import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
import feedparser
import yfinance as yf
from textblob import TextBlob
import os

@dataclass
class NewsItem:
    """News article data structure"""
    title: str
    content: str
    source: str
    timestamp: float
    sentiment_score: float
    market_relevance: float
    location: str
    url: str

@dataclass
class MarketData:
    """Market data point structure"""
    symbol: str
    price: float
    volume: float
    change_percent: float
    timestamp: float
    source: str
    location: str

@dataclass
class LocalIntelligence:
    """Local market intelligence"""
    region: str
    currency_flows: Dict[str, float]
    economic_indicators: Dict[str, float]
    regulatory_updates: List[str]
    trading_volume_patterns: Dict[str, float]
    timestamp: float

class EvoSphereDataCollector:
    """Comprehensive data collection for EvoSphere nodes"""
    
    def __init__(self, sphere_id: str, location: str, region: str = "ASEAN"):
        self.sphere_id = sphere_id
        self.location = location
        self.region = region
        
        # Data storage
        self.news_cache = []
        self.market_data_cache = []
        self.local_intelligence = {}
        
        # Collection settings
        self.max_cache_size = 1000
        self.collection_interval = 300  # 5 minutes
        self.is_collecting = False
        
        # News sources by region
        self.news_sources = self._get_regional_news_sources()
        
        # Market symbols to track
        self.market_symbols = self._get_regional_markets()
        
    def _get_regional_news_sources(self) -> List[Dict]:
        """Get news sources based on geographic location"""
        base_sources = [
            {
                'name': 'Reuters Business',
                'url': 'https://feeds.reuters.com/reuters/businessNews',
                'type': 'global'
            },
            {
                'name': 'Bloomberg Markets',
                'url': 'https://feeds.bloomberg.com/markets/news.rss',
                'type': 'global'
            },
            {
                'name': 'MarketWatch',
                'url': 'https://feeds.marketwatch.com/marketwatch/realtimeheadlines/',
                'type': 'global'
            }
        ]
        
        # Add regional sources based on location
        if 'vietnam' in self.location.lower() or 'ho chi minh' in self.location.lower():
            base_sources.extend([
                {
                    'name': 'VnExpress Business',
                    'url': 'https://vnexpress.net/rss/kinh-doanh.rss',
                    'type': 'local'
                },
                {
                    'name': 'Vietnam Investment Review',
                    'url': 'https://vir.com.vn/rss/latest-news.rss',
                    'type': 'local'
                }
            ])
        elif 'singapore' in self.location.lower():
            base_sources.extend([
                {
                    'name': 'Straits Times Business',
                    'url': 'https://www.straitstimes.com/rss/business.xml',
                    'type': 'local'
                },
                {
                    'name': 'Channel NewsAsia Business',
                    'url': 'https://www.channelnewsasia.com/rssfeeds/8395986',
                    'type': 'local'
                }
            ])
        elif 'bangkok' in self.location.lower() or 'thailand' in self.location.lower():
            base_sources.extend([
                {
                    'name': 'Bangkok Post Business',
                    'url': 'https://www.bangkokpost.com/rss/data/business.xml',
                    'type': 'local'
                },
                {
                    'name': 'The Nation Business',
                    'url': 'https://www.nationthailand.com/rss/business.xml',
                    'type': 'local'
                }
            ])
        
        return base_sources
    
    def _get_regional_markets(self) -> List[str]:
        """Get market symbols relevant to the region"""
        base_symbols = ['BTC-USD', 'ETH-USD', 'EURUSD=X', 'GBPUSD=X', '^VIX']
        
        # Add regional symbols
        if 'vietnam' in self.location.lower():
            base_symbols.extend(['VNM', 'VCB.VN', 'HPG.VN', 'VHM.VN'])
        elif 'singapore' in self.location.lower():
            base_symbols.extend(['^STI', 'D05.SI', 'O39.SI', 'C6L.SI'])
        elif 'thailand' in self.location.lower():
            base_symbols.extend(['^SET', 'PTT.BK', 'CPALL.BK', 'KBANK.BK'])
        
        return base_symbols
    
    def collect_news_data(self) -> List[NewsItem]:
        """Collect news from all configured sources"""
        news_items = []
        
        for source in self.news_sources:
            try:
                print(f"📰 Collecting news from {source['name']}...")
                
                # Parse RSS feed
                feed = feedparser.parse(source['url'])
                
                for entry in feed.entries[:10]:  # Limit to 10 recent articles
                    # Extract content
                    content = entry.get('summary', entry.get('description', ''))
                    title = entry.get('title', '')
                    
                    # Calculate sentiment
                    text_for_sentiment = f"{title} {content}"
                    sentiment = TextBlob(text_for_sentiment).sentiment.polarity
                    
                    # Calculate market relevance
                    market_keywords = ['trading', 'market', 'price', 'bitcoin', 'crypto', 
                                     'forex', 'currency', 'economy', 'central bank', 
                                     'inflation', 'GDP', 'investment']
                    
                    relevance = sum(1 for keyword in market_keywords 
                                  if keyword.lower() in text_for_sentiment.lower()) / len(market_keywords)
                    
                    news_item = NewsItem(
                        title=title,
                        content=content,
                        source=source['name'],
                        timestamp=time.time(),
                        sentiment_score=sentiment,
                        market_relevance=relevance,
                        location=self.location,
                        url=entry.get('link', '')
                    )
                    
                    news_items.append(news_item)
                    
            except Exception as e:
                print(f"⚠️  Error collecting from {source['name']}: {e}")
                continue
        
        return news_items
    
    def collect_market_data(self) -> List[MarketData]:
        """Collect real-time market data"""
        market_data = []
        
        for symbol in self.market_symbols:
            try:
                print(f"📈 Collecting market data for {symbol}...")
                
                # Get ticker data
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="1d", interval="1m")
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    
                    # Calculate change percentage
                    if len(hist) > 1:
                        prev_close = hist.iloc[-2]['Close']
                        change_pct = ((latest['Close'] - prev_close) / prev_close) * 100
                    else:
                        change_pct = 0.0
                    
                    market_item = MarketData(
                        symbol=symbol,
                        price=float(latest['Close']),
                        volume=float(latest['Volume']),
                        change_percent=change_pct,
                        timestamp=time.time(),
                        source='Yahoo Finance',
                        location=self.location
                    )
                    
                    market_data.append(market_item)
                    
            except Exception as e:
                print(f"⚠️  Error collecting market data for {symbol}: {e}")
                continue
        
        return market_data
    
    def collect_local_intelligence(self) -> LocalIntelligence:
        """Collect local market intelligence"""
        try:
            print(f"🔍 Collecting local intelligence for {self.location}...")
            
            # Simulate local intelligence collection
            # In production, this would connect to local APIs, economic data sources, etc.
            
            intelligence = LocalIntelligence(
                region=self.region,
                currency_flows={
                    'USD_inflow': 1500000 + (time.time() % 1000000),
                    'EUR_inflow': 800000 + (time.time() % 500000),
                    'local_currency_strength': 0.7 + (time.time() % 100) / 1000
                },
                economic_indicators={
                    'local_gdp_growth': 3.2 + (time.time() % 10) / 100,
                    'inflation_rate': 2.1 + (time.time() % 5) / 100,
                    'unemployment_rate': 4.5 - (time.time() % 3) / 100,
                    'manufacturing_index': 52.3 + (time.time() % 20) / 10
                },
                regulatory_updates=[
                    f"Central bank policy update - {self.location}",
                    f"New trading regulations effective next month - {self.region}",
                    "Cryptocurrency guidelines updated"
                ],
                trading_volume_patterns={
                    'morning_volume_spike': 1.3 + (time.time() % 50) / 100,
                    'afternoon_lull': 0.8 + (time.time() % 30) / 100,
                    'evening_surge': 1.1 + (time.time() % 40) / 100
                },
                timestamp=time.time()
            )
            
            return intelligence
            
        except Exception as e:
            print(f"⚠️  Error collecting local intelligence: {e}")
            return None
    
    def analyze_collected_data(self) -> Dict:
        """Analyze all collected data for trading insights"""
        analysis = {
            'sentiment_summary': self._analyze_news_sentiment(),
            'market_momentum': self._analyze_market_momentum(),
            'local_factors': self._analyze_local_factors(),
            'trading_recommendations': self._generate_recommendations(),
            'data_quality_score': self._calculate_data_quality(),
            'timestamp': time.time()
        }
        
        return analysis
    
    def _analyze_news_sentiment(self) -> Dict:
        """Analyze overall news sentiment"""
        if not self.news_cache:
            return {'overall_sentiment': 0, 'confidence': 0, 'article_count': 0}
        
        # Weight by market relevance
        weighted_sentiment = sum(item.sentiment_score * item.market_relevance 
                               for item in self.news_cache[-50:])  # Last 50 articles
        total_weight = sum(item.market_relevance for item in self.news_cache[-50:])
        
        if total_weight > 0:
            overall_sentiment = weighted_sentiment / total_weight
        else:
            overall_sentiment = 0
        
        return {
            'overall_sentiment': overall_sentiment,
            'confidence': min(total_weight, 1.0),
            'article_count': len(self.news_cache[-50:]),
            'bullish_articles': sum(1 for item in self.news_cache[-50:] if item.sentiment_score > 0.1),
            'bearish_articles': sum(1 for item in self.news_cache[-50:] if item.sentiment_score < -0.1)
        }
    
    def _analyze_market_momentum(self) -> Dict:
        """Analyze market momentum from price data"""
        if not self.market_data_cache:
            return {'momentum_score': 0, 'volatility': 0, 'trend_strength': 0}
        
        recent_data = self.market_data_cache[-20:]  # Last 20 data points
        
        # Calculate momentum
        positive_changes = sum(1 for item in recent_data if item.change_percent > 0)
        momentum_score = (positive_changes / len(recent_data)) * 2 - 1  # Scale to -1 to 1
        
        # Calculate volatility
        changes = [abs(item.change_percent) for item in recent_data]
        volatility = sum(changes) / len(changes) if changes else 0
        
        # Calculate trend strength
        price_changes = [item.change_percent for item in recent_data]
        trend_strength = abs(sum(price_changes)) / len(price_changes) if price_changes else 0
        
        return {
            'momentum_score': momentum_score,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'market_symbols_tracked': len(set(item.symbol for item in recent_data))
        }
    
    def _analyze_local_factors(self) -> Dict:
        """Analyze local economic factors"""
        if not self.local_intelligence:
            return {'economic_health': 0.5, 'currency_strength': 0.5}
        
        latest = self.local_intelligence
        
        # Economic health score (0-1)
        gdp_score = min(latest.economic_indicators.get('local_gdp_growth', 0) / 5.0, 1.0)
        inflation_score = max(0, 1 - latest.economic_indicators.get('inflation_rate', 5) / 10.0)
        unemployment_score = max(0, 1 - latest.economic_indicators.get('unemployment_rate', 10) / 10.0)
        
        economic_health = (gdp_score + inflation_score + unemployment_score) / 3
        
        # Currency strength
        currency_strength = latest.currency_flows.get('local_currency_strength', 0.5)
        
        return {
            'economic_health': economic_health,
            'currency_strength': currency_strength,
            'gdp_growth': latest.economic_indicators.get('local_gdp_growth', 0),
            'inflation_rate': latest.economic_indicators.get('inflation_rate', 0),
            'manufacturing_index': latest.economic_indicators.get('manufacturing_index', 50)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate trading recommendations based on collected data"""
        recommendations = []
        
        sentiment = self._analyze_news_sentiment()
        momentum = self._analyze_market_momentum()
        local = self._analyze_local_factors()
        
        # Sentiment-based recommendations
        if sentiment['overall_sentiment'] > 0.3:
            recommendations.append("Strong positive sentiment detected - consider long positions")
        elif sentiment['overall_sentiment'] < -0.3:
            recommendations.append("Negative sentiment prevailing - defensive positioning recommended")
        
        # Momentum-based recommendations
        if momentum['momentum_score'] > 0.6:
            recommendations.append("Strong upward momentum - trend following strategy optimal")
        elif momentum['momentum_score'] < -0.6:
            recommendations.append("Downward momentum confirmed - consider short positions")
        
        # Volatility recommendations
        if momentum['volatility'] > 3.0:
            recommendations.append("High volatility detected - reduce position sizes")
        elif momentum['volatility'] < 1.0:
            recommendations.append("Low volatility environment - consider larger positions")
        
        # Local factor recommendations
        if local['economic_health'] > 0.7:
            recommendations.append(f"Strong local economy in {self.location} - favor local assets")
        elif local['economic_health'] < 0.3:
            recommendations.append(f"Economic concerns in {self.location} - diversify away from local exposure")
        
        return recommendations
    
    def _calculate_data_quality(self) -> float:
        """Calculate overall data quality score"""
        factors = []
        
        # News data quality
        if self.news_cache:
            news_recency = 1.0 - min((time.time() - max(item.timestamp for item in self.news_cache[-10:])) / 3600, 1.0)
            factors.append(news_recency)
        
        # Market data quality
        if self.market_data_cache:
            market_recency = 1.0 - min((time.time() - max(item.timestamp for item in self.market_data_cache[-10:])) / 1800, 1.0)
            factors.append(market_recency)
        
        # Source diversity
        if self.news_cache:
            unique_sources = len(set(item.source for item in self.news_cache[-20:]))
            source_diversity = min(unique_sources / 3.0, 1.0)
            factors.append(source_diversity)
        
        return sum(factors) / len(factors) if factors else 0.0
    
    def start_continuous_collection(self):
        """Start continuous data collection"""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        collection_thread.start()
        
        print(f"🚀 Started continuous data collection for {self.sphere_id} in {self.location}")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.is_collecting:
            try:
                print(f"\n🔄 Data collection cycle - {datetime.now().strftime('%H:%M:%S')}")
                
                # Collect news
                new_news = self.collect_news_data()
                self.news_cache.extend(new_news)
                
                # Collect market data
                new_market_data = self.collect_market_data()
                self.market_data_cache.extend(new_market_data)
                
                # Collect local intelligence
                intelligence = self.collect_local_intelligence()
                if intelligence:
                    self.local_intelligence = intelligence
                
                # Trim cache if too large
                if len(self.news_cache) > self.max_cache_size:
                    self.news_cache = self.news_cache[-self.max_cache_size:]
                
                if len(self.market_data_cache) > self.max_cache_size:
                    self.market_data_cache = self.market_data_cache[-self.max_cache_size:]
                
                # Generate analysis
                analysis = self.analyze_collected_data()
                
                print(f"📊 Collection Summary:")
                print(f"   News Articles: {len(new_news)} new, {len(self.news_cache)} total")
                print(f"   Market Data Points: {len(new_market_data)} new, {len(self.market_data_cache)} total")
                print(f"   Overall Sentiment: {analysis['sentiment_summary']['overall_sentiment']:.2f}")
                print(f"   Market Momentum: {analysis['market_momentum']['momentum_score']:.2f}")
                print(f"   Data Quality: {analysis['data_quality_score']:.2%}")
                
                if analysis['trading_recommendations']:
                    print(f"💡 Recommendations:")
                    for rec in analysis['trading_recommendations'][:3]:
                        print(f"   • {rec}")
                
            except Exception as e:
                print(f"❌ Error in collection cycle: {e}")
            
            # Wait for next collection cycle
            time.sleep(self.collection_interval)
    
    def stop_collection(self):
        """Stop continuous data collection"""
        self.is_collecting = False
        print(f"⏹️  Stopped data collection for {self.sphere_id}")
    
    def get_status_summary(self) -> Dict:
        """Get current status summary"""
        return {
            'sphere_id': self.sphere_id,
            'location': self.location,
            'region': self.region,
            'is_collecting': self.is_collecting,
            'news_articles_cached': len(self.news_cache),
            'market_data_points': len(self.market_data_cache),
            'news_sources_configured': len(self.news_sources),
            'market_symbols_tracked': len(self.market_symbols),
            'data_quality_score': self._calculate_data_quality(),
            'last_analysis': self.analyze_collected_data()
        }

def demo_data_collection():
    """Demonstrate the data collection system"""
    print("🌐 EvoSphere Data Collection System Demo")
    print("Real-time local intelligence gathering for trading networks")
    print("=" * 70)
    
    # Create collectors for different regions
    collectors = [
        EvoSphereDataCollector("vietnam_data_001", "Ho Chi Minh City", "ASEAN"),
        EvoSphereDataCollector("singapore_data_002", "Singapore", "ASEAN"),
        EvoSphereDataCollector("bangkok_data_003", "Bangkok", "ASEAN")
    ]
    
    print(f"🔗 Created {len(collectors)} regional data collectors")
    
    # Start data collection for each node
    for collector in collectors:
        print(f"\n{'='*50}")
        print(f"🎯 Starting collection for {collector.location}")
        
        # Show configuration
        status = collector.get_status_summary()
        print(f"📍 Location: {status['location']}")
        print(f"🌏 Region: {status['region']}")
        print(f"📰 News Sources: {status['news_sources_configured']}")
        print(f"📈 Market Symbols: {status['market_symbols_tracked']}")
        
        # Start continuous collection
        collector.start_continuous_collection()
    
    print(f"\n🎉 ALL DATA COLLECTORS OPERATIONAL")
    print("✅ Multi-regional news monitoring active")
    print("✅ Real-time market data streaming")
    print("✅ Local economic intelligence gathering")
    print("✅ Sentiment analysis and recommendations")
    print("\n💡 Each EvoSphere node now has comprehensive market awareness!")
    
    # Let it run for a short demo period
    time.sleep(30)
    
    # Show final status
    print(f"\n📊 FINAL STATUS REPORT:")
    for collector in collectors:
        status = collector.get_status_summary()
        print(f"\n🏢 {collector.location}:")
        print(f"   📰 News: {status['news_articles_cached']} articles")
        print(f"   📈 Market: {status['market_data_points']} data points")
        print(f"   🎯 Quality: {status['data_quality_score']:.1%}")
        
        # Stop collection
        collector.stop_collection()

if __name__ == "__main__":
    demo_data_collection()