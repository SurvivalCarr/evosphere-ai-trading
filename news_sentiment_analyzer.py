"""
News Sentiment Analysis for Forex Trading
Uses free RSS feeds and TextBlob for sentiment analysis
"""

import feedparser
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
import logging
import re
from typing import Dict, List, Optional, Tuple
import time
from urllib.parse import urljoin
import json

class NewsSentimentAnalyzer:
    """Analyzes financial news sentiment for forex trading decisions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Free financial news RSS feeds
        self.news_sources = {
            'reuters_markets': 'https://feeds.reuters.com/reuters/businessNews',
            'reuters_forex': 'https://feeds.reuters.com/reuters/UKbusinessNews',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/realtimeheadlines/',
            'cnbc': 'https://feeds.feedburner.com/cnbc/finance',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
        }
        
        # Forex-related keywords for filtering relevant news
        self.forex_keywords = [
            'dollar', 'euro', 'eur/usd', 'eurusd', 'currency', 'forex', 'fx',
            'federal reserve', 'fed', 'ecb', 'european central bank',
            'interest rate', 'inflation', 'gdp', 'unemployment',
            'trade war', 'economic growth', 'monetary policy',
            'brexit', 'european union', 'eu', 'us economy', 'eurozone'
        ]
        
        # Cache for news sentiment to avoid repeated API calls
        self.sentiment_cache = {}
        self.cache_duration = 300  # 5 minutes
        
    def fetch_news_headlines(self, hours_back: int = 6) -> List[Dict]:
        """Fetch recent financial news headlines from multiple sources"""
        all_headlines = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        for source_name, feed_url in self.news_sources.items():
            try:
                self.logger.info(f"Fetching news from {source_name}")
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    # Parse publication date
                    try:
                        if hasattr(entry, 'published_parsed'):
                            pub_date = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed'):
                            pub_date = datetime(*entry.updated_parsed[:6])
                        else:
                            pub_date = datetime.now()
                    except:
                        pub_date = datetime.now()
                    
                    # Only include recent news
                    if pub_date > cutoff_time:
                        headline = {
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', ''),
                            'source': source_name,
                            'published': pub_date,
                            'link': entry.get('link', '')
                        }
                        
                        # Filter for forex-relevant news
                        if self._is_forex_relevant(headline['title'] + ' ' + headline['summary']):
                            all_headlines.append(headline)
                            
            except Exception as e:
                self.logger.warning(f"Error fetching from {source_name}: {e}")
                continue
        
        # Sort by publication date (newest first)
        all_headlines.sort(key=lambda x: x['published'], reverse=True)
        return all_headlines[:20]  # Return top 20 most recent relevant headlines
    
    def _is_forex_relevant(self, text: str) -> bool:
        """Check if news text is relevant to forex trading"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.forex_keywords)
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of given text using TextBlob"""
        try:
            # Clean the text
            cleaned_text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
            cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text)  # Remove special chars
            
            # Analyze sentiment
            blob = TextBlob(cleaned_text)
            
            # TextBlob returns polarity (-1 to 1) and subjectivity (0 to 1)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Convert to trading-relevant scores
            sentiment_score = polarity  # -1 (very bearish) to 1 (very bullish)
            confidence = 1 - subjectivity  # Higher confidence for objective statements
            
            # Classify sentiment
            if sentiment_score > 0.1:
                sentiment_label = "bullish"
            elif sentiment_score < -0.1:
                sentiment_label = "bearish"
            else:
                sentiment_label = "neutral"
            
            return {
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'sentiment_label': sentiment_label,
                'polarity': polarity,
                'subjectivity': subjectivity
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'sentiment_label': "neutral",
                'polarity': 0.0,
                'subjectivity': 0.0
            }
    
    def get_market_sentiment(self, hours_back: int = 6) -> Dict[str, float]:
        """Get overall market sentiment for the specified time period"""
        cache_key = f"market_sentiment_{hours_back}"
        
        # Check cache
        if cache_key in self.sentiment_cache:
            cached_time, cached_result = self.sentiment_cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return cached_result
        
        # Fetch fresh news
        headlines = self.fetch_news_headlines(hours_back)
        
        if not headlines:
            return {
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'sentiment_label': 'neutral',
                'news_count': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0
            }
        
        sentiments = []
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for headline in headlines:
            # Analyze sentiment of title and summary combined
            text = f"{headline['title']} {headline['summary']}"
            sentiment = self.analyze_sentiment(text)
            
            # Weight recent news more heavily
            hours_old = (datetime.now() - headline['published']).total_seconds() / 3600
            time_weight = max(0.1, 1.0 - (hours_old / 24))  # Decay over 24 hours
            
            weighted_sentiment = sentiment['sentiment_score'] * sentiment['confidence'] * time_weight
            sentiments.append(weighted_sentiment)
            
            # Count sentiment categories
            if sentiment['sentiment_label'] == 'bullish':
                bullish_count += 1
            elif sentiment['sentiment_label'] == 'bearish':
                bearish_count += 1
            else:
                neutral_count += 1
        
        # Calculate overall sentiment
        if sentiments:
            overall_sentiment = sum(sentiments) / len(sentiments)
            overall_confidence = sum(abs(s) for s in sentiments) / len(sentiments)
        else:
            overall_sentiment = 0.0
            overall_confidence = 0.0
        
        # Determine overall sentiment label
        if overall_sentiment > 0.1:
            sentiment_label = "bullish"
        elif overall_sentiment < -0.1:
            sentiment_label = "bearish"
        else:
            sentiment_label = "neutral"
        
        result = {
            'overall_sentiment': overall_sentiment,
            'confidence': overall_confidence,
            'sentiment_label': sentiment_label,
            'news_count': len(headlines),
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'latest_headlines': [h['title'] for h in headlines[:5]]  # Top 5 headlines
        }
        
        # Cache the result
        self.sentiment_cache[cache_key] = (time.time(), result)
        
        return result
    
    def get_sentiment_signal(self, hours_back: int = 6) -> Tuple[float, str]:
        """Get trading signal based on news sentiment"""
        sentiment_data = self.get_market_sentiment(hours_back)
        
        sentiment_score = sentiment_data['overall_sentiment']
        confidence = sentiment_data['confidence']
        
        # Convert sentiment to trading signal
        # Strong bullish sentiment suggests buying EUR/USD
        # Strong bearish sentiment suggests selling EUR/USD
        
        signal_strength = sentiment_score * confidence
        
        if signal_strength > 0.3:
            signal = "strong_buy"
        elif signal_strength > 0.1:
            signal = "buy"
        elif signal_strength < -0.3:
            signal = "strong_sell"
        elif signal_strength < -0.1:
            signal = "sell"
        else:
            signal = "hold"
        
        return signal_strength, signal
    
    def get_economic_calendar_events(self) -> List[Dict]:
        """Get upcoming economic events (simplified version using free sources)"""
        # This would ideally connect to a free economic calendar API
        # For now, we'll return a placeholder structure
        events = []
        
        try:
            # You could integrate with free APIs like:
            # - Trading Economics (limited free tier)
            # - Economic Calendar APIs
            # For now, we'll focus on news sentiment
            pass
        except Exception as e:
            self.logger.warning(f"Could not fetch economic calendar: {e}")
        
        return events

def test_sentiment_analyzer():
    """Test the news sentiment analyzer"""
    analyzer = NewsSentimentAnalyzer()
    
    print("Testing News Sentiment Analyzer...")
    
    # Test basic sentiment analysis
    test_texts = [
        "USD strengthens as Fed signals more aggressive rate hikes",
        "Euro falls on recession fears and ECB dovish stance",
        "Markets remain stable amid mixed economic data"
    ]
    
    for text in test_texts:
        sentiment = analyzer.analyze_sentiment(text)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment['sentiment_label']} (score: {sentiment['sentiment_score']:.2f})")
        print()
    
    # Test market sentiment
    print("Current Market Sentiment:")
    market_sentiment = analyzer.get_market_sentiment(6)
    print(f"Overall: {market_sentiment['sentiment_label']} ({market_sentiment['overall_sentiment']:.2f})")
    print(f"Confidence: {market_sentiment['confidence']:.2f}")
    print(f"News analyzed: {market_sentiment['news_count']}")
    print()
    
    # Test trading signal
    signal_strength, signal = analyzer.get_sentiment_signal()
    print(f"Trading Signal: {signal} (strength: {signal_strength:.2f})")

if __name__ == "__main__":
    test_sentiment_analyzer()