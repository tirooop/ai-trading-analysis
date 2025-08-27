from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from polygon import RESTClient
import os
from dotenv import load_dotenv

# Load environment variables from API.env
load_dotenv('API.env')

@dataclass
class MarketContext:
    """Market context information"""
    symbol: str
    timestamp: datetime
    price_data: Dict[str, pd.DataFrame]  # Multiple timeframe data
    sector_performance: Dict[str, float]
    market_breadth: Dict[str, float]
    vix_level: float
    news_sentiment: Dict[str, float]
    option_chain: Optional[pd.DataFrame] = None

class MarketContextBuilder:
    """Builds comprehensive market context for analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            print("Warning: No Polygon API Key provided. Using mock data.")
        self.client = RESTClient(self.api_key) if self.api_key else None
        
    def build_context(self, 
                     symbol: str,
                     timeframes: List[str] = ["1", "5", "15", "60"]) -> MarketContext:
        """
        Build comprehensive market context
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes in minutes
            
        Returns:
            MarketContext object
        """
        if not self.api_key:
            return self._build_mock_context(symbol, timeframes)
            
        # Get price data for multiple timeframes
        price_data = self._get_multi_timeframe_data(symbol, timeframes)
        
        # Get sector performance
        sector_performance = self._get_sector_performance(symbol)
        
        # Get market breadth
        market_breadth = self._calculate_market_breadth()
        
        # Get VIX level
        vix_level = self._get_vix_level()
        
        # Get news sentiment
        news_sentiment = self._get_news_sentiment(symbol)
        
        # Get option chain if available
        option_chain = self._get_option_chain(symbol)
        
        return MarketContext(
            symbol=symbol,
            timestamp=datetime.now(),
            price_data=price_data,
            sector_performance=sector_performance,
            market_breadth=market_breadth,
            vix_level=vix_level,
            news_sentiment=news_sentiment,
            option_chain=option_chain
        )
    
    def _build_mock_context(self, symbol: str, timeframes: List[str]) -> MarketContext:
        """Build mock market context for testing"""
        # Generate mock price data
        price_data = {}
        for tf in timeframes:
            dates = pd.date_range(end=datetime.now(), periods=100, freq=f"{tf}min")
            data = {
                'open': np.random.normal(100, 1, 100),
                'high': np.random.normal(101, 1, 100),
                'low': np.random.normal(99, 1, 100),
                'close': np.random.normal(100, 1, 100),
                'volume': np.random.randint(1000, 10000, 100)
            }
            df = pd.DataFrame(data, index=dates)
            price_data[tf] = df
        
        # Mock sector performance
        sectors = ['Technology', 'Financial', 'Healthcare', 'Energy', 'Industrial']
        sector_performance = {sector: np.random.uniform(-0.02, 0.02) for sector in sectors}
        
        # Mock market breadth
        market_breadth = {
            'advance_decline_ratio': np.random.uniform(0.4, 0.6),
            'advances': np.random.randint(100, 200),
            'declines': np.random.randint(100, 200)
        }
        
        # Mock VIX level
        vix_level = np.random.uniform(15, 25)
        
        # Mock news sentiment
        news_sentiment = {
            'overall': np.random.uniform(0.3, 0.7),
            'recent': np.random.uniform(0.3, 0.7),
            'volume': np.random.uniform(0.3, 0.7)
        }
        
        # Mock option chain
        strikes = np.arange(90, 110, 2)
        option_chain = pd.DataFrame({
            'strike': strikes,
            'expiry': [datetime.now() + timedelta(days=30)] * len(strikes),
            'type': ['call'] * len(strikes),
            'volume': np.random.randint(100, 1000, len(strikes)),
            'open_interest': np.random.randint(1000, 10000, len(strikes)),
            'implied_volatility': np.random.uniform(0.2, 0.4, len(strikes))
        })
        
        return MarketContext(
            symbol=symbol,
            timestamp=datetime.now(),
            price_data=price_data,
            sector_performance=sector_performance,
            market_breadth=market_breadth,
            vix_level=vix_level,
            news_sentiment=news_sentiment,
            option_chain=option_chain
        )
    
    def _get_multi_timeframe_data(self, 
                                symbol: str, 
                                timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Get price data for multiple timeframes"""
        data = {}
        for tf in timeframes:
            # Get minute bars
            bars = self.client.get_aggs(
                symbol,
                multiplier=int(tf),
                timespan="minute",
                from_=datetime.now() - timedelta(days=5),
                to=datetime.now()
            )
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            } for bar in bars])
            
            df.set_index('timestamp', inplace=True)
            data[tf] = df
            
        return data
    
    def _get_sector_performance(self, symbol: str) -> Dict[str, float]:
        """Get sector performance metrics"""
        # Get sector ETF performance
        sectors = {
            'XLK': 'Technology',
            'XLF': 'Financial',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrial',
            'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary',
            'XLB': 'Materials',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate'
        }
        
        performance = {}
        for etf, sector in sectors.items():
            try:
                # Get daily change
                bars = self.client.get_aggs(
                    etf,
                    multiplier=1,
                    timespan="day",
                    from_=datetime.now() - timedelta(days=2),
                    to=datetime.now()
                )
                
                if len(bars) >= 2:
                    change = (bars[-1].close - bars[-2].close) / bars[-2].close
                    performance[sector] = change
            except Exception as e:
                print(f"Error getting sector performance for {sector}: {e}")
                
        return performance
    
    def _calculate_market_breadth(self) -> Dict[str, float]:
        """Calculate market breadth indicators"""
        try:
            # Get S&P 500 components
            sp500 = self.client.get_ticker_details("SPY")
            
            # Calculate advance-decline ratio
            advances = 0
            declines = 0
            
            for symbol in sp500.components:
                try:
                    bars = self.client.get_aggs(
                        symbol,
                        multiplier=1,
                        timespan="day",
                        from_=datetime.now() - timedelta(days=2),
                        to=datetime.now()
                    )
                    
                    if len(bars) >= 2:
                        if bars[-1].close > bars[-2].close:
                            advances += 1
                        else:
                            declines += 1
                except Exception:
                    continue
            
            ad_ratio = advances / (advances + declines) if (advances + declines) > 0 else 0.5
            
            return {
                'advance_decline_ratio': ad_ratio,
                'advances': advances,
                'declines': declines
            }
            
        except Exception as e:
            print(f"Error calculating market breadth: {e}")
            return {
                'advance_decline_ratio': 0.5,
                'advances': 0,
                'declines': 0
            }
    
    def _get_vix_level(self) -> float:
        """Get current VIX level"""
        try:
            bars = self.client.get_aggs(
                "^VIX",
                multiplier=1,
                timespan="minute",
                from_=datetime.now() - timedelta(minutes=5),
                to=datetime.now()
            )
            
            return bars[-1].close if bars else 20.0
            
        except Exception as e:
            print(f"Error getting VIX level: {e}")
            return 20.0
    
    def _get_news_sentiment(self, symbol: str) -> Dict[str, float]:
        """Get news sentiment analysis"""
        # TODO: Implement news sentiment analysis
        # This could be integrated with a news API and sentiment analysis service
        return {
            'overall': 0.5,
            'recent': 0.5,
            'volume': 0.5
        }
    
    def _get_option_chain(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get option chain data"""
        try:
            # Get option chain
            options = self.client.get_options_contracts(
                underlying_ticker=symbol,
                expiration_date=datetime.now() + timedelta(days=30)
            )
            
            if not options:
                return None
            
            # Convert to DataFrame
            data = []
            for opt in options:
                data.append({
                    'strike': opt.strike_price,
                    'expiry': opt.expiration_date,
                    'type': opt.contract_type,
                    'volume': opt.volume,
                    'open_interest': opt.open_interest,
                    'implied_volatility': opt.implied_volatility
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error getting option chain: {e}")
            return None 