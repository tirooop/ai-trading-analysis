from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from dotenv import load_dotenv
import requests
from .market_context_builder import MarketContextBuilder, MarketContext
from .llm_prompt_builder import LLMPromptBuilder, MarketData

# Load environment variables from API.env
load_dotenv('API.env')

@dataclass
class AnalysisResult:
    """Analysis result with detailed breakdown"""
    timestamp: datetime
    symbol: str
    direction: str  # "BULLISH", "BEARISH", or "NEUTRAL"
    confidence: float  # 0-1
    logic_chain: List[Dict[str, str]]  # List of reasoning steps
    risk_factors: List[Dict[str, str]]  # List of risk factors
    trade_suggestions: List[Dict[str, str]]  # List of trade suggestions
    market_context: Dict[str, any]  # Market context summary
    llm_analysis: Dict[str, any]  # Raw LLM analysis
    timeframe_analysis: Dict[str, Dict[str, any]]  # Multi-timeframe analysis

class AIAnalystV2:
    """Enhanced AI analyst with comprehensive market analysis"""
    
    def __init__(self, api_key: Optional[str] = None, llm_api_key: Optional[str] = None):
        self.context_builder = MarketContextBuilder(api_key)
        self.prompt_builder = LLMPromptBuilder()
        self.llm_api_key = llm_api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.llm_api_key:
            raise ValueError("DeepSeek API Key is required")
        
        # Supported timeframes for analysis
        self.timeframes = ['1', '3', '15', 'D']
        
    def analyze(self, symbol: str) -> AnalysisResult:
        """
        Perform comprehensive market analysis
        
        Args:
            symbol: Trading symbol to analyze
            
        Returns:
            AnalysisResult with detailed analysis
        """
        # Build market context for all timeframes
        context = self.context_builder.build_context(symbol)
        
        # Prepare market data for LLM
        market_data = self._prepare_market_data(context)
        
        # Get LLM analysis
        llm_analysis = self._get_llm_analysis(market_data)
        
        # Analyze market structure across timeframes
        structure_analysis = self._analyze_market_structure(context)
        
        # Analyze multi-timeframe trends
        trend_analysis = self._analyze_multi_timeframe_trends(context)
        
        # Analyze sector strength
        sector_analysis = self._analyze_sector_strength(context)
        
        # Analyze market breadth
        breadth_analysis = self._analyze_market_breadth(context)
        
        # Analyze volatility
        volatility_analysis = self._analyze_volatility(context)
        
        # Generate trading signals
        signals = self._generate_trading_signals(
            structure_analysis,
            trend_analysis,
            sector_analysis,
            breadth_analysis,
            volatility_analysis
        )
        
        # Build logic chain
        logic_chain = self._build_logic_chain(
            structure_analysis,
            trend_analysis,
            sector_analysis,
            breadth_analysis,
            volatility_analysis
        )
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(context)
        
        # Generate trade suggestions
        trade_suggestions = self._generate_trade_suggestions(
            signals,
            context,
            risk_factors
        )
        
        # Determine overall direction and confidence
        direction, confidence = self._determine_direction_and_confidence(signals)
        
        return AnalysisResult(
            timestamp=datetime.now(),
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            logic_chain=logic_chain,
            risk_factors=risk_factors,
            trade_suggestions=trade_suggestions,
            market_context=self._summarize_market_context(context),
            llm_analysis=llm_analysis,
            timeframe_analysis=self._analyze_timeframes(context)
        )
    
    def _call_llm_api(self, prompt_text: str) -> str:
        """
        Call DeepSeek API for market analysis
        
        Args:
            prompt_text: The prompt to send to the API
            
        Returns:
            API response text
        """
        try:
            response = requests.post(
                "https://api.siliconflow.cn/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.llm_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "You are an expert trader with 20 years of experience."},
                        {"role": "user", "content": prompt_text}
                    ],
                    "temperature": 0.2
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                print(f"Error calling DeepSeek API: {response.status_code}")
                return json.dumps({"error": f"API call failed with status {response.status_code}"})
                
        except Exception as e:
            print(f"Error calling DeepSeek API: {e}")
            return json.dumps({"error": str(e)})
    
    def _get_llm_analysis(self, market_data: MarketData) -> Dict[str, any]:
        """Get analysis from LLM"""
        # Build prompt
        prompt = self.prompt_builder.build_prompt(market_data)
        
        # Call API
        response = self._call_llm_api(prompt)
        
        # Parse response
        try:
            return self.prompt_builder.parse_response(response)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return {"error": str(e)}
    
    def _analyze_timeframes(self, context: MarketContext) -> Dict[str, Dict[str, any]]:
        """Analyze market data across multiple timeframes"""
        timeframe_analysis = {}
        
        for timeframe in self.timeframes:
            if timeframe not in context.price_data:
                continue
                
            data = context.price_data[timeframe]
            
            # Calculate technical indicators
            rsi = self._calculate_rsi(data['close'])
            macd = self._calculate_macd(data['close'])
            ema_20 = data['close'].ewm(span=20).mean()
            ema_50 = data['close'].ewm(span=50).mean()
            
            # Calculate volume change
            volume_ma = data['volume'].rolling(20).mean()
            volume_change = (data['volume'].iloc[-1] / volume_ma.iloc[-1] - 1) * 100
            
            # Store analysis
            timeframe_analysis[timeframe] = {
                'rsi': rsi.iloc[-1],
                'macd': macd.iloc[-1],
                'ema_trend': "BULLISH" if ema_20.iloc[-1] > ema_50.iloc[-1] else "BEARISH",
                'volume_change': volume_change,
                'support_levels': self._find_support_levels(data),
                'resistance_levels': self._find_resistance_levels(data)
            }
        
        return timeframe_analysis
    
    def _calculate_trend_agreement(self, timeframe_analysis: Dict[str, Dict[str, any]]) -> float:
        """Calculate trend agreement score across timeframes"""
        if not timeframe_analysis:
            return 0.5
            
        # Count bullish and bearish trends
        bullish_count = 0
        bearish_count = 0
        
        for analysis in timeframe_analysis.values():
            if analysis['ema_trend'] == "BULLISH":
                bullish_count += 1
            elif analysis['ema_trend'] == "BEARISH":
                bearish_count += 1
        
        total = len(timeframe_analysis)
        if total == 0:
            return 0.5
            
        # Calculate agreement score
        max_count = max(bullish_count, bearish_count)
        return max_count / total
    
    def _calculate_volatility_coherence(self, timeframe_analysis: Dict[str, Dict[str, any]]) -> float:
        """Calculate volatility coherence across timeframes"""
        if not timeframe_analysis:
            return 0.5
            
        # Calculate RSI standard deviation
        rsi_values = [analysis['rsi'] for analysis in timeframe_analysis.values()]
        rsi_std = np.std(rsi_values)
        
        # Normalize to 0-1 scale (assuming RSI typically ranges from 20 to 80)
        coherence = 1 - (rsi_std / 30)  # 30 is max expected std for RSI
        return max(0, min(1, coherence))
    
    def _enhance_confidence(self, 
                          base_confidence: float,
                          timeframe_analysis: Dict[str, Dict[str, any]]) -> float:
        """Enhance confidence based on multi-timeframe analysis"""
        trend_agreement = self._calculate_trend_agreement(timeframe_analysis)
        volatility_coherence = self._calculate_volatility_coherence(timeframe_analysis)
        
        # Weight the factors
        weights = {
            'base': 0.4,
            'trend_agreement': 0.4,
            'volatility_coherence': 0.2
        }
        
        enhanced_confidence = (
            base_confidence * weights['base'] +
            trend_agreement * weights['trend_agreement'] +
            volatility_coherence * weights['volatility_coherence']
        )
        
        return min(1.0, enhanced_confidence)
    
    def _prepare_market_data(self, context: MarketContext) -> MarketData:
        """Prepare market data for LLM analysis"""
        # Get latest price data
        latest_data = context.price_data['1'].iloc[-1]
        
        # Calculate technical indicators
        rsi = self._calculate_rsi(context.price_data['1']['close']).iloc[-1]
        macd = self._calculate_macd(context.price_data['1']['close']).iloc[-1]
        
        # Determine EMA trend
        ema_20 = context.price_data['1']['close'].ewm(span=20).mean().iloc[-1]
        ema_50 = context.price_data['1']['close'].ewm(span=50).mean().iloc[-1]
        ema_trend = "BULLISH" if ema_20 > ema_50 else "BEARISH"
        
        # Determine MACD signal
        macd_signal = "BULLISH" if macd > 0 else "BEARISH"
        
        # Calculate volume factor
        volume_ma = context.price_data['1']['volume'].rolling(20).mean().iloc[-1]
        volume_factor = "HIGH" if latest_data['volume'] > volume_ma * 1.5 else "LOW" if latest_data['volume'] < volume_ma * 0.5 else "NORMAL"
        
        return MarketData(
            symbol=context.symbol,
            current_price=latest_data['close'],
            ema_trend=ema_trend,
            macd_signal=macd_signal,
            rsi=rsi,
            volume_factor=volume_factor,
            support_levels=self._find_support_levels(context.price_data['1']),
            resistance_levels=self._find_resistance_levels(context.price_data['1']),
            market_sentiment=self._determine_market_sentiment(context),
            sector_strength=context.sector_performance,
            news_summary=self._summarize_news(context.news_sentiment),
            vix_level=context.vix_level,
            market_breadth=context.market_breadth
        )
    
    def _determine_market_sentiment(self, context: MarketContext) -> str:
        """Determine overall market sentiment"""
        # Combine multiple factors
        factors = []
        
        # VIX level
        if context.vix_level > 25:
            factors.append("HIGH_VOLATILITY")
        elif context.vix_level < 15:
            factors.append("LOW_VOLATILITY")
            
        # Market breadth
        if context.market_breadth['advance_decline_ratio'] > 0.6:
            factors.append("STRONG_BREADTH")
        elif context.market_breadth['advance_decline_ratio'] < 0.4:
            factors.append("WEAK_BREADTH")
            
        # Sector performance
        strong_sectors = sum(1 for p in context.sector_performance.values() if p > 0.01)
        weak_sectors = sum(1 for p in context.sector_performance.values() if p < -0.01)
        
        if strong_sectors > weak_sectors:
            factors.append("SECTOR_STRENGTH")
        elif weak_sectors > strong_sectors:
            factors.append("SECTOR_WEAKNESS")
            
        # News sentiment
        if context.news_sentiment['overall'] > 0.6:
            factors.append("POSITIVE_NEWS")
        elif context.news_sentiment['overall'] < 0.4:
            factors.append("NEGATIVE_NEWS")
            
        # Determine overall sentiment
        if len(factors) == 0:
            return "NEUTRAL"
            
        positive_factors = sum(1 for f in factors if f in ["STRONG_BREADTH", "SECTOR_STRENGTH", "POSITIVE_NEWS", "LOW_VOLATILITY"])
        negative_factors = sum(1 for f in factors if f in ["WEAK_BREADTH", "SECTOR_WEAKNESS", "NEGATIVE_NEWS", "HIGH_VOLATILITY"])
        
        if positive_factors > negative_factors:
            return "BULLISH"
        elif negative_factors > positive_factors:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _summarize_news(self, news_sentiment: Dict[str, float]) -> str:
        """Summarize news sentiment"""
        if news_sentiment['overall'] > 0.6:
            return "市场新闻整体偏多"
        elif news_sentiment['overall'] < 0.4:
            return "市场新闻整体偏空"
        else:
            return "市场新闻中性"
    
    def _analyze_market_structure(self, context: MarketContext) -> Dict[str, any]:
        """Analyze market structure across timeframes"""
        structure = {}
        
        for timeframe, data in context.price_data.items():
            # Calculate key levels
            highs = data['high'].rolling(20).max()
            lows = data['low'].rolling(20).min()
            
            # Identify support and resistance
            support_levels = self._find_support_levels(data)
            resistance_levels = self._find_resistance_levels(data)
            
            # Identify chart patterns
            patterns = self._identify_chart_patterns(data)
            
            structure[timeframe] = {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'patterns': patterns,
                'trend_strength': self._calculate_trend_strength(data)
            }
            
        return structure
    
    def _analyze_multi_timeframe_trends(self, context: MarketContext) -> Dict[str, any]:
        """Analyze trends across multiple timeframes"""
        trends = {}
        
        for timeframe, data in context.price_data.items():
            # Calculate trend indicators
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()
            sma_200 = data['close'].rolling(200).mean()
            
            # Calculate momentum
            rsi = self._calculate_rsi(data['close'])
            macd = self._calculate_macd(data['close'])
            
            trends[timeframe] = {
                'trend': self._determine_trend(sma_20, sma_50, sma_200),
                'momentum': {
                    'rsi': rsi.iloc[-1],
                    'macd': macd.iloc[-1]
                },
                'trend_strength': self._calculate_trend_strength(data)
            }
            
        return trends
    
    def _analyze_sector_strength(self, context: MarketContext) -> Dict[str, any]:
        """Analyze sector strength and correlations"""
        # Calculate sector performance
        sector_performance = context.sector_performance
        
        # Calculate correlations
        correlations = {}
        for sector, performance in sector_performance.items():
            correlations[sector] = {
                'performance': performance,
                'strength': 'STRONG' if performance > 0.01 else 'WEAK' if performance < -0.01 else 'NEUTRAL'
            }
            
        return {
            'sector_performance': correlations,
            'overall_strength': np.mean([p for p in sector_performance.values()])
        }
    
    def _analyze_market_breadth(self, context: MarketContext) -> Dict[str, any]:
        """Analyze market breadth indicators"""
        breadth = context.market_breadth
        
        return {
            'advance_decline_ratio': breadth['advance_decline_ratio'],
            'market_strength': 'STRONG' if breadth['advance_decline_ratio'] > 0.6 else 'WEAK' if breadth['advance_decline_ratio'] < 0.4 else 'NEUTRAL',
            'advances': breadth['advances'],
            'declines': breadth['declines']
        }
    
    def _analyze_volatility(self, context: MarketContext) -> Dict[str, any]:
        """Analyze volatility conditions"""
        vix = context.vix_level
        
        return {
            'vix_level': vix,
            'volatility_state': 'HIGH' if vix > 25 else 'LOW' if vix < 15 else 'NORMAL',
            'risk_level': 'HIGH' if vix > 30 else 'LOW' if vix < 15 else 'MODERATE'
        }
    
    def _generate_trading_signals(self, *analyses) -> Dict[str, any]:
        """Generate trading signals from all analyses"""
        signals = {
            'structure': self._generate_structure_signals(analyses[0]),
            'trend': self._generate_trend_signals(analyses[1]),
            'sector': self._generate_sector_signals(analyses[2]),
            'breadth': self._generate_breadth_signals(analyses[3]),
            'volatility': self._generate_volatility_signals(analyses[4])
        }
        
        return signals
    
    def _build_logic_chain(self, *analyses) -> List[Dict[str, str]]:
        """Build reasoning chain from analyses"""
        logic_chain = []
        
        # Add structure analysis
        structure = analyses[0]
        for timeframe, data in structure.items():
            logic_chain.append({
                'timeframe': timeframe,
                'analysis': 'Market Structure',
                'conclusion': f"Market structure shows {data['trend_strength']} trend with {len(data['support_levels'])} support and {len(data['resistance_levels'])} resistance levels"
            })
        
        # Add trend analysis
        trends = analyses[1]
        for timeframe, data in trends.items():
            logic_chain.append({
                'timeframe': timeframe,
                'analysis': 'Trend Analysis',
                'conclusion': f"Trend is {data['trend']} with {data['momentum']['rsi']:.2f} RSI and {data['momentum']['macd']:.2f} MACD"
            })
        
        # Add sector analysis
        sector = analyses[2]
        logic_chain.append({
            'analysis': 'Sector Analysis',
            'conclusion': f"Sector strength is {sector['overall_strength']:.2%} with {sum(1 for s in sector['sector_performance'].values() if s['strength'] == 'STRONG')} strong sectors"
        })
        
        # Add breadth analysis
        breadth = analyses[3]
        logic_chain.append({
            'analysis': 'Market Breadth',
            'conclusion': f"Market breadth is {breadth['market_strength']} with {breadth['advance_decline_ratio']:.2f} advance-decline ratio"
        })
        
        # Add volatility analysis
        volatility = analyses[4]
        logic_chain.append({
            'analysis': 'Volatility Analysis',
            'conclusion': f"Volatility is {volatility['volatility_state']} with VIX at {volatility['vix_level']:.1f}"
        })
        
        return logic_chain
    
    def _identify_risk_factors(self, context: MarketContext) -> List[Dict[str, str]]:
        """Identify potential risk factors"""
        risk_factors = []
        
        # Check volatility
        if context.vix_level > 25:
            risk_factors.append({
                'factor': 'High Volatility',
                'description': f"VIX is elevated at {context.vix_level:.1f}",
                'impact': 'HIGH'
            })
        
        # Check market breadth
        if context.market_breadth['advance_decline_ratio'] < 0.4:
            risk_factors.append({
                'factor': 'Weak Market Breadth',
                'description': "Market breadth is weak with more declining stocks",
                'impact': 'MEDIUM'
            })
        
        # Check sector performance
        weak_sectors = sum(1 for p in context.sector_performance.values() if p < -0.01)
        if weak_sectors > 5:
            risk_factors.append({
                'factor': 'Sector Weakness',
                'description': f"{weak_sectors} sectors showing weakness",
                'impact': 'MEDIUM'
            })
        
        return risk_factors
    
    def _generate_trade_suggestions(self, 
                                  signals: Dict[str, any],
                                  context: MarketContext,
                                  risk_factors: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Generate trade suggestions based on analysis"""
        suggestions = []
        
        # Determine overall bias
        bias = self._determine_direction_and_confidence(signals)[0]
        
        # Generate option strategies if option chain is available
        if context.option_chain is not None:
            if bias == "BULLISH":
                suggestions.append({
                    'strategy': 'Bull Call Spread',
                    'description': 'Buy lower strike call, sell higher strike call',
                    'confidence': 'HIGH' if len(risk_factors) == 0 else 'MEDIUM'
                })
            elif bias == "BEARISH":
                suggestions.append({
                    'strategy': 'Bear Put Spread',
                    'description': 'Buy higher strike put, sell lower strike put',
                    'confidence': 'HIGH' if len(risk_factors) == 0 else 'MEDIUM'
                })
            else:
                suggestions.append({
                    'strategy': 'Iron Condor',
                    'description': 'Sell OTM put spread and OTM call spread',
                    'confidence': 'MEDIUM'
                })
        
        # Add directional suggestions
        if bias != "NEUTRAL":
            suggestions.append({
                'strategy': f"{bias} Position",
                'description': f"Take {bias.lower()} position with appropriate stop loss",
                'confidence': 'HIGH' if len(risk_factors) == 0 else 'MEDIUM'
            })
        
        return suggestions
    
    def _determine_direction_and_confidence(self, signals: Dict[str, any]) -> Tuple[str, float]:
        """Determine overall direction and confidence"""
        # Calculate direction scores
        bullish_score = 0
        bearish_score = 0
        
        # Structure signals
        if signals['structure'].get('trend', '') == 'BULLISH':
            bullish_score += 1
        elif signals['structure'].get('trend', '') == 'BEARISH':
            bearish_score += 1
        
        # Trend signals
        if signals['trend'].get('trend', '') == 'BULLISH':
            bullish_score += 1
        elif signals['trend'].get('trend', '') == 'BEARISH':
            bearish_score += 1
        
        # Sector signals
        if signals['sector'].get('overall_strength', 0) > 0:
            bullish_score += 1
        elif signals['sector'].get('overall_strength', 0) < 0:
            bearish_score += 1
        
        # Breadth signals
        if signals['breadth'].get('market_strength', '') == 'STRONG':
            bullish_score += 1
        elif signals['breadth'].get('market_strength', '') == 'WEAK':
            bearish_score += 1
        
        # Determine direction
        if bullish_score > bearish_score:
            direction = "BULLISH"
            confidence = bullish_score / (bullish_score + bearish_score)
        elif bearish_score > bullish_score:
            direction = "BEARISH"
            confidence = bearish_score / (bullish_score + bearish_score)
        else:
            direction = "NEUTRAL"
            confidence = 0.5
        
        return direction, confidence
    
    def _summarize_market_context(self, context: MarketContext) -> Dict[str, any]:
        """Summarize market context for reporting"""
        return {
            'timestamp': context.timestamp.isoformat(),
            'symbol': context.symbol,
            'vix_level': context.vix_level,
            'market_breadth': context.market_breadth,
            'sector_performance': context.sector_performance,
            'news_sentiment': context.news_sentiment
        }
    
    # Technical analysis helper methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        return exp1 - exp2
    
    def _find_support_levels(self, data: pd.DataFrame) -> List[float]:
        """Find support levels"""
        # Simple implementation - can be enhanced with more sophisticated methods
        lows = data['low'].rolling(20).min()
        return lows.dropna().unique().tolist()
    
    def _find_resistance_levels(self, data: pd.DataFrame) -> List[float]:
        """Find resistance levels"""
        # Simple implementation - can be enhanced with more sophisticated methods
        highs = data['high'].rolling(20).max()
        return highs.dropna().unique().tolist()
    
    def _identify_chart_patterns(self, data: pd.DataFrame) -> List[str]:
        """Identify chart patterns"""
        patterns = []
        # TODO: Implement pattern recognition
        return patterns
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> str:
        """Calculate trend strength"""
        # Simple implementation - can be enhanced
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        
        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            return "STRONG_UPTREND"
        elif sma_20.iloc[-1] < sma_50.iloc[-1]:
            return "STRONG_DOWNTREND"
        else:
            return "SIDEWAYS"
    
    def _determine_trend(self, sma_20: pd.Series, sma_50: pd.Series, sma_200: pd.Series) -> str:
        """Determine trend based on moving averages"""
        if sma_20.iloc[-1] > sma_50.iloc[-1] > sma_200.iloc[-1]:
            return "BULLISH"
        elif sma_20.iloc[-1] < sma_50.iloc[-1] < sma_200.iloc[-1]:
            return "BEARISH"
        else:
            return "NEUTRAL" 