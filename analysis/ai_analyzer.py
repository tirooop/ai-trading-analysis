"""
AI 分析器 - 集成 DeepSeek API
"""

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass

@dataclass
class SignalStrength:
    """Signal strength configuration"""
    min_strength: float = 0.6  # Minimum signal strength to consider
    max_strength: float = 1.0  # Maximum possible signal strength

@dataclass
class SignalConfig:
    """Configuration for signal generation"""
    lookback_periods: int = 20  # Number of periods to look back
    volume_threshold: float = 1.5  # Volume threshold multiplier
    trend_threshold: float = 0.02  # Trend threshold (2%)
    volatility_threshold: float = 0.015  # Volatility threshold (1.5%)

class AIAnalyzer:
    """AI-powered market analyzer for generating trading signals"""
    
    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self.signal_strength = SignalStrength()
        
    def analyze_market_data(self, 
                          df: pd.DataFrame,
                          symbol: str) -> Dict:
        """
        Analyze market data and generate trading signals
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            
        Returns:
            Dict containing analysis results and signals
        """
        # Calculate technical indicators
        df = self._calculate_indicators(df)
        
        # Generate signals
        signals = self._generate_signals(df)
        
        # Calculate signal strength
        strength = self._calculate_signal_strength(df, signals)
        
        # Generate market analysis
        analysis = self._analyze_market_conditions(df)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'signals': signals,
            'strength': strength,
            'analysis': analysis,
            'indicators': self._get_indicator_summary(df)
        }
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def _generate_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate trading signals based on technical analysis"""
        signals = {
            'trend': self._analyze_trend(df),
            'momentum': self._analyze_momentum(df),
            'volatility': self._analyze_volatility(df),
            'volume': self._analyze_volume(df)
        }
        
        # Combine signals
        combined_signal = self._combine_signals(signals)
        
        return {
            'individual': signals,
            'combined': combined_signal
        }
    
    def _calculate_signal_strength(self, 
                                 df: pd.DataFrame, 
                                 signals: Dict) -> float:
        """Calculate overall signal strength"""
        # Get the latest values
        latest = df.iloc[-1]
        
        # Calculate strength based on multiple factors
        trend_strength = abs(signals['individual']['trend'])
        momentum_strength = abs(signals['individual']['momentum'])
        volatility_strength = signals['individual']['volatility']
        volume_strength = signals['individual']['volume']
        
        # Weighted average of strengths
        strength = (
            0.4 * trend_strength +
            0.3 * momentum_strength +
            0.2 * volatility_strength +
            0.1 * volume_strength
        )
        
        return min(max(strength, 0), 1)
    
    def _analyze_market_conditions(self, df: pd.DataFrame) -> Dict:
        """Analyze overall market conditions"""
        latest = df.iloc[-1]
        
        return {
            'trend': self._get_trend_description(latest),
            'momentum': self._get_momentum_description(latest),
            'volatility': self._get_volatility_description(latest),
            'volume': self._get_volume_description(latest),
            'support_resistance': self._identify_support_resistance(df)
        }
    
    def _get_indicator_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary of technical indicators"""
        latest = df.iloc[-1]
        
        return {
            'rsi': latest['rsi'],
            'macd': latest['macd'],
            'macd_signal': latest['macd_signal'],
            'bb_upper': latest['bb_upper'],
            'bb_middle': latest['bb_middle'],
            'bb_lower': latest['bb_lower'],
            'volume_sma': latest['volume_sma'],
            'volume_ratio': latest['volume_ratio']
        }
    
    # Technical Analysis Methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, 
                       prices: pd.Series, 
                       fast: int = 12, 
                       slow: int = 26, 
                       signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    def _calculate_bollinger_bands(self, 
                                 prices: pd.Series, 
                                 period: int = 20, 
                                 std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    # Signal Generation Methods
    def _analyze_trend(self, df: pd.DataFrame) -> float:
        """Analyze price trend"""
        latest = df.iloc[-1]
        sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
        sma_50 = df['close'].rolling(window=50).mean().iloc[-1]
        
        # Calculate trend strength
        price_vs_sma20 = (latest['close'] - sma_20) / sma_20
        sma20_vs_sma50 = (sma_20 - sma_50) / sma_50
        
        return (price_vs_sma20 + sma20_vs_sma50) / 2
    
    def _analyze_momentum(self, df: pd.DataFrame) -> float:
        """Analyze price momentum"""
        latest = df.iloc[-1]
        
        # RSI momentum
        rsi_signal = (latest['rsi'] - 50) / 50
        
        # MACD momentum
        macd_signal = latest['macd'] / latest['close']
        
        return (rsi_signal + macd_signal) / 2
    
    def _analyze_volatility(self, df: pd.DataFrame) -> float:
        """Analyze market volatility"""
        latest = df.iloc[-1]
        
        # Bollinger Band width
        bb_width = (latest['bb_upper'] - latest['bb_lower']) / latest['bb_middle']
        
        # Price position in BB
        bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
        
        return (bb_width + bb_position) / 2
    
    def _analyze_volume(self, df: pd.DataFrame) -> float:
        """Analyze trading volume"""
        latest = df.iloc[-1]
        
        # Volume ratio
        volume_signal = latest['volume_ratio'] - 1
        
        # Volume trend
        volume_trend = df['volume'].pct_change().rolling(window=5).mean().iloc[-1]
        
        return (volume_signal + volume_trend) / 2
    
    def _combine_signals(self, signals: Dict[str, float]) -> float:
        """Combine individual signals into a single signal"""
        weights = {
            'trend': 0.4,
            'momentum': 0.3,
            'volatility': 0.2,
            'volume': 0.1
        }
        
        combined = sum(signals[key] * weight for key, weight in weights.items())
        return max(min(combined, 1), -1)
    
    # Market Analysis Methods
    def _get_trend_description(self, latest: pd.Series) -> str:
        """Get description of current trend"""
        if latest['close'] > latest['bb_upper']:
            return "Strong Uptrend"
        elif latest['close'] > latest['bb_middle']:
            return "Moderate Uptrend"
        elif latest['close'] < latest['bb_lower']:
            return "Strong Downtrend"
        elif latest['close'] < latest['bb_middle']:
            return "Moderate Downtrend"
        else:
            return "Sideways"
    
    def _get_momentum_description(self, latest: pd.Series) -> str:
        """Get description of current momentum"""
        if latest['rsi'] > 70:
            return "Overbought"
        elif latest['rsi'] < 30:
            return "Oversold"
        elif latest['macd'] > latest['macd_signal']:
            return "Bullish Momentum"
        else:
            return "Bearish Momentum"
    
    def _get_volatility_description(self, latest: pd.Series) -> str:
        """Get description of current volatility"""
        bb_width = (latest['bb_upper'] - latest['bb_lower']) / latest['bb_middle']
        if bb_width > 0.05:
            return "High Volatility"
        elif bb_width > 0.03:
            return "Moderate Volatility"
        else:
            return "Low Volatility"
    
    def _get_volume_description(self, latest: pd.Series) -> str:
        """Get description of current volume"""
        if latest['volume_ratio'] > 2:
            return "Very High Volume"
        elif latest['volume_ratio'] > 1.5:
            return "High Volume"
        elif latest['volume_ratio'] < 0.5:
            return "Low Volume"
        else:
            return "Normal Volume"
    
    def _identify_support_resistance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Identify support and resistance levels"""
        # Simple implementation using recent highs and lows
        recent_high = df['high'].rolling(window=20).max().iloc[-1]
        recent_low = df['low'].rolling(window=20).min().iloc[-1]
        
        return {
            'resistance': recent_high,
            'support': recent_low
        }
            
    def analyze_market_context(self, 
                             technical_data: Dict,
                             option_data: Dict,
                             news_data: Optional[Dict] = None) -> Dict:
        """
        使用 AI 分析市场环境
        Args:
            technical_data: 技术面数据
            option_data: 期权面数据
            news_data: 新闻数据（可选）
        Returns:
            AI 分析结果
        """
        # 构建 prompt
        prompt = self._build_analysis_prompt(
            technical_data,
            option_data,
            news_data
        )
        
        # TODO: 调用 DeepSeek API
        # response = self._call_deepseek_api(prompt)
        
        # 解析响应
        analysis = {
            'market_sentiment': 'bullish',  # AI 分析的市场情绪
            'trend_strength': 0.85,         # 趋势强度评分
            'risk_level': 'medium',         # 风险水平评估
            'key_factors': [                # 关键影响因素
                '技术面呈现强势上涨趋势',
                '期权链显示机构大量做多',
                '市场情绪偏向乐观'
            ],
            'trading_suggestions': [         # 交易建议
                {
                    'type': 'LONG_CALL',
                    'confidence': 0.87,
                    'reason': '技术面和期权面都支持上涨',
                    'risk_reward_ratio': 3.5
                }
            ]
        }
        
        return analysis
        
    def generate_option_strategy(self,
                               symbol: str,
                               price_data: pd.DataFrame,
                               option_chain: pd.DataFrame,
                               risk_preference: str = 'moderate') -> Dict:
        """
        生成期权策略建议
        Args:
            symbol: 股票代码
            price_data: 价格数据
            option_chain: 期权链数据
            risk_preference: 风险偏好 ('conservative'/'moderate'/'aggressive')
        Returns:
            策略建议
        """
        # 构建 prompt
        prompt = self._build_strategy_prompt(
            symbol,
            price_data,
            option_chain,
            risk_preference
        )
        
        # TODO: 调用 DeepSeek API
        # response = self._call_deepseek_api(prompt)
        
        # 解析响应
        strategy = {
            'type': 'BULL_CALL_SPREAD',
            'legs': [
                {
                    'action': 'BUY',
                    'strike': 180,
                    'expiry': '2024-05-17',
                    'contracts': 1
                },
                {
                    'action': 'SELL',
                    'strike': 190,
                    'expiry': '2024-05-17',
                    'contracts': 1
                }
            ],
            'max_loss': 500,
            'max_profit': 1000,
            'probability_of_profit': 0.65,
            'rationale': [
                '技术面支持继续上涨',
                '波动率适中，适合构建垂直价差',
                '风险收益比合理'
            ]
        }
        
        return strategy
        
    def _build_analysis_prompt(self,
                             technical_data: Dict,
                             option_data: Dict,
                             news_data: Optional[Dict]) -> str:
        """构建市场分析 prompt"""
        prompt = f"""
        请分析以下市场数据并给出交易建议：
        
        技术面数据：
        - 当前价格：${technical_data['current_price']}
        - RSI：{technical_data['rsi']}
        - MACD：{technical_data['macd']}
        - 支撑位：${technical_data['support']}
        - 压力位：${technical_data['resistance']}
        
        期权数据：
        - 看涨看跌比率：{option_data['call_put_ratio']}
        - 隐含波动率：{option_data['implied_volatility']}%
        - 期权链异常活动：{option_data['unusual_activity']}
        """
        
        if news_data:
            prompt += f"""
            市场新闻：
            - 最新消息：{news_data['latest_news']}
            - 市场情绪：{news_data['sentiment']}
            """
            
        return prompt
        
    def _build_strategy_prompt(self,
                             symbol: str,
                             price_data: pd.DataFrame,
                             option_chain: pd.DataFrame,
                             risk_preference: str) -> str:
        """构建策略生成 prompt"""
        prompt = f"""
        请为 {symbol} 设计一个期权交易策略，风险偏好为 {risk_preference}：
        
        价格数据：
        - 当前价格：${price_data['close'].iloc[-1]}
        - 52周高点：${price_data['high'].max()}
        - 52周低点：${price_data['low'].min()}
        
        期权数据：
        - 可用期权到期日：{option_chain['expiry'].unique().tolist()}
        - 当前隐含波动率范围：{option_chain['implied_volatility'].min():.1f}% - {option_chain['implied_volatility'].max():.1f}%
        
        请考虑以下因素：
        1. 风险收益比
        2. 最大损失控制
        3. 胜率要求
        4. 时间衰减影响
        """
        
        return prompt 

    def generate_trade_signal(self, context: Dict) -> Dict:
        """
        Generate trading signal based on market context
        
        Args:
            context: Dictionary containing market context data
                - symbol: Trading symbol
                - price_data: DataFrame with OHLCV data
                - technical_signals: Dictionary of technical indicators
                - options_data: Dictionary of option chain data
                - current_price: Current price
                - timestamp: Current timestamp
                
        Returns:
            Dictionary containing trading signal and analysis
        """
        try:
            # Extract data from context
            symbol = context['symbol']
            price_data = context['price_data']
            technical_signals = context['technical_signals']
            options_data = context.get('options_data', {})
            
            # Analyze market data
            analysis = self.analyze_market_data(price_data, symbol)
            
            # Generate option strategy if options data is available
            strategy = None
            if options_data:
                strategy = self.generate_option_strategy(
                    symbol=symbol,
                    price_data=price_data,
                    option_chain=options_data,
                    risk_preference='moderate'
                )
            
            # Combine all signals
            signal = {
                'symbol': symbol,
                'timestamp': context['timestamp'],
                'current_price': context['current_price'],
                'strength': analysis['strength'],
                'direction': 'BUY' if analysis['strength'] > 0.7 else 'SELL' if analysis['strength'] < 0.3 else 'HOLD',
                'analysis': analysis['analysis'],
                'indicators': analysis['indicators'],
                'strategy': strategy
            }
            
            return signal
            
        except Exception as e:
            print(f"Error generating trade signal: {str(e)}")
            return {
                'symbol': context['symbol'],
                'timestamp': context['timestamp'],
                'current_price': context['current_price'],
                'strength': 0.5,
                'direction': 'HOLD',
                'error': str(e)
            } 