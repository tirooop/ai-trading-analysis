"""
期权信号检测器
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import logging

from data.market_data import MarketDataSource
from analysis.technical_analysis import TechnicalAnalyzer
from analysis.ai_analyzer import AIAnalyzer

class OptionSignalDetector:
    """期权信号检测器"""
    
    def __init__(self, 
                 data_source: MarketDataSource,
                 technical_analyzer: TechnicalAnalyzer,
                 ai_analyzer: Optional[AIAnalyzer] = None):
        """
        初始化期权信号检测器
        Args:
            data_source: 市场数据源
            technical_analyzer: 技术分析器
            ai_analyzer: AI分析器（可选）
        """
        self.data_source = data_source
        self.technical_analyzer = technical_analyzer
        self.ai_analyzer = ai_analyzer
        
        # 加载配置
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        with open('config/option_signal_config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    def _calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """计算EMA"""
        return data.ewm(span=period, adjust=False).mean()
        
    def _calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _check_ema_crossover(self, 
                           data: pd.DataFrame,
                           short_period: int,
                           long_period: int) -> bool:
        """检查EMA交叉信号"""
        ema_short = self._calculate_ema(data['close'], short_period)
        ema_long = self._calculate_ema(data['close'], long_period)
        
        # 检查最新两根K线的交叉
        prev_short = ema_short.iloc[-2]
        prev_long = ema_long.iloc[-2]
        curr_short = ema_short.iloc[-1]
        curr_long = ema_long.iloc[-1]
        
        return (prev_short <= prev_long) and (curr_short > curr_long)
        
    def _check_rsi_signal(self, 
                         data: pd.DataFrame,
                         threshold: float,
                         direction: str) -> bool:
        """检查RSI信号"""
        rsi = self._calculate_rsi(data['close'])
        current_rsi = rsi.iloc[-1]
        
        if direction.lower() == 'long':
            return current_rsi > threshold
        else:
            return current_rsi < threshold
            
    def _check_volume_signal(self,
                           data: pd.DataFrame,
                           multiplier: float) -> bool:
        """检查成交量信号"""
        avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
        current_volume = data['volume'].iloc[-1]
        
        return current_volume > (avg_volume * multiplier)
        
    def _calculate_signal_confidence(self,
                                   ema_signal: bool,
                                   rsi_signal: bool,
                                   volume_signal: bool) -> float:
        """计算信号置信度"""
        confidence = 0.0
        
        if ema_signal:
            confidence += 40
        if rsi_signal:
            confidence += 30
        if volume_signal:
            confidence += 30
            
        return confidence
        
    def detect_signals(self,
                      symbol: str,
                      contract_type: str,
                      strike: float,
                      expiry: str,
                      strategy: Dict) -> Optional[Dict]:
        """
        检测期权信号
        Args:
            symbol: 股票代码
            contract_type: 期权类型（call/put）
            strike: 行权价
            expiry: 到期日
            strategy: 策略配置
        Returns:
            信号信息字典，如果没有信号则返回None
        """
        try:
            # 获取期权数据
            data = self.data_source.get_option_data(
                symbol=symbol,
                contract_type=contract_type,
                strike=strike,
                expiry=expiry,
                interval='1m'  # 使用1分钟数据
            )
            
            if data is None or len(data) < 30:  # 确保有足够的数据
                return None
                
            # 检查EMA交叉
            ema_signal = self._check_ema_crossover(
                data,
                strategy['ema_short'],
                strategy['ema_long']
            )
            
            # 检查RSI
            rsi_signal = self._check_rsi_signal(
                data,
                strategy['rsi_threshold'],
                strategy['direction']
            )
            
            # 检查成交量
            volume_signal = self._check_volume_signal(
                data,
                strategy['vol_multiplier']
            )
            
            # 计算置信度
            confidence = self._calculate_signal_confidence(
                ema_signal,
                rsi_signal,
                volume_signal
            )
            
            # 如果置信度低于阈值，不生成信号
            if confidence < self.config['monitoring']['min_confidence']:
                return None
                
            # 生成信号信息
            signal = {
                'symbol': symbol,
                'contract_type': contract_type,
                'strike': strike,
                'expiry': expiry,
                'strategy_name': strategy['name'],
                'direction': strategy['direction'],
                'price': data['close'].iloc[-1],
                'confidence': confidence,
                'signals': {
                    'ema_crossover': ema_signal,
                    'rsi_signal': rsi_signal,
                    'volume_signal': volume_signal
                },
                'timestamp': datetime.now()
            }
            
            # 如果有AI分析器，添加AI分析
            if self.ai_analyzer:
                ai_analysis = self.ai_analyzer.analyze_option_signal(signal)
                signal['ai_analysis'] = ai_analysis
                
            return signal
            
        except Exception as e:
            logging.error(f"检测期权信号时出错: {str(e)}")
            return None
            
    def monitor_contracts(self) -> List[Dict]:
        """
        监控所有配置的合约
        Returns:
            信号列表
        """
        signals = []
        
        for item in self.config['watchlist']:
            symbol = item['symbol']
            
            for contract in item['contracts']:
                for strategy in contract['strategies']:
                    signal = self.detect_signals(
                        symbol=symbol,
                        contract_type=contract['type'],
                        strike=contract['strike'],
                        expiry=contract['expiry'],
                        strategy=strategy
                    )
                    
                    if signal:
                        signals.append(signal)
                        
        return signals 