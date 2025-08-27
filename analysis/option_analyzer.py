"""
期权分析器
"""

from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import time

from data.market_data import MarketDataSource
from analysis.technical_analysis import TechnicalAnalyzer
from analysis.ai_analyzer import AIAnalyzer
from utils.unified_notifier import UnifiedNotifier

class OptionAnalyzer:
    """期权分析器"""
    
    def __init__(self, 
                 data_source: MarketDataSource,
                 notifier: UnifiedNotifier,
                 ai_analyzer: Optional[AIAnalyzer] = None):
        """
        初始化期权分析器
        Args:
            data_source: 市场数据源
            notifier: 通知器
            ai_analyzer: AI分析器（可选）
        """
        self.data_source = data_source
        self.notifier = notifier
        self.ai_analyzer = ai_analyzer
        
    def analyze_entry_opportunity(self, symbol: str) -> Dict:
        """
        分析入场机会
        结合技术面和期权面的指标
        """
        # 获取市场数据
        hist_data = self.data_source.get_historical_data(
            symbol=symbol,
            start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
            end_date=datetime.now()
        )
        
        option_chain = self.data_source.get_option_chain(symbol)
        quote = self.data_source.get_realtime_quote(symbol)
        
        # 技术分析
        tech_analyzer = TechnicalAnalyzer(hist_data)
        
        # 计算技术指标
        support, resistance = tech_analyzer.calculate_support_resistance()
        momentum = tech_analyzer.calculate_momentum_indicators()
        patterns = tech_analyzer.identify_chart_patterns()
        
        # 计算EMA
        ema20 = tech_analyzer.calculate_ema(20)
        ema50 = tech_analyzer.calculate_ema(50)
        
        # 分析期权数据
        sentiment = self._analyze_option_sentiment(option_chain)
        unusual_activity = self._detect_unusual_activity(option_chain)
        
        # 准备AI分析数据
        technical_data = {
            'current_price': quote['last'],
            'rsi': momentum['rsi'],
            'macd': momentum['macd'],
            'support': support,
            'resistance': resistance,
            'ema20': ema20.iloc[-1],
            'ema50': ema50.iloc[-1]
        }
        
        option_data = {
            'call_put_ratio': sentiment['call_put_ratio'],
            'implied_volatility': sentiment['iv_skew'],
            'unusual_activity': unusual_activity
        }
        
        # AI分析（如果可用）
        ai_analysis = None
        if self.ai_analyzer:
            try:
                ai_analysis = self.ai_analyzer.analyze_market_context(
                    technical_data=technical_data,
                    option_data=option_data
                )
            except Exception as e:
                print(f"AI分析出错: {str(e)}")
        
        # 生成交易信号
        signals = self._generate_trading_signals(
            current_price=quote['last'],
            support=support,
            resistance=resistance,
            momentum=momentum,
            patterns=patterns,
            sentiment=sentiment,
            unusual_activity=unusual_activity,
            ema20=ema20.iloc[-1],
            ema50=ema50.iloc[-1],
            ai_analysis=ai_analysis
        )
        
        return {
            'signals': signals,
            'technical': {
                'support': support,
                'resistance': resistance,
                'momentum': momentum,
                'patterns': patterns,
                'ema20': ema20.iloc[-1],
                'ema50': ema50.iloc[-1]
            },
            'sentiment': sentiment,
            'unusual_activity': unusual_activity,
            'ai_analysis': ai_analysis
        }
        
    def _analyze_option_sentiment(self, option_chain: pd.DataFrame) -> Dict:
        """分析期权市场情绪"""
        # 计算看涨看跌比率
        calls = option_chain[option_chain['type'] == 'call']
        puts = option_chain[option_chain['type'] == 'put']
        
        call_volume = calls['volume'].sum()
        put_volume = puts['volume'].sum()
        
        call_oi = calls['open_interest'].sum()
        put_oi = puts['open_interest'].sum()
        
        # 计算平均隐含波动率
        call_iv = calls['implied_volatility'].mean()
        put_iv = puts['implied_volatility'].mean()
        
        return {
            'call_put_ratio': call_volume / put_volume if put_volume > 0 else float('inf'),
            'oi_ratio': call_oi / put_oi if put_oi > 0 else float('inf'),
            'iv_skew': call_iv - put_iv,
            'overall_sentiment': 'bullish' if call_volume > put_volume else 'bearish'
        }
        
    def _detect_unusual_activity(self, option_chain: pd.DataFrame) -> List[Dict]:
        """检测异常期权活动"""
        unusual_activities = []
        
        # 计算成交量均值和标准差
        volume_mean = option_chain['volume'].mean()
        volume_std = option_chain['volume'].std()
        
        # 寻找异常成交量
        threshold = volume_mean + 2 * volume_std
        unusual = option_chain[option_chain['volume'] > threshold]
        
        for _, row in unusual.iterrows():
            unusual_activities.append({
                'strike': row['strike'],
                'type': row['type'],
                'volume': row['volume'],
                'open_interest': row['open_interest'],
                'implied_volatility': row['implied_volatility']
            })
            
        return unusual_activities
        
    def _generate_trading_signals(self,
                                current_price: float,
                                support: float,
                                resistance: float,
                                momentum: Dict,
                                patterns: List[Dict],
                                sentiment: Dict,
                                unusual_activity: List[Dict],
                                ema20: float,
                                ema50: float,
                                ai_analysis: Optional[Dict] = None) -> List[Dict]:
        """生成交易信号"""
        signals = []
        
        # 1. EMA金叉/死叉信号
        if ema20 > ema50 and self._was_below(ema20, ema50):
            signals.append({
                'type': 'LONG_CALL',
                'confidence': 0.8,
                'reason': 'EMA20上穿EMA50，形成金叉',
                'signal_type': 'TECHNICAL'
            })
        elif ema20 < ema50 and self._was_above(ema20, ema50):
            signals.append({
                'type': 'LONG_PUT',
                'confidence': 0.8,
                'reason': 'EMA20下穿EMA50，形成死叉',
                'signal_type': 'TECHNICAL'
            })
            
        # 2. 支撑压力位信号
        if abs(current_price - support) / support < 0.02:
            if momentum['rsi'] < 30:
                signals.append({
                    'type': 'LONG_CALL',
                    'confidence': 0.85,
                    'reason': '价格接近支撑位且RSI超卖',
                    'signal_type': 'TECHNICAL'
                })
                
        if abs(current_price - resistance) / resistance < 0.02:
            if momentum['rsi'] > 70:
                signals.append({
                    'type': 'LONG_PUT',
                    'confidence': 0.85,
                    'reason': '价格接近压力位且RSI超买',
                    'signal_type': 'TECHNICAL'
                })
                
        # 3. 期权异常活动信号
        for activity in unusual_activity:
            volume_oi_ratio = activity['volume'] / activity['open_interest']
            if volume_oi_ratio > 3:
                signals.append({
                    'type': f"FOLLOW_{activity['type'].upper()}",
                    'strike': activity['strike'],
                    'confidence': 0.75,
                    'reason': f"期权链出现大额交易，成交量/持仓量比率 {volume_oi_ratio:.1f}",
                    'signal_type': 'OPTION_FLOW'
                })
                
        # 4. 综合AI分析信号
        if ai_analysis and ai_analysis['trading_suggestions']:
            for suggestion in ai_analysis['trading_suggestions']:
                signals.append({
                    'type': suggestion['type'],
                    'confidence': suggestion['confidence'],
                    'reason': suggestion['reason'],
                    'risk_reward_ratio': suggestion.get('risk_reward_ratio', None),
                    'signal_type': 'AI'
                })
                
        # 对信号进行评分和排序
        scored_signals = self._score_signals(signals)
        return sorted(scored_signals, key=lambda x: x['score'], reverse=True)
        
    def _score_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        对交易信号进行评分
        考虑多个因素：信号类型、置信度、风险收益比等
        """
        for signal in signals:
            base_score = signal['confidence'] * 100
            
            # 根据信号类型调整分数
            type_multiplier = {
                'TECHNICAL': 1.0,
                'OPTION_FLOW': 1.2,
                'AI': 1.3
            }
            
            signal['score'] = base_score * type_multiplier.get(signal['signal_type'], 1.0)
            
            # 如果有风险收益比，进一步调整分数
            if 'risk_reward_ratio' in signal:
                signal['score'] *= min(signal['risk_reward_ratio'] / 2, 1.5)
                
        return signals
        
    def _was_below(self, series1: pd.Series, series2: pd.Series) -> bool:
        """检查前一个周期是否在下方"""
        return series1.iloc[-2] < series2.iloc[-2]
        
    def _was_above(self, series1: pd.Series, series2: pd.Series) -> bool:
        """检查前一个周期是否在上方"""
        return series1.iloc[-2] > series2.iloc[-2]
        
    def monitor_and_notify(self, symbol: str, check_interval: int = 300):
        """
        持续监控并发送通知
        Args:
            symbol: 股票代码
            check_interval: 检查间隔(秒)
        """
        while True:
            try:
                # 分析入场机会
                analysis = self.analyze_entry_opportunity(symbol)
                
                # 如果有交易信号，发送通知
                if analysis['signals']:
                    for signal in analysis['signals']:
                        # 只推送高分信号
                        if signal['score'] >= 80:
                            self._send_signal_notification(symbol, signal, analysis)
                
            except Exception as e:
                self.notifier.send_message(
                    f"监控过程中出现错误: {str(e)}",
                    "错误警告"
                )
                
            time.sleep(check_interval)
            
    def _send_signal_notification(self, symbol: str, signal: Dict, analysis: Dict):
        """发送信号通知"""
        # 构建通知内容
        technical = analysis['technical']
        
        message = f"""
📥 入场提醒信号

🎯 标的：{symbol}
🔍 类型：{signal['type']}
💰 目标价位：{'%.2f' % (technical['resistance'] if 'CALL' in signal['type'] else technical['support'])}

📈 技术分析：
- EMA20/50: {'多头' if technical['ema20'] > technical['ema50'] else '空头'}排列
- RSI: {technical['momentum']['rsi']:.1f}
- 支撑位：{'%.2f' % technical['support']}
- 阻力位：{'%.2f' % technical['resistance']}

🔥 综合评分：{signal['score']:.0f}/100
📊 置信度：{signal['confidence']:.2%}
        """
        
        if 'risk_reward_ratio' in signal:
            message += f"📌 风险收益比：1:{signal['risk_reward_ratio']:.1f}\n"
            
        message += f"\n📝 信号说明：\n{signal['reason']}"
        
        # 如果有AI分析，添加AI见解
        if analysis['ai_analysis']:
            ai = analysis['ai_analysis']
            message += f"\n\n🤖 AI分析：\n"
            message += f"市场情绪：{ai['market_sentiment']}\n"
            message += f"趋势强度：{ai['trend_strength']:.0%}\n"
            message += "关键因素：\n"
            for factor in ai['key_factors']:
                message += f"- {factor}\n"
                
        self.notifier.send_message(message, "期权交易信号") 