from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import json

@dataclass
class MarketData:
    """Market data for analysis"""
    symbol: str
    current_price: float
    ema_trend: str
    macd_signal: str
    rsi: float
    volume_factor: str
    support_levels: List[float]
    resistance_levels: List[float]
    market_sentiment: str
    sector_strength: Dict[str, float]
    news_summary: str
    vix_level: float
    market_breadth: Dict[str, float]

class LLMPromptBuilder:
    """Builds professional trading analysis prompts"""
    
    def __init__(self):
        self.base_prompt = """
你是一位经验丰富的对冲基金交易员，具备20年实盘操盘经验。请基于以下市场数据，给出专业的交易分析：

{market_data}

请从以下角度给出判断：
1. 当前是否值得介入？为什么？
2. 看涨/看跌理由？（至少3点）
3. 止损止盈建议？
4. 置信度评分（0~100）？
5. 风险提示？

请以JSON格式输出分析结果，包含以下字段：
{
    "should_enter": true/false,
    "direction": "BULLISH/BEARISH/NEUTRAL",
    "confidence_score": 0-100,
    "reasons": ["理由1", "理由2", "理由3"],
    "risk_factors": ["风险1", "风险2"],
    "recommendation": {
        "entry_price": float,
        "stop_loss": float,
        "take_profit": float
    }
}
"""
    
    def build_prompt(self, market_data: MarketData) -> str:
        """Build a complete analysis prompt"""
        # Format market data section
        market_data_str = f"""
标的：{market_data.symbol}
当前价格：${market_data.current_price:.2f}

技术指标：
- EMA趋势：{market_data.ema_trend}
- MACD信号：{market_data.macd_signal}
- RSI：{market_data.rsi:.1f}
- 成交量变化：{market_data.volume_factor}

关键价位：
- 支撑位：{', '.join(f'${level:.2f}' for level in market_data.support_levels)}
- 阻力位：{', '.join(f'${level:.2f}' for level in market_data.resistance_levels)}

市场环境：
- 市场情绪：{market_data.market_sentiment}
- VIX指数：{market_data.vix_level:.1f}
- 市场广度：{market_data.market_breadth['advance_decline_ratio']:.2f}

板块强度：
{self._format_sector_strength(market_data.sector_strength)}

新闻摘要：
{market_data.news_summary}
"""
        
        # Combine with base prompt
        return self.base_prompt.format(market_data=market_data_str)
    
    def _format_sector_strength(self, sector_strength: Dict[str, float]) -> str:
        """Format sector strength data"""
        formatted = []
        for sector, strength in sector_strength.items():
            strength_str = "强势" if strength > 0.01 else "弱势" if strength < -0.01 else "中性"
            formatted.append(f"- {sector}: {strength_str} ({strength:.2%})")
        return "\n".join(formatted)
    
    def parse_response(self, response: str) -> Dict:
        """Parse LLM response into structured format"""
        try:
            # Extract JSON from response
            json_str = response[response.find("{"):response.rfind("}")+1]
            return json.loads(json_str)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return {
                "error": "Failed to parse LLM response",
                "raw_response": response
            }
    
    def build_followup_prompt(self, 
                            original_analysis: Dict,
                            new_market_data: MarketData) -> str:
        """Build a followup prompt for analysis update"""
        return f"""
基于之前的分析：
{json.dumps(original_analysis, indent=2, ensure_ascii=False)}

请根据最新的市场数据更新分析：
{self.build_prompt(new_market_data)}

特别注意：
1. 与之前分析相比，市场条件发生了哪些变化？
2. 是否需要调整之前的建议？
3. 新的风险因素？

请同样以JSON格式输出更新后的分析结果。
""" 