from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
import json
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class FactorScore:
    """Score for a single analysis factor"""
    name: str
    score: float  # -1 to 1
    weight: float  # 0 to 1
    confidence: float  # 0 to 1
    description: str
    trend: str  # BULLISH/BEARISH/NEUTRAL

@dataclass
class FusionResult:
    """Result of signal fusion"""
    timestamp: datetime
    symbol: str
    final_score: float  # -1 to 1
    risk_level: str  # LOW/MEDIUM/HIGH
    action: str  # BUY/SELL/HOLD
    confidence: float  # 0 to 1
    factor_scores: List[FactorScore]
    reasoning: str
    recommendation: str

class SignalFusionEngine:
    """Engine for fusing multiple analysis signals"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.weights = self.config.get('factor_weights', {
            'technical_analysis': 0.3,
            'market_sentiment': 0.2,
            'sector_analysis': 0.15,
            'news_sentiment': 0.15,
            'volatility_analysis': 0.1,
            'llm_analysis': 0.1
        })
        
        # Define risk thresholds
        self.risk_thresholds = {
            'LOW': 0.3,
            'MEDIUM': 0.6,
            'HIGH': 1.0
        }
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def fuse_signals(self,
                    technical_analysis: Dict,
                    market_sentiment: Dict,
                    sector_analysis: Dict,
                    news_sentiment: Dict,
                    volatility_analysis: Dict,
                    llm_analysis: Dict) -> FusionResult:
        """
        Fuse multiple analysis signals into a unified signal
        
        Args:
            technical_analysis: Technical analysis results
            market_sentiment: Market sentiment analysis
            sector_analysis: Sector performance analysis
            news_sentiment: News sentiment analysis
            volatility_analysis: Volatility analysis
            llm_analysis: LLM analysis results
            
        Returns:
            Fused signal result
        """
        try:
            # Calculate factor scores
            factor_scores = []
            
            # Technical Analysis
            tech_score = self._calculate_technical_score(technical_analysis)
            factor_scores.append(FactorScore(
                name="Technical Analysis",
                score=tech_score,
                weight=self.weights['technical_analysis'],
                confidence=technical_analysis.get('confidence', 0.7),
                description=self._get_technical_description(technical_analysis),
                trend=self._get_trend(tech_score)
            ))
            
            # Market Sentiment
            sentiment_score = self._calculate_sentiment_score(market_sentiment)
            factor_scores.append(FactorScore(
                name="Market Sentiment",
                score=sentiment_score,
                weight=self.weights['market_sentiment'],
                confidence=market_sentiment.get('confidence', 0.7),
                description=self._get_sentiment_description(market_sentiment),
                trend=self._get_trend(sentiment_score)
            ))
            
            # Sector Analysis
            sector_score = self._calculate_sector_score(sector_analysis)
            factor_scores.append(FactorScore(
                name="Sector Analysis",
                score=sector_score,
                weight=self.weights['sector_analysis'],
                confidence=sector_analysis.get('confidence', 0.7),
                description=self._get_sector_description(sector_analysis),
                trend=self._get_trend(sector_score)
            ))
            
            # News Sentiment
            news_score = self._calculate_news_score(news_sentiment)
            factor_scores.append(FactorScore(
                name="News Sentiment",
                score=news_score,
                weight=self.weights['news_sentiment'],
                confidence=news_sentiment.get('confidence', 0.7),
                description=self._get_news_description(news_sentiment),
                trend=self._get_trend(news_score)
            ))
            
            # Volatility Analysis
            vol_score = self._calculate_volatility_score(volatility_analysis)
            factor_scores.append(FactorScore(
                name="Volatility Analysis",
                score=vol_score,
                weight=self.weights['volatility_analysis'],
                confidence=volatility_analysis.get('confidence', 0.7),
                description=self._get_volatility_description(volatility_analysis),
                trend=self._get_trend(vol_score)
            ))
            
            # LLM Analysis
            llm_score = self._calculate_llm_score(llm_analysis)
            factor_scores.append(FactorScore(
                name="LLM Analysis",
                score=llm_score,
                weight=self.weights['llm_analysis'],
                confidence=llm_analysis.get('confidence', 0.7),
                description=self._get_llm_description(llm_analysis),
                trend=self._get_trend(llm_score)
            ))
            
            # Calculate final score
            final_score = self._calculate_final_score(factor_scores)
            
            # Determine action and risk level
            action = self._determine_action(final_score)
            risk_level = self._determine_risk_level(volatility_analysis)
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(factor_scores)
            
            # Generate reasoning and recommendation
            reasoning = self._generate_reasoning(factor_scores)
            recommendation = self._generate_recommendation(
                action, risk_level, factor_scores
            )
            
            return FusionResult(
                timestamp=datetime.now(),
                symbol=technical_analysis.get('symbol', 'UNKNOWN'),
                final_score=final_score,
                risk_level=risk_level,
                action=action,
                confidence=confidence,
                factor_scores=factor_scores,
                reasoning=reasoning,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error fusing signals: {str(e)}")
            raise
    
    def _calculate_technical_score(self, analysis: Dict) -> float:
        """Calculate technical analysis score"""
        # Combine multiple technical indicators
        scores = []
        
        # Trend indicators
        if 'ema_trend' in analysis:
            scores.append(1.0 if analysis['ema_trend'] == 'BULLISH' else -1.0)
        
        if 'macd_signal' in analysis:
            scores.append(1.0 if analysis['macd_signal'] == 'BULLISH' else -1.0)
        
        # Momentum indicators
        if 'rsi' in analysis:
            rsi = analysis['rsi']
            if rsi > 70:
                scores.append(-1.0)  # Overbought
            elif rsi < 30:
                scores.append(1.0)   # Oversold
            else:
                scores.append(0.0)   # Neutral
        
        # Volume analysis
        if 'volume_factor' in analysis:
            vol_factor = analysis['volume_factor']
            if vol_factor > 1.5:
                scores.append(1.0)   # High volume
            elif vol_factor < 0.5:
                scores.append(-1.0)  # Low volume
            else:
                scores.append(0.0)   # Normal volume
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_sentiment_score(self, sentiment: Dict) -> float:
        """Calculate market sentiment score"""
        if not sentiment:
            return 0.0
            
        # Combine multiple sentiment indicators
        scores = []
        
        if 'market_breadth' in sentiment:
            breadth = sentiment['market_breadth']
            if breadth > 0.6:
                scores.append(1.0)   # Strong breadth
            elif breadth < 0.4:
                scores.append(-1.0)  # Weak breadth
            else:
                scores.append(0.0)   # Neutral
        
        if 'vix_level' in sentiment:
            vix = sentiment['vix_level']
            if vix < 15:
                scores.append(1.0)   # Low volatility
            elif vix > 25:
                scores.append(-1.0)  # High volatility
            else:
                scores.append(0.0)   # Normal
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_sector_score(self, analysis: Dict) -> float:
        """Calculate sector analysis score"""
        if not analysis:
            return 0.0
            
        # Calculate sector strength
        if 'sector_strength' in analysis:
            strength = analysis['sector_strength']
            if strength > 0.7:
                return 1.0
            elif strength < 0.3:
                return -1.0
            else:
                return 0.0
        
        return 0.0
    
    def _calculate_news_score(self, sentiment: Dict) -> float:
        """Calculate news sentiment score"""
        if not sentiment:
            return 0.0
            
        # Use overall sentiment score
        if 'overall_sentiment' in sentiment:
            return sentiment['overall_sentiment']
        
        return 0.0
    
    def _calculate_volatility_score(self, analysis: Dict) -> float:
        """Calculate volatility analysis score"""
        if not analysis:
            return 0.0
            
        # Combine multiple volatility indicators
        scores = []
        
        if 'vix_level' in analysis:
            vix = analysis['vix_level']
            if vix < 15:
                scores.append(1.0)   # Low volatility
            elif vix > 25:
                scores.append(-1.0)  # High volatility
            else:
                scores.append(0.0)   # Normal
        
        if 'historical_volatility' in analysis:
            hist_vol = analysis['historical_volatility']
            if hist_vol < 0.2:
                scores.append(1.0)   # Low volatility
            elif hist_vol > 0.4:
                scores.append(-1.0)  # High volatility
            else:
                scores.append(0.0)   # Normal
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_llm_score(self, analysis: Dict) -> float:
        """Calculate LLM analysis score"""
        if not analysis:
            return 0.0
            
        # Use LLM confidence and direction
        if 'direction' in analysis and 'confidence' in analysis:
            direction = analysis['direction']
            confidence = analysis['confidence']
            
            if direction == 'BULLISH':
                return confidence
            elif direction == 'BEARISH':
                return -confidence
            else:
                return 0.0
        
        return 0.0
    
    def _calculate_final_score(self, factor_scores: List[FactorScore]) -> float:
        """Calculate final weighted score"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor in factor_scores:
            weighted_sum += factor.score * factor.weight * factor.confidence
            total_weight += factor.weight * factor.confidence
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _determine_action(self, final_score: float) -> str:
        """Determine trading action based on final score"""
        if final_score > 0.3:
            return "BUY"
        elif final_score < -0.3:
            return "SELL"
        else:
            return "HOLD"
    
    def _determine_risk_level(self, volatility_analysis: Dict) -> str:
        """Determine risk level based on volatility analysis"""
        if not volatility_analysis:
            return "MEDIUM"
            
        vix = volatility_analysis.get('vix_level', 20)
        hist_vol = volatility_analysis.get('historical_volatility', 0.3)
        
        # Combine VIX and historical volatility
        risk_score = (vix / 30 + hist_vol) / 2
        
        if risk_score < self.risk_thresholds['LOW']:
            return "LOW"
        elif risk_score < self.risk_thresholds['MEDIUM']:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _calculate_confidence(self, factor_scores: List[FactorScore]) -> float:
        """Calculate overall confidence"""
        if not factor_scores:
            return 0.0
            
        # Weight confidence by factor importance
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor in factor_scores:
            weighted_sum += factor.confidence * factor.weight
            total_weight += factor.weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_reasoning(self, factor_scores: List[FactorScore]) -> str:
        """Generate reasoning for the signal"""
        reasoning = []
        
        # Sort factors by absolute score
        sorted_factors = sorted(
            factor_scores,
            key=lambda x: abs(x.score),
            reverse=True
        )
        
        # Add top factors to reasoning
        for factor in sorted_factors[:3]:
            if abs(factor.score) > 0.3:  # Only include significant factors
                reasoning.append(f"{factor.name}: {factor.description}")
        
        return "\n".join(reasoning) if reasoning else "No significant factors identified"
    
    def _generate_recommendation(self,
                               action: str,
                               risk_level: str,
                               factor_scores: List[FactorScore]) -> str:
        """Generate trading recommendation"""
        recommendation = []
        
        # Add action recommendation
        if action == "BUY":
            recommendation.append("Consider taking a long position")
        elif action == "SELL":
            recommendation.append("Consider taking a short position")
        else:
            recommendation.append("Consider waiting for better entry")
        
        # Add risk management
        if risk_level == "HIGH":
            recommendation.append("Use tight stop losses and smaller position sizes")
        elif risk_level == "LOW":
            recommendation.append("Normal position sizing is appropriate")
        
        # Add specific strategy suggestions
        if action != "HOLD":
            recommendation.append(self._get_strategy_suggestions(action, factor_scores))
        
        return "\n".join(recommendation)
    
    def _get_strategy_suggestions(self,
                                action: str,
                                factor_scores: List[FactorScore]) -> str:
        """Get specific strategy suggestions"""
        suggestions = []
        
        # Check technical factors
        tech_factors = [f for f in factor_scores if f.name == "Technical Analysis"]
        if tech_factors:
            tech = tech_factors[0]
            if abs(tech.score) > 0.5:
                if action == "BUY":
                    suggestions.append("Consider call options or bull spreads")
                else:
                    suggestions.append("Consider put options or bear spreads")
        
        # Check volatility
        vol_factors = [f for f in factor_scores if f.name == "Volatility Analysis"]
        if vol_factors:
            vol = vol_factors[0]
            if vol.score < -0.5:  # High volatility
                suggestions.append("Consider selling premium strategies")
            elif vol.score > 0.5:  # Low volatility
                suggestions.append("Consider buying options or debit spreads")
        
        return "\n".join(suggestions) if suggestions else "Use standard option strategies"
    
    def _get_trend(self, score: float) -> str:
        """Get trend direction from score"""
        if score > 0.3:
            return "BULLISH"
        elif score < -0.3:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _get_technical_description(self, analysis: Dict) -> str:
        """Get technical analysis description"""
        descriptions = []
        
        if 'ema_trend' in analysis:
            descriptions.append(f"EMA Trend: {analysis['ema_trend']}")
        
        if 'macd_signal' in analysis:
            descriptions.append(f"MACD: {analysis['macd_signal']}")
        
        if 'rsi' in analysis:
            rsi = analysis['rsi']
            if rsi > 70:
                descriptions.append("RSI: Overbought")
            elif rsi < 30:
                descriptions.append("RSI: Oversold")
        
        return ", ".join(descriptions) if descriptions else "No significant technical signals"
    
    def _get_sentiment_description(self, sentiment: Dict) -> str:
        """Get market sentiment description"""
        if not sentiment:
            return "No sentiment data available"
            
        descriptions = []
        
        if 'market_breadth' in sentiment:
            breadth = sentiment['market_breadth']
            if breadth > 0.6:
                descriptions.append("Strong market breadth")
            elif breadth < 0.4:
                descriptions.append("Weak market breadth")
        
        if 'vix_level' in sentiment:
            vix = sentiment['vix_level']
            if vix < 15:
                descriptions.append("Low market fear")
            elif vix > 25:
                descriptions.append("High market fear")
        
        return ", ".join(descriptions) if descriptions else "Neutral market sentiment"
    
    def _get_sector_description(self, analysis: Dict) -> str:
        """Get sector analysis description"""
        if not analysis:
            return "No sector data available"
            
        if 'sector_strength' in analysis:
            strength = analysis['sector_strength']
            if strength > 0.7:
                return "Strong sector performance"
            elif strength < 0.3:
                return "Weak sector performance"
            else:
                return "Neutral sector performance"
        
        return "No sector strength data available"
    
    def _get_news_description(self, sentiment: Dict) -> str:
        """Get news sentiment description"""
        if not sentiment:
            return "No news data available"
            
        if 'overall_sentiment' in sentiment:
            sentiment_score = sentiment['overall_sentiment']
            if sentiment_score > 0.3:
                return "Positive news sentiment"
            elif sentiment_score < -0.3:
                return "Negative news sentiment"
            else:
                return "Neutral news sentiment"
        
        return "No news sentiment data available"
    
    def _get_volatility_description(self, analysis: Dict) -> str:
        """Get volatility analysis description"""
        if not analysis:
            return "No volatility data available"
            
        descriptions = []
        
        if 'vix_level' in analysis:
            vix = analysis['vix_level']
            if vix < 15:
                descriptions.append("Low VIX")
            elif vix > 25:
                descriptions.append("High VIX")
        
        if 'historical_volatility' in analysis:
            hist_vol = analysis['historical_volatility']
            if hist_vol < 0.2:
                descriptions.append("Low historical volatility")
            elif hist_vol > 0.4:
                descriptions.append("High historical volatility")
        
        return ", ".join(descriptions) if descriptions else "Normal volatility levels"
    
    def _get_llm_description(self, analysis: Dict) -> str:
        """Get LLM analysis description"""
        if not analysis:
            return "No LLM analysis available"
            
        if 'direction' in analysis and 'confidence' in analysis:
            direction = analysis['direction']
            confidence = analysis['confidence']
            return f"{direction} with {confidence:.0%} confidence"
        
        return "No clear LLM direction" 