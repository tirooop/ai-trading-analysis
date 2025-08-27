from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import json
import os
from datetime import datetime
import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import threading
import queue
import time

from .market_context_builder import MarketContextBuilder, MarketContext
from .llm_prompt_builder import LLMPromptBuilder, MarketData
from .ai_analyst_v2 import AIAnalystV2, AnalysisResult
from .signal_fusion_engine import SignalFusionEngine, FusionResult
from utils.unified_notifier import UnifiedNotifier, NotificationConfig

# Load environment variables from API.env
load_dotenv('API.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ExecutionConfig:
    """Configuration for signal execution"""
    symbols: List[str]
    timeframes: List[str] = None
    save_to_db: bool = False
    notify_on_signal: bool = True
    min_confidence: float = 0.7
    output_dir: str = "output"
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['1', '3', '15', 'D']
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

@dataclass
class SignalConfig:
    """Configuration for signal generation and execution"""
    symbols: List[str]
    interval: int  # seconds
    min_confidence: float
    risk_levels: Dict[str, float]
    notification_config: NotificationConfig

class SignalExecutor:
    """Executes trading signals based on fused analysis"""
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self.market_context = MarketContextBuilder()
        self.ai_analyst = AIAnalystV2()
        self.fusion_engine = SignalFusionEngine()
        self.notifier = UnifiedNotifier(config.notification_config)
        
        self.signal_queue = queue.Queue()
        self.is_running = False
        self.thread = None
        
        # Store recent signals
        self.recent_signals = {}
        self.max_signals = 100  # Keep last 100 signals per symbol
    
    def start(self):
        """Start the signal executor"""
        if self.is_running:
            logger.warning("Signal executor is already running")
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Signal executor started")
    
    def stop(self):
        """Stop the signal executor"""
        if not self.is_running:
            logger.warning("Signal executor is not running")
            return
            
        self.is_running = False
        if self.thread:
            self.thread.join()
        logger.info("Signal executor stopped")
    
    def _run(self):
        """Main execution loop"""
        while self.is_running:
            try:
                # Process each symbol
                for symbol in self.config.symbols:
                    self._process_symbol(symbol)
                
                # Wait for next interval
                time.sleep(self.config.interval)
                
            except Exception as e:
                logger.error(f"Error in signal executor: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def _process_symbol(self, symbol: str):
        """Process a single symbol"""
        try:
            # Build market context
            context = self.market_context.build_context(symbol)
            
            # Get AI analysis
            analysis = self.ai_analyst.analyze(context)
            
            # Fuse signals
            fusion_result = self.fusion_engine.fuse_signals(
                technical_analysis=analysis.technical_analysis,
                market_sentiment=analysis.market_sentiment,
                sector_analysis=analysis.sector_analysis,
                news_sentiment=analysis.news_sentiment,
                volatility_analysis=analysis.volatility_analysis,
                llm_analysis=analysis.llm_analysis
            )
            
            # Store signal
            self._store_signal(symbol, fusion_result)
            
            # Check if signal meets confidence threshold
            if fusion_result.confidence >= self.config.min_confidence:
                # Add to execution queue
                self.signal_queue.put((symbol, fusion_result))
                
                # Send notification
                self._send_notification(symbol, fusion_result)
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {str(e)}")
    
    def _store_signal(self, symbol: str, signal):
        """Store signal in recent signals"""
        if symbol not in self.recent_signals:
            self.recent_signals[symbol] = []
            
        self.recent_signals[symbol].append(signal)
        
        # Keep only recent signals
        if len(self.recent_signals[symbol]) > self.max_signals:
            self.recent_signals[symbol] = self.recent_signals[symbol][-self.max_signals:]
    
    def _send_notification(self, symbol: str, signal):
        """Send notification for signal"""
        try:
            self.notifier.send_option_entry_signal(
                symbol=symbol,
                direction=signal.action,
                confidence=signal.confidence,
                reasoning=signal.reasoning,
                recommendation=signal.recommendation
            )
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
    
    def get_recent_signals(self, symbol: str, limit: int = 10) -> List:
        """Get recent signals for a symbol"""
        if symbol not in self.recent_signals:
            return []
            
        return self.recent_signals[symbol][-limit:]
    
    def get_signal_queue_size(self) -> int:
        """Get current size of signal queue"""
        return self.signal_queue.qsize()
    
    def get_next_signal(self) -> Optional[tuple]:
        """Get next signal from queue"""
        try:
            return self.signal_queue.get_nowait()
        except queue.Empty:
            return None

def create_executor(symbols: List[str],
                   timeframes: Optional[List[str]] = None,
                   save_to_db: bool = False,
                   notify_on_signal: bool = True,
                   min_confidence: float = 0.7,
                   output_dir: str = "output") -> SignalExecutor:
    """
    Create a SignalExecutor with the given configuration
    
    Args:
        symbols: List of symbols to analyze
        timeframes: List of timeframes to analyze
        save_to_db: Whether to save results to database
        notify_on_signal: Whether to send notifications
        min_confidence: Minimum confidence for notifications
        output_dir: Directory to save results
        
    Returns:
        Configured SignalExecutor instance
    """
    config = ExecutionConfig(
        symbols=symbols,
        timeframes=timeframes,
        save_to_db=save_to_db,
        notify_on_signal=notify_on_signal,
        min_confidence=min_confidence,
        output_dir=output_dir
    )
    return SignalExecutor(config) 