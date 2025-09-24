"""Sentiment analysis and emotion detection."""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import structlog
import re
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from ..models.content import (
    SentimentAnalysis, 
    SentimentLabel, 
    EmotionLabel, 
    ExtractedContent
)
from ..config import settings

logger = structlog.get_logger(__name__)

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass


class SentimentAnalyzer:
    """Multilingual sentiment analysis and emotion detection."""
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.sentiment_pipeline = None
        self.emotion_pipeline = None
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.model_loaded = False
        
        # Load models asynchronously
        asyncio.create_task(self._initialize_models())
    
    async def _initialize_models(self):
        """Initialize sentiment and emotion analysis models."""
        try:
            # Load sentiment analysis model
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Load emotion analysis model
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            
            self.model_loaded = True
            logger.info("Sentiment analysis models loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load sentiment models", error=str(e))
            # Fallback to TextBlob and VADER
            self.model_loaded = False
    
    async def analyze_sentiment(
        self, 
        content: ExtractedContent, 
        language: str = "en"
    ) -> SentimentAnalysis:
        """Analyze sentiment and emotions in content."""
        try:
            # Prepare text for analysis
            text = f"{content.title} {content.summary or ''} {content.content[:2000]}"
            
            if self.model_loaded:
                # Use transformer models
                sentiment_result = await self._analyze_with_transformers(text)
                emotion_result = await self._analyze_emotions_with_transformers(text)
            else:
                # Use fallback methods
                sentiment_result = await self._analyze_with_fallback(text)
                emotion_result = await self._analyze_emotions_with_fallback(text)
            
            # Combine results
            sentiment_analysis = SentimentAnalysis(
                sentiment=sentiment_result["sentiment"],
                confidence=sentiment_result["confidence"],
                emotions=emotion_result,
                subjectivity=sentiment_result["subjectivity"],
                polarity=sentiment_result["polarity"]
            )
            
            logger.info("Sentiment analysis completed",
                       content_id=content.id,
                       sentiment=sentiment_analysis.sentiment.value,
                       confidence=sentiment_analysis.confidence,
                       language=language)
            
            return sentiment_analysis
            
        except Exception as e:
            logger.error("Sentiment analysis failed",
                        content_id=content.id,
                        error=str(e))
            
            # Return neutral sentiment as fallback
            return SentimentAnalysis(
                sentiment=SentimentLabel.NEUTRAL,
                confidence=0.5,
                subjectivity=0.5,
                polarity=0.0
            )
    
    async def _analyze_with_transformers(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using transformer models."""
        try:
            # Get sentiment scores
            sentiment_scores = self.sentiment_pipeline(text)
            
            # Find the highest scoring sentiment
            best_sentiment = max(sentiment_scores[0], key=lambda x: x['score'])
            
            # Map to our sentiment labels
            sentiment_mapping = {
                'LABEL_0': SentimentLabel.NEGATIVE,
                'LABEL_1': SentimentLabel.NEUTRAL,
                'LABEL_2': SentimentLabel.POSITIVE
            }
            
            sentiment = sentiment_mapping.get(best_sentiment['label'], SentimentLabel.NEUTRAL)
            confidence = best_sentiment['score']
            
            # Calculate subjectivity and polarity using TextBlob as additional signal
            blob = TextBlob(text)
            subjectivity = blob.sentiment.subjectivity
            polarity = blob.sentiment.polarity
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "subjectivity": subjectivity,
                "polarity": polarity
            }
            
        except Exception as e:
            logger.error("Transformer sentiment analysis failed", error=str(e))
            return await self._analyze_with_fallback(text)
    
    async def _analyze_with_fallback(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using fallback methods."""
        try:
            # Use VADER sentiment analyzer
            vader_scores = self.vader_analyzer.polarity_scores(text)
            
            # Determine sentiment based on compound score
            compound = vader_scores['compound']
            if compound >= 0.05:
                sentiment = SentimentLabel.POSITIVE
            elif compound <= -0.05:
                sentiment = SentimentLabel.NEGATIVE
            else:
                sentiment = SentimentLabel.NEUTRAL
            
            # Use TextBlob for additional analysis
            blob = TextBlob(text)
            subjectivity = blob.sentiment.subjectivity
            polarity = blob.sentiment.polarity
            
            # Calculate confidence based on compound score magnitude
            confidence = min(abs(compound) * 2, 1.0)
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "subjectivity": subjectivity,
                "polarity": polarity
            }
            
        except Exception as e:
            logger.error("Fallback sentiment analysis failed", error=str(e))
            return {
                "sentiment": SentimentLabel.NEUTRAL,
                "confidence": 0.5,
                "subjectivity": 0.5,
                "polarity": 0.0
            }
    
    async def _analyze_emotions_with_transformers(self, text: str) -> Dict[EmotionLabel, float]:
        """Analyze emotions using transformer models."""
        try:
            emotion_scores = self.emotion_pipeline(text)
            
            # Map emotion labels
            emotion_mapping = {
                'joy': EmotionLabel.JOY,
                'sadness': EmotionLabel.SADNESS,
                'anger': EmotionLabel.ANGER,
                'fear': EmotionLabel.FEAR,
                'surprise': EmotionLabel.SURPRISE,
                'disgust': EmotionLabel.DISGUST,
                'neutral': EmotionLabel.NEUTRAL
            }
            
            emotions = {}
            for emotion_data in emotion_scores[0]:
                emotion_name = emotion_data['label'].lower()
                if emotion_name in emotion_mapping:
                    emotions[emotion_mapping[emotion_name]] = emotion_data['score']
            
            return emotions
            
        except Exception as e:
            logger.error("Transformer emotion analysis failed", error=str(e))
            return await self._analyze_emotions_with_fallback(text)
    
    async def _analyze_emotions_with_fallback(self, text: str) -> Dict[EmotionLabel, float]:
        """Analyze emotions using fallback methods."""
        try:
            emotions = {}
            
            # Use keyword-based emotion detection
            emotion_keywords = {
                EmotionLabel.JOY: ['happy', 'joy', 'excited', 'celebrate', 'wonderful', 'amazing', 'great', 'fantastic'],
                EmotionLabel.SADNESS: ['sad', 'depressed', 'unhappy', 'grief', 'mourn', 'cry', 'tears', 'sorrow'],
                EmotionLabel.ANGER: ['angry', 'mad', 'furious', 'rage', 'outrage', 'annoyed', 'irritated', 'frustrated'],
                EmotionLabel.FEAR: ['afraid', 'scared', 'fear', 'terrified', 'worried', 'anxious', 'nervous', 'panic'],
                EmotionLabel.SURPRISE: ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'sudden', 'wow'],
                EmotionLabel.DISGUST: ['disgusted', 'revolted', 'sick', 'nauseated', 'repulsed', 'gross', 'disgusting']
            }
            
            text_lower = text.lower()
            
            for emotion, keywords in emotion_keywords.items():
                keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
                if keyword_count > 0:
                    # Calculate emotion intensity based on keyword frequency
                    intensity = min(keyword_count / len(keywords), 1.0)
                    emotions[emotion] = intensity
            
            # If no emotions detected, set neutral
            if not emotions:
                emotions[EmotionLabel.NEUTRAL] = 1.0
            
            return emotions
            
        except Exception as e:
            logger.error("Fallback emotion analysis failed", error=str(e))
            return {EmotionLabel.NEUTRAL: 1.0}
    
    async def analyze_sentiment_batch(
        self, 
        contents: List[ExtractedContent]
    ) -> List[SentimentAnalysis]:
        """Analyze sentiment for multiple contents."""
        try:
            tasks = [
                self.analyze_sentiment(content)
                for content in contents
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            sentiment_analyses = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error("Sentiment analysis failed for content",
                                content_id=contents[i].id,
                                error=str(result))
                    # Add neutral sentiment as fallback
                    sentiment_analyses.append(SentimentAnalysis(
                        sentiment=SentimentLabel.NEUTRAL,
                        confidence=0.5,
                        subjectivity=0.5,
                        polarity=0.0
                    ))
                else:
                    sentiment_analyses.append(result)
            
            return sentiment_analyses
            
        except Exception as e:
            logger.error("Batch sentiment analysis failed", error=str(e))
            return []
    
    def get_sentiment_statistics(self, analyses: List[SentimentAnalysis]) -> Dict[str, Any]:
        """Get statistics about sentiment analyses."""
        if not analyses:
            return {}
        
        stats = {
            "total_analyses": len(analyses),
            "sentiment_distribution": {
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "mixed": 0
            },
            "average_confidence": 0.0,
            "average_subjectivity": 0.0,
            "average_polarity": 0.0,
            "emotion_distribution": {}
        }
        
        total_confidence = 0.0
        total_subjectivity = 0.0
        total_polarity = 0.0
        
        for analysis in analyses:
            # Sentiment distribution
            stats["sentiment_distribution"][analysis.sentiment.value] += 1
            
            # Averages
            total_confidence += analysis.confidence
            total_subjectivity += analysis.subjectivity
            total_polarity += analysis.polarity
            
            # Emotion distribution
            for emotion, intensity in analysis.emotions.items():
                if emotion not in stats["emotion_distribution"]:
                    stats["emotion_distribution"][emotion.value] = 0
                stats["emotion_distribution"][emotion.value] += 1
        
        stats["average_confidence"] = total_confidence / len(analyses)
        stats["average_subjectivity"] = total_subjectivity / len(analyses)
        stats["average_polarity"] = total_polarity / len(analyses)
        
        return stats
    
    async def detect_sentiment_shift(
        self, 
        old_analysis: SentimentAnalysis, 
        new_analysis: SentimentAnalysis
    ) -> Dict[str, Any]:
        """Detect sentiment shift between two analyses."""
        try:
            # Calculate sentiment change
            sentiment_change = {
                "old_sentiment": old_analysis.sentiment.value,
                "new_sentiment": new_analysis.sentiment.value,
                "confidence_change": new_analysis.confidence - old_analysis.confidence,
                "polarity_change": new_analysis.polarity - old_analysis.polarity,
                "subjectivity_change": new_analysis.subjectivity - old_analysis.subjectivity
            }
            
            # Detect significant changes
            significant_changes = []
            
            if abs(sentiment_change["polarity_change"]) > 0.3:
                significant_changes.append("polarity")
            
            if abs(sentiment_change["confidence_change"]) > 0.2:
                significant_changes.append("confidence")
            
            if old_analysis.sentiment != new_analysis.sentiment:
                significant_changes.append("sentiment_label")
            
            sentiment_change["significant_changes"] = significant_changes
            sentiment_change["has_significant_change"] = len(significant_changes) > 0
            
            return sentiment_change
            
        except Exception as e:
            logger.error("Sentiment shift detection failed", error=str(e))
            return {}
