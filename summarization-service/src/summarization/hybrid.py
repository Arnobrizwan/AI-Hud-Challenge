"""
Hybrid Summarization Engine
Combines extractive and abstractive methods for optimal results
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import ProcessedContent, SummarizationMethod
from .extractive import BertExtractiveSummarizer
from .abstractive import VertexAISummarizer
from config.settings import settings

logger = logging.getLogger(__name__)


class HybridSummarizer:
    """Hybrid summarization combining extractive and abstractive methods"""
    
    def __init__(self):
        """Initialize the hybrid summarizer"""
        self.extractive_summarizer = BertExtractiveSummarizer()
        self.abstractive_summarizer = VertexAISummarizer()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self._initialized = False
        
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("Initializing hybrid summarizer...")
            
            # Initialize both summarizers
            await self.extractive_summarizer.initialize()
            await self.abstractive_summarizer.initialize()
            
            self._initialized = True
            logger.info("Hybrid summarizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid summarizer: {str(e)}")
            raise
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            await self.extractive_summarizer.cleanup()
            await self.abstractive_summarizer.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    async def summarize(
        self, 
        content: ProcessedContent, 
        target_length: int = 120,
        strategy: str = "adaptive"
    ) -> str:
        """
        Generate hybrid summary using optimal combination of methods
        
        Args:
            content: Processed content to summarize
            target_length: Target summary length in words
            strategy: Hybrid strategy ('adaptive', 'weighted', 'sequential')
            
        Returns:
            Hybrid summary text
        """
        if not self._initialized:
            raise RuntimeError("Hybrid summarizer not initialized")
        
        try:
            if strategy == "adaptive":
                return await self._adaptive_hybrid(content, target_length)
            elif strategy == "weighted":
                return await self._weighted_hybrid(content, target_length)
            elif strategy == "sequential":
                return await self._sequential_hybrid(content, target_length)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Hybrid summarization failed: {str(e)}")
            # Fallback to extractive only
            return await self.extractive_summarizer.summarize(content.text, target_length)
    
    async def _adaptive_hybrid(
        self, 
        content: ProcessedContent, 
        target_length: int
    ) -> str:
        """Adaptive hybrid strategy that chooses the best approach"""
        try:
            # Analyze content characteristics
            content_analysis = await self._analyze_content(content)
            
            # Determine optimal strategy based on content
            if content_analysis['is_news_article']:
                # News articles: prefer extractive for factual accuracy
                extractive_weight = 0.7
                abstractive_weight = 0.3
            elif content_analysis['is_technical']:
                # Technical content: balanced approach
                extractive_weight = 0.5
                abstractive_weight = 0.5
            elif content_analysis['is_narrative']:
                # Narrative content: prefer abstractive for coherence
                extractive_weight = 0.3
                abstractive_weight = 0.7
            else:
                # Default: balanced approach
                extractive_weight = 0.5
                abstractive_weight = 0.5
            
            # Generate both summaries
            extractive_summary = await self.extractive_summarizer.summarize(
                content.text, target_length
            )
            abstractive_summary = await self.abstractive_summarizer.summarize(
                content, target_length
            )
            
            # Combine based on weights
            combined_summary = await self._combine_summaries(
                extractive_summary, abstractive_summary,
                extractive_weight, abstractive_weight, target_length
            )
            
            return combined_summary
            
        except Exception as e:
            logger.error(f"Adaptive hybrid failed: {str(e)}")
            raise
    
    async def _weighted_hybrid(
        self, 
        content: ProcessedContent, 
        target_length: int
    ) -> str:
        """Weighted hybrid strategy with fixed weights"""
        try:
            # Generate both summaries
            extractive_summary = await self.extractive_summarizer.summarize(
                content.text, target_length
            )
            abstractive_summary = await self.abstractive_summarizer.summarize(
                content, target_length
            )
            
            # Use equal weights for balanced approach
            combined_summary = await self._combine_summaries(
                extractive_summary, abstractive_summary,
                0.5, 0.5, target_length
            )
            
            return combined_summary
            
        except Exception as e:
            logger.error(f"Weighted hybrid failed: {str(e)}")
            raise
    
    async def _sequential_hybrid(
        self, 
        content: ProcessedContent, 
        target_length: int
    ) -> str:
        """Sequential hybrid strategy: extractive first, then abstractive"""
        try:
            # Step 1: Generate extractive summary
            extractive_summary = await self.extractive_summarizer.summarize(
                content.text, target_length * 2  # Longer for better context
            )
            
            # Step 2: Use extractive summary as input for abstractive
            # Create new content object with extractive summary
            extractive_content = ProcessedContent(
                text=extractive_summary,
                title=content.title,
                author=content.author,
                source=content.source,
                published_at=content.published_at,
                language=content.language,
                content_type=content.content_type,
                metadata=content.metadata
            )
            
            # Step 3: Generate abstractive summary from extractive summary
            final_summary = await self.abstractive_summarizer.summarize(
                extractive_content, target_length
            )
            
            return final_summary
            
        except Exception as e:
            logger.error(f"Sequential hybrid failed: {str(e)}")
            raise
    
    async def _analyze_content(self, content: ProcessedContent) -> Dict[str, Any]:
        """Analyze content characteristics to determine optimal strategy"""
        try:
            text = content.text.lower()
            
            # News article indicators
            news_indicators = [
                'breaking', 'reported', 'according to', 'sources say',
                'officials', 'announced', 'confirmed', 'statement'
            ]
            is_news_article = any(indicator in text for indicator in news_indicators)
            
            # Technical content indicators
            technical_indicators = [
                'algorithm', 'methodology', 'analysis', 'data', 'research',
                'study', 'experiment', 'results', 'findings', 'conclusion'
            ]
            is_technical = any(indicator in text for indicator in technical_indicators)
            
            # Narrative content indicators
            narrative_indicators = [
                'story', 'narrative', 'character', 'plot', 'scene',
                'dialogue', 'describes', 'tells', 'recounts'
            ]
            is_narrative = any(indicator in text for indicator in narrative_indicators)
            
            # Content length analysis
            word_count = len(content.text.split())
            sentence_count = len(content.text.split('.'))
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            return {
                'is_news_article': is_news_article,
                'is_technical': is_technical,
                'is_narrative': is_narrative,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_sentence_length': avg_sentence_length
            }
            
        except Exception as e:
            logger.error(f"Content analysis failed: {str(e)}")
            return {
                'is_news_article': False,
                'is_technical': False,
                'is_narrative': False,
                'word_count': 0,
                'sentence_count': 0,
                'avg_sentence_length': 0
            }
    
    async def _combine_summaries(
        self, 
        extractive_summary: str, 
        abstractive_summary: str,
        extractive_weight: float, 
        abstractive_weight: float,
        target_length: int
    ) -> str:
        """Combine extractive and abstractive summaries"""
        try:
            # If weights are very different, use the dominant method
            if extractive_weight > 0.8:
                return extractive_summary
            elif abstractive_weight > 0.8:
                return abstractive_summary
            
            # Otherwise, combine using sentence-level fusion
            return await self._sentence_level_fusion(
                extractive_summary, abstractive_summary, target_length
            )
            
        except Exception as e:
            logger.error(f"Summary combination failed: {str(e)}")
            # Fallback to the longer summary
            return extractive_summary if len(extractive_summary) > len(abstractive_summary) else abstractive_summary
    
    async def _sentence_level_fusion(
        self, 
        extractive_summary: str, 
        abstractive_summary: str,
        target_length: int
    ) -> str:
        """Fuse summaries at sentence level"""
        try:
            # Split into sentences
            extractive_sentences = [s.strip() for s in extractive_summary.split('.') if s.strip()]
            abstractive_sentences = [s.strip() for s in abstractive_summary.split('.') if s.strip()]
            
            # Calculate sentence similarities
            all_sentences = extractive_sentences + abstractive_sentences
            if len(all_sentences) < 2:
                return extractive_summary or abstractive_summary
            
            # Vectorize sentences
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Select best sentences
            selected_sentences = []
            used_indices = set()
            current_length = 0
            
            # Start with the most important sentences from each summary
            for i, sentence in enumerate(extractive_sentences):
                if current_length >= target_length:
                    break
                if i not in used_indices:
                    selected_sentences.append(sentence)
                    used_indices.add(i)
                    current_length += len(sentence.split())
            
            # Add unique sentences from abstractive summary
            for i, sentence in enumerate(abstractive_sentences):
                if current_length >= target_length:
                    break
                abs_index = i + len(extractive_sentences)
                if abs_index not in used_indices:
                    # Check if this sentence is too similar to already selected ones
                    is_similar = False
                    for used_idx in used_indices:
                        if similarity_matrix[abs_index][used_idx] > 0.7:
                            is_similar = True
                            break
                    
                    if not is_similar:
                        selected_sentences.append(sentence)
                        used_indices.add(abs_index)
                        current_length += len(sentence.split())
            
            # Join sentences
            result = '. '.join(selected_sentences)
            if not result.endswith('.'):
                result += '.'
            
            return result
            
        except Exception as e:
            logger.error(f"Sentence-level fusion failed: {str(e)}")
            # Fallback to simple concatenation
            return f"{extractive_summary} {abstractive_summary}"
    
    async def generate_multiple_variants(
        self, 
        content: ProcessedContent, 
        target_lengths: List[int]
    ) -> List[Dict[str, Any]]:
        """Generate multiple hybrid summary variants"""
        variants = []
        
        strategies = ["adaptive", "weighted", "sequential"]
        
        for target_length in target_lengths:
            for strategy in strategies:
                try:
                    summary = await self.summarize(
                        content=content,
                        target_length=target_length,
                        strategy=strategy
                    )
                    
                    variants.append({
                        "text": summary,
                        "strategy": strategy,
                        "target_length": target_length,
                        "actual_length": len(summary.split()),
                        "method": SummarizationMethod.HYBRID
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to generate {strategy} variant: {str(e)}")
                    continue
        
        return variants
    
    async def get_status(self) -> Dict[str, Any]:
        """Get hybrid summarizer status"""
        return {
            "initialized": self._initialized,
            "extractive_available": self.extractive_summarizer._initialized if hasattr(self.extractive_summarizer, '_initialized') else False,
            "abstractive_available": self.abstractive_summarizer._initialized if hasattr(self.abstractive_summarizer, '_initialized') else False,
            "strategies": ["adaptive", "weighted", "sequential"]
        }
