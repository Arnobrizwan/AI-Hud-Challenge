"""
BERT-based Extractive Summarization
High-quality extractive summarization using BERT and transformer models
"""

import asyncio
import logging
from typing import List, Optional, Tuple

import nltk
import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer, pipeline

from config.settings import settings

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except:
    pass


class BertExtractiveSummarizer:
    """BERT-based extractive summarization with advanced scoring"""

    def __init__(self):
        """Initialize the extractive summarizer"""
        self.model = None
        self.tokenizer = None
        self.sentence_model = None
        self.device = "cuda" if torch.cuda.is_available() and settings.USE_GPU else "cpu"
        self.max_length = 512
        self._initialized = False

    async def initialize(self):
        """Initialize models and tokenizers"""
        try:
            logger.info("Initializing BERT extractive summarizer...")

            # Initialize BERT model for sentence scoring
            self.tokenizer = BertTokenizer.from_pretrained(settings.BERT_MODEL_PATH)
            self.model = BertModel.from_pretrained(settings.BERT_MODEL_PATH)
            self.model.to(self.device)
            self.model.eval()

            # Initialize sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

            self._initialized = True
            logger.info("BERT extractive summarizer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize BERT summarizer: {str(e)}")
            raise

    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.model:
                del self.model
            if self.sentence_model:
                del self.sentence_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def summarize(
        self, text: str, target_length: int = 120, min_sentences: int = 2, max_sentences: int = 10
    ) -> str:
        """
        Generate extractive summary using BERT-based scoring

        Args:
            text: Input text to summarize
            target_length: Target summary length in words
            min_sentences: Minimum number of sentences in summary
            max_sentences: Maximum number of sentences in summary

        Returns:
            Extracted summary text
        """
        if not self._initialized:
            raise RuntimeError("Summarizer not initialized")

        try:
            # Preprocess text
            sentences = self._preprocess_sentences(text)

            if len(sentences) <= min_sentences:
                return text

            # Calculate sentence scores using multiple methods
            scores = await self._calculate_sentence_scores(text, sentences)

            # Select best sentences
            selected_sentences = self._select_sentences(
                sentences, scores, target_length, min_sentences, max_sentences
            )

            # Reorder sentences to maintain coherence
            ordered_sentences = self._reorder_sentences(selected_sentences, sentences)

            # Generate final summary
            summary = " ".join(ordered_sentences)

            return summary.strip()

        except Exception as e:
            logger.error(f"Extractive summarization failed: {str(e)}")
            # Fallback to simple extraction
            return self._fallback_extraction(text, target_length)

    def _preprocess_sentences(self, text: str) -> List[str]:
        """Preprocess and split text into sentences"""
        try:
            # Clean text
            text = text.replace("\n", " ").replace("\t", " ")
            text = " ".join(text.split())  # Remove extra whitespace

            # Split into sentences
            sentences = sent_tokenize(text)

            # Filter out very short sentences
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

            return sentences

        except Exception as e:
            logger.error(f"Sentence preprocessing failed: {str(e)}")
            # Fallback to simple splitting
            return [s.strip() for s in text.split(".") if s.strip()]

    async def _calculate_sentence_scores(self, text: str, sentences: List[str]) -> List[float]:
        """Calculate comprehensive sentence scores"""
        try:
            # Get sentence embeddings
            sentence_embeddings = self.sentence_model.encode(sentences)

            # Calculate various scoring metrics
            scores = []

            for i, sentence in enumerate(sentences):
                score = 0.0

                # 1. Position score (sentences at beginning and end are often important)
                position_score = self._calculate_position_score(i, len(sentences))
                score += position_score * 0.1

                # 2. Length score (moderate length sentences are preferred)
                length_score = self._calculate_length_score(sentence)
                score += length_score * 0.1

                # 3. Keyword score (sentences with important keywords)
                keyword_score = self._calculate_keyword_score(sentence, text)
                score += keyword_score * 0.2

                # 4. Semantic similarity to document
                semantic_score = self._calculate_semantic_score(
                    sentence_embeddings[i], sentence_embeddings
                )
                score += semantic_score * 0.3

                # 5. Named entity score (sentences with entities are important)
                entity_score = self._calculate_entity_score(sentence)
                score += entity_score * 0.1

                # 6. BERT-based importance score
                bert_score = await self._calculate_bert_score(sentence, text)
                score += bert_score * 0.2

                scores.append(score)

            # Normalize scores
            if scores:
                max_score = max(scores)
                if max_score > 0:
                    scores = [s / max_score for s in scores]

            return scores

        except Exception as e:
            logger.error(f"Sentence scoring failed: {str(e)}")
            # Return uniform scores as fallback
            return [1.0] * len(sentences)

    def _calculate_position_score(self, position: int, total_sentences: int) -> float:
        """Calculate position-based score"""
        if total_sentences <= 1:
            return 1.0

        # Higher score for first and last sentences
        if position == 0 or position == total_sentences - 1:
            return 1.0
        elif position < total_sentences * 0.1 or position > total_sentences * 0.9:
            return 0.8
        else:
            return 0.5

    def _calculate_length_score(self, sentence: str) -> float:
        """Calculate length-based score"""
        word_count = len(sentence.split())

        # Optimal length is around 15-25 words
        if 15 <= word_count <= 25:
            return 1.0
        elif 10 <= word_count <= 30:
            return 0.8
        elif 5 <= word_count <= 40:
            return 0.6
        else:
            return 0.3

    def _calculate_keyword_score(self, sentence: str, full_text: str) -> float:
        """Calculate keyword-based score"""
        try:
            # Get stopwords
            stop_words = set(stopwords.words("english"))

            # Extract words from sentence and full text
            sentence_words = set(
                word.lower()
                for word in word_tokenize(sentence)
                if word.isalpha() and word.lower() not in stop_words
            )
            text_words = word_tokenize(full_text.lower())

            # Calculate word frequencies
            word_freq = {}
            for word in text_words:
                if word.isalpha() and word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1

            # Calculate score based on word importance
            if not sentence_words:
                return 0.0

            total_freq = sum(word_freq.values())
            if total_freq == 0:
                return 0.0

            sentence_score = sum(word_freq.get(word, 0) for word in sentence_words)
            max_possible_score = len(sentence_words) * max(word_freq.values()) if word_freq else 1

            return sentence_score / max_possible_score if max_possible_score > 0 else 0.0

        except Exception as e:
            logger.error(f"Keyword scoring failed: {str(e)}")
            return 0.5

    def _calculate_semantic_score(
        self, sentence_embedding: np.ndarray, all_embeddings: np.ndarray
    ) -> float:
        """Calculate semantic similarity score"""
        try:
            # Calculate cosine similarity with all other sentences
            similarities = cosine_similarity([sentence_embedding], all_embeddings)[0]

            # Remove self-similarity
            similarities = np.delete(similarities, np.argmax(similarities))

            # Return average similarity (higher = more representative)
            return float(np.mean(similarities)) if len(similarities) > 0 else 0.0

        except Exception as e:
            logger.error(f"Semantic scoring failed: {str(e)}")
            return 0.5

    def _calculate_entity_score(self, sentence: str) -> float:
        """Calculate named entity score"""
        try:
            # Simple entity detection (can be enhanced with spaCy)
            entity_indicators = [
                # Capitalized words (potential proper nouns)
                len([word for word in sentence.split() if word[0].isupper() and len(word) > 2]),
                # Numbers
                len([word for word in sentence.split() if word.isdigit()]),
                # Common entity patterns
                sentence.count('"'),  # Quoted text
                sentence.count("("),  # Parenthetical information
            ]

            # Normalize score
            max_indicators = max(entity_indicators) if entity_indicators else 1
            return min(sum(entity_indicators) / (max_indicators * 4), 1.0)

        except Exception as e:
            logger.error(f"Entity scoring failed: {str(e)}")
            return 0.5

    async def _calculate_bert_score(self, sentence: str, full_text: str) -> float:
        """Calculate BERT-based importance score"""
        try:
            # Tokenize inputs
            sentence_tokens = self.tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self.device)

            text_tokens = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                # Get BERT embeddings
                sentence_output = self.model(**sentence_tokens)
                text_output = self.model(**text_tokens)

                # Use [CLS] token embeddings
                sentence_embedding = sentence_output.last_hidden_state[:, 0, :]
                text_embedding = text_output.last_hidden_state[:, 0, :]

                # Calculate cosine similarity
                similarity = torch.cosine_similarity(sentence_embedding, text_embedding)

                return float(similarity.cpu().numpy()[0])

        except Exception as e:
            logger.error(f"BERT scoring failed: {str(e)}")
            return 0.5

    def _select_sentences(
        self,
        sentences: List[str],
        scores: List[float],
        target_length: int,
        min_sentences: int,
        max_sentences: int,
    ) -> List[str]:
        """Select best sentences for summary"""
        try:
            # Create sentence-score pairs
            sentence_scores = list(zip(sentences, scores))

            # Sort by score (descending)
            sentence_scores.sort(key=lambda x: x[1], reverse=True)

            selected_sentences = []
            current_length = 0

            for sentence, score in sentence_scores:
                sentence_length = len(sentence.split())

                # Check if adding this sentence would exceed target length
                if (
                    current_length + sentence_length > target_length
                    and len(selected_sentences) >= min_sentences
                ):
                    break

                # Check if we've reached max sentences
                if len(selected_sentences) >= max_sentences:
                    break

                selected_sentences.append(sentence)
                current_length += sentence_length

            # Ensure minimum sentences
            if len(selected_sentences) < min_sentences:
                remaining = min_sentences - len(selected_sentences)
                for sentence, _ in sentence_scores[
                    len(selected_sentences) : len(selected_sentences) + remaining
                ]:
                    if sentence not in selected_sentences:
                        selected_sentences.append(sentence)

            return selected_sentences

        except Exception as e:
            logger.error(f"Sentence selection failed: {str(e)}")
            # Fallback to first few sentences
            return sentences[:min_sentences]

    def _reorder_sentences(
        self, selected_sentences: List[str], original_sentences: List[str]
    ) -> List[str]:
        """Reorder selected sentences to maintain original order"""
        try:
            # Create ordered list based on original position
            ordered = []
            for original_sentence in original_sentences:
                if original_sentence in selected_sentences:
                    ordered.append(original_sentence)

            return ordered

        except Exception as e:
            logger.error(f"Sentence reordering failed: {str(e)}")
            return selected_sentences

    def _fallback_extraction(self, text: str, target_length: int) -> str:
        """Fallback extraction method"""
        try:
            sentences = text.split(".")
            sentences = [s.strip() for s in sentences if s.strip()]

            # Simple selection: first few sentences
            selected = []
            current_length = 0

            for sentence in sentences:
                if current_length >= target_length:
                    break
                selected.append(sentence)
                current_length += len(sentence.split())

            return ". ".join(selected) + "."

        except Exception as e:
            logger.error(f"Fallback extraction failed: {str(e)}")
            return text[: target_length * 5]  # Rough character limit

    async def get_status(self) -> dict:
        """Get summarizer status"""
        return {
            "initialized": self._initialized,
            "device": self.device,
            "model": settings.BERT_MODEL_PATH,
            "max_length": self.max_length,
        }
