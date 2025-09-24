"""
Content Preprocessing Module
Advanced text preprocessing and normalization for summarization
"""

import asyncio
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from config.settings import settings

from .models import ContentType, Language, ProcessedContent

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
except BaseException:
    pass


class ContentPreprocessor:
    """Advanced content preprocessing for summarization"""

    def __init__(self):
        """Initialize the preprocessor"""
        self.nlp = None
        self.lemmatizer = None
        self.stop_words = None
        self._initialized = False

    async def initialize(self) -> Dict[str, Any]:
    """Initialize preprocessing tools"""
        try:
            logger.info("Initializing content preprocessor...")

            # Initialize spaCy model
            try:
                self.nlp = spacy.load(settings.SPACY_MODEL)
            except OSError:
                logger.warning(
                    f"spaCy model {settings.SPACY_MODEL} not found, using basic processing"
                )
                self.nlp = None

            # Initialize lemmatizer
            self.lemmatizer = WordNetLemmatizer()

            # Initialize stop words
            self.stop_words = set(stopwords.words("english"))

            self._initialized = True
            logger.info("Content preprocessor initialized successfully")

except Exception as e:
            logger.error(
                f"Failed to initialize content preprocessor: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
    """Clean up resources"""
        try:
            if self.nlp:
                del self.nlp
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def preprocess(self, content: ProcessedContent) -> ProcessedContent:
        """Preprocess content for summarization"""

        if not self._initialized:
            raise RuntimeError("Content preprocessor not initialized")

        try:
            # Create a copy to avoid modifying original
            processed = ProcessedContent(
                text=content.text,
                title=content.title,
                author=content.author,
                source=content.source,
                published_at=content.published_at,
                language=content.language,
                content_type=content.content_type,
                metadata=content.metadata.copy(),
            )

            # Clean and normalize text
            processed.text = await self._clean_text(processed.text)

            # Detect content type if not specified
            if processed.content_type == ContentType.GENERAL:
                processed.content_type = await self._detect_content_type(processed.text)

            # Extract additional metadata
            processed.metadata.update(await self._extract_metadata(processed.text))

            # Validate content length
            if len(processed.text.split()) < 10:
                raise ValueError("Content too short for summarization")

            return processed

except Exception as e:
            logger.error(f"Content preprocessing failed: {str(e)}")
            return content  # Return original if preprocessing fails

    async def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        try:
            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text)

            # Remove special characters but keep punctuation
            text = re.sub(r'[^\w\s.,!?;:()\-"\']', " ", text)

            # Normalize unicode
            text = unicodedata.normalize("NFKD", text)

            # Remove excessive punctuation
            text = re.sub(r"[.]{2,}", ".", text)
            text = re.sub(r"[!]{2,}", "!", text)
            text = re.sub(r"[?]{2,}", "?", text)

            # Clean up quotes
            text = re.sub(r"[\u201c\u201d]", '"', text)
            text = re.sub(r"[\u2018\u2019]", "'", text)

            # Remove extra spaces
            text = " ".join(text.split())

            return text.strip()

except Exception as e:
            logger.error(f"Text cleaning failed: {str(e)}")
            return text

    async def _detect_content_type(self, text: str) -> ContentType:
        """Detect content type based on text characteristics"""
        try:
            text_lower = text.lower()

            # News article indicators
            news_indicators = [
                "breaking",
                "reported",
                "according to",
                "sources say",
                "officials",
                "announced",
                "confirmed",
                "statement",
                "press release",
                "news",
                "update",
            ]

            # Blog post indicators
            blog_indicators = [
                "i think",
                "in my opinion",
                "personally",
                "i believe",
                "from my experience",
                "i've learned",
                "tips",
                "guide",
                "tutorial",
                "how to",
                "why",
                "what",
            ]

            # Academic paper indicators
            academic_indicators = [
                "abstract",
                "introduction",
                "methodology",
                "results",
                "conclusion",
                "references",
                "study",
                "research",
                "analysis",
                "findings",
                "hypothesis",
                "data",
            ]

            # Social media indicators
            social_indicators = [
                "hashtag",
                "@",
                "retweet",
                "like",
                "share",
                "follow",
                "unfollow",
                "dm",
                "status",
                "post",
            ]

            # Product description indicators
            product_indicators = [
                "buy",
                "purchase",
                "price",
                "sale",
                "discount",
                "features",
                "specifications",
                "warranty",
                "shipping",
                "product",
                "item",
                "order",
            ]

            # Count indicators
            news_count = sum(
                1 for indicator in news_indicators if indicator in text_lower)
            blog_count = sum(
                1 for indicator in blog_indicators if indicator in text_lower)
            academic_count = sum(
                1 for indicator in academic_indicators if indicator in text_lower)
            social_count = sum(
                1 for indicator in social_indicators if indicator in text_lower)
            product_count = sum(
                1 for indicator in product_indicators if indicator in text_lower)

            # Determine content type
            counts = {
                ContentType.NEWS_ARTICLE: news_count,
                ContentType.BLOG_POST: blog_count,
                ContentType.ACADEMIC_PAPER: academic_count,
                ContentType.SOCIAL_MEDIA: social_count,
                ContentType.PRODUCT_DESCRIPTION: product_count,
            }

            max_type = max(counts, key=counts.get)
            max_count = counts[max_type]

            # Return detected type if confidence is high enough
            if max_count >= 2:
                return max_type
else:
                return ContentType.GENERAL

except Exception as e:
            logger.error(f"Content type detection failed: {str(e)}")
            return ContentType.GENERAL

    async def _extract_metadata(self, text: str) -> Dict[str, Any]:
    """Extract additional metadata from text"""
        try:
            metadata = {}

            # Extract basic statistics
            metadata["word_count"] = len(word_tokenize(text))
            metadata["sentence_count"] = len(sent_tokenize(text))
            metadata["character_count"] = len(text)

            # Calculate average sentence length
            if metadata["sentence_count"] > 0:
                metadata["avg_sentence_length"] = (
                    metadata["word_count"] / metadata["sentence_count"]
                )
else:
                metadata["avg_sentence_length"] = 0

            # Extract reading time estimate (assuming 200 WPM)
            metadata["estimated_reading_time"] = metadata["word_count"] / 200

            # Extract language complexity indicators
            if self.nlp:
                doc = self.nlp(text)

                # Count different POS tags
                pos_counts = {}
                for token in doc:
                    pos = token.pos_
                    pos_counts[pos] = pos_counts.get(pos, 0) + 1

                metadata["pos_counts"] = pos_counts

                # Extract named entities
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                metadata["named_entities"] = entities

                # Calculate lexical diversity
                unique_words = set(token.lemma_.lower()
                                   for token in doc if token.is_alpha)
                total_words = len([token for token in doc if token.is_alpha])
                if total_words > 0:
                    metadata["lexical_diversity"] = len(
                        unique_words) / total_words
else:
                    metadata["lexical_diversity"] = 0

            # Extract key phrases (simple approach)
            key_phrases = await self._extract_key_phrases(text)
            metadata["key_phrases"] = key_phrases

            return metadata

except Exception as e:
            logger.error(f"Metadata extraction failed: {str(e)}")
            return {}

    async def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        try:
            if not self.nlp:
                return []

            doc = self.nlp(text)

            # Extract noun phrases
            noun_phrases = []
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 4:  # Limit phrase length
                    noun_phrases.append(chunk.text)

            # Extract named entities
            entities = [
                ent.text for ent in doc.ents if ent.label_ in [
                    "PERSON", "ORG", "GPE", "EVENT"]]

            # Combine and deduplicate
            key_phrases = list(set(noun_phrases + entities))

            # Sort by frequency
            phrase_counts = {}
            for phrase in key_phrases:
                phrase_counts[phrase] = text.lower().count(phrase.lower())

            sorted_phrases = sorted(
                phrase_counts.items(),
                key=lambda x: x[1],
                reverse=True)

            return [phrase for phrase,
                    count in sorted_phrases[:10]]  # Top 10 phrases

except Exception as e:
            logger.error(f"Key phrase extraction failed: {str(e)}")
            return []

    async def get_status(self) -> Dict[str, Any]:
    """Get preprocessor status"""
        return {
            "initialized": self._initialized,
            "spacy_available": self.nlp is not None,
            "lemmatizer_available": self.lemmatizer is not None,
            "stop_words_available": self.stop_words is not None,
        }
