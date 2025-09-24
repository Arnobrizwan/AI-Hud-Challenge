"""
Advanced Headline Generation with T5 and Multiple Style Variants
Comprehensive headline generation with quality scoring and A/B testing
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from summarization.models import HeadlineStyle, Language, ProcessedContent
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

from config.settings import settings

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("vader_lexicon", quiet=True)
except BaseException:
    pass


class HeadlineGenerator:
    """Advanced headline generation with style variants and quality scoring"""

    def __init__(self):
        """Initialize the headline generator"""
        self.t5_model = None
        self.t5_tokenizer = None
        self.sentence_model = None
        self.sentiment_analyzer = None
        self.device = "cuda" if torch.cuda.is_available() and settings.USE_GPU else "cpu"
        self._initialized = False

        # Style-specific prompts and parameters
        self.style_configs = {
            HeadlineStyle.NEWS: {
                "prompt": "Generate a factual news headline:",
                "temperature": 0.1,
                "max_length": 15,
                "keywords": ["breaking", "reports", "announces", "confirms"],
            },
            HeadlineStyle.ENGAGING: {
                "prompt": "Generate an engaging, clickworthy headline:",
                "temperature": 0.7,
                "max_length": 20,
                "keywords": ["shocking", "amazing", "incredible", "must-see"],
            },
            HeadlineStyle.QUESTION: {
                "prompt": "Generate a question-based headline:",
                "temperature": 0.5,
                "max_length": 18,
                "keywords": ["what", "how", "why", "when", "where"],
            },
            HeadlineStyle.NEUTRAL: {
                "prompt": "Generate a neutral, objective headline:",
                "temperature": 0.2,
                "max_length": 16,
                "keywords": ["analysis", "overview", "summary", "report"],
            },
            HeadlineStyle.URGENT: {
                "prompt": "Generate an urgent, breaking news headline:",
                "temperature": 0.3,
                "max_length": 14,
                "keywords": ["urgent", "breaking", "alert", "immediate"],
            },
        }

    async def initialize(self) -> Dict[str, Any]:
    """Initialize models and tokenizers"""
        try:
            logger.info("Initializing headline generator...")

            # Initialize T5 model
            self.t5_tokenizer = T5Tokenizer.from_pretrained(
                settings.T5_MODEL_PATH)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(
                settings.T5_MODEL_PATH)
            self.t5_model.to(self.device)
            self.t5_model.eval()

            # Initialize sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

            # Initialize sentiment analyzer
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1,
            )

            self._initialized = True
            logger.info("Headline generator initialized successfully")

except Exception as e:
            logger.error(f"Failed to initialize headline generator: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
    """Clean up resources"""
        try:
            if self.t5_model:
                del self.t5_model
            if self.sentence_model:
                del self.sentence_model
            if self.sentiment_analyzer:
                del self.sentiment_analyzer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def generate_headlines(
        self, content: ProcessedContent, styles: List[str], num_variants: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple headline variants with different styles

        Args:
            content: Processed content to generate headlines for
            styles: List of headline styles to generate
            num_variants: Number of variants per style

        Returns:
            List of generated headlines with scores and metrics
        """
        if not self._initialized:
            raise RuntimeError("Headline generator not initialized")

        try:
            all_headlines = []

            for style in styles:
                try:
                    style_enum = HeadlineStyle(style)
                    style_config = self.style_configs[style_enum]

                    # Generate multiple candidates for this style
                    candidates = await self._generate_style_candidates(
                        content, style_enum, num_variants
                    )

                    # Score and rank candidates
                    scored_candidates = await self._score_headlines(candidates, content, style_enum)

                    # Select best headlines
                    best_headlines = sorted(
                        scored_candidates,
                        key=lambda x: x["score"],
                        reverse=True)[
                        :num_variants]

                    all_headlines.extend(best_headlines)

except Exception as e:
                    logger.error(
                        f"Failed to generate {style} headlines: {str(e)}")
                    continue

            return all_headlines

except Exception as e:
            logger.error(f"Headline generation failed: {str(e)}")
            raise

    async def _generate_style_candidates(
            self,
            content: ProcessedContent,
            style: HeadlineStyle,
            num_candidates: int) -> List[str]:
        """Generate headline candidates for a specific style"""
        try:
            style_config = self.style_configs[style]

            # Prepare input text (truncate if too long)
            input_text = content.text[:1000]  # Limit input length

            # Create prompt
            prompt = f"{style_config['prompt']} {input_text}"

            # Generate multiple candidates with different parameters
            candidates = []

            for i in range(num_candidates):
                try:
                    # Vary temperature slightly for diversity
                    temperature = style_config["temperature"] + (i * 0.1)
                    temperature = min(temperature, 1.0)

                    # Tokenize input
                    inputs = self.t5_tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True).to(
                        self.device)

                    # Generate headline
                    with torch.no_grad():
                        outputs = self.t5_model.generate(
                            inputs.input_ids,
                            max_length=style_config["max_length"],
                            temperature=temperature,
                            do_sample=True,
                            top_p=0.9,
                            top_k=50,
                            num_return_sequences=1,
                            pad_token_id=self.t5_tokenizer.pad_token_id,
                            eos_token_id=self.t5_tokenizer.eos_token_id,
                        )

                    # Decode headline
                    headline = self.t5_tokenizer.decode(
                        outputs[0], skip_special_tokens=True)

                    # Clean up headline
                    headline = self._clean_headline(headline)

                    if headline and len(
                            headline.split()) > 3:  # Minimum length check
                        candidates.append(headline)

except Exception as e:
                    logger.error(f"Failed to generate candidate {i}: {str(e)}")
                    continue

            # If we don't have enough candidates, generate fallback ones
            while len(candidates) < num_candidates:
                fallback_headline = self._generate_fallback_headline(
                    content, style)
                if fallback_headline not in candidates:
                    candidates.append(fallback_headline)

            return candidates[:num_candidates]

except Exception as e:
            logger.error(f"Style candidate generation failed: {str(e)}")
            return []

    def _clean_headline(self, headline: str) -> str:
        """Clean and format headline"""
        try:
            # Remove prompt text if present
            headline = headline.replace(
                "Generate a factual news headline:", "")
            headline = headline.replace(
                "Generate an engaging, clickworthy headline:", "")
            headline = headline.replace(
                "Generate a question-based headline:", "")
            headline = headline.replace(
                "Generate a neutral, objective headline:", "")
            headline = headline.replace(
                "Generate an urgent, breaking news headline:", "")

            # Clean up whitespace
            headline = " ".join(headline.split())

            # Remove quotes if present
            headline = headline.strip("\"'")

            # Ensure proper capitalization
            headline = headline.capitalize()

            # Remove trailing punctuation except for questions
            if not headline.endswith("?"):
                headline = headline.rstrip(".!")

            return headline.strip()

except Exception as e:
            logger.error(f"Headline cleaning failed: {str(e)}")
            return headline

    def _generate_fallback_headline(
            self,
            content: ProcessedContent,
            style: HeadlineStyle) -> str:
        """Generate fallback headline using simple extraction"""
        try:
            # Extract first sentence or key phrases
            sentences = sent_tokenize(content.text)
            if not sentences:
                return "Important News Update"

            first_sentence = sentences[0]

            # Extract key words
            words = word_tokenize(first_sentence.lower())
            stop_words = set(stopwords.words("english"))
            key_words = [
                w for w in words if w.isalpha() and w not in stop_words]

            # Create headline based on style
            if style == HeadlineStyle.QUESTION:
                return f"What {key_words[0] if key_words else 'happened'}?"
            elif style == HeadlineStyle.URGENT:
                return f"Breaking: {first_sentence[:50]}..."
            elif style == HeadlineStyle.ENGAGING:
                return f"Amazing: {first_sentence[:40]}..."
else:
                return first_sentence[:60] + \
                    "..." if len(first_sentence) > 60 else first_sentence

except Exception as e:
            logger.error(f"Fallback headline generation failed: {str(e)}")
            return "Important News Update"

    async def _score_headlines(
        self, candidates: List[str], content: ProcessedContent, style: HeadlineStyle
    ) -> List[Dict[str, Any]]:
        """Score and rank headline candidates"""
        try:
            scored_headlines = []

            for candidate in candidates:
                scores = {}

                # 1. Factual accuracy (semantic similarity to content)
                scores["factual_accuracy"] = await self._check_factual_accuracy(
                    candidate, content.text
                )

                # 2. Semantic similarity to title
                scores["semantic_similarity"] = await self._compute_semantic_similarity(
                    candidate, content.title or content.text[:100]
                )

                # 3. Readability score
                scores["readability"] = self._compute_readability_score(
                    candidate)

                # 4. Engagement prediction
                scores["engagement"] = await self._predict_engagement(candidate, style)

                # 5. Length appropriateness
                scores["length_appropriateness"] = self._score_length_appropriateness(
                    candidate)

                # 6. Grammatical quality
                scores["grammatical_quality"] = await self._check_grammar(candidate)

                # 7. Style adherence
                scores["style_adherence"] = self._check_style_adherence(
                    candidate, style)

                # 8. Sentiment preservation
                scores["sentiment_preservation"] = await self._check_sentiment_preservation(
                    candidate, content.text
                )

                # Calculate weighted total score
                weights = {
                    "factual_accuracy": 0.25,
                    "semantic_similarity": 0.15,
                    "readability": 0.15,
                    "engagement": 0.15,
                    "length_appropriateness": 0.10,
                    "grammatical_quality": 0.10,
                    "style_adherence": 0.05,
                    "sentiment_preservation": 0.05,
                }

                total_score = sum(
                    scores[metric] * weight for metric,
                    weight in weights.items())

                scored_headlines.append(
                    {
                        "text": candidate,
                        "style": style.value,
                        "score": total_score,
                        "metrics": scores,
                    }
                )

            return scored_headlines

except Exception as e:
            logger.error(f"Headline scoring failed: {str(e)}")
            return [
                {"text": candidate, "style": style.value, "score": 0.5, "metrics": {}}
                for candidate in candidates
            ]

    async def _check_factual_accuracy(
            self, headline: str, content: str) -> float:
        """Check if headline facts are present in content"""
        try:
            # Extract entities from headline and content
            headline_words = set(word.lower() for word in word_tokenize(
                headline) if word.isalpha() and len(word) > 2)
            content_words = set(word.lower() for word in word_tokenize(
                content) if word.isalpha() and len(word) > 2)

            # Calculate overlap
            if not headline_words:
                return 0.0

            overlap = len(headline_words.intersection(content_words))
            return overlap / len(headline_words)

except Exception as e:
            logger.error(f"Factual accuracy check failed: {str(e)}")
            return 0.5

    async def _compute_semantic_similarity(
            self, headline: str, title: str) -> float:
        """Compute semantic similarity between headline and title"""
        try:
            if not title:
                return 0.5

            # Get embeddings
            headline_embedding = self.sentence_model.encode([headline])
            title_embedding = self.sentence_model.encode([title])

            # Calculate cosine similarity
            similarity = cosine_similarity(
                headline_embedding, title_embedding)[0][0]
            return float(similarity)

except Exception as e:
            logger.error(f"Semantic similarity computation failed: {str(e)}")
            return 0.5

    def _compute_readability_score(self, headline: str) -> float:
        """Compute readability score for headline"""
        try:
            words = word_tokenize(headline)
            sentences = sent_tokenize(headline)

            if not words or not sentences:
                return 0.0

            # Simple readability metrics
            avg_words_per_sentence = len(words) / len(sentences)
            avg_syllables_per_word = sum(
                self._count_syllables(word) for word in words) / len(words)

            # Flesch Reading Ease approximation
            score = 206.835 - (1.015 * avg_words_per_sentence) - \
                (84.6 * avg_syllables_per_word)

            # Normalize to 0-1 scale
            return max(0, min(1, score / 100))

except Exception as e:
            logger.error(f"Readability computation failed: {str(e)}")
            return 0.5

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel

        # Handle silent 'e'
        if word.endswith("e") and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)

    async def _predict_engagement(
            self,
            headline: str,
            style: HeadlineStyle) -> float:
        """Predict engagement potential of headline"""
        try:
            # Check for engagement indicators
            engagement_indicators = [
                "amazing",
                "shocking",
                "incredible",
                "breaking",
                "urgent",
                "exclusive",
                "revealed",
                "uncovered",
                "exposed",
                "secret",
                "must-see",
                "can't miss",
                "essential",
                "critical",
                "important",
            ]

            headline_lower = headline.lower()
            indicator_count = sum(
                1 for indicator in engagement_indicators if indicator in headline_lower)

            # Check for emotional words
            emotional_words = [
                "love",
                "hate",
                "fear",
                "anger",
                "joy",
                "sadness",
                "surprise",
                "excitement",
                "worry",
                "hope",
                "disappointment",
                "relief",
            ]

            emotional_count = sum(
                1 for word in emotional_words if word in headline_lower)

            # Check for numbers (often increase engagement)
            number_count = len(re.findall(r"\d+", headline))

            # Calculate engagement score
            base_score = 0.5
            indicator_score = min(0.3, indicator_count * 0.1)
            emotional_score = min(0.2, emotional_count * 0.1)
            number_score = min(0.1, number_count * 0.05)

            return base_score + indicator_score + emotional_score + number_score

except Exception as e:
            logger.error(f"Engagement prediction failed: {str(e)}")
            return 0.5

    def _score_length_appropriateness(self, headline: str) -> float:
        """Score headline length appropriateness"""
        try:
            word_count = len(headline.split())

            # Optimal length is 6-12 words
            if 6 <= word_count <= 12:
                return 1.0
            elif 4 <= word_count <= 15:
                return 0.8
            elif 3 <= word_count <= 18:
                return 0.6
else:
                return 0.3

except Exception as e:
            logger.error(f"Length scoring failed: {str(e)}")
            return 0.5

    async def _check_grammar(self, headline: str) -> float:
        """Check grammatical quality of headline"""
        try:
            # Simple grammar checks
            score = 1.0

            # Check for proper capitalization
            if not headline[0].isupper():
                score -= 0.2

            # Check for proper ending punctuation
            if not headline.endswith((".", "!", "?")):
                score -= 0.1

            # Check for double spaces
            if "  " in headline:
                score -= 0.1

            # Check for repeated words
            words = headline.lower().split()
            if len(words) != len(set(words)):
                score -= 0.2

            return max(0, score)

except Exception as e:
            logger.error(f"Grammar check failed: {str(e)}")
            return 0.5

    def _check_style_adherence(
            self,
            headline: str,
            style: HeadlineStyle) -> float:
        """Check if headline adheres to specified style"""
        try:
            style_config = self.style_configs[style]
            headline_lower = headline.lower()

            # Check for style-specific keywords
            keyword_matches = sum(
                1 for keyword in style_config["keywords"] if keyword in headline_lower)

            # Check length adherence
            word_count = len(headline.split())
            max_length = style_config["max_length"]
            length_score = 1.0 - abs(word_count - max_length) / max_length
            length_score = max(0, length_score)

            # Check style-specific patterns
            pattern_score = 0.5
            if style == HeadlineStyle.QUESTION and headline.endswith("?"):
                pattern_score = 1.0
            elif style == HeadlineStyle.URGENT and any(
                word in headline_lower for word in ["breaking", "urgent", "alert"]
            ):
                pattern_score = 1.0
            elif style == HeadlineStyle.ENGAGING and any(
                word in headline_lower for word in ["amazing", "incredible", "shocking"]
            ):
                pattern_score = 1.0

            # Combine scores
            keyword_score = min(1.0, keyword_matches * 0.3)
            return (keyword_score + length_score + pattern_score) / 3

except Exception as e:
            logger.error(f"Style adherence check failed: {str(e)}")
            return 0.5

    async def _check_sentiment_preservation(
            self, headline: str, content: str) -> float:
        """Check if headline preserves content sentiment"""
        try:
            # Analyze sentiment of headline and content
            headline_sentiment = self.sentiment_analyzer(headline)[0]
            content_sentiment = self.sentiment_analyzer(
                content[:500])[0]  # Sample content

            # Compare sentiment labels
            if headline_sentiment["label"] == content_sentiment["label"]:
                # Check confidence scores
                confidence_diff = abs(
                    headline_sentiment["score"] -
                    content_sentiment["score"])
                return max(0, 1.0 - confidence_diff)
else:
                return 0.3  # Different sentiment labels

except Exception as e:
            logger.error(f"Sentiment preservation check failed: {str(e)}")
            return 0.5

    async def get_status(self) -> Dict[str, Any]:
    """Get headline generator status"""
        return {
            "initialized": self._initialized,
            "model": settings.T5_MODEL_PATH,
            "device": self.device,
            "supported_styles": [style.value for style in HeadlineStyle],
            "max_candidates_per_style": 10,
        }
