"""
Bias Detection and Neutrality Scoring System
Comprehensive bias analysis for political, gender, racial, and sentiment biases
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
import spacy
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from config.settings import settings

from .models import BiasAnalysis, ProcessedContent

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except BaseException:
    pass


class BiasDetector:
    """Comprehensive bias detection and neutrality scoring"""

    def __init__(self):
        """Initialize the bias detector"""
        self.political_classifier = None
        self.gender_classifier = None
        self.sentiment_analyzer = None
        self.nlp = None
        self.sentence_model = None
        self._initialized = False

        # Bias indicators and patterns
        self.political_indicators = {
            "left": [
                "progressive",
                "liberal",
                "democrat",
                "socialist",
                "equality",
                "justice",
                "reform",
                "change",
                "environmental",
                "climate",
                "workers",
                "unions",
                "healthcare",
                "education",
                "welfare",
            ],
            "right": [
                "conservative",
                "republican",
                "traditional",
                "patriot",
                "freedom",
                "liberty",
                "market",
                "business",
                "family",
                "values",
                "security",
                "defense",
                "military",
                "law",
                "order",
                "economy",
            ],
        }

        self.gender_indicators = {
            "male": [
                "he",
                "him",
                "his",
                "man",
                "men",
                "boy",
                "boys",
                "father",
                "husband",
                "brother",
                "son",
                "gentleman",
                "sir",
                "mr",
            ],
            "female": [
                "she",
                "her",
                "woman",
                "women",
                "girl",
                "girls",
                "mother",
                "wife",
                "sister",
                "daughter",
                "lady",
                "madam",
                "ms",
                "mrs",
            ],
        }

        self.racial_indicators = {
            "positive": [
                "diverse",
                "inclusive",
                "multicultural",
                "equality",
                "justice",
                "representation",
                "opportunity",
                "success",
                "achievement",
            ],
            "negative": [
                "stereotype",
                "discrimination",
                "bias",
                "prejudice",
                "racism",
                "inequality",
                "exclusion",
                "marginalization",
                "oppression",
            ],
        }

    async def initialize(self) -> Dict[str, Any]:
        """Initialize bias detection models"""
        try:
            logger.info("Initializing bias detector...")

            # Initialize political bias classifier
            try:
                self.political_classifier = pipeline(
                    "text-classification",
                    model="microsoft/DialoGPT-medium",
                    device=0 if settings.USE_GPU else -1,
                )
            except Exception as e:
                logger.warning(f"Political classifier not available: {str(e)}")

            # Initialize gender bias classifier
            try:
                self.gender_classifier = pipeline(
                    "text-classification",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if settings.USE_GPU else -1,
                )
            except Exception as e:
                logger.warning(f"Gender classifier not available: {str(e)}")

            # Initialize sentiment analyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()

            # Initialize spaCy model
            try:
                self.nlp = spacy.load(settings.SPACY_MODEL)
            except OSError:
                logger.warning(f"spaCy model {settings.SPACY_MODEL} not found")
                self.nlp = None

            # Initialize sentence transformer
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

            self._initialized = True
            logger.info("Bias detector initialized successfully")

except Exception as e:
            logger.error(f"Failed to initialize bias detector: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
        """Clean up resources"""
        try:
            if self.political_classifier:
                del self.political_classifier
            if self.gender_classifier:
                del self.gender_classifier
            if self.sentence_model:
                del self.sentence_model
            if self.nlp:
                del self.nlp
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def analyze_bias(self, text: str) -> BiasAnalysis:
        """Comprehensive bias analysis of text"""

        if not self._initialized:
            raise RuntimeError("Bias detector not initialized")

        try:
            # Analyze different types of bias
            political_bias = await self._analyze_political_bias(text)
            gender_bias = await self._analyze_gender_bias(text)
            racial_bias = await self._analyze_racial_bias(text)
            sentiment_bias = await self._analyze_sentiment_bias(text)

            # Calculate overall bias
            overall_bias = (political_bias + gender_bias +
                            racial_bias + sentiment_bias) / 4

            # Calculate neutrality score
            neutrality_score = 1.0 - overall_bias

            # Detect specific bias types
            detected_biases = self._detect_bias_types(
                political_bias, gender_bias, racial_bias, sentiment_bias
            )

            return BiasAnalysis(
                political_bias=political_bias,
                gender_bias=gender_bias,
                racial_bias=racial_bias,
                sentiment_bias=sentiment_bias,
                overall_bias=overall_bias,
                neutrality_score=neutrality_score,
                detected_biases=detected_biases,
            )

except Exception as e:
            logger.error(f"Bias analysis failed: {str(e)}")
            return BiasAnalysis(
                political_bias=0.0,
                gender_bias=0.0,
                racial_bias=0.0,
                sentiment_bias=0.0,
                overall_bias=0.0,
                neutrality_score=1.0,
                detected_biases=[],
            )

    async def _analyze_political_bias(self, text: str) -> float:
        """Analyze political bias in text"""
        try:
            # Method 1: Keyword-based analysis
            keyword_bias = self._analyze_political_keywords(text)

            # Method 2: Sentiment analysis of political terms
            sentiment_bias = await self._analyze_political_sentiment(text)

            # Method 3: Entity analysis
            entity_bias = await self._analyze_political_entities(text)

            # Combine methods
            political_bias = (keyword_bias + sentiment_bias + entity_bias) / 3

            return min(1.0, max(0.0, political_bias))

except Exception as e:
            logger.error(f"Political bias analysis failed: {str(e)}")
            return 0.0

    def _analyze_political_keywords(self, text: str) -> float:
        """Analyze political bias using keyword patterns"""
        try:
            text_lower = text.lower()
            words = word_tokenize(text_lower)

            left_count = sum(
                1 for word in words if word in self.political_indicators["left"])
            right_count = sum(
                1 for word in words if word in self.political_indicators["right"])

            total_political_words = left_count + right_count

            if total_political_words == 0:
                return 0.0

            # Calculate bias ratio
            bias_ratio = abs(left_count - right_count) / total_political_words

            return bias_ratio

except Exception as e:
            logger.error(f"Political keyword analysis failed: {str(e)}")
            return 0.0

    async def _analyze_political_sentiment(self, text: str) -> float:
        """Analyze sentiment bias in political terms"""
        try:
            # Extract sentences with political terms
            sentences = sent_tokenize(text)
            political_sentences = []

            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(
                    term in sentence_lower
                    for term in self.political_indicators["left"]
                    + self.political_indicators["right"]
                ):
                    political_sentences.append(sentence)

            if not political_sentences:
                return 0.0

            # Analyze sentiment of political sentences
            sentiments = []
            for sentence in political_sentences:
                sentiment = self.sentiment_analyzer.polarity_scores(sentence)
                sentiments.append(sentiment["compound"])

            # Calculate sentiment bias (deviation from neutral)
            avg_sentiment = np.mean(sentiments)
            sentiment_bias = abs(avg_sentiment)

            return min(1.0, sentiment_bias)

except Exception as e:
            logger.error(f"Political sentiment analysis failed: {str(e)}")
            return 0.0

    async def _analyze_political_entities(self, text: str) -> float:
        """Analyze political bias in named entities"""
        try:
            if not self.nlp:
                return 0.0

            doc = self.nlp(text)

            # Extract political entities
            political_entities = []
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE"]:
                    political_entities.append(ent.text)

            if not political_entities:
                return 0.0

            # Analyze sentiment of entity contexts
            entity_sentiments = []
            for entity in political_entities:
                # Find sentences containing the entity
                for sentence in sent_tokenize(text):
                    if entity in sentence:
                        sentiment = self.sentiment_analyzer.polarity_scores(
                            sentence)
                        entity_sentiments.append(sentiment["compound"])
                        break

            if not entity_sentiments:
                return 0.0

            # Calculate entity bias
            avg_entity_sentiment = np.mean(entity_sentiments)
            entity_bias = abs(avg_entity_sentiment)

            return min(1.0, entity_bias)

except Exception as e:
            logger.error(f"Political entity analysis failed: {str(e)}")
            return 0.0

    async def _analyze_gender_bias(self, text: str) -> float:
        """Analyze gender bias in text"""
        try:
            # Method 1: Pronoun analysis
            pronoun_bias = self._analyze_gender_pronouns(text)

            # Method 2: Role/stereotype analysis
            role_bias = self._analyze_gender_roles(text)

            # Method 3: Sentiment analysis by gender
            sentiment_bias = await self._analyze_gender_sentiment(text)

            # Combine methods
            gender_bias = (pronoun_bias + role_bias + sentiment_bias) / 3

            return min(1.0, max(0.0, gender_bias))

except Exception as e:
            logger.error(f"Gender bias analysis failed: {str(e)}")
            return 0.0

    def _analyze_gender_pronouns(self, text: str) -> float:
        """Analyze gender bias in pronoun usage"""
        try:
            text_lower = text.lower()
            words = word_tokenize(text_lower)

            male_pronouns = sum(
                1 for word in words if word in self.gender_indicators["male"])
            female_pronouns = sum(
                1 for word in words if word in self.gender_indicators["female"])

            total_pronouns = male_pronouns + female_pronouns

            if total_pronouns == 0:
                return 0.0

            # Calculate pronoun bias
            pronoun_bias = abs(
                male_pronouns - female_pronouns) / total_pronouns

            return pronoun_bias

except Exception as e:
            logger.error(f"Gender pronoun analysis failed: {str(e)}")
            return 0.0

    def _analyze_gender_roles(self, text: str) -> float:
        """Analyze gender bias in role descriptions"""
        try:
            # Common gender-biased role patterns
            male_roles = [
                "ceo",
                "president",
                "leader",
                "boss",
                "manager",
                "director",
                "engineer",
                "scientist",
                "doctor",
                "lawyer",
                "professor",
            ]

            female_roles = [
                "nurse",
                "teacher",
                "secretary",
                "assistant",
                "receptionist",
                "housewife",
                "mother",
                "caregiver",
                "nanny",
                "cleaner",
            ]

            text_lower = text.lower()
            words = word_tokenize(text_lower)

            male_role_count = sum(1 for word in words if word in male_roles)
            female_role_count = sum(
                1 for word in words if word in female_roles)

            total_roles = male_role_count + female_role_count

            if total_roles == 0:
                return 0.0

            # Calculate role bias
            role_bias = abs(male_role_count - female_role_count) / total_roles

            return role_bias

except Exception as e:
            logger.error(f"Gender role analysis failed: {str(e)}")
            return 0.0

    async def _analyze_gender_sentiment(self, text: str) -> float:
        """Analyze sentiment bias by gender"""
        try:
            # Extract sentences with gender indicators
            sentences = sent_tokenize(text)
            male_sentences = []
            female_sentences = []

            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(
                        term in sentence_lower for term in self.gender_indicators["male"]):
                    male_sentences.append(sentence)
                elif any(term in sentence_lower for term in self.gender_indicators["female"]):
                    female_sentences.append(sentence)

            if not male_sentences and not female_sentences:
                return 0.0

            # Analyze sentiment for each gender
            male_sentiments = []
            female_sentiments = []

            for sentence in male_sentences:
                sentiment = self.sentiment_analyzer.polarity_scores(sentence)
                male_sentiments.append(sentiment["compound"])

            for sentence in female_sentences:
                sentiment = self.sentiment_analyzer.polarity_scores(sentence)
                female_sentiments.append(sentiment["compound"])

            # Calculate sentiment bias
            if male_sentiments and female_sentiments:
                avg_male_sentiment = np.mean(male_sentiments)
                avg_female_sentiment = np.mean(female_sentiments)
                sentiment_bias = abs(avg_male_sentiment - avg_female_sentiment)
else:
                sentiment_bias = 0.0

            return min(1.0, sentiment_bias)

except Exception as e:
            logger.error(f"Gender sentiment analysis failed: {str(e)}")
            return 0.0

    async def _analyze_racial_bias(self, text: str) -> float:
        """Analyze racial bias in text"""
        try:
            # Method 1: Keyword analysis
            keyword_bias = self._analyze_racial_keywords(text)

            # Method 2: Sentiment analysis
            sentiment_bias = await self._analyze_racial_sentiment(text)

            # Method 3: Representation analysis
            representation_bias = await self._analyze_racial_representation(text)

            # Combine methods
            racial_bias = (keyword_bias + sentiment_bias +
                           representation_bias) / 3

            return min(1.0, max(0.0, racial_bias))

except Exception as e:
            logger.error(f"Racial bias analysis failed: {str(e)}")
            return 0.0

    def _analyze_racial_keywords(self, text: str) -> float:
        """Analyze racial bias using keyword patterns"""
        try:
            text_lower = text.lower()
            words = word_tokenize(text_lower)

            positive_count = sum(
                1 for word in words if word in self.racial_indicators["positive"])
            negative_count = sum(
                1 for word in words if word in self.racial_indicators["negative"])

            total_racial_words = positive_count + negative_count

            if total_racial_words == 0:
                return 0.0

            # Calculate bias ratio
            bias_ratio = negative_count / total_racial_words

            return bias_ratio

except Exception as e:
            logger.error(f"Racial keyword analysis failed: {str(e)}")
            return 0.0

    async def _analyze_racial_sentiment(self, text: str) -> float:
        """Analyze sentiment bias in racial contexts"""
        try:
            # Extract sentences with racial terms
            sentences = sent_tokenize(text)
            racial_sentences = []

            racial_terms = [
                "race",
                "ethnic",
                "cultural",
                "diversity",
                "minority",
                "african",
                "asian",
                "hispanic",
                "caucasian",
                "indigenous",
            ]

            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(term in sentence_lower for term in racial_terms):
                    racial_sentences.append(sentence)

            if not racial_sentences:
                return 0.0

            # Analyze sentiment
            sentiments = []
            for sentence in racial_sentences:
                sentiment = self.sentiment_analyzer.polarity_scores(sentence)
                sentiments.append(sentiment["compound"])

            # Calculate sentiment bias
            avg_sentiment = np.mean(sentiments)
            sentiment_bias = abs(avg_sentiment)

            return min(1.0, sentiment_bias)

except Exception as e:
            logger.error(f"Racial sentiment analysis failed: {str(e)}")
            return 0.0

    async def _analyze_racial_representation(self, text: str) -> float:
        """Analyze racial representation bias"""
        try:
            if not self.nlp:
                return 0.0

            doc = self.nlp(text)

            # Extract person entities
            person_entities = [
                ent.text for ent in doc.ents if ent.label_ == "PERSON"]

            if not person_entities:
                return 0.0

            # Analyze sentiment of person contexts
            person_sentiments = []
            for person in person_entities:
                for sentence in sent_tokenize(text):
                    if person in sentence:
                        sentiment = self.sentiment_analyzer.polarity_scores(
                            sentence)
                        person_sentiments.append(sentiment["compound"])
                        break

            if not person_sentiments:
                return 0.0

            # Calculate representation bias
            avg_sentiment = np.mean(person_sentiments)
            representation_bias = abs(avg_sentiment)

            return min(1.0, representation_bias)

except Exception as e:
            logger.error(f"Racial representation analysis failed: {str(e)}")
            return 0.0

    async def _analyze_sentiment_bias(self, text: str) -> float:
        """Analyze sentiment bias in text"""
        try:
            # Analyze overall sentiment
            sentiment = self.sentiment_analyzer.polarity_scores(text)

            # Calculate sentiment bias (deviation from neutral)
            compound_score = sentiment["compound"]
            sentiment_bias = abs(compound_score)

            # Analyze sentiment distribution across sentences
            sentences = sent_tokenize(text)
            if len(sentences) > 1:
                sentence_sentiments = []
                for sentence in sentences:
                    sent = self.sentiment_analyzer.polarity_scores(sentence)
                    sentence_sentiments.append(sent["compound"])

                # Calculate sentiment variance
                sentiment_variance = np.var(sentence_sentiments)
                sentiment_bias = max(sentiment_bias, sentiment_variance)

            return min(1.0, sentiment_bias)

except Exception as e:
            logger.error(f"Sentiment bias analysis failed: {str(e)}")
            return 0.0

    def _detect_bias_types(
            self,
            political_bias: float,
            gender_bias: float,
            racial_bias: float,
            sentiment_bias: float) -> List[str]:
        """Detect specific types of bias present"""
        detected_biases = []

        if political_bias > 0.3:
            detected_biases.append("political")

        if gender_bias > 0.3:
            detected_biases.append("gender")

        if racial_bias > 0.3:
            detected_biases.append("racial")

        if sentiment_bias > 0.3:
            detected_biases.append("sentiment")

        return detected_biases

    async def get_status(self) -> Dict[str, Any]:
        """Get bias detector status"""
        return {
            "initialized": self._initialized,
            "political_classifier_available": self.political_classifier is not None,
            "gender_classifier_available": self.gender_classifier is not None,
            "sentiment_analyzer_available": self.sentiment_analyzer is not None,
            "spacy_available": self.nlp is not None,
            "sentence_transformer_available": self.sentence_model is not None,
        }
