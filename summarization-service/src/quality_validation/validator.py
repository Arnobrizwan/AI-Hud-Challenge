"""
Comprehensive Quality Validation System
ROUGE, BERTScore, factual consistency, and readability assessment
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
import spacy
from bert_score import score as bert_score
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from summarization.models import ProcessedContent, QualityMetrics
from textstat import flesch_kincaid_grade, flesch_reading_ease

from config.settings import settings

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
except BaseException:
    pass


class SummaryQualityValidator:
    """Comprehensive quality validation for summaries and headlines"""

    def __init__(self):
        """Initialize the quality validator"""
        self.rouge_scorer = None
        self.sentence_model = None
        self.nlp = None
        self.tfidf_vectorizer = None
        self._initialized = False

    async def initialize(self) -> Dict[str, Any]:
    """Initialize models and tools"""
        try:
            logger.info("Initializing quality validator...")

            # Initialize ROUGE scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )

            # Initialize sentence transformer
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

            # Initialize spaCy model
            try:
                self.nlp = spacy.load(settings.SPACY_MODEL)
            except OSError:
                logger.warning(
                    f"spaCy model {settings.SPACY_MODEL} not found, using basic processing"
                )
                self.nlp = None

            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000, stop_words="english", ngram_range=(1, 2)
            )

            self._initialized = True
            logger.info("Quality validator initialized successfully")

except Exception as e:
            logger.error(f"Failed to initialize quality validator: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
    """Clean up resources"""
        try:
            if self.sentence_model:
                del self.sentence_model
            if self.nlp:
                del self.nlp
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def validate_summary_quality(
            self,
            original: str,
            summary: str) -> QualityMetrics:
        """Comprehensive quality assessment of summary"""

        if not self._initialized:
            raise RuntimeError("Quality validator not initialized")

        try:
            # Calculate all quality metrics
            rouge_scores = await self._calculate_rouge_scores(original, summary)
            bert_scores = await self._calculate_bert_scores(original, summary)
            factual_consistency = await self._calculate_factual_consistency(original, summary)
            readability = await self._calculate_readability(summary)
            coverage = await self._calculate_coverage(original, summary)
            abstractiveness = await self._calculate_abstractiveness(original, summary)

            # Calculate overall score
            overall_score = self._calculate_overall_score(
                rouge_scores,
                bert_scores,
                factual_consistency,
                readability,
                coverage,
                abstractiveness,
            )

            return QualityMetrics(
                rouge1_f1=rouge_scores["rouge1_f1"],
                rouge2_f1=rouge_scores["rouge2_f1"],
                rougeL_f1=rouge_scores["rougeL_f1"],
                bertscore_f1=bert_scores["bertscore_f1"],
                factual_consistency=factual_consistency,
                readability=readability,
                coverage=coverage,
                abstractiveness=abstractiveness,
                overall_score=overall_score,
            )

except Exception as e:
            logger.error(f"Quality validation failed: {str(e)}")
            # Return default metrics
            return QualityMetrics(
                rouge1_f1=0.0,
                rouge2_f1=0.0,
                rougeL_f1=0.0,
                bertscore_f1=0.0,
                factual_consistency=0.0,
                readability=0.0,
                coverage=0.0,
                abstractiveness=0.0,
                overall_score=0.0,
            )

    async def _calculate_rouge_scores(
            self, original: str, summary: str) -> Dict[str, float]:
        """Calculate ROUGE scores for n-gram overlap"""
        try:
            scores = self.rouge_scorer.score(original, summary)

            return {
                "rouge1_f1": scores["rouge1"].fmeasure,
                "rouge2_f1": scores["rouge2"].fmeasure,
                "rougeL_f1": scores["rougeL"].fmeasure,
            }

except Exception as e:
            logger.error(f"ROUGE calculation failed: {str(e)}")
            return {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0}

    async def _calculate_bert_scores(
            self, original: str, summary: str) -> Dict[str, float]:
        """Calculate BERTScore for semantic similarity"""
        try:
            # BERTScore calculation
            P, R, F1 = bert_score(
                [summary], [original], lang="en", verbose=False)

            return {"bertscore_f1": float(F1.item())}

except Exception as e:
            logger.error(f"BERTScore calculation failed: {str(e)}")
            return {"bertscore_f1": 0.0}

    async def _calculate_factual_consistency(
            self, original: str, summary: str) -> float:
        """Calculate factual consistency between original and summary"""
        try:
            # Extract entities from both texts
            original_entities = self._extract_entities(original)
            summary_entities = self._extract_entities(summary)

            # Check entity consistency
            entity_consistency = self._check_entity_consistency(
                original_entities, summary_entities)

            # Check numerical consistency
            numerical_consistency = self._check_numerical_consistency(
                original, summary)

            # Check temporal consistency
            temporal_consistency = self._check_temporal_consistency(
                original, summary)

            # Check semantic consistency
            semantic_consistency = await self._check_semantic_consistency(original, summary)

            # Combine scores
            consistency_score = (
                entity_consistency * 0.3
                + numerical_consistency * 0.2
                + temporal_consistency * 0.2
                + semantic_consistency * 0.3
            )

            return consistency_score

except Exception as e:
            logger.error(f"Factual consistency calculation failed: {str(e)}")
            return 0.5

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        try:
            if self.nlp:
                doc = self.nlp(text)
                entities = {
                    "PERSON": [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
                    "ORG": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
                    "GPE": [ent.text for ent in doc.ents if ent.label_ == "GPE"],
                    "DATE": [ent.text for ent in doc.ents if ent.label_ == "DATE"],
                    "MONEY": [ent.text for ent in doc.ents if ent.label_ == "MONEY"],
                    "PERCENT": [ent.text for ent in doc.ents if ent.label_ == "PERCENT"],
                }
        else:
                # Fallback: simple entity extraction
                entities = self._simple_entity_extraction(text)

            return entities

except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return {
                "PERSON": [],
                "ORG": [],
                "GPE": [],
                "DATE": [],
                "MONEY": [],
                "PERCENT": []}

    def _simple_entity_extraction(self, text: str) -> Dict[str, List[str]]:
        """Simple entity extraction without spaCy"""
        entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],
            "DATE": [],
            "MONEY": [],
            "PERCENT": []}

        # Extract dates
        date_pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b"
        entities["DATE"] = re.findall(date_pattern, text)

        # Extract money
        money_pattern = r"\$[\d,]+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*dollars?"
        entities["MONEY"] = re.findall(money_pattern, text, re.IGNORECASE)

        # Extract percentages
        percent_pattern = r"\d+(?:\.\d+)?%"
        entities["PERCENT"] = re.findall(percent_pattern, text)

        return entities

    def _check_entity_consistency(
            self,
            original_entities: Dict,
            summary_entities: Dict) -> float:
        """Check consistency of entities between original and summary"""
        try:
            total_consistency = 0.0
            entity_types = 0

            for entity_type in original_entities:
                if not original_entities[entity_type]:
                    continue

                entity_types += 1
                original_set = set(original_entities[entity_type])
                summary_set = set(summary_entities[entity_type])

                if not original_set:
                    continue

                # Calculate Jaccard similarity
                intersection = len(original_set.intersection(summary_set))
                union = len(original_set.union(summary_set))

                if union > 0:
                    consistency = intersection / union
                    total_consistency += consistency

            return total_consistency / entity_types if entity_types > 0 else 0.5

except Exception as e:
            logger.error(f"Entity consistency check failed: {str(e)}")
            return 0.5

    def _check_numerical_consistency(
            self, original: str, summary: str) -> float:
        """Check consistency of numbers between original and summary"""
        try:
            # Extract numbers from both texts
            original_numbers = re.findall(r"\d+(?:\.\d+)?", original)
            summary_numbers = re.findall(r"\d+(?:\.\d+)?", summary)

            if not original_numbers:
                return 1.0  # No numbers to check

            # Check if summary numbers are subset of original numbers
            original_set = set(original_numbers)
            summary_set = set(summary_numbers)

            # Calculate precision and recall
            if not summary_set:
                return 0.5  # No numbers in summary

            precision = len(summary_set.intersection(
                original_set)) / len(summary_set)
            recall = len(summary_set.intersection(
                original_set)) / len(original_set)

            # F1 score
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                return f1
else:
                return 0.0

except Exception as e:
            logger.error(f"Numerical consistency check failed: {str(e)}")
            return 0.5

    def _check_temporal_consistency(
            self, original: str, summary: str) -> float:
        """Check temporal consistency between original and summary"""
        try:
            # Extract temporal expressions
            temporal_patterns = [
                r"\b(?:yesterday|today|tomorrow)\b",
                r"\b(?:last|next)\s+(?:week|month|year)\b",
                r"\b(?:in|on|at)\s+\d{4}\b",
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
            ]

            original_temporal = []
            summary_temporal = []

            for pattern in temporal_patterns:
                original_temporal.extend(re.findall(
                    pattern, original, re.IGNORECASE))
                summary_temporal.extend(
                    re.findall(
                        pattern,
                        summary,
                        re.IGNORECASE))

            if not original_temporal:
                return 1.0  # No temporal expressions to check

            # Check consistency
            original_set = set(t.lower() for t in original_temporal)
            summary_set = set(t.lower() for t in summary_temporal)

            if not summary_set:
                return 0.5  # No temporal expressions in summary

            intersection = len(original_set.intersection(summary_set))
            union = len(original_set.union(summary_set))

            return intersection / union if union > 0 else 0.5

except Exception as e:
            logger.error(f"Temporal consistency check failed: {str(e)}")
            return 0.5

    async def _check_semantic_consistency(
            self, original: str, summary: str) -> float:
        """Check semantic consistency using sentence embeddings"""
        try:
            # Get embeddings
            original_embedding = self.sentence_model.encode([original])
            summary_embedding = self.sentence_model.encode([summary])

            # Calculate cosine similarity
            similarity = cosine_similarity(
                original_embedding, summary_embedding)[0][0]

            return float(similarity)

except Exception as e:
            logger.error(f"Semantic consistency check failed: {str(e)}")
            return 0.5

    async def _calculate_readability(self, text: str) -> float:
        """Calculate readability score"""
        try:
            # Flesch Reading Ease
            flesch_score = flesch_reading_ease(text)

            # Flesch-Kincaid Grade Level
            fk_grade = flesch_kincaid_grade(text)

            # Normalize scores to 0-1 scale
            # Flesch score: 0-100, higher is better
            flesch_normalized = max(0, min(1, flesch_score / 100))

            # FK grade: 0-20+, lower is better
            fk_normalized = max(0, min(1, 1 - (fk_grade / 20)))

            # Combine scores
            readability_score = (flesch_normalized + fk_normalized) / 2

            return readability_score

except Exception as e:
            logger.error(f"Readability calculation failed: {str(e)}")
            return 0.5

    async def _calculate_coverage(self, original: str, summary: str) -> float:
        """Calculate information coverage of summary"""
        try:
            # Split into sentences
            original_sentences = sent_tokenize(original)
            summary_sentences = sent_tokenize(summary)

            if not original_sentences:
                return 0.0

            # Calculate TF-IDF similarity for each summary sentence
            similarities = []

            for summary_sentence in summary_sentences:
                max_similarity = 0.0

                for original_sentence in original_sentences:
                    # Calculate cosine similarity
                    try:
                        summary_embedding = self.sentence_model.encode(
                            [summary_sentence])
                        original_embedding = self.sentence_model.encode(
                            [original_sentence])
                        similarity = cosine_similarity(
                            summary_embedding, original_embedding)[0][0]
                        max_similarity = max(max_similarity, similarity)
                    except BaseException:
                        continue

                similarities.append(max_similarity)

            # Coverage is average similarity
            coverage = np.mean(similarities) if similarities else 0.0

            return float(coverage)

except Exception as e:
            logger.error(f"Coverage calculation failed: {str(e)}")
            return 0.5

    async def _calculate_abstractiveness(
            self, original: str, summary: str) -> float:
        """Calculate abstractiveness (how much new phrasing vs extraction)"""
        try:
            # Tokenize texts
            original_tokens = word_tokenize(original.lower())
            summary_tokens = word_tokenize(summary.lower())

            if not original_tokens or not summary_tokens:
                return 0.0

            # Remove stopwords
            stop_words = set(stopwords.words("english"))
            original_tokens = [
                t for t in original_tokens if t not in stop_words]
            summary_tokens = [t for t in summary_tokens if t not in stop_words]

            # Calculate n-gram overlap
            def get_ngrams(tokens, n):
                return set(tuple(tokens[i: i + n])
                           for i in range(len(tokens) - n + 1))

            # 1-gram overlap
            original_1grams = get_ngrams(original_tokens, 1)
            summary_1grams = get_ngrams(summary_tokens, 1)
            overlap_1gram = len(original_1grams.intersection(summary_1grams))
            total_1gram = len(original_1grams.union(summary_1grams))

            # 2-gram overlap
            original_2grams = get_ngrams(original_tokens, 2)
            summary_2grams = get_ngrams(summary_tokens, 2)
            overlap_2gram = len(original_2grams.intersection(summary_2grams))
            total_2gram = len(original_2grams.union(summary_2grams))

            # Calculate abstractiveness (1 - overlap ratio)
            if total_1gram > 0 and total_2gram > 0:
                overlap_ratio = (overlap_1gram / total_1gram +
                                 overlap_2gram / total_2gram) / 2
                abstractiveness = 1 - overlap_ratio
else:
                abstractiveness = 0.5

            return max(0, min(1, abstractiveness))

except Exception as e:
            logger.error(f"Abstractiveness calculation failed: {str(e)}")
            return 0.5

    def _calculate_overall_score(
        self,
        rouge_scores: Dict[str, float],
        bert_scores: Dict[str, float],
        factual_consistency: float,
        readability: float,
        coverage: float,
        abstractiveness: float,
    ) -> float:
        """Calculate overall quality score"""
        try:
            # Weighted combination of metrics
            weights = {
                "rouge1_f1": 0.15,
                "rouge2_f1": 0.10,
                "rougeL_f1": 0.10,
                "bertscore_f1": 0.20,
                "factual_consistency": 0.20,
                "readability": 0.10,
                "coverage": 0.10,
                "abstractiveness": 0.05,
            }

            score = (
                rouge_scores["rouge1_f1"] * weights["rouge1_f1"]
                + rouge_scores["rouge2_f1"] * weights["rouge2_f1"]
                + rouge_scores["rougeL_f1"] * weights["rougeL_f1"]
                + bert_scores["bertscore_f1"] * weights["bertscore_f1"]
                + factual_consistency * weights["factual_consistency"]
                + readability * weights["readability"]
                + coverage * weights["coverage"]
                + abstractiveness * weights["abstractiveness"]
            )

            return round(score, 3)

except Exception as e:
            logger.error(f"Overall score calculation failed: {str(e)}")
            return 0.5

    async def validate_summary_quality_async(
            self, original: str, summary: str) -> QualityMetrics:
        """Async wrapper for quality validation"""
        return await self.validate_summary_quality(original, summary)

    async def get_status(self) -> Dict[str, Any]:
    """Get validator status"""
        return {
            "initialized": self._initialized,
            "rouge_available": self.rouge_scorer is not None,
            "bert_score_available": True,
            "spacy_available": self.nlp is not None,
            "sentence_transformer_available": self.sentence_model is not None,
        }
