"""
Factual Consistency Checking Module
Advanced consistency validation between source and summary
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings

from .models import ConsistencyScores, ProcessedContent

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
except:
    pass


class FactualConsistencyChecker:
    """Advanced factual consistency checking between source and summary"""

    def __init__(self):
        """Initialize the consistency checker"""
        self.nlp = None
        self.sentence_model = None
        self._initialized = False

    async def initialize(self):
        """Initialize consistency checking tools"""
        try:
            logger.info("Initializing factual consistency checker...")

            # Initialize spaCy model
            try:
                self.nlp = spacy.load(settings.SPACY_MODEL)
            except OSError:
                logger.warning(f"spaCy model {settings.SPACY_MODEL} not found")
                self.nlp = None

            # Initialize sentence transformer
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

            self._initialized = True
            logger.info("Factual consistency checker initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize consistency checker: {str(e)}")
            raise

    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.sentence_model:
                del self.sentence_model
            if self.nlp:
                del self.nlp
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def check_consistency(self, source_text: str, summary_text: str) -> ConsistencyScores:
        """Check factual consistency between source and summary"""

        if not self._initialized:
            raise RuntimeError("Consistency checker not initialized")

        try:
            # Check different types of consistency
            entity_consistency = await self._check_entity_consistency(source_text, summary_text)
            numerical_consistency = await self._check_numerical_consistency(
                source_text, summary_text
            )
            temporal_consistency = await self._check_temporal_consistency(source_text, summary_text)
            entailment_score = await self._check_entailment(source_text, summary_text)

            # Calculate overall consistency
            overall_consistency = (
                entity_consistency * 0.3
                + numerical_consistency * 0.2
                + temporal_consistency * 0.2
                + entailment_score * 0.3
            )

            return ConsistencyScores(
                entity_consistency=entity_consistency,
                numerical_consistency=numerical_consistency,
                temporal_consistency=temporal_consistency,
                entailment_score=entailment_score,
                overall_consistency=overall_consistency,
            )

        except Exception as e:
            logger.error(f"Consistency checking failed: {str(e)}")
            return ConsistencyScores(
                entity_consistency=0.0,
                numerical_consistency=0.0,
                temporal_consistency=0.0,
                entailment_score=0.0,
                overall_consistency=0.0,
            )

    async def _check_entity_consistency(self, source: str, summary: str) -> float:
        """Check consistency of named entities"""
        try:
            if not self.nlp:
                return 0.5  # Fallback score

            # Extract entities from both texts
            source_doc = self.nlp(source)
            summary_doc = self.nlp(summary)

            source_entities = {
                "PERSON": [ent.text for ent in source_doc.ents if ent.label_ == "PERSON"],
                "ORG": [ent.text for ent in source_doc.ents if ent.label_ == "ORG"],
                "GPE": [ent.text for ent in source_doc.ents if ent.label_ == "GPE"],
                "DATE": [ent.text for ent in source_doc.ents if ent.label_ == "DATE"],
                "MONEY": [ent.text for ent in source_doc.ents if ent.label_ == "MONEY"],
                "PERCENT": [ent.text for ent in source_doc.ents if ent.label_ == "PERCENT"],
            }

            summary_entities = {
                "PERSON": [ent.text for ent in summary_doc.ents if ent.label_ == "PERSON"],
                "ORG": [ent.text for ent in summary_doc.ents if ent.label_ == "ORG"],
                "GPE": [ent.text for ent in summary_doc.ents if ent.label_ == "GPE"],
                "DATE": [ent.text for ent in summary_doc.ents if ent.label_ == "DATE"],
                "MONEY": [ent.text for ent in summary_doc.ents if ent.label_ == "MONEY"],
                "PERCENT": [ent.text for ent in summary_doc.ents if ent.label_ == "PERCENT"],
            }

            # Calculate consistency for each entity type
            total_consistency = 0.0
            entity_types = 0

            for entity_type in source_entities:
                if not source_entities[entity_type]:
                    continue

                entity_types += 1
                source_set = set(source_entities[entity_type])
                summary_set = set(summary_entities[entity_type])

                # Calculate Jaccard similarity
                intersection = len(source_set.intersection(summary_set))
                union = len(source_set.union(summary_set))

                if union > 0:
                    consistency = intersection / union
                    total_consistency += consistency

            return total_consistency / entity_types if entity_types > 0 else 0.5

        except Exception as e:
            logger.error(f"Entity consistency check failed: {str(e)}")
            return 0.5

    async def _check_numerical_consistency(self, source: str, summary: str) -> float:
        """Check consistency of numbers between source and summary"""
        try:
            # Extract numbers from both texts
            source_numbers = re.findall(r"\d+(?:\.\d+)?", source)
            summary_numbers = re.findall(r"\d+(?:\.\d+)?", summary)

            if not source_numbers:
                return 1.0  # No numbers to check

            # Convert to sets for comparison
            source_set = set(source_numbers)
            summary_set = set(summary_numbers)

            if not summary_set:
                return 0.5  # No numbers in summary

            # Calculate precision and recall
            intersection = source_set.intersection(summary_set)
            precision = len(intersection) / len(summary_set)
            recall = len(intersection) / len(source_set)

            # F1 score
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                return f1
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Numerical consistency check failed: {str(e)}")
            return 0.5

    async def _check_temporal_consistency(self, source: str, summary: str) -> float:
        """Check temporal consistency between source and summary"""
        try:
            # Extract temporal expressions
            temporal_patterns = [
                r"\b(?:yesterday|today|tomorrow)\b",
                r"\b(?:last|next)\s+(?:week|month|year)\b",
                r"\b(?:in|on|at)\s+\d{4}\b",
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
                r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
            ]

            source_temporal = []
            summary_temporal = []

            for pattern in temporal_patterns:
                source_temporal.extend(re.findall(pattern, source, re.IGNORECASE))
                summary_temporal.extend(re.findall(pattern, summary, re.IGNORECASE))

            if not source_temporal:
                return 1.0  # No temporal expressions to check

            # Normalize temporal expressions
            source_set = set(t.lower().strip() for t in source_temporal)
            summary_set = set(t.lower().strip() for t in summary_temporal)

            if not summary_set:
                return 0.5  # No temporal expressions in summary

            # Calculate consistency
            intersection = source_set.intersection(summary_set)
            union = source_set.union(summary_set)

            return len(intersection) / len(union) if union else 0.5

        except Exception as e:
            logger.error(f"Temporal consistency check failed: {str(e)}")
            return 0.5

    async def _check_entailment(self, source: str, summary: str) -> float:
        """Check natural language entailment between source and summary"""
        try:
            # Split into sentences
            source_sentences = sent_tokenize(source)
            summary_sentences = sent_tokenize(summary)

            if not source_sentences or not summary_sentences:
                return 0.5

            # Calculate semantic similarity between sentences
            source_embeddings = self.sentence_model.encode(source_sentences)
            summary_embeddings = self.sentence_model.encode(summary_sentences)

            # Calculate pairwise similarities
            similarities = cosine_similarity(summary_embeddings, source_embeddings)

            # For each summary sentence, find the most similar source sentence
            max_similarities = np.max(similarities, axis=1)

            # Calculate average entailment score
            entailment_score = np.mean(max_similarities)

            return float(entailment_score)

        except Exception as e:
            logger.error(f"Entailment check failed: {str(e)}")
            return 0.5

    async def get_status(self) -> Dict[str, Any]:
        """Get consistency checker status"""
        return {
            "initialized": self._initialized,
            "spacy_available": self.nlp is not None,
            "sentence_transformer_available": self.sentence_model is not None,
        }
