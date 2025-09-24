"""Hierarchical topic classification with confidence scoring."""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from ..config import settings
from ..models.content import ExtractedContent, Topic, TopicCategory

logger = structlog.get_logger(__name__)


class TopicClassifier:
    """Hierarchical topic classification with confidence scoring."""

    def __init__(self):
        """Initialize the topic classifier."""
        self.taxonomy = self._load_topic_taxonomy()
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.model_loaded = False

        # Load or train model
        asyncio.create_task(self._initialize_model())

    def _load_topic_taxonomy(self) -> Dict[str, Any]:
        """Load hierarchical topic taxonomy."""
        return {
            "technology": {
                "artificial_intelligence": {
                    "machine_learning": [
                        "deep_learning",
                        "neural_networks",
                        "nlp",
                        "computer_vision",
                    ],
                    "robotics": ["autonomous_vehicles", "industrial_robots", "service_robots"],
                    "data_science": ["big_data", "analytics", "visualization"],
                },
                "software": {
                    "programming": ["python", "javascript", "java", "go", "rust"],
                    "web_development": ["frontend", "backend", "fullstack"],
                    "mobile_development": ["ios", "android", "react_native", "flutter"],
                },
                "hardware": {
                    "computers": ["laptops", "desktops", "servers"],
                    "mobile_devices": ["smartphones", "tablets", "wearables"],
                    "gaming": ["consoles", "pc_gaming", "vr_ar"],
                },
                "cybersecurity": {
                    "network_security": ["firewalls", "vpn", "intrusion_detection"],
                    "data_protection": ["encryption", "privacy", "gdpr"],
                    "threat_intelligence": ["malware", "phishing", "social_engineering"],
                },
            },
            "business": {
                "finance": {
                    "banking": ["retail_banking", "investment_banking", "fintech"],
                    "investment": ["stocks", "bonds", "cryptocurrency", "real_estate"],
                    "accounting": ["bookkeeping", "taxation", "auditing"],
                },
                "marketing": {
                    "digital_marketing": ["seo", "sem", "social_media", "content_marketing"],
                    "advertising": ["online_ads", "tv_ads", "print_ads", "outdoor_ads"],
                    "branding": ["brand_strategy", "logo_design", "brand_identity"],
                },
                "management": {
                    "leadership": ["team_management", "project_management", "strategic_planning"],
                    "operations": ["supply_chain", "logistics", "quality_control"],
                    "hr": ["recruitment", "training", "performance_management"],
                },
            },
            "politics": {
                "domestic_politics": {
                    "elections": ["presidential", "congressional", "local"],
                    "policy": ["healthcare", "education", "infrastructure", "environment"],
                    "government": ["executive", "legislative", "judicial"],
                },
                "international_politics": {
                    "diplomacy": ["treaties", "alliances", "trade_agreements"],
                    "conflicts": ["wars", "sanctions", "peace_negotiations"],
                    "organizations": ["un", "nato", "eu", "g7", "g20"],
                },
            },
            "health": {
                "medicine": {
                    "diseases": ["covid19", "cancer", "diabetes", "heart_disease"],
                    "treatments": ["drugs", "surgery", "therapy", "vaccines"],
                    "research": ["clinical_trials", "medical_research", "drug_development"],
                },
                "wellness": {
                    "fitness": ["exercise", "gym", "yoga", "running"],
                    "nutrition": ["diet", "supplements", "healthy_eating"],
                    "mental_health": ["depression", "anxiety", "therapy", "meditation"],
                },
            },
            "science": {
                "physics": {
                    "quantum_physics": ["quantum_mechanics", "quantum_computing"],
                    "astrophysics": ["cosmology", "space_exploration", "exoplanets"],
                    "materials": ["nanotechnology", "superconductors", "graphene"],
                },
                "biology": {
                    "genetics": ["dna", "genome", "gene_therapy", "crispr"],
                    "ecology": ["climate_change", "biodiversity", "conservation"],
                    "evolution": ["natural_selection", "speciation", "fossils"],
                },
                "chemistry": {
                    "organic_chemistry": ["drugs", "polymers", "biochemistry"],
                    "inorganic_chemistry": ["metals", "catalysts", "materials"],
                    "physical_chemistry": ["thermodynamics", "kinetics", "spectroscopy"],
                },
            },
        }

    async def _initialize_model(self) -> Dict[str, Any]:
        """Initialize or load the topic classification model."""
        try:
            model_path = os.path.join(
                settings.model_cache_dir,
                "topic_classifier.joblib")

            if os.path.exists(model_path):
                # Load existing model
                model_data = joblib.load(model_path)
                self.model = model_data["model"]
                self.vectorizer = model_data["vectorizer"]
                self.label_encoder = model_data["label_encoder"]
                self.model_loaded = True
                logger.info("Topic classifier model loaded from cache")
            else:
                # Train new model
                await self._train_model()

        except Exception as e:
            logger.error("Failed to initialize topic classifier", error=str(e))
            # Fallback to simple keyword-based classification
            self.model_loaded = False

    async def _train_model(self) -> Dict[str, Any]:
        """Train the topic classification model."""
        try:
            # This is a simplified training process
            # In practice, you'd use a large dataset of labeled content

            # Create training data from taxonomy
            training_data = self._create_training_data()

            if not training_data:
                logger.warning(
                    "No training data available, using keyword-based classification")
                self.model_loaded = False
                return

            # Prepare features and labels
            texts = [item["text"] for item in training_data]
            labels = [item["labels"] for item in training_data]

            # Create vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=10000, ngram_range=(1, 2), stop_words="english"
            )

            # Create multi-label classifier
            self.model = OneVsRestClassifier(
                LogisticRegression(
                    random_state=42, max_iter=1000))

            # Create pipeline
            pipeline = Pipeline(
                [("vectorizer", self.vectorizer), ("classifier", self.model)])

            # Train model
            pipeline.fit(texts, labels)

            # Save model
            os.makedirs(settings.model_cache_dir, exist_ok=True)
            model_path = os.path.join(
                settings.model_cache_dir,
                "topic_classifier.joblib")

            joblib.dump(
                {
                    "model": pipeline,
                    "vectorizer": self.vectorizer,
                    "label_encoder": self.label_encoder,
                },
                model_path,
            )

            self.model_loaded = True
            logger.info("Topic classifier model trained and saved")

        except Exception as e:
            logger.error("Failed to train topic classifier", error=str(e))
            self.model_loaded = False

    def _create_training_data(self) -> List[Dict[str, Any]]:
        """Create training data from taxonomy."""
        # This is a simplified implementation
        # In practice, you'd use a large dataset of real content

        training_data = []

        # Create sample texts for each topic
        sample_texts = {
            "artificial_intelligence": [
                "Machine learning algorithms are revolutionizing data analysis",
                "Deep learning neural networks can recognize complex patterns",
                "Natural language processing enables computers to understand human language",
            ],
            "programming": [
                "Python is a versatile programming language for data science",
                "JavaScript powers modern web applications and mobile apps",
                "Go is known for its simplicity and performance in backend systems",
            ],
            "finance": [
                "Investment banking involves helping companies raise capital",
                "Cryptocurrency markets are highly volatile and speculative",
                "Financial planning helps individuals achieve their long-term goals",
            ],
            "politics": [
                "Election campaigns require significant funding and organization",
                "Government policy affects economic growth and social welfare",
                "International diplomacy requires careful negotiation and compromise",
            ],
            "health": [
                "Medical research is advancing treatments for chronic diseases",
                "Mental health awareness is increasing in modern society",
                "Preventive healthcare can reduce long-term medical costs",
            ],
        }

        for topic, texts in sample_texts.items():
            for text in texts:
                # Create multi-label encoding
                labels = self._get_topic_labels(topic)
                training_data.append({"text": text, "labels": labels})

        return training_data

    def _get_topic_labels(self, topic: str) -> List[int]:
        """Get binary labels for a topic."""
        # This is a simplified implementation
        # In practice, you'd have a proper label encoding system

        all_topics = self._flatten_taxonomy()
        topic_index = all_topics.index(topic) if topic in all_topics else 0

        # Create binary vector
        labels = [0] * len(all_topics)
        labels[topic_index] = 1

        return labels

    def _flatten_taxonomy(self) -> List[str]:
        """Flatten the hierarchical taxonomy into a list."""
        topics = []

        def extract_topics(taxonomy_dict, prefix=""):
            for key, value in taxonomy_dict.items():
                topic_name = f"{prefix}_{key}" if prefix else key
                topics.append(topic_name)

                if isinstance(value, dict):
                    extract_topics(value, topic_name)
                elif isinstance(value, list):
                    for sub_topic in value:
                        topics.append(f"{topic_name}_{sub_topic}")

        extract_topics(self.taxonomy)
        return topics

    async def classify_topics(
            self,
            content: ExtractedContent,
            language: str = "en") -> List[Topic]:
        """Classify topics for content."""
        try:
            # Prepare text for classification
            text = f"{content.title} {content.summary or ''} {content.content[:1000]}"

            if self.model_loaded:
                # Use trained model
                predictions = await self._predict_with_model(text)
            else:
                # Use keyword-based classification
                predictions = await self._predict_with_keywords(text)

            # Convert predictions to Topic objects
            topics = []
            for prediction in predictions:
                if prediction["confidence"] > settings.topic_confidence_threshold:
                    topic = Topic(
                        id=prediction["id"],
                        name=prediction["name"],
                        category=prediction["category"],
                        confidence=prediction["confidence"],
                        hierarchy_path=prediction["hierarchy_path"],
                        keywords=prediction.get("keywords", []),
                    )
                    topics.append(topic)

            # Sort by confidence and limit results
            topics.sort(key=lambda x: x.confidence, reverse=True)
            topics = topics[: settings.max_topics_per_document]

            logger.info(
                "Topic classification completed",
                content_id=content.id,
                topics_found=len(topics),
                language=language,
            )

            return topics

        except Exception as e:
            logger.error(
                "Topic classification failed",
                content_id=content.id,
                error=str(e))
            return []

    async def _predict_with_model(self, text: str) -> List[Dict[str, Any]]:
        """Predict topics using trained model."""
        try:
            # Get predictions
            predictions = self.model.predict_proba([text])[0]

            # Get topic names
            all_topics = self._flatten_taxonomy()

            results = []
            for i, confidence in enumerate(predictions):
                if confidence > 0.1:  # Low threshold for initial filtering
                    topic_name = all_topics[i]
                    topic_info = self._get_topic_info(topic_name)

                    results.append(
                        {
                            "id": topic_name,
                            "name": topic_info["name"],
                            "category": topic_info["category"],
                            "confidence": float(confidence),
                            "hierarchy_path": topic_info["hierarchy_path"],
                            "keywords": topic_info.get("keywords", []),
                        }
                    )

            return results

        except Exception as e:
            logger.error("Model prediction failed", error=str(e))
            return []

    async def _predict_with_keywords(self, text: str) -> List[Dict[str, Any]]:
        """Predict topics using keyword matching."""
        try:
            text_lower = text.lower()
            results = []

            # Define keyword patterns for each topic
            topic_keywords = {
                "technology": {
                    "keywords": [
                        "technology",
                        "tech",
                        "software",
                        "hardware",
                        "computer",
                        "digital",
                        "ai",
                        "artificial intelligence",
                    ],
                    "weight": 1.0,
                },
                "business": {
                    "keywords": [
                        "business",
                        "company",
                        "corporate",
                        "finance",
                        "marketing",
                        "management",
                        "strategy",
                    ],
                    "weight": 1.0,
                },
                "politics": {
                    "keywords": [
                        "politics",
                        "political",
                        "government",
                        "election",
                        "policy",
                        "democracy",
                        "republican",
                        "democrat",
                    ],
                    "weight": 1.0,
                },
                "health": {
                    "keywords": [
                        "health",
                        "medical",
                        "medicine",
                        "healthcare",
                        "doctor",
                        "hospital",
                        "disease",
                        "treatment",
                    ],
                    "weight": 1.0,
                },
                "science": {
                    "keywords": [
                        "science",
                        "scientific",
                        "research",
                        "study",
                        "experiment",
                        "discovery",
                        "physics",
                        "chemistry",
                        "biology",
                    ],
                    "weight": 1.0,
                },
            }

            for topic, info in topic_keywords.items():
                keyword_matches = sum(
                    1 for keyword in info["keywords"] if keyword in text_lower)
                if keyword_matches > 0:
                    confidence = min(keyword_matches /
                                     len(info["keywords"]) *
                                     info["weight"], 1.0)

                    results.append(
                        {
                            "id": topic,
                            "name": topic.title(),
                            "category": TopicCategory(topic),
                            "confidence": confidence,
                            "hierarchy_path": [topic],
                            "keywords": info["keywords"],
                        }
                    )

            return results

        except Exception as e:
            logger.error("Keyword prediction failed", error=str(e))
            return []

    def _get_topic_info(self, topic_name: str) -> Dict[str, Any]:
        """Get detailed information about a topic."""
        # Parse hierarchy from topic name
        hierarchy_parts = topic_name.split("_")

        # Find category
        category = TopicCategory.OTHER
        if hierarchy_parts[0] in [cat.value for cat in TopicCategory]:
            category = TopicCategory(hierarchy_parts[0])

        return {
            "name": topic_name.replace("_", " ").title(),
            "category": category,
            "hierarchy_path": hierarchy_parts,
            "keywords": [],
        }

    async def get_topic_statistics(self) -> Dict[str, Any]:
        """Get topic classification statistics."""
        return {
            "model_loaded": self.model_loaded,
            "total_topics": len(self._flatten_taxonomy()),
            "categories": len(TopicCategory),
            "taxonomy_depth": self._get_taxonomy_depth(),
        }

    def _get_taxonomy_depth(self) -> int:
        """Get the maximum depth of the taxonomy."""

        def get_depth(taxonomy_dict, current_depth=0):
            max_depth = current_depth
            for value in taxonomy_dict.values():
                if isinstance(value, dict):
                    depth = get_depth(value, current_depth + 1)
                    max_depth = max(max_depth, depth)
            return max_depth

        return get_depth(self.taxonomy)
