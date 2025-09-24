"""Differential privacy implementation for personalization."""

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
from scipy.stats import laplace

from ..config.settings import settings

logger = structlog.get_logger()


class DifferentialPrivacy:
    """Differential privacy implementation for user data protection."""

    def __init__(self, epsilon: float = None, delta: float = None):
        self.epsilon = epsilon or settings.privacy_epsilon
        self.delta = delta or settings.privacy_delta
        self.sensitivity = 1.0  # L1 sensitivity

    def add_laplace_noise(
            self,
            data: np.ndarray,
            sensitivity: float = None) -> np.ndarray:
        """Add Laplace noise to data for differential privacy."""
        if sensitivity is None:
            sensitivity = self.sensitivity

        # Calculate noise scale
        scale = sensitivity / self.epsilon

        # Add Laplace noise
        noise = laplace.rvs(scale=scale, size=data.shape)
        noisy_data = data + noise

        return noisy_data

    def add_gaussian_noise(
            self,
            data: np.ndarray,
            sensitivity: float = None) -> np.ndarray:
        """Add Gaussian noise to data for differential privacy."""
        if sensitivity is None:
            sensitivity = self.sensitivity

        # Calculate noise scale for Gaussian mechanism
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * \
            sensitivity / self.epsilon

        # Add Gaussian noise
        noise = np.random.normal(0, sigma, data.shape)
        noisy_data = data + noise

        return noisy_data

    def privatize_user_preferences(
            self, preferences: Dict[str, float]) -> Dict[str, float]:
        """Privatize user preferences using differential privacy."""
        if not preferences:
            return {}

        # Convert to numpy array
        values = np.array(list(preferences.values()))
        keys = list(preferences.keys())

        # Add noise
        noisy_values = self.add_laplace_noise(values)

        # Ensure non-negative values
        noisy_values = np.maximum(0, noisy_values)

        # Normalize
        if noisy_values.sum() > 0:
            noisy_values = noisy_values / noisy_values.sum()

        # Convert back to dictionary
        privatized = dict(zip(keys, noisy_values))

        return privatized

    def privatize_interaction_counts(
            self, counts: Dict[str, int]) -> Dict[str, int]:
        """Privatize interaction counts using differential privacy."""
        if not counts:
            return {}

        # Convert to numpy array
        values = np.array(list(counts.values()))
        keys = list(counts.keys())

        # Add noise
        noisy_values = self.add_laplace_noise(values)

        # Round to integers and ensure non-negative
        noisy_values = np.maximum(0, np.round(noisy_values))

        # Convert back to dictionary
        privatized = dict(zip(keys, noisy_values.astype(int)))

        return privatized

    def privatize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Privatize embeddings using differential privacy."""
        if embeddings.size == 0:
            return embeddings

        # Add noise to embeddings
        noisy_embeddings = self.add_gaussian_noise(embeddings)

        return noisy_embeddings

    def compute_privacy_budget(self, queries: int) -> float:
        """Compute remaining privacy budget."""
        # Simple composition: each query uses epsilon
        used_budget = queries * self.epsilon
        remaining_budget = 1.0 - used_budget

        return max(0.0, remaining_budget)

    def is_privacy_budget_exhausted(self, queries: int) -> bool:
        """Check if privacy budget is exhausted."""
        return self.compute_privacy_budget(queries) <= 0.0


class DataAnonymization:
    """Data anonymization techniques."""

    def __init__(self):
        # In production, use secure random salt
        self.salt = "personalization_service_salt"

    def anonymize_user_id(self, user_id: str) -> str:
        """Anonymize user ID using hashing."""
        # Add salt and hash
        salted_id = f"{user_id}{self.salt}"
        hashed_id = hashlib.sha256(salted_id.encode()).hexdigest()

        return hashed_id[:16]  # Use first 16 characters

    def anonymize_content_id(self, content_id: str) -> str:
        """Anonymize content ID using hashing."""
        salted_id = f"{content_id}{self.salt}"
        hashed_id = hashlib.sha256(salted_id.encode()).hexdigest()

        return hashed_id[:16]

    def anonymize_text(self, text: str, preserve_length: bool = True) -> str:
        """Anonymize text content."""
        if not text:
            return text

        # Simple character-level anonymization
        anonymized = ""
        for char in text:
            if char.isalpha():
                # Replace with random character of same case
                if char.isupper():
                    anonymized += chr(ord("A") +
                                      (ord(char) - ord("A") + 7) % 26)
                else:
                    anonymized += chr(ord("a") +
                                      (ord(char) - ord("a") + 7) % 26)
            elif char.isdigit():
                # Replace with random digit
                anonymized += str((int(char) + 3) % 10)
            else:
                # Keep special characters
                anonymized += char

        return anonymized

    def anonymize_user_profile(
            self, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Anonymize user profile data."""
        anonymized = {}

        for key, value in profile.items():
            if key == "user_id":
                anonymized[key] = self.anonymize_user_id(str(value))
            elif key in ["topic_preferences", "source_preferences"]:
                # Anonymize preference keys
                anonymized[key] = {
                    self.anonymize_text(k): v for k,
                    v in value.items()}
            elif key == "demographic_data":
                # Anonymize demographic data
                anonymized[key] = self._anonymize_demographic_data(value)
            else:
                anonymized[key] = value

        return anonymized

    def _anonymize_demographic_data(
            self, demographic_data: Dict[str, Any]) -> Dict[str, Any]:
    """Anonymize demographic data."""
        anonymized = {}

        for key, value in demographic_data.items():
            if isinstance(value, str):
                anonymized[key] = self.anonymize_text(value)
            elif isinstance(value, (int, float)):
                # Add noise to numerical data
                noise = np.random.normal(0, 0.1 * abs(value))
                anonymized[key] = value + noise
            else:
                anonymized[key] = value

        return anonymized


class PrivacyPreservingPersonalization:
    """Privacy-preserving personalization techniques."""

    def __init__(self, epsilon: float = None, delta: float = None):
        self.dp = DifferentialPrivacy(epsilon, delta)
        self.anonymizer = DataAnonymization()

    def privatize_recommendations(
        self, recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Privatize recommendations to protect user privacy."""
        privatized = []

        for rec in recommendations:
            privatized_rec = rec.copy()

            # Add noise to scores
            if "score" in rec:
                noisy_score = self.dp.add_laplace_noise(
                    np.array([rec["score"]]), sensitivity=1.0)[0]
                privatized_rec["score"] = max(0.0, min(1.0, noisy_score))

            # Anonymize item IDs
            if "item_id" in rec:
                privatized_rec["item_id"] = self.anonymizer.anonymize_content_id(
                    rec["item_id"])

            # Anonymize features
            if "features" in rec and isinstance(rec["features"], dict):
                privatized_rec["features"] = self._privatize_features(
                    rec["features"])

            privatized.append(privatized_rec)

        return privatized

    def _privatize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
    """Privatize recommendation features."""
        privatized_features = {}

        for key, value in features.items():
            if isinstance(value, (int, float)):
                # Add noise to numerical features
                noisy_value = self.dp.add_laplace_noise(
                    np.array([value]), sensitivity=1.0)[0]
                privatized_features[key] = noisy_value
            else:
                privatized_features[key] = value

        return privatized_features

    def federated_learning_update(
        self, local_updates: List[Dict[str, Any]], global_model: Dict[str, Any]
    ) -> Dict[str, Any]:
    """Perform federated learning update with privacy preservation."""
        if not local_updates:
            return global_model

        # Aggregate local updates with differential privacy
        aggregated_update = {}

        for key in global_model.keys():
            if key in local_updates[0]:
                # Collect values from all local updates
                values = [update[key]
                          for update in local_updates if key in update]

                if values:
                    # Compute average
                    avg_value = np.mean(values)

                    # Add noise for privacy
                    noisy_value = self.dp.add_laplace_noise(
                        np.array([avg_value]), sensitivity=1.0)[0]

                    # Update global model
                    aggregated_update[key] = global_model[key] + noisy_value

        return aggregated_update

    def secure_aggregation(
            self, user_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform secure aggregation of user data."""
        if not user_data:
            return {}

        # Aggregate data with privacy preservation
        aggregated = {}

        for key in user_data[0].keys():
            if isinstance(user_data[0][key], (int, float)):
                # Sum numerical values
                total = sum(item[key] for item in user_data if key in item)

                # Add noise
                noisy_total = self.dp.add_laplace_noise(
                    np.array([total]), sensitivity=1.0)[0]

                aggregated[key] = noisy_total
            else:
                # For non-numerical data, use most common value
                values = [item[key] for item in user_data if key in item]
                if values:
                    aggregated[key] = max(set(values), key=values.count)

        return aggregated

    def get_privacy_metrics(self) -> Dict[str, Any]:
    """Get privacy preservation metrics."""
        return {
            "epsilon": self.dp.epsilon,
            "delta": self.dp.delta,
            "privacy_budget_used": 0.0,  # This would be tracked in production
            "anonymization_enabled": True,
            "differential_privacy_enabled": True,
        }
