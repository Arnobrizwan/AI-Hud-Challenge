"""
Graph-Based Abuse Detector
Analyze user connections and network patterns for abuse detection
"""

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from safety_engine.config import get_abuse_config
from safety_engine.models import GraphSignals
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class GraphBasedAbuseDetector:
    """Detect abuse patterns using graph analysis"""

    def __init__(self):
        self.config = get_abuse_config()
        self.is_initialized = False

        # Graph storage
        self.user_graph = nx.Graph()
        self.user_attributes = {}
        self.edge_attributes = {}

        # Clustering model
        self.clustering_model = DBSCAN(eps=0.5, min_samples=3)
        self.scaler = StandardScaler()

        # Abuse patterns
        self.suspicious_patterns = set()
        self.known_abusers = set()

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the graph analyzer"""
        try:
            self.is_initialized = True
            logger.info("Graph-based abuse detector initialized")

        except Exception as e:
            logger.error(f"Failed to initialize graph analyzer: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup resources"""
        try:
            self.user_graph.clear()
            self.user_attributes.clear()
            self.edge_attributes.clear()
            self.suspicious_patterns.clear()
            self.known_abusers.clear()

            self.is_initialized = False
            logger.info("Graph analyzer cleanup completed")

        except Exception as e:
            logger.error(f"Error during graph analyzer cleanup: {str(e)}")

    async def analyze_user_graph(
        self, user_id: str, connection_data: Optional[Dict[str, Any]]
    ) -> GraphSignals:
        """Analyze user's graph position for abuse patterns"""

        if not self.is_initialized:
            raise RuntimeError("Graph analyzer not initialized")

        try:
            # Update graph with new connection data
            if connection_data:
                await self.update_user_connections(user_id, connection_data)

            # Calculate graph-based signals
            suspicious_connections = await self.count_suspicious_connections(user_id)
            cluster_anomaly = await self.detect_cluster_anomaly(user_id)
            centrality_anomaly = await self.detect_centrality_anomaly(user_id)

            # Calculate overall abuse probability
            abuse_probability = self.calculate_abuse_probability(
                suspicious_connections, cluster_anomaly, centrality_anomaly
            )

            return GraphSignals(
                abuse_probability=abuse_probability,
                suspicious_connections=suspicious_connections,
                cluster_anomaly=cluster_anomaly,
                centrality_anomaly=centrality_anomaly,
            )

        except Exception as e:
            logger.error(f"Graph analysis failed: {str(e)}")
            return GraphSignals(abuse_probability=0.0)

    async def update_user_connections(
            self, user_id: str, connection_data: Dict[str, Any]) -> None:
        """Update user's connections in the graph"""
        try:
            # Add user node if not exists
            if not self.user_graph.has_node(user_id):
                self.user_graph.add_node(user_id)
                self.user_attributes[user_id] = {}

            # Update user attributes
            if "user_attributes" in connection_data:
                self.user_attributes[user_id].update(
                    connection_data["user_attributes"])

            # Add connections
            if "connections" in connection_data:
                for connection in connection_data["connections"]:
                    connected_user = connection.get("user_id")
                    if connected_user and connected_user != user_id:
                        # Add edge
                        self.user_graph.add_edge(user_id, connected_user)

                        # Store edge attributes
                        edge_key = tuple(sorted([user_id, connected_user]))
                        self.edge_attributes[edge_key] = connection.get(
                            "attributes", {})

            # Update known abusers
            if connection_data.get("is_abuser", False):
                self.known_abusers.add(user_id)

        except Exception as e:
            logger.error(f"Connection update failed: {str(e)}")

    async def count_suspicious_connections(self, user_id: str) -> int:
        """Count suspicious connections for a user"""
        try:
            if not self.user_graph.has_node(user_id):
                return 0

            suspicious_count = 0
            neighbors = list(self.user_graph.neighbors(user_id))

            for neighbor in neighbors:
                # Check if neighbor is a known abuser
                if neighbor in self.known_abusers:
                    suspicious_count += 1

                # Check neighbor's attributes for suspicious patterns
                neighbor_attrs = self.user_attributes.get(neighbor, {})
                if self.is_suspicious_user(neighbor_attrs):
                    suspicious_count += 1

                # Check edge attributes
                edge_key = tuple(sorted([user_id, neighbor]))
                edge_attrs = self.edge_attributes.get(edge_key, {})
                if self.is_suspicious_connection(edge_attrs):
                    suspicious_count += 1

            return suspicious_count

        except Exception as e:
            logger.error(f"Suspicious connection counting failed: {str(e)}")
            return 0

    async def detect_cluster_anomaly(self, user_id: str) -> bool:
        """Detect if user is in an anomalous cluster"""
        try:
            if not self.user_graph.has_node(user_id):
                return False

            # Get user's neighborhood
            neighbors = list(self.user_graph.neighbors(user_id))
            if len(neighbors) < 2:
                return False

            # Extract features for clustering
            features = []
            for neighbor in neighbors:
                neighbor_attrs = self.user_attributes.get(neighbor, {})
                feature_vector = self.extract_user_features(neighbor_attrs)
                features.append(feature_vector)

            if len(features) < 2:
                return False

            # Normalize features
            features_array = np.array(features)
            normalized_features = self.scaler.fit_transform(features_array)

            # Perform clustering
            cluster_labels = self.clustering_model.fit_predict(
                normalized_features)

            # Check for anomalies (outliers or very small clusters)
            unique_labels, counts = np.unique(
                cluster_labels, return_counts=True)

            # Anomaly if user's neighbors form very small clusters or are
            # mostly outliers
            outlier_count = np.sum(cluster_labels == -1)  # DBSCAN outliers
            # Clusters with < 3 members
            small_cluster_count = np.sum(counts[counts > 0] < 3)

            anomaly_ratio = (
                outlier_count + small_cluster_count) / len(neighbors)

            return anomaly_ratio > self.config.cluster_anomaly_threshold

        except Exception as e:
            logger.error(f"Cluster anomaly detection failed: {str(e)}")
            return False

    async def detect_centrality_anomaly(self, user_id: str) -> bool:
        """Detect if user has anomalous centrality measures"""
        try:
            if not self.user_graph.has_node(user_id):
                return False

            # Calculate centrality measures
            degree_centrality = nx.degree_centrality(self.user_graph)
            betweenness_centrality = nx.betweenness_centrality(self.user_graph)
            closeness_centrality = nx.closeness_centrality(self.user_graph)

            user_degree = degree_centrality.get(user_id, 0)
            user_betweenness = betweenness_centrality.get(user_id, 0)
            user_closeness = closeness_centrality.get(user_id, 0)

            # Calculate network statistics
            all_degrees = list(degree_centrality.values())
            all_betweenness = list(betweenness_centrality.values())
            all_closeness = list(closeness_centrality.values())

            if not all_degrees:
                return False

            # Check for anomalies
            degree_anomaly = self.is_statistical_anomaly(
                user_degree, all_degrees)
            betweenness_anomaly = self.is_statistical_anomaly(
                user_betweenness, all_betweenness)
            closeness_anomaly = self.is_statistical_anomaly(
                user_closeness, all_closeness)

            # Anomaly if any centrality measure is unusual
            return degree_anomaly or betweenness_anomaly or closeness_anomaly

        except Exception as e:
            logger.error(f"Centrality anomaly detection failed: {str(e)}")
            return False

    def is_suspicious_user(self, user_attrs: Dict[str, Any]) -> bool:
        """Check if user attributes indicate suspicious behavior"""
        try:
            suspicious_indicators = 0

            # Check account age
            if "account_age_days" in user_attrs:
                if user_attrs["account_age_days"] < 7:  # Very new account
                    suspicious_indicators += 1

            # Check activity level
            if "activity_score" in user_attrs:
                if user_attrs["activity_score"] > 0.9:  # Very high activity
                    suspicious_indicators += 1

            # Check reputation
            if "reputation_score" in user_attrs:
                if user_attrs["reputation_score"] < 0.3:  # Low reputation
                    suspicious_indicators += 1

            # Check verification status
            if "is_verified" in user_attrs:
                if not user_attrs["is_verified"]:  # Unverified account
                    suspicious_indicators += 1

            # Check for suspicious patterns
            if "suspicious_patterns" in user_attrs:
                if user_attrs["suspicious_patterns"]:
                    suspicious_indicators += 2

            return suspicious_indicators >= 2

        except Exception as e:
            logger.error(f"Suspicious user check failed: {str(e)}")
            return False

    def is_suspicious_connection(self, edge_attrs: Dict[str, Any]) -> bool:
        """Check if connection attributes indicate suspicious behavior"""
        try:
            suspicious_indicators = 0

            # Check connection frequency
            if "connection_frequency" in edge_attrs:
                if edge_attrs["connection_frequency"] > 100:  # Very frequent connections
                    suspicious_indicators += 1

            # Check connection timing
            if "connection_timing" in edge_attrs:
                if edge_attrs["connection_timing"] == "unusual_hours":  # Unusual timing
                    suspicious_indicators += 1

            # Check connection type
            if "connection_type" in edge_attrs:
                if edge_attrs["connection_type"] in ["automated", "bot"]:
                    suspicious_indicators += 1

            # Check for mutual connections
            if "mutual_connections" in edge_attrs:
                if edge_attrs["mutual_connections"] < 2:  # Few mutual connections
                    suspicious_indicators += 1

            return suspicious_indicators >= 2

        except Exception as e:
            logger.error(f"Suspicious connection check failed: {str(e)}")
            return False

    def extract_user_features(self, user_attrs: Dict[str, Any]) -> List[float]:
        """Extract numerical features from user attributes"""
        try:
            features = []

            # Account age (normalized)
            features.append(user_attrs.get("account_age_days", 0) / 365.0)

            # Activity score
            features.append(user_attrs.get("activity_score", 0.5))

            # Reputation score
            features.append(user_attrs.get("reputation_score", 0.5))

            # Verification status
            features.append(
                1.0 if user_attrs.get(
                    "is_verified",
                    False) else 0.0)

            # Suspicious patterns count
            features.append(len(user_attrs.get("suspicious_patterns", [])))

            # Connection count
            features.append(user_attrs.get("connection_count", 0) / 100.0)

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return [0.0] * 6

    def is_statistical_anomaly(
            self,
            value: float,
            distribution: List[float]) -> bool:
        """Check if a value is statistically anomalous"""
        try:
            if not distribution:
                return False

            mean_val = np.mean(distribution)
            std_val = np.std(distribution)

            if std_val == 0:
                return False

            # Z-score
            z_score = abs(value - mean_val) / std_val

            # Anomaly if z-score > 2 (2 standard deviations)
            return z_score > 2.0

        except Exception as e:
            logger.error(f"Statistical anomaly check failed: {str(e)}")
            return False

    def calculate_abuse_probability(
            self,
            suspicious_connections: int,
            cluster_anomaly: bool,
            centrality_anomaly: bool) -> float:
        """Calculate overall abuse probability from graph signals"""
        try:
            # Weight different signals
            weights = {
                "suspicious_connections": 0.4,
                "cluster_anomaly": 0.3,
                "centrality_anomaly": 0.3,
            }

            # Calculate weighted score
            score = 0.0

            # Suspicious connections score
            suspicious_score = min(
                suspicious_connections / 5.0,
                1.0)  # Normalize to [0, 1]
            score += suspicious_score * weights["suspicious_connections"]

            # Cluster anomaly score
            if cluster_anomaly:
                score += weights["cluster_anomaly"]

            # Centrality anomaly score
            if centrality_anomaly:
                score += weights["centrality_anomaly"]

            return min(score, 1.0)

        except Exception as e:
            logger.error(f"Abuse probability calculation failed: {str(e)}")
            return 0.0

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get overall graph statistics"""
        try:
            if self.user_graph.number_of_nodes() == 0:
                return {"message": "No graph data available"}

            stats = {
                "total_nodes": self.user_graph.number_of_nodes(),
                "total_edges": self.user_graph.number_of_edges(),
                "average_degree": np.mean([d for n, d in self.user_graph.degree()]),
                "density": nx.density(self.user_graph),
                "connected_components": nx.number_connected_components(self.user_graph),
                "known_abusers": len(self.known_abusers),
                "suspicious_patterns": len(self.suspicious_patterns),
            }

            # Calculate clustering coefficient
            try:
                stats["average_clustering"] = nx.average_clustering(
                    self.user_graph)
            except BaseException:
                stats["average_clustering"] = 0.0

            return stats

        except Exception as e:
            logger.error(f"Graph statistics calculation failed: {str(e)}")
            return {"error": str(e)}

    async def identify_abuse_clusters(self) -> List[List[str]]:
        """Identify potential abuse clusters in the graph"""
        try:
            if self.user_graph.number_of_nodes() < 3:
                return []

            # Extract features for all users
            features = []
            user_ids = []

            for user_id in self.user_graph.nodes():
                user_attrs = self.user_attributes.get(user_id, {})
                feature_vector = self.extract_user_features(user_attrs)
                features.append(feature_vector)
                user_ids.append(user_id)

            if len(features) < 3:
                return []

            # Normalize features
            features_array = np.array(features)
            normalized_features = self.scaler.fit_transform(features_array)

            # Perform clustering
            cluster_labels = self.clustering_model.fit_predict(
                normalized_features)

            # Group users by cluster
            clusters = defaultdict(list)
            for user_id, label in zip(user_ids, cluster_labels):
                clusters[label].append(user_id)

            # Return clusters with more than 2 members (excluding outliers)
            abuse_clusters = [
                cluster for label,
                cluster in clusters.items() if label != -
                1 and len(cluster) > 2]

            return abuse_clusters

        except Exception as e:
            logger.error(f"Abuse cluster identification failed: {str(e)}")
            return []
