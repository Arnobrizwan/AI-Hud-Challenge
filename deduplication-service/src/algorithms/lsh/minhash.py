"""MinHash implementation for LSH."""

import hashlib
from typing import List, Set, Union

import mmh3
import numpy as np
from datasketch import MinHash


class MinHashGenerator:
    """MinHash generator for content fingerprinting."""

    def __init__(self, num_perm: int = 128, seed: int = 42):
        """Initialize MinHash generator.

        Args:
            num_perm: Number of permutations for MinHash
            seed: Random seed for reproducibility
        """
        self.num_perm = num_perm
        self.seed = seed
        self._hash_funcs = self._generate_hash_functions()

    def _generate_hash_functions(self) -> List[tuple]:
        """Generate hash functions for MinHash."""
        np.random.seed(self.seed)
        hash_funcs = []

        for i in range(self.num_perm):
            # Generate random coefficients for linear hash functions
            a = np.random.randint(1, 2**32)
            b = np.random.randint(0, 2**32)
            hash_funcs.append((a, b))

        return hash_funcs

    def _hash(self, value: Union[str, int], a: int, b: int) -> int:
        """Apply hash function to value."""
        if isinstance(value, str):
            # Use mmh3 for string hashing
            return mmh3.hash(value, seed=a) ^ b
        else:
            return (a * value + b) % (2**32)

    def _get_shingles(self, text: str, k: int = 3) -> Set[str]:
        """Extract k-shingles from text.

        Args:
            text: Input text
            k: Shingle size

        Returns:
            Set of k-shingles
        """
        if len(text) < k:
            return {text}

        shingles = set()
        for i in range(len(text) - k + 1):
            shingle = text[i : i + k].lower()
            shingles.add(shingle)

        return shingles

    def generate_minhash(self, text: str, k: int = 3) -> MinHash:
        """Generate MinHash signature for text.

        Args:
            text: Input text
            k: Shingle size

        Returns:
            MinHash signature
        """
        shingles = self._get_shingles(text, k)
        minhash = MinHash(num_perm=self.num_perm, seed=self.seed)

        for shingle in shingles:
            minhash.update(shingle.encode("utf-8"))

        return minhash

    def generate_minhash_from_tokens(self, tokens: List[str]) -> MinHash:
        """Generate MinHash signature from token list.

        Args:
            tokens: List of tokens

        Returns:
            MinHash signature
        """
        minhash = MinHash(num_perm=self.num_perm, seed=self.seed)

        for token in tokens:
            minhash.update(token.encode("utf-8"))

        return minhash

    def jaccard_similarity(self, minhash1: MinHash, minhash2: MinHash) -> float:
        """Calculate Jaccard similarity between two MinHash signatures.

        Args:
            minhash1: First MinHash signature
            minhash2: Second MinHash signature

        Returns:
            Jaccard similarity (0.0 to 1.0)
        """
        return minhash1.jaccard(minhash2)

    def estimate_jaccard(self, minhash1: MinHash, minhash2: MinHash) -> float:
        """Estimate Jaccard similarity using MinHash.

        Args:
            minhash1: First MinHash signature
            minhash2: Second MinHash signature

        Returns:
            Estimated Jaccard similarity
        """
        return minhash1.jaccard(minhash2)


class ContentFingerprinter:
    """Content fingerprinting using multiple techniques."""

    def __init__(self, minhash_generator: MinHashGenerator):
        """Initialize content fingerprinter.

        Args:
            minhash_generator: MinHash generator instance
        """
        self.minhash_gen = minhash_generator

    def compute_content_hash(self, content: str) -> str:
        """Compute content hash for exact matching.

        Args:
            content: Article content

        Returns:
            Content hash
        """
        # Normalize content
        normalized = self._normalize_content(content)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def compute_title_hash(self, title: str) -> str:
        """Compute title hash for exact matching.

        Args:
            title: Article title

        Returns:
            Title hash
        """
        normalized = self._normalize_content(title)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def compute_fingerprint(self, content: str, title: str = None) -> dict:
        """Compute comprehensive content fingerprint.

        Args:
            content: Article content
            title: Article title (optional)

        Returns:
            Dictionary with various fingerprints
        """
        fingerprint = {
            "content_hash": self.compute_content_hash(content),
            "content_minhash": self.minhash_gen.generate_minhash(content),
            "content_length": len(content),
            "word_count": len(content.split()),
        }

        if title:
            fingerprint.update(
                {
                    "title_hash": self.compute_title_hash(title),
                    "title_minhash": self.minhash_gen.generate_minhash(title),
                    "title_length": len(title),
                }
            )

        return fingerprint

    def _normalize_content(self, text: str) -> str:
        """Normalize text for consistent hashing.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove special characters but keep alphanumeric and spaces
        import re

        text = re.sub(r"[^\w\s]", " ", text)

        return text.strip()

    def compute_similarity(self, fingerprint1: dict, fingerprint2: dict) -> dict:
        """Compute similarity between two fingerprints.

        Args:
            fingerprint1: First content fingerprint
            fingerprint2: Second content fingerprint

        Returns:
            Dictionary with similarity scores
        """
        similarities = {}

        # Content hash similarity (exact match)
        similarities["content_hash_match"] = fingerprint1["content_hash"] == fingerprint2["content_hash"]

        # Title hash similarity (exact match)
        if "title_hash" in fingerprint1 and "title_hash" in fingerprint2:
            similarities["title_hash_match"] = fingerprint1["title_hash"] == fingerprint2["title_hash"]

        # MinHash similarity
        if "content_minhash" in fingerprint1 and "content_minhash" in fingerprint2:
            similarities["content_jaccard"] = self.minhash_gen.jaccard_similarity(
                fingerprint1["content_minhash"], fingerprint2["content_minhash"]
            )

        # Title MinHash similarity
        if "title_minhash" in fingerprint1 and "title_minhash" in fingerprint2:
            similarities["title_jaccard"] = self.minhash_gen.jaccard_similarity(
                fingerprint1["title_minhash"], fingerprint2["title_minhash"]
            )

        # Length similarity
        similarities["length_similarity"] = self._compute_length_similarity(
            fingerprint1["content_length"], fingerprint2["content_length"]
        )

        return similarities

    def _compute_length_similarity(self, len1: int, len2: int) -> float:
        """Compute length similarity between two texts.

        Args:
            len1: First text length
            len2: Second text length

        Returns:
            Length similarity (0.0 to 1.0)
        """
        if len1 == 0 and len2 == 0:
            return 1.0

        if len1 == 0 or len2 == 0:
            return 0.0

        # Use ratio of smaller to larger length
        return min(len1, len2) / max(len1, len2)
