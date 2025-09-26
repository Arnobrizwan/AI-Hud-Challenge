"""Tests for MinHash implementation."""

from src.algorithms.lsh.minhash import ContentFingerprinter, MinHashGenerator


class TestMinHashGenerator:
    """Test MinHash generator."""

    def test_init(self):
        """Test MinHash generator initialization."""
        generator = MinHashGenerator(num_perm=128, seed=42)
        assert generator.num_perm == 128
        assert generator.seed == 42
        assert len(generator._hash_funcs) == 128

    def test_get_shingles(self):
        """Test shingle extraction."""
        generator = MinHashGenerator()
        text = "hello world"
        shingles = generator._get_shingles(text, k=3)
        expected = {"hel", "ell", "llo", "lo ", "o w", " wo", "wor", "orl", "rld"}
        assert shingles == expected

    def test_get_shingles_short_text(self):
        """Test shingle extraction with short text."""
        generator = MinHashGenerator()
        text = "hi"
        shingles = generator._get_shingles(text, k=3)
        assert shingles == {"hi"}

    def test_generate_minhash(self):
        """Test MinHash generation."""
        generator = MinHashGenerator(num_perm=64, seed=42)
        text = "hello world"
        minhash = generator.generate_minhash(text)
        assert minhash is not None
        assert len(minhash.hashvalues) == 64

    def test_generate_minhash_from_tokens(self):
        """Test MinHash generation from tokens."""
        generator = MinHashGenerator(num_perm=64, seed=42)
        tokens = ["hello", "world", "test"]
        minhash = generator.generate_minhash_from_tokens(tokens)
        assert minhash is not None
        assert len(minhash.hashvalues) == 64

    def test_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        generator = MinHashGenerator(num_perm=64, seed=42)

        # Identical texts
        text1 = "hello world"
        text2 = "hello world"
        minhash1 = generator.generate_minhash(text1)
        minhash2 = generator.generate_minhash(text2)
        similarity = generator.jaccard_similarity(minhash1, minhash2)
        assert similarity == 1.0

        # Different texts
        text3 = "goodbye world"
        minhash3 = generator.generate_minhash(text3)
        similarity = generator.jaccard_similarity(minhash1, minhash3)
        assert 0.0 <= similarity <= 1.0

    def test_estimate_jaccard(self):
        """Test Jaccard estimation."""
        generator = MinHashGenerator(num_perm=64, seed=42)
        text1 = "hello world"
        text2 = "hello world"
        minhash1 = generator.generate_minhash(text1)
        minhash2 = generator.generate_minhash(text2)
        similarity = generator.estimate_jaccard(minhash1, minhash2)
        assert similarity == 1.0


class TestContentFingerprinter:
    """Test content fingerprinter."""

    def test_init(self, minhash_generator):
        """Test fingerprinter initialization."""
        fingerprinter = ContentFingerprinter(minhash_generator)
        assert fingerprinter.minhash_gen == minhash_generator

    def test_compute_content_hash(self, content_fingerprinter):
        """Test content hash computation."""
        content = "This is a test article content."
        content_hash = content_fingerprinter.compute_content_hash(content)
        assert isinstance(content_hash, str)
        assert len(content_hash) == 64  # SHA256 hash length

        # Same content should produce same hash
        content_hash2 = content_fingerprinter.compute_content_hash(content)
        assert content_hash == content_hash2

    def test_compute_title_hash(self, content_fingerprinter):
        """Test title hash computation."""
        title = "Test Article Title"
        title_hash = content_fingerprinter.compute_title_hash(title)
        assert isinstance(title_hash, str)
        assert len(title_hash) == 64

    def test_compute_fingerprint(self, content_fingerprinter):
        """Test fingerprint computation."""
        content = "This is a test article content."
        title = "Test Article Title"

        fingerprint = content_fingerprinter.compute_fingerprint(content, title)

        assert "content_hash" in fingerprint
        assert "content_minhash" in fingerprint
        assert "content_length" in fingerprint
        assert "word_count" in fingerprint
        assert "title_hash" in fingerprint
        assert "title_minhash" in fingerprint
        assert "title_length" in fingerprint

        assert fingerprint["content_length"] == len(content)
        assert fingerprint["word_count"] == len(content.split())
        assert fingerprint["title_length"] == len(title)

    def test_compute_similarity(self, content_fingerprinter):
        """Test similarity computation."""
        content1 = "This is a test article content."
        title1 = "Test Article Title"

        content2 = "This is a test article content with more text."
        title2 = "Test Article Title - Updated"

        fingerprint1 = content_fingerprinter.compute_fingerprint(content1, title1)
        fingerprint2 = content_fingerprinter.compute_fingerprint(content2, title2)

        similarities = content_fingerprinter.compute_similarity(fingerprint1, fingerprint2)

        assert "content_hash_match" in similarities
        assert "title_hash_match" in similarities
        assert "content_jaccard" in similarities
        assert "title_jaccard" in similarities
        assert "length_similarity" in similarities

        assert isinstance(similarities["content_hash_match"], bool)
        assert isinstance(similarities["title_hash_match"], bool)
        assert 0.0 <= similarities["content_jaccard"] <= 1.0
        assert 0.0 <= similarities["title_jaccard"] <= 1.0
        assert 0.0 <= similarities["length_similarity"] <= 1.0

    def test_normalize_content(self, content_fingerprinter):
        """Test content normalization."""
        text = "  Hello, World!  This is a test.  "
        normalized = content_fingerprinter._normalize_content(text)
        expected = "hello  world  this is a test"
        assert normalized == expected

    def test_compute_length_similarity(self, content_fingerprinter):
        """Test length similarity computation."""
        # Identical lengths
        sim = content_fingerprinter._compute_length_similarity(100, 100)
        assert sim == 1.0

        # Different lengths
        sim = content_fingerprinter._compute_length_similarity(100, 200)
        assert sim == 0.5

        # Zero lengths
        sim = content_fingerprinter._compute_length_similarity(0, 0)
        assert sim == 1.0

        sim = content_fingerprinter._compute_length_similarity(0, 100)
        assert sim == 0.0
