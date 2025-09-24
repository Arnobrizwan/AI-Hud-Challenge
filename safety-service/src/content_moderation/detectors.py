"""
Content Moderation Detectors
Various detectors for different types of content violations
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from safety_engine.config import get_content_config

logger = logging.getLogger(__name__)


@dataclass
class ModerationResult:
    """Base moderation result"""

    score: float
    confidence: float
    detected_issues: List[str] = None


class BaseContentDetector:
    """Base class for content detectors"""

    def __init__(self):
        self.config = get_content_config()
        self.is_initialized = False

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the detector"""
        self.is_initialized = True

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup resources"""
        self.is_initialized = False


class ToxicityDetector(BaseContentDetector):
    """Detect toxic content in text"""

    def __init__(self):
        super().__init__()
        self.toxic_keywords = [
            "hate",
            "stupid",
            "idiot",
            "moron",
            "dumb",
            "retard",
            "fool",
            "worthless",
            "useless",
            "pathetic",
            "disgusting",
            "gross",
            "kill yourself",
            "die",
            "suicide",
            "murder",
            "violence",
        ]
        self.toxic_patterns = [
            r"\b(f\*ck|fuck|shit|damn|hell|bitch|asshole)\b",
            r"\b(you\s+are\s+so\s+stupid)\b",
            r"\b(i\s+hate\s+you)\b",
            r"\b(go\s+die)\b",
            r"\b(kill\s+yourself)\b",
        ]

    async def analyze(self, text: str) -> ModerationResult:
        """Analyze text for toxicity"""
        try:
            if not text:
                return ModerationResult(
                    score=0.0, confidence=1.0, detected_issues=[])

            text_lower = text.lower()
            toxicity_score = 0.0
            detected_issues = []

            # Check for toxic keywords
            keyword_matches = sum(
                1 for keyword in self.toxic_keywords if keyword in text_lower)
            if keyword_matches > 0:
                toxicity_score += min(keyword_matches * 0.2, 0.6)
                detected_issues.append(
                    f"Toxic keywords detected: {keyword_matches}")

            # Check for toxic patterns
            pattern_matches = 0
            for pattern in self.toxic_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    pattern_matches += 1

            if pattern_matches > 0:
                toxicity_score += min(pattern_matches * 0.3, 0.4)
                detected_issues.append(
                    f"Toxic patterns detected: {pattern_matches}")

            # Calculate confidence based on text length and matches
            confidence = min(
                1.0, (keyword_matches + pattern_matches) / max(len(text.split()), 1))

            return ModerationResult(
                score=min(toxicity_score, 1.0),
                confidence=confidence,
                detected_issues=detected_issues,
            )

        except Exception as e:
            logger.error(f"Toxicity detection failed: {str(e)}")
            return ModerationResult(
                score=0.0, confidence=0.0, detected_issues=[])


class HateSpeechDetector(BaseContentDetector):
    """Detect hate speech in text"""

    def __init__(self):
        super().__init__()
        self.hate_keywords = [
            "nazi",
            "fascist",
            "racist",
            "bigot",
            "homophobe",
            "transphobe",
            "sexist",
            "misogynist",
            "antisemitic",
            "islamophobe",
        ]
        self.hate_patterns = [
            r"\b(all\s+\w+\s+are\s+\w+)\b",  # All X are Y
            r"\b(\w+\s+should\s+die)\b",  # X should die
            r"\b(\w+\s+are\s+inferior)\b",  # X are inferior
            r"\b(go\s+back\s+to\s+\w+)\b",  # Go back to X
        ]
        self.protected_groups = [
            "race",
            "religion",
            "gender",
            "sexual orientation",
            "disability",
            "ethnicity",
            "nationality",
            "age",
            "political views",
        ]

    async def analyze(self, text: str) -> ModerationResult:
        """Analyze text for hate speech"""
        try:
            if not text:
                return ModerationResult(
                    score=0.0, confidence=1.0, detected_issues=[])

            text_lower = text.lower()
            hate_score = 0.0
            detected_issues = []

            # Check for hate keywords
            keyword_matches = sum(
                1 for keyword in self.hate_keywords if keyword in text_lower)
            if keyword_matches > 0:
                hate_score += min(keyword_matches * 0.3, 0.7)
                detected_issues.append(
                    f"Hate speech keywords detected: {keyword_matches}")

            # Check for hate patterns
            pattern_matches = 0
            for pattern in self.hate_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    pattern_matches += len(matches)
                    detected_issues.append(
                        f"Hate speech patterns detected: {matches}")

            if pattern_matches > 0:
                hate_score += min(pattern_matches * 0.2, 0.3)

            # Check for protected group targeting
            group_targeting = 0
            for group in self.protected_groups:
                if group in text_lower:
                    # Look for negative context around the group
                    context_pattern = rf"\b(negative|bad|wrong|inferior|hate|dislike)\s+{group}\b"
                    if re.search(context_pattern, text_lower):
                        group_targeting += 1
                        detected_issues.append(
                            f"Protected group targeting detected: {group}")

            if group_targeting > 0:
                hate_score += min(group_targeting * 0.4, 0.5)

            # Calculate confidence
            confidence = min(1.0, (keyword_matches +
                                   pattern_matches +
                                   group_targeting) /
                             max(len(text.split()), 1), )

            return ModerationResult(
                score=min(
                    hate_score,
                    1.0),
                confidence=confidence,
                detected_issues=detected_issues)

        except Exception as e:
            logger.error(f"Hate speech detection failed: {str(e)}")
            return ModerationResult(
                score=0.0, confidence=0.0, detected_issues=[])


class SpamDetector(BaseContentDetector):
    """Detect spam content in text"""

    def __init__(self):
        super().__init__()
        self.spam_keywords = [
            "click here",
            "buy now",
            "free money",
            "make money",
            "work from home",
            "guaranteed",
            "no risk",
            "limited time",
            "act now",
            "urgent",
            "congratulations",
            "you won",
            "prize",
            "lottery",
            "winner",
        ]
        self.spam_patterns = [
            r"\b(click\s+here\s+now)\b",
            r"\b(buy\s+now\s+only)\b",
            r"\b(free\s+money\s+guaranteed)\b",
            r"\b(make\s+money\s+fast)\b",
            r"\b(work\s+from\s+home\s+earn)\b",
        ]
        self.url_pattern = r"https?://[^\s]+"
        self.email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

    async def analyze(self, text: str) -> ModerationResult:
        """Analyze text for spam"""
        try:
            if not text:
                return ModerationResult(
                    score=0.0, confidence=1.0, detected_issues=[])

            text_lower = text.lower()
            spam_score = 0.0
            detected_issues = []

            # Check for spam keywords
            keyword_matches = sum(
                1 for keyword in self.spam_keywords if keyword in text_lower)
            if keyword_matches > 0:
                spam_score += min(keyword_matches * 0.2, 0.5)
                detected_issues.append(
                    f"Spam keywords detected: {keyword_matches}")

            # Check for spam patterns
            pattern_matches = 0
            for pattern in self.spam_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    pattern_matches += 1

            if pattern_matches > 0:
                spam_score += min(pattern_matches * 0.3, 0.4)
                detected_issues.append(
                    f"Spam patterns detected: {pattern_matches}")

            # Check for excessive URLs
            url_count = len(re.findall(self.url_pattern, text))
            if url_count > 3:
                spam_score += min(url_count * 0.1, 0.3)
                detected_issues.append(f"Excessive URLs detected: {url_count}")

            # Check for excessive emails
            email_count = len(re.findall(self.email_pattern, text))
            if email_count > 2:
                spam_score += min(email_count * 0.15, 0.2)
                detected_issues.append(
                    f"Excessive emails detected: {email_count}")

            # Check for repetitive content
            words = text_lower.split()
            if len(words) > 10:
                word_freq = {}
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1

                max_freq = max(word_freq.values())
                if max_freq > len(words) * 0.3:  # 30% repetition threshold
                    spam_score += 0.3
                    detected_issues.append("Repetitive content detected")

            # Check for excessive capitalization
            if len(text) > 10:
                caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
                if caps_ratio > 0.7:  # 70% caps threshold
                    spam_score += 0.2
                    detected_issues.append("Excessive capitalization detected")

            # Calculate confidence
            confidence = min(
                1.0,
                (keyword_matches + pattern_matches + url_count + email_count)
                / max(len(text.split()), 1),
            )

            return ModerationResult(
                score=min(
                    spam_score,
                    1.0),
                confidence=confidence,
                detected_issues=detected_issues)

        except Exception as e:
            logger.error(f"Spam detection failed: {str(e)}")
            return ModerationResult(
                score=0.0, confidence=0.0, detected_issues=[])


class MisinformationDetector(BaseContentDetector):
    """Detect misinformation in text"""

    def __init__(self):
        super().__init__()
        self.misinformation_keywords = [
            "fake news",
            "conspiracy",
            "hoax",
            "lies",
            "cover up",
            "secret",
            "hidden truth",
            "they don't want you to know",
            "mainstream media lies",
            "government lies",
        ]
        self.claim_patterns = [
            r"\b(studies\s+show)\b",
            r"\b(scientists\s+prove)\b",
            r"\b(doctors\s+say)\b",
            r"\b(experts\s+agree)\b",
            r"\b(research\s+proves)\b",
        ]
        self.urgency_patterns = [
            r"\b(urgent|breaking|shocking|revealed|exposed)\b",
            r"\b(you\s+won\'t\s+believe)\b",
            r"\b(this\s+will\s+shock\s+you)\b",
        ]

    async def analyze(self, text: str) -> ModerationResult:
        """Analyze text for misinformation"""
        try:
            if not text:
                return ModerationResult(
                    score=0.0, confidence=1.0, detected_issues=[])

            text_lower = text.lower()
            misinformation_score = 0.0
            detected_issues = []

            # Check for misinformation keywords
            keyword_matches = sum(
                1 for keyword in self.misinformation_keywords if keyword in text_lower)
            if keyword_matches > 0:
                misinformation_score += min(keyword_matches * 0.3, 0.6)
                detected_issues.append(
                    f"Misinformation keywords detected: {keyword_matches}")

            # Check for unsubstantiated claims
            claim_matches = 0
            for pattern in self.claim_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    claim_matches += 1

            if claim_matches > 0:
                misinformation_score += min(claim_matches * 0.2, 0.3)
                detected_issues.append(
                    f"Unsubstantiated claims detected: {claim_matches}")

            # Check for urgency patterns
            urgency_matches = 0
            for pattern in self.urgency_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    urgency_matches += 1

            if urgency_matches > 0:
                misinformation_score += min(urgency_matches * 0.15, 0.2)
                detected_issues.append(
                    f"Urgency patterns detected: {urgency_matches}")

            # Check for excessive punctuation (often used in misinformation)
            if len(text) > 10:
                exclamation_ratio = text.count("!") / len(text)
                if exclamation_ratio > 0.05:  # 5% exclamation threshold
                    misinformation_score += 0.1
                    detected_issues.append(
                        "Excessive exclamation marks detected")

            # Calculate confidence
            confidence = min(1.0, (keyword_matches +
                                   claim_matches +
                                   urgency_matches) /
                             max(len(text.split()), 1))

            return ModerationResult(
                score=min(misinformation_score, 1.0),
                confidence=confidence,
                detected_issues=detected_issues,
            )

        except Exception as e:
            logger.error(f"Misinformation detection failed: {str(e)}")
            return ModerationResult(
                score=0.0, confidence=0.0, detected_issues=[])


class AdultContentDetector(BaseContentDetector):
    """Detect adult content in text and media"""

    def __init__(self):
        super().__init__()
        self.adult_keywords = [
            "porn",
            "pornography",
            "sex",
            "sexual",
            "nude",
            "naked",
            "adult",
            "explicit",
            "xxx",
            "nsfw",
            "erotic",
        ]
        self.adult_patterns = [
            r"\b(adult\s+content)\b",
            r"\b(not\s+safe\s+for\s+work)\b",
            r"\b(explicit\s+content)\b",
        ]

    async def analyze(self, text: str) -> ModerationResult:
        """Analyze text for adult content"""
        try:
            if not text:
                return ModerationResult(
                    score=0.0, confidence=1.0, detected_issues=[])

            text_lower = text.lower()
            adult_score = 0.0
            detected_issues = []

            # Check for adult keywords
            keyword_matches = sum(
                1 for keyword in self.adult_keywords if keyword in text_lower)
            if keyword_matches > 0:
                adult_score += min(keyword_matches * 0.4, 0.8)
                detected_issues.append(
                    f"Adult content keywords detected: {keyword_matches}")

            # Check for adult patterns
            pattern_matches = 0
            for pattern in self.adult_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    pattern_matches += 1

            if pattern_matches > 0:
                adult_score += min(pattern_matches * 0.3, 0.2)
                detected_issues.append(
                    f"Adult content patterns detected: {pattern_matches}")

            # Calculate confidence
            confidence = min(
                1.0, (keyword_matches + pattern_matches) / max(len(text.split()), 1))

            return ModerationResult(
                score=min(
                    adult_score,
                    1.0),
                confidence=confidence,
                detected_issues=detected_issues)

        except Exception as e:
            logger.error(f"Adult content detection failed: {str(e)}")
            return ModerationResult(
                score=0.0, confidence=0.0, detected_issues=[])

    async def analyze_image(self, image_url: str) -> ModerationResult:
        """Analyze image for adult content"""
        try:
            # In a real implementation, this would use computer vision APIs
            # For now, we'll simulate based on URL patterns
            adult_score = 0.0
            detected_issues = []

            # Check URL for adult content indicators
            if any(keyword in image_url.lower()
                   for keyword in self.adult_keywords):
                adult_score = 0.8
                detected_issues.append("Adult content detected in image URL")

            return ModerationResult(
                score=adult_score,
                confidence=0.7 if adult_score > 0 else 0.9,
                detected_issues=detected_issues,
            )

        except Exception as e:
            logger.error(f"Image adult content detection failed: {str(e)}")
            return ModerationResult(
                score=0.0, confidence=0.0, detected_issues=[])

    async def analyze_video(self, video_url: str) -> ModerationResult:
        """Analyze video for adult content"""
        try:
            # In a real implementation, this would use video analysis APIs
            # For now, we'll simulate based on URL patterns
            adult_score = 0.0
            detected_issues = []

            # Check URL for adult content indicators
            if any(keyword in video_url.lower()
                   for keyword in self.adult_keywords):
                adult_score = 0.8
                detected_issues.append("Adult content detected in video URL")

            return ModerationResult(
                score=adult_score,
                confidence=0.7 if adult_score > 0 else 0.9,
                detected_issues=detected_issues,
            )

        except Exception as e:
            logger.error(f"Video adult content detection failed: {str(e)}")
            return ModerationResult(
                score=0.0, confidence=0.0, detected_issues=[])


class ViolenceDetector(BaseContentDetector):
    """Detect violent content in text and media"""

    def __init__(self):
        super().__init__()
        self.violence_keywords = [
            "kill",
            "murder",
            "death",
            "violence",
            "fight",
            "attack",
            "weapon",
            "gun",
            "knife",
            "bomb",
            "explosion",
            "blood",
            "gore",
            "torture",
            "abuse",
            "assault",
        ]
        self.violence_patterns = [
            r"\b(kill\s+\w+)\b",
            r"\b(attack\s+\w+)\b",
            r"\b(violence\s+against)\b",
            r"\b(weapon\s+of\s+choice)\b",
        ]

    async def analyze(self, text: str) -> ModerationResult:
        """Analyze text for violence"""
        try:
            if not text:
                return ModerationResult(
                    score=0.0, confidence=1.0, detected_issues=[])

            text_lower = text.lower()
            violence_score = 0.0
            detected_issues = []

            # Check for violence keywords
            keyword_matches = sum(
                1 for keyword in self.violence_keywords if keyword in text_lower)
            if keyword_matches > 0:
                violence_score += min(keyword_matches * 0.3, 0.7)
                detected_issues.append(
                    f"Violence keywords detected: {keyword_matches}")

            # Check for violence patterns
            pattern_matches = 0
            for pattern in self.violence_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    pattern_matches += 1

            if pattern_matches > 0:
                violence_score += min(pattern_matches * 0.2, 0.3)
                detected_issues.append(
                    f"Violence patterns detected: {pattern_matches}")

            # Calculate confidence
            confidence = min(
                1.0, (keyword_matches + pattern_matches) / max(len(text.split()), 1))

            return ModerationResult(
                score=min(violence_score, 1.0),
                confidence=confidence,
                detected_issues=detected_issues,
            )

        except Exception as e:
            logger.error(f"Violence detection failed: {str(e)}")
            return ModerationResult(
                score=0.0, confidence=0.0, detected_issues=[])

    async def analyze_image(self, image_url: str) -> ModerationResult:
        """Analyze image for violence"""
        try:
            # In a real implementation, this would use computer vision APIs
            # For now, we'll simulate based on URL patterns
            violence_score = 0.0
            detected_issues = []

            # Check URL for violence indicators
            if any(keyword in image_url.lower()
                   for keyword in self.violence_keywords):
                violence_score = 0.7
                detected_issues.append("Violence detected in image URL")

            return ModerationResult(
                score=violence_score,
                confidence=0.7 if violence_score > 0 else 0.9,
                detected_issues=detected_issues,
            )

        except Exception as e:
            logger.error(f"Image violence detection failed: {str(e)}")
            return ModerationResult(
                score=0.0, confidence=0.0, detected_issues=[])

    async def analyze_video(self, video_url: str) -> ModerationResult:
        """Analyze video for violence"""
        try:
            # In a real implementation, this would use video analysis APIs
            # For now, we'll simulate based on URL patterns
            violence_score = 0.0
            detected_issues = []

            # Check URL for violence indicators
            if any(keyword in video_url.lower()
                   for keyword in self.violence_keywords):
                violence_score = 0.7
                detected_issues.append("Violence detected in video URL")

            return ModerationResult(
                score=violence_score,
                confidence=0.7 if violence_score > 0 else 0.9,
                detected_issues=detected_issues,
            )

        except Exception as e:
            logger.error(f"Video violence detection failed: {str(e)}")
            return ModerationResult(
                score=0.0, confidence=0.0, detected_issues=[])


class ExternalModerationAPIs:
    """External moderation API integration"""

    def __init__(self):
        self.config = get_content_config()
        self.is_initialized = False
        self.enabled = self.config.external_apis

    async def initialize(self) -> Dict[str, Any]:
        """Initialize external APIs"""
        self.is_initialized = True

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup resources"""
        self.is_initialized = False

    async def moderate_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Moderate text using external APIs"""
        try:
            if not self.enabled or not self.is_initialized:
                return None

            # In a real implementation, this would call external APIs like:
            # - Google Perspective API
            # - Microsoft Content Moderator
            # - AWS Comprehend
            # - Azure Content Safety

            # For now, return None (no external moderation)
            return None

        except Exception as e:
            logger.error(f"External text moderation failed: {str(e)}")
            return None

    async def moderate_image(self, image_url: str) -> Optional[Dict[str, Any]]:
        """Moderate image using external APIs"""
        try:
            if not self.enabled or not self.is_initialized:
                return None

            # In a real implementation, this would call external APIs
            return None

        except Exception as e:
            logger.error(f"External image moderation failed: {str(e)}")
            return None

    async def moderate_video(self, video_url: str) -> Optional[Dict[str, Any]]:
        """Moderate video using external APIs"""
        try:
            if not self.enabled or not self.is_initialized:
                return None

            # In a real implementation, this would call external APIs
            return None

        except Exception as e:
            logger.error(f"External video moderation failed: {str(e)}")
            return None
