"""
Content Moderation Engine
AI-powered content moderation and safety
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from safety_engine.config import get_content_config
from safety_engine.models import (
    ContentModerationRequest,
    ContentViolation,
    ModerationAction,
    ModerationResult,
    TextModerationResult,
)

from .detectors import (
    AdultContentDetector,
    HateSpeechDetector,
    MisinformationDetector,
    SpamDetector,
    ToxicityDetector,
    ViolenceDetector,
)
from .external_apis import ExternalModerationAPIs

logger = logging.getLogger(__name__)


class ContentModerationEngine:
    """AI-powered content moderation and safety"""

    def __init__(self):
        self.config = get_content_config()
        self.is_initialized = False

        # Content detectors
        self.toxicity_detector = ToxicityDetector()
        self.hate_speech_detector = HateSpeechDetector()
        self.spam_detector = SpamDetector()
        self.misinformation_detector = MisinformationDetector()
        self.adult_content_detector = AdultContentDetector()
        self.violence_detector = ViolenceDetector()

        # External APIs
        self.external_apis = ExternalModerationAPIs()

    async def initialize(self):
        """Initialize the content moderation engine"""
        try:
            # Initialize all detectors
            await self.toxicity_detector.initialize()
            await self.hate_speech_detector.initialize()
            await self.spam_detector.initialize()
            await self.misinformation_detector.initialize()
            await self.adult_content_detector.initialize()
            await self.violence_detector.initialize()
            await self.external_apis.initialize()

            self.is_initialized = True
            logger.info("Content moderation engine initialized")

        except Exception as e:
            logger.error(f"Failed to initialize content moderation engine: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.toxicity_detector.cleanup()
            await self.hate_speech_detector.cleanup()
            await self.spam_detector.cleanup()
            await self.misinformation_detector.cleanup()
            await self.adult_content_detector.cleanup()
            await self.violence_detector.cleanup()
            await self.external_apis.cleanup()

            self.is_initialized = False
            logger.info("Content moderation engine cleanup completed")

        except Exception as e:
            logger.error(f"Error during content moderation engine cleanup: {str(e)}")

    async def moderate_content(self, content: Any) -> ModerationResult:
        """Comprehensive content moderation"""

        if not self.is_initialized:
            raise RuntimeError("Content moderation engine not initialized")

        try:
            moderation_results = {}

            # Text content moderation
            if hasattr(content, "text_content") and content.text_content:
                text_results = await self.moderate_text_content(content.text_content)
                moderation_results["text"] = text_results

            # Image content moderation
            if hasattr(content, "image_urls") and content.image_urls:
                image_results = await self.moderate_image_content(content.image_urls)
                moderation_results["images"] = image_results

            # Video content moderation
            if hasattr(content, "video_urls") and content.video_urls:
                video_results = await self.moderate_video_content(content.video_urls)
                moderation_results["videos"] = video_results

            # URL safety check
            if hasattr(content, "external_urls") and content.external_urls:
                url_results = await self.check_url_safety(content.external_urls)
                moderation_results["urls"] = url_results

            # Calculate overall safety score
            overall_safety_score = self.calculate_content_safety_score(moderation_results)

            # Determine moderation action
            moderation_action = self.determine_moderation_action(
                overall_safety_score, moderation_results
            )

            # Extract violations
            violations = self.extract_violations(moderation_results)

            return ModerationResult(
                content_id=getattr(content, "id", "unknown"),
                overall_safety_score=overall_safety_score,
                moderation_results=moderation_results,
                recommended_action=moderation_action,
                violations=violations,
                moderation_timestamp=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Content moderation failed: {str(e)}")
            raise

    async def moderate_text_content(self, text: str) -> TextModerationResult:
        """Moderate text content for various safety issues"""
        try:
            # Run parallel moderation checks
            moderation_checks = await asyncio.gather(
                self.toxicity_detector.analyze(text),
                self.hate_speech_detector.analyze(text),
                self.spam_detector.analyze(text),
                self.misinformation_detector.analyze(text),
                return_exceptions=True,
            )

            toxicity_result, hate_speech_result, spam_result, misinformation_result = (
                moderation_checks
            )

            # External API checks (if enabled)
            external_results = None
            if self.config.external_apis:
                external_results = await self.external_apis.moderate_text(text)

            # Extract scores and handle exceptions
            toxicity_score = (
                toxicity_result.toxicity_score
                if not isinstance(toxicity_result, Exception)
                else 0.0
            )
            hate_speech_score = (
                hate_speech_result.hate_score
                if not isinstance(hate_speech_result, Exception)
                else 0.0
            )
            spam_score = spam_result.spam_score if not isinstance(spam_result, Exception) else 0.0
            misinformation_score = (
                misinformation_result.misinformation_score
                if not isinstance(misinformation_result, Exception)
                else 0.0
            )

            # Extract detected issues
            detected_issues = self.extract_detected_issues(moderation_checks)

            return TextModerationResult(
                text=text,
                toxicity_score=toxicity_score,
                hate_speech_score=hate_speech_score,
                spam_score=spam_score,
                misinformation_score=misinformation_score,
                external_results=external_results,
                detected_issues=detected_issues,
            )

        except Exception as e:
            logger.error(f"Text content moderation failed: {str(e)}")
            return TextModerationResult(
                text=text,
                toxicity_score=0.0,
                hate_speech_score=0.0,
                spam_score=0.0,
                misinformation_score=0.0,
                external_results=None,
                detected_issues=[],
            )

    async def moderate_image_content(self, image_urls: List[str]) -> Dict[str, Any]:
        """Moderate image content for safety issues"""
        try:
            image_results = {}

            for i, image_url in enumerate(image_urls):
                # Run parallel image moderation checks
                image_checks = await asyncio.gather(
                    self.adult_content_detector.analyze_image(image_url),
                    self.violence_detector.analyze_image(image_url),
                    return_exceptions=True,
                )

                adult_result, violence_result = image_checks

                # Extract scores
                adult_score = (
                    adult_result.adult_score if not isinstance(adult_result, Exception) else 0.0
                )
                violence_score = (
                    violence_result.violence_score
                    if not isinstance(violence_result, Exception)
                    else 0.0
                )

                image_results[f"image_{i}"] = {
                    "url": image_url,
                    "adult_content_score": adult_score,
                    "violence_score": violence_score,
                    "is_safe": adult_score < self.config.adult_content_threshold
                    and violence_score < self.config.violence_threshold,
                }

            return image_results

        except Exception as e:
            logger.error(f"Image content moderation failed: {str(e)}")
            return {}

    async def moderate_video_content(self, video_urls: List[str]) -> Dict[str, Any]:
        """Moderate video content for safety issues"""
        try:
            video_results = {}

            for i, video_url in enumerate(video_urls):
                # Run parallel video moderation checks
                video_checks = await asyncio.gather(
                    self.adult_content_detector.analyze_video(video_url),
                    self.violence_detector.analyze_video(video_url),
                    return_exceptions=True,
                )

                adult_result, violence_result = video_checks

                # Extract scores
                adult_score = (
                    adult_result.adult_score if not isinstance(adult_result, Exception) else 0.0
                )
                violence_score = (
                    violence_result.violence_score
                    if not isinstance(violence_result, Exception)
                    else 0.0
                )

                video_results[f"video_{i}"] = {
                    "url": video_url,
                    "adult_content_score": adult_score,
                    "violence_score": violence_score,
                    "is_safe": adult_score < self.config.adult_content_threshold
                    and violence_score < self.config.violence_threshold,
                }

            return video_results

        except Exception as e:
            logger.error(f"Video content moderation failed: {str(e)}")
            return {}

    async def check_url_safety(self, urls: List[str]) -> Dict[str, Any]:
        """Check URL safety"""
        try:
            url_results = {}

            for i, url in enumerate(urls):
                # Check URL safety
                is_safe = await self.check_single_url_safety(url)

                url_results[f"url_{i}"] = {
                    "url": url,
                    "is_safe": is_safe,
                    "risk_level": "low" if is_safe else "high",
                }

            return url_results

        except Exception as e:
            logger.error(f"URL safety check failed: {str(e)}")
            return {}

    async def check_single_url_safety(self, url: str) -> bool:
        """Check safety of a single URL"""
        try:
            # Basic URL safety checks
            suspicious_domains = ["malware.com", "phishing.com", "spam.com"]
            suspicious_keywords = ["malware", "phishing", "spam", "virus"]

            # Check domain
            for domain in suspicious_domains:
                if domain in url.lower():
                    return False

            # Check keywords
            for keyword in suspicious_keywords:
                if keyword in url.lower():
                    return False

            # Check URL structure
            if "http://" in url and not url.startswith("https://"):
                return False  # Prefer HTTPS

            return True

        except Exception as e:
            logger.error(f"URL safety check failed for {url}: {str(e)}")
            return False

    def calculate_content_safety_score(self, moderation_results: Dict[str, Any]) -> float:
        """Calculate overall content safety score"""
        try:
            if not moderation_results:
                return 1.0  # No content to moderate

            scores = []

            # Text moderation score
            if "text" in moderation_results:
                text_result = moderation_results["text"]
                text_score = self.calculate_text_safety_score(text_result)
                scores.append(text_score)

            # Image moderation score
            if "images" in moderation_results:
                image_score = self.calculate_media_safety_score(moderation_results["images"])
                scores.append(image_score)

            # Video moderation score
            if "videos" in moderation_results:
                video_score = self.calculate_media_safety_score(moderation_results["videos"])
                scores.append(video_score)

            # URL safety score
            if "urls" in moderation_results:
                url_score = self.calculate_url_safety_score(moderation_results["urls"])
                scores.append(url_score)

            if not scores:
                return 1.0

            # Return average safety score
            return sum(scores) / len(scores)

        except Exception as e:
            logger.error(f"Content safety score calculation failed: {str(e)}")
            return 0.0

    def calculate_text_safety_score(self, text_result: TextModerationResult) -> float:
        """Calculate safety score for text content"""
        try:
            # Get individual scores
            toxicity_score = text_result.toxicity_score
            hate_speech_score = text_result.hate_speech_score
            spam_score = text_result.spam_score
            misinformation_score = text_result.misinformation_score

            # Calculate weighted safety score
            weights = {"toxicity": 0.3, "hate_speech": 0.3, "spam": 0.2, "misinformation": 0.2}

            safety_score = 1.0
            safety_score -= toxicity_score * weights["toxicity"]
            safety_score -= hate_speech_score * weights["hate_speech"]
            safety_score -= spam_score * weights["spam"]
            safety_score -= misinformation_score * weights["misinformation"]

            return max(0.0, min(1.0, safety_score))

        except Exception as e:
            logger.error(f"Text safety score calculation failed: {str(e)}")
            return 0.0

    def calculate_media_safety_score(self, media_results: Dict[str, Any]) -> float:
        """Calculate safety score for media content"""
        try:
            if not media_results:
                return 1.0

            scores = []
            for item in media_results.values():
                if isinstance(item, dict) and "is_safe" in item:
                    scores.append(1.0 if item["is_safe"] else 0.0)

            return sum(scores) / len(scores) if scores else 1.0

        except Exception as e:
            logger.error(f"Media safety score calculation failed: {str(e)}")
            return 0.0

    def calculate_url_safety_score(self, url_results: Dict[str, Any]) -> float:
        """Calculate safety score for URLs"""
        try:
            if not url_results:
                return 1.0

            scores = []
            for item in url_results.values():
                if isinstance(item, dict) and "is_safe" in item:
                    scores.append(1.0 if item["is_safe"] else 0.0)

            return sum(scores) / len(scores) if scores else 1.0

        except Exception as e:
            logger.error(f"URL safety score calculation failed: {str(e)}")
            return 0.0

    def determine_moderation_action(
        self, safety_score: float, moderation_results: Dict[str, Any]
    ) -> ModerationAction:
        """Determine appropriate moderation action based on safety score"""
        try:
            if safety_score >= self.config.safety_threshold:
                return ModerationAction.ALLOW
            elif safety_score >= 0.6:
                return ModerationAction.WARN
            elif safety_score >= 0.4:
                return ModerationAction.FLAG
            elif safety_score >= 0.2:
                return ModerationAction.BLOCK
            else:
                return ModerationAction.REMOVE

        except Exception as e:
            logger.error(f"Moderation action determination failed: {str(e)}")
            return ModerationAction.FLAG

    def extract_violations(self, moderation_results: Dict[str, Any]) -> List[ContentViolation]:
        """Extract content violations from moderation results"""
        try:
            violations = []

            # Text violations
            if "text" in moderation_results:
                text_result = moderation_results["text"]
                violations.extend(self.extract_text_violations(text_result))

            # Media violations
            if "images" in moderation_results:
                violations.extend(
                    self.extract_media_violations(moderation_results["images"], "image")
                )

            if "videos" in moderation_results:
                violations.extend(
                    self.extract_media_violations(moderation_results["videos"], "video")
                )

            # URL violations
            if "urls" in moderation_results:
                violations.extend(self.extract_url_violations(moderation_results["urls"]))

            return violations

        except Exception as e:
            logger.error(f"Violation extraction failed: {str(e)}")
            return []

    def extract_text_violations(self, text_result: TextModerationResult) -> List[ContentViolation]:
        """Extract violations from text moderation results"""
        try:
            violations = []

            # Toxicity violation
            if text_result.toxicity_score > self.config.toxicity_threshold:
                violations.append(
                    ContentViolation(
                        violation_type="toxicity",
                        severity="high" if text_result.toxicity_score > 0.8 else "medium",
                        confidence=text_result.toxicity_score,
                        description=f"Toxic content detected (score: {text_result.toxicity_score:.2f})",
                        affected_content=(
                            text_result.text[:100] + "..."
                            if len(text_result.text) > 100
                            else text_result.text
                        ),
                    )
                )

            # Hate speech violation
            if text_result.hate_speech_score > self.config.hate_speech_threshold:
                violations.append(
                    ContentViolation(
                        violation_type="hate_speech",
                        severity="high" if text_result.hate_speech_score > 0.8 else "medium",
                        confidence=text_result.hate_speech_score,
                        description=f"Hate speech detected (score: {text_result.hate_speech_score:.2f})",
                        affected_content=(
                            text_result.text[:100] + "..."
                            if len(text_result.text) > 100
                            else text_result.text
                        ),
                    )
                )

            # Spam violation
            if text_result.spam_score > self.config.spam_threshold:
                violations.append(
                    ContentViolation(
                        violation_type="spam",
                        severity="medium" if text_result.spam_score > 0.8 else "low",
                        confidence=text_result.spam_score,
                        description=f"Spam content detected (score: {text_result.spam_score:.2f})",
                        affected_content=(
                            text_result.text[:100] + "..."
                            if len(text_result.text) > 100
                            else text_result.text
                        ),
                    )
                )

            # Misinformation violation
            if text_result.misinformation_score > self.config.misinformation_threshold:
                violations.append(
                    ContentViolation(
                        violation_type="misinformation",
                        severity="high" if text_result.misinformation_score > 0.8 else "medium",
                        confidence=text_result.misinformation_score,
                        description=f"Misinformation detected (score: {text_result.misinformation_score:.2f})",
                        affected_content=(
                            text_result.text[:100] + "..."
                            if len(text_result.text) > 100
                            else text_result.text
                        ),
                    )
                )

            return violations

        except Exception as e:
            logger.error(f"Text violation extraction failed: {str(e)}")
            return []

    def extract_media_violations(
        self, media_results: Dict[str, Any], media_type: str
    ) -> List[ContentViolation]:
        """Extract violations from media moderation results"""
        try:
            violations = []

            for item_id, item in media_results.items():
                if not isinstance(item, dict):
                    continue

                # Adult content violation
                if item.get("adult_content_score", 0) > self.config.adult_content_threshold:
                    violations.append(
                        ContentViolation(
                            violation_type="adult_content",
                            severity="high" if item["adult_content_score"] > 0.8 else "medium",
                            confidence=item["adult_content_score"],
                            description=f'Adult content detected in {media_type} (score: {item["adult_content_score"]:.2f})',
                            affected_content=item.get("url", "Unknown URL"),
                        )
                    )

                # Violence violation
                if item.get("violence_score", 0) > self.config.violence_threshold:
                    violations.append(
                        ContentViolation(
                            violation_type="violence",
                            severity="high" if item["violence_score"] > 0.8 else "medium",
                            confidence=item["violence_score"],
                            description=f'Violent content detected in {media_type} (score: {item["violence_score"]:.2f})',
                            affected_content=item.get("url", "Unknown URL"),
                        )
                    )

            return violations

        except Exception as e:
            logger.error(f"Media violation extraction failed: {str(e)}")
            return []

    def extract_url_violations(self, url_results: Dict[str, Any]) -> List[ContentViolation]:
        """Extract violations from URL safety results"""
        try:
            violations = []

            for item_id, item in url_results.items():
                if not isinstance(item, dict):
                    continue

                if not item.get("is_safe", True):
                    violations.append(
                        ContentViolation(
                            violation_type="unsafe_url",
                            severity="medium",
                            confidence=0.8,
                            description=f'Unsafe URL detected: {item.get("risk_level", "unknown")} risk',
                            affected_content=item.get("url", "Unknown URL"),
                        )
                    )

            return violations

        except Exception as e:
            logger.error(f"URL violation extraction failed: {str(e)}")
            return []

    def extract_detected_issues(self, moderation_checks: List[Any]) -> List[str]:
        """Extract detected issues from moderation checks"""
        try:
            issues = []

            for check in moderation_checks:
                if isinstance(check, Exception):
                    continue

                if hasattr(check, "detected_issues"):
                    issues.extend(check.detected_issues)
                elif hasattr(check, "issues"):
                    issues.extend(check.issues)

            return list(set(issues))  # Remove duplicates

        except Exception as e:
            logger.error(f"Issue extraction failed: {str(e)}")
            return []
