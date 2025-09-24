"""Content quality signals and trustworthiness scoring."""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
import structlog
import textstat
from bs4 import BeautifulSoup

from ..config import settings
from ..models.content import ContentSignal, ExtractedContent, TrustworthinessScore

logger = structlog.get_logger(__name__)


class SignalExtractor:
    """Content quality signals and trustworthiness scoring."""

    def __init__(self):
        """Initialize the signal extractor."""
        self.bias_keywords = self._load_bias_keywords()
        self.political_keywords = self._load_political_keywords()
        self.expertise_indicators = self._load_expertise_indicators()
        self.fact_checking_patterns = self._load_fact_checking_patterns()

    def _load_bias_keywords(self) -> Dict[str, List[str]]:
        """Load bias detection keywords."""
        return {
            "left_bias": [
                "progressive",
                "liberal",
                "democrat",
                "left-wing",
                "socialist",
                "equality",
                "social justice",
                "climate change",
                "renewable energy",
            ],
            "right_bias": [
                "conservative",
                "republican",
                "right-wing",
                "traditional",
                "free market",
                "small government",
                "patriotism",
                "family values",
            ],
            "emotional_language": [
                "shocking",
                "outrageous",
                "incredible",
                "amazing",
                "terrible",
                "horrible",
                "fantastic",
                "devastating",
                "stunning",
                "appalling",
            ],
            "subjective_indicators": [
                "i believe",
                "i think",
                "in my opinion",
                "personally",
                "obviously",
                "clearly",
                "undoubtedly",
                "certainly",
            ],
        }

    def _load_political_keywords(self) -> Dict[str, List[str]]:
        """Load political leaning keywords."""
        return {
            "left": [
                "progressive",
                "liberal",
                "democrat",
                "socialist",
                "green party",
                "climate action",
                "social justice",
                "universal healthcare",
            ],
            "right": [
                "conservative",
                "republican",
                "libertarian",
                "tea party",
                "free market",
                "small government",
                "traditional values",
            ],
            "center": ["moderate", "centrist", "bipartisan", "compromise", "balanced"],
        }

    def _load_expertise_indicators(self) -> List[str]:
        """Load expertise and authority indicators."""
        return [
            "research",
            "study",
            "data",
            "statistics",
            "analysis",
            "report",
            "expert",
            "professor",
            "doctor",
            "scientist",
            "researcher",
            "peer-reviewed",
            "journal",
            "university",
            "institute",
            "laboratory",
            "clinical trial",
            "experiment",
            "findings",
            "conclusions",
        ]

    def _load_fact_checking_patterns(self) -> List[str]:
        """Load fact-checking and citation patterns."""
        return [
            r"according to",
            r"research shows",
            r"studies indicate",
            r"data suggests",
            r"statistics show",
            r"reports indicate",
            r"findings reveal",
            r"evidence shows",
            r"analysis reveals",
            r"investigation found",
        ]

    async def extract_signals(
        self, content: ExtractedContent, language: str = "en"
    ) -> ContentSignal:
        """Extract comprehensive content quality signals."""
        try:
            # Prepare text for analysis
            full_text = f"{content.title} {content.summary or ''} {content.content}"

            # Extract various signals in parallel
            tasks = [
                self._calculate_readability(full_text),
                self._count_factual_claims(full_text),
                self._count_citations(full_text),
                self._analyze_bias(full_text),
                self._detect_political_leaning(full_text),
                self._predict_engagement(full_text, content),
                self._calculate_virality_potential(full_text, content),
                self._assess_content_freshness(content),
                self._calculate_authority_score(full_text, content),
                self._extract_expertise_indicators(full_text),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            readability_score = results[0] if not isinstance(results[0], Exception) else 0.5
            factual_claims = results[1] if not isinstance(results[1], Exception) else 0
            citations_count = results[2] if not isinstance(results[2], Exception) else 0
            bias_score = results[3] if not isinstance(results[3], Exception) else 0.0
            political_leaning = results[4] if not isinstance(results[4], Exception) else None
            engagement_prediction = results[5] if not isinstance(results[5], Exception) else 0.5
            virality_potential = results[6] if not isinstance(results[6], Exception) else 0.5
            content_freshness = results[7] if not isinstance(results[7], Exception) else 0.5
            authority_score = results[8] if not isinstance(results[8], Exception) else 0.5
            expertise_indicators = results[9] if not isinstance(results[9], Exception) else []

            # Create content signal
            signal = ContentSignal(
                readability_score=readability_score,
                factual_claims=factual_claims,
                citations_count=citations_count,
                bias_score=bias_score,
                political_leaning=political_leaning,
                engagement_prediction=engagement_prediction,
                virality_potential=virality_potential,
                content_freshness=content_freshness,
                authority_score=authority_score,
                expertise_indicators=expertise_indicators,
            )

            logger.info(
                "Content signals extracted",
                content_id=content.id,
                readability=readability_score,
                factual_claims=factual_claims,
                citations=citations_count,
                bias_score=bias_score,
            )

            return signal

        except Exception as e:
            logger.error("Signal extraction failed", content_id=content.id, error=str(e))

            # Return default signals as fallback
            return ContentSignal(
                readability_score=0.5,
                factual_claims=0,
                citations_count=0,
                bias_score=0.0,
                engagement_prediction=0.5,
                virality_potential=0.5,
                content_freshness=0.5,
                authority_score=0.5,
            )

    async def _calculate_readability(self, text: str) -> float:
        """Calculate readability score using multiple metrics."""
        try:
            # Flesch Reading Ease Score (0-100, higher is easier)
            flesch_score = textstat.flesch_reading_ease(text)

            # Flesch-Kincaid Grade Level
            fk_grade = textstat.flesch_kincaid_grade(text)

            # SMOG Index
            smog_score = textstat.smog_index(text)

            # Coleman-Liau Index
            cl_score = textstat.coleman_liau_index(text)

            # Average Grade Level
            avg_grade = (fk_grade + smog_score + cl_score) / 3

            # Convert to 0-1 scale (inverted, so higher is better)
            # Assuming target audience is 8th grade (13-14 years old)
            readability_score = max(0, min(1, (14 - avg_grade) / 14))

            return readability_score

        except Exception as e:
            logger.error("Readability calculation failed", error=str(e))
            return 0.5

    async def _count_factual_claims(self, text: str) -> int:
        """Count factual claims in the text."""
        try:
            claim_patterns = [
                r"is\s+\w+",
                r"are\s+\w+",
                r"was\s+\w+",
                r"were\s+\w+",
                r"will\s+be\s+\w+",
                r"has\s+been\s+\w+",
                r"have\s+been\s+\w+",
                r"can\s+be\s+\w+",
                r"should\s+be\s+\w+",
                r"must\s+be\s+\w+",
            ]

            claim_count = 0
            for pattern in claim_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                claim_count += len(matches)

            return claim_count

        except Exception as e:
            logger.error("Factual claims counting failed", error=str(e))
            return 0

    async def _count_citations(self, text: str) -> int:
        """Count citations and references in the text."""
        try:
            citation_patterns = [
                r"\([^)]*\d{4}[^)]*\)",  # (Author, 2023)
                r"\[[^\]]*\d{4}[^\]]*\]",  # [Author, 2023]
                r"according to [^.]*",  # according to source
                r"as reported by [^.]*",  # as reported by source
                r"studies show",  # studies show
                r"research indicates",  # research indicates
                r"data suggests",  # data suggests
                r"statistics show",  # statistics show
            ]

            citation_count = 0
            for pattern in citation_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                citation_count += len(matches)

            return citation_count

        except Exception as e:
            logger.error("Citation counting failed", error=str(e))
            return 0

    async def _analyze_bias(self, text: str) -> float:
        """Analyze bias in the text (-1 = left bias, 1 = right bias)."""
        try:
            text_lower = text.lower()

            # Count bias keywords
            left_count = sum(
                1 for keyword in self.bias_keywords["left_bias"] if keyword in text_lower
            )
            right_count = sum(
                1 for keyword in self.bias_keywords["right_bias"] if keyword in text_lower
            )

            # Count emotional language
            emotional_count = sum(
                1 for keyword in self.bias_keywords["emotional_language"] if keyword in text_lower
            )

            # Count subjective indicators
            subjective_count = sum(
                1
                for keyword in self.bias_keywords["subjective_indicators"]
                if keyword in text_lower
            )

            # Calculate bias score
            total_bias_indicators = left_count + right_count + emotional_count + subjective_count

            if total_bias_indicators == 0:
                return 0.0  # Neutral

            # Normalize bias score
            bias_score = (right_count - left_count) / total_bias_indicators

            # Adjust for emotional and subjective language
            if emotional_count > 0 or subjective_count > 0:
                bias_score *= 1.2  # Amplify bias if emotional/subjective

            return max(-1.0, min(1.0, bias_score))

        except Exception as e:
            logger.error("Bias analysis failed", error=str(e))
            return 0.0

    async def _detect_political_leaning(self, text: str) -> Optional[str]:
        """Detect political leaning of the content."""
        try:
            text_lower = text.lower()

            left_score = sum(
                1 for keyword in self.political_keywords["left"] if keyword in text_lower
            )
            right_score = sum(
                1 for keyword in self.political_keywords["right"] if keyword in text_lower
            )
            center_score = sum(
                1 for keyword in self.political_keywords["center"] if keyword in text_lower
            )

            total_score = left_score + right_score + center_score

            if total_score == 0:
                return None  # No political indicators found

            # Determine leaning
            if left_score > right_score and left_score > center_score:
                return "left"
            elif right_score > left_score and right_score > center_score:
                return "right"
            elif center_score > left_score and center_score > right_score:
                return "center"
            else:
                return "mixed"

        except Exception as e:
            logger.error("Political leaning detection failed", error=str(e))
            return None

    async def _predict_engagement(self, text: str, content: ExtractedContent) -> float:
        """Predict engagement potential of the content."""
        try:
            engagement_score = 0.0

            # Title factors
            title_length = len(content.title)
            if 20 <= title_length <= 60:  # Optimal title length
                engagement_score += 0.2

            # Question in title
            if "?" in content.title:
                engagement_score += 0.1

            # Number in title
            if re.search(r"\d+", content.title):
                engagement_score += 0.1

            # Content length (optimal range)
            content_length = len(text)
            if 500 <= content_length <= 2000:
                engagement_score += 0.2
            elif 2000 < content_length <= 5000:
                engagement_score += 0.1

            # Lists and bullet points
            list_count = text.count("\n-") + text.count("\n•") + text.count("\n*")
            if list_count > 0:
                engagement_score += min(0.2, list_count * 0.05)

            # Subheadings
            subheading_count = len(re.findall(r"\n#+\s+", text))
            if subheading_count > 0:
                engagement_score += min(0.1, subheading_count * 0.02)

            # Images (if metadata available)
            if hasattr(content, "metadata") and "image_count" in content.metadata:
                image_count = content.metadata["image_count"]
                engagement_score += min(0.2, image_count * 0.05)

            return min(1.0, engagement_score)

        except Exception as e:
            logger.error("Engagement prediction failed", error=str(e))
            return 0.5

    async def _calculate_virality_potential(self, text: str, content: ExtractedContent) -> float:
        """Calculate virality potential of the content."""
        try:
            virality_score = 0.0

            # Emotional words
            emotional_words = [
                "viral",
                "trending",
                "breaking",
                "shocking",
                "amazing",
                "incredible",
                "unbelievable",
                "stunning",
                "devastating",
                "outrageous",
                "controversial",
            ]

            text_lower = text.lower()
            emotional_count = sum(1 for word in emotional_words if word in text_lower)
            virality_score += min(0.3, emotional_count * 0.05)

            # Controversial topics
            controversial_topics = [
                "politics",
                "scandal",
                "controversy",
                "outrage",
                "backlash",
                "criticism",
                "debate",
                "dispute",
                "conflict",
                "crisis",
            ]

            controversial_count = sum(1 for topic in controversial_topics if topic in text_lower)
            virality_score += min(0.2, controversial_count * 0.05)

            # Social media indicators
            social_indicators = ["share", "retweet", "like", "comment", "follow"]
            social_count = sum(1 for indicator in social_indicators if indicator in text_lower)
            virality_score += min(0.1, social_count * 0.02)

            # Time sensitivity
            time_sensitive = [
                "breaking",
                "urgent",
                "immediate",
                "now",
                "today",
                "latest",
                "just in",
                "recently",
                "newly",
                "fresh",
            ]

            time_count = sum(1 for word in time_sensitive if word in text_lower)
            virality_score += min(0.2, time_count * 0.05)

            # Content type
            if content.content_type.value in ["news", "social_media"]:
                virality_score += 0.1

            return min(1.0, virality_score)

        except Exception as e:
            logger.error("Virality potential calculation failed", error=str(e))
            return 0.5

    async def _assess_content_freshness(self, content: ExtractedContent) -> float:
        """Assess content freshness based on publication date."""
        try:
            if not content.published_date:
                return 0.5  # Unknown freshness

            now = datetime.utcnow()
            age_days = (now - content.published_date).days

            # Calculate freshness score
            if age_days <= 1:
                freshness = 1.0
            elif age_days <= 7:
                freshness = 0.8
            elif age_days <= 30:
                freshness = 0.6
            elif age_days <= 90:
                freshness = 0.4
            elif age_days <= 365:
                freshness = 0.2
            else:
                freshness = 0.1

            return freshness

        except Exception as e:
            logger.error("Content freshness assessment failed", error=str(e))
            return 0.5

    async def _calculate_authority_score(self, text: str, content: ExtractedContent) -> float:
        """Calculate authority and credibility score."""
        try:
            authority_score = 0.0

            # Author credibility
            if content.author:
                # Check for professional titles
                professional_titles = [
                    "dr",
                    "professor",
                    "prof",
                    "doctor",
                    "phd",
                    "md",
                    "rn",
                    "expert",
                    "specialist",
                    "analyst",
                    "researcher",
                ]

                author_lower = content.author.lower()
                title_count = sum(1 for title in professional_titles if title in author_lower)
                authority_score += min(0.3, title_count * 0.1)

            # Source credibility
            if content.source:
                # Check for known credible sources
                credible_sources = [
                    "reuters",
                    "ap",
                    "bbc",
                    "cnn",
                    "nytimes",
                    "washington post",
                    "wall street journal",
                    "bloomberg",
                    "forbes",
                    "techcrunch",
                ]

                source_lower = content.source.lower()
                if any(source in source_lower for source in credible_sources):
                    authority_score += 0.2

            # URL credibility
            if content.url:
                try:
                    parsed_url = urlparse(content.url)
                    domain = parsed_url.netloc.lower()

                    # Check for trusted domains
                    trusted_domains = [
                        ".edu",
                        ".gov",
                        ".org",
                        "reuters.com",
                        "ap.org",
                        "bbc.com",
                        "nytimes.com",
                        "washingtonpost.com",
                    ]

                    if any(domain.endswith(trusted) for trusted in trusted_domains):
                        authority_score += 0.2

                except Exception:
                    pass

            # Content quality indicators
            expertise_count = sum(
                1 for indicator in self.expertise_indicators if indicator in text.lower()
            )
            authority_score += min(0.3, expertise_count * 0.05)

            return min(1.0, authority_score)

        except Exception as e:
            logger.error("Authority score calculation failed", error=str(e))
            return 0.5

    async def _extract_expertise_indicators(self, text: str) -> List[str]:
        """Extract expertise and authority indicators from text."""
        try:
            text_lower = text.lower()
            found_indicators = []

            for indicator in self.expertise_indicators:
                if indicator in text_lower:
                    found_indicators.append(indicator)

            return found_indicators

        except Exception as e:
            logger.error("Expertise indicators extraction failed", error=str(e))
            return []

    async def compute_trustworthiness(
        self, content: ExtractedContent, language: str = "en"
    ) -> TrustworthinessScore:
        """Compute comprehensive trustworthiness score."""
        try:
            # Get content signals first
            signals = await self.extract_signals(content, language)

            # Calculate individual trustworthiness components
            source_reliability = await self._assess_source_reliability(content)
            fact_checking_score = await self._assess_fact_checking(content)
            citation_quality = await self._assess_citation_quality(content)
            author_credibility = await self._assess_author_credibility(content)
            content_quality = await self._assess_content_quality(content, signals)

            # Calculate overall trustworthiness score
            overall_score = (
                source_reliability * 0.25
                + fact_checking_score * 0.20
                + citation_quality * 0.20
                + author_credibility * 0.15
                + content_quality * 0.20
            )

            # Identify bias indicators and warning flags
            bias_indicators = await self._identify_bias_indicators(content, signals)
            warning_flags = await self._identify_warning_flags(content, signals)

            trust_score = TrustworthinessScore(
                overall_score=overall_score,
                source_reliability=source_reliability,
                fact_checking_score=fact_checking_score,
                citation_quality=citation_quality,
                author_credibility=author_credibility,
                content_quality=content_quality,
                bias_indicators=bias_indicators,
                warning_flags=warning_flags,
            )

            logger.info(
                "Trustworthiness score computed",
                content_id=content.id,
                overall_score=overall_score,
                source_reliability=source_reliability,
            )

            return trust_score

        except Exception as e:
            logger.error("Trustworthiness computation failed", content_id=content.id, error=str(e))

            # Return neutral trust score as fallback
            return TrustworthinessScore(
                overall_score=0.5,
                source_reliability=0.5,
                fact_checking_score=0.5,
                citation_quality=0.5,
                author_credibility=0.5,
                content_quality=0.5,
            )

    async def _assess_source_reliability(self, content: ExtractedContent) -> float:
        """Assess source reliability."""
        try:
            reliability_score = 0.5  # Default neutral score

            if content.source:
                # Check against known reliable sources
                reliable_sources = [
                    "reuters",
                    "associated press",
                    "ap",
                    "bbc",
                    "cnn",
                    "new york times",
                    "washington post",
                    "wall street journal",
                    "bloomberg",
                    "forbes",
                    "techcrunch",
                    "wired",
                ]

                source_lower = content.source.lower()
                if any(source in source_lower for source in reliable_sources):
                    reliability_score = 0.9
                else:
                    reliability_score = 0.6  # Unknown source

            if content.url:
                try:
                    parsed_url = urlparse(content.url)
                    domain = parsed_url.netloc.lower()

                    # Check domain trustworthiness
                    if domain.endswith(".edu") or domain.endswith(".gov"):
                        reliability_score = 0.95
                    elif domain.endswith(".org"):
                        reliability_score = 0.8
                    elif any(trusted in domain for trusted in ["reuters", "ap", "bbc", "cnn"]):
                        reliability_score = 0.9

                except Exception:
                    pass

            return reliability_score

        except Exception as e:
            logger.error("Source reliability assessment failed", error=str(e))
            return 0.5

    async def _assess_fact_checking(self, content: ExtractedContent) -> float:
        """Assess fact-checking quality."""
        try:
            text = f"{content.title} {content.content}"
            fact_checking_score = 0.0

            # Count fact-checking patterns
            fact_patterns = [
                r"according to [^.]*",
                r"studies show",
                r"research indicates",
                r"data suggests",
                r"statistics show",
                r"reports indicate",
                r"findings reveal",
                r"evidence shows",
            ]

            pattern_count = 0
            for pattern in fact_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                pattern_count += len(matches)

            # Calculate score based on pattern frequency
            if pattern_count > 5:
                fact_checking_score = 0.9
            elif pattern_count > 3:
                fact_checking_score = 0.7
            elif pattern_count > 1:
                fact_checking_score = 0.5
            else:
                fact_checking_score = 0.3

            return fact_checking_score

        except Exception as e:
            logger.error("Fact-checking assessment failed", error=str(e))
            return 0.5

    async def _assess_citation_quality(self, content: ExtractedContent) -> float:
        """Assess citation quality."""
        try:
            text = f"{content.title} {content.content}"
            citation_score = 0.0

            # Count different types of citations
            citation_patterns = {
                "academic": [r"\([^)]*\d{4}[^)]*\)", r"\[[^\]]*\d{4}[^\]]*\]"],
                "url": [r"https?://[^\s]+"],
                "general": [r"according to", r"as reported by", r"studies show"],
            }

            total_citations = 0
            for pattern_list in citation_patterns.values():
                for pattern in pattern_list:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    total_citations += len(matches)

            # Weight academic citations higher
            academic_citations = 0
            for pattern in citation_patterns["academic"]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                academic_citations += len(matches)

            # Calculate score
            if academic_citations > 3:
                citation_score = 0.9
            elif total_citations > 5:
                citation_score = 0.7
            elif total_citations > 2:
                citation_score = 0.5
            else:
                citation_score = 0.3

            return citation_score

        except Exception as e:
            logger.error("Citation quality assessment failed", error=str(e))
            return 0.5

    async def _assess_author_credibility(self, content: ExtractedContent) -> float:
        """Assess author credibility."""
        try:
            if not content.author:
                return 0.3  # No author information

            author_lower = content.author.lower()
            credibility_score = 0.5  # Default score

            # Professional titles
            professional_titles = [
                "dr",
                "professor",
                "prof",
                "doctor",
                "phd",
                "md",
                "rn",
                "expert",
                "specialist",
                "analyst",
                "researcher",
                "scientist",
            ]

            title_count = sum(1 for title in professional_titles if title in author_lower)
            if title_count > 0:
                credibility_score += 0.3

            # Institutional affiliations
            institutions = [
                "university",
                "college",
                "institute",
                "laboratory",
                "hospital",
                "research center",
                "think tank",
                "ngo",
                "government",
            ]

            institution_count = sum(1 for inst in institutions if inst in author_lower)
            if institution_count > 0:
                credibility_score += 0.2

            return min(1.0, credibility_score)

        except Exception as e:
            logger.error("Author credibility assessment failed", error=str(e))
            return 0.5

    async def _assess_content_quality(
        self, content: ExtractedContent, signals: ContentSignal
    ) -> float:
        """Assess overall content quality."""
        try:
            quality_score = 0.0

            # Readability score
            quality_score += signals.readability_score * 0.3

            # Authority score
            quality_score += signals.authority_score * 0.3

            # Expertise indicators
            expertise_score = min(1.0, len(signals.expertise_indicators) / 10)
            quality_score += expertise_score * 0.2

            # Content length (optimal range)
            content_length = len(content.content)
            if 500 <= content_length <= 2000:
                length_score = 1.0
            elif 2000 < content_length <= 5000:
                length_score = 0.8
            elif 100 <= content_length < 500:
                length_score = 0.6
            else:
                length_score = 0.4

            quality_score += length_score * 0.2

            return min(1.0, quality_score)

        except Exception as e:
            logger.error("Content quality assessment failed", error=str(e))
            return 0.5

    async def _identify_bias_indicators(
        self, content: ExtractedContent, signals: ContentSignal
    ) -> List[str]:
        """Identify bias indicators in content."""
        try:
            indicators = []

            # High bias score
            if abs(signals.bias_score) > 0.5:
                indicators.append("high_bias_score")

            # Political leaning
            if signals.political_leaning:
                indicators.append(f"political_leaning_{signals.political_leaning}")

            # Emotional language
            text = f"{content.title} {content.content}".lower()
            emotional_count = sum(
                1 for word in self.bias_keywords["emotional_language"] if word in text
            )
            if emotional_count > 3:
                indicators.append("excessive_emotional_language")

            # Subjective language
            subjective_count = sum(
                1 for word in self.bias_keywords["subjective_indicators"] if word in text
            )
            if subjective_count > 2:
                indicators.append("subjective_language")

            return indicators

        except Exception as e:
            logger.error("Bias indicators identification failed", error=str(e))
            return []

    async def _identify_warning_flags(
        self, content: ExtractedContent, signals: ContentSignal
    ) -> List[str]:
        """Identify warning flags for content quality."""
        try:
            flags = []

            # Low readability
            if signals.readability_score < 0.3:
                flags.append("low_readability")

            # No citations
            if signals.citations_count == 0:
                flags.append("no_citations")

            # High bias
            if abs(signals.bias_score) > 0.7:
                flags.append("high_bias")

            # Low authority
            if signals.authority_score < 0.3:
                flags.append("low_authority")

            # No expertise indicators
            if len(signals.expertise_indicators) == 0:
                flags.append("no_expertise_indicators")

            # Very short content
            if len(content.content) < 200:
                flags.append("very_short_content")

            # Very long content
            if len(content.content) > 10000:
                flags.append("very_long_content")

            return flags

        except Exception as e:
            logger.error("Warning flags identification failed", error=str(e))
            return []
