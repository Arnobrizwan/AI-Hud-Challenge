"""
Vertex AI Integration for Abstractive Summarization
Advanced abstractive summarization using PaLM 2 and Vertex AI
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import google.cloud.aiplatform as aiplatform
import vertexai
from google.api_core import exceptions as gcp_exceptions
from google.cloud import aiplatform_v1
from vertexai.preview.language_models import TextGenerationModel

from config.settings import settings

from .models import ProcessedContent, SummarizationMethod

logger = logging.getLogger(__name__)


class VertexAISummarizer:
    """Vertex AI integration for advanced abstractive summarization"""

    def __init__(self):
        """Initialize the Vertex AI summarizer"""
        self.project_id = settings.GOOGLE_CLOUD_PROJECT
        self.location = settings.GOOGLE_CLOUD_REGION
        self.endpoint = settings.VERTEX_AI_ENDPOINT
        self.model_name = "text-bison@001"  # PaLM 2 model
        self.client = None
        self.prediction_client = None
        self._initialized = False

    async def initialize(self):
        """Initialize Vertex AI client and models"""
        try:
            logger.info("Initializing Vertex AI summarizer...")

            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.location)

            # Initialize prediction service client
            self.prediction_client = aiplatform_v1.PredictionServiceClient()

            # Initialize text generation model
            self.model = TextGenerationModel.from_pretrained(self.model_name)

            self._initialized = True
            logger.info("Vertex AI summarizer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI summarizer: {str(e)}")
            raise

    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.prediction_client:
                self.prediction_client = None
            if self.model:
                self.model = None
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def summarize(
        self,
        content: ProcessedContent,
        target_length: int = 120,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> str:
        """
        Generate abstractive summary using PaLM 2

        Args:
            content: Processed content to summarize
            target_length: Target summary length in words
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated abstractive summary
        """
        if not self._initialized:
            raise RuntimeError("Vertex AI summarizer not initialized")

        try:
            # Construct optimized prompt
            prompt = self._construct_summarization_prompt(content, target_length)

            # Generate summary using PaLM 2
            response = await self._generate_with_retry(
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.8,
                top_k=40,
            )

            # Post-process and validate
            summary = self._post_process_summary(response, content, target_length)

            return summary

        except Exception as e:
            logger.error(f"Abstractive summarization failed: {str(e)}")
            raise

    def _construct_summarization_prompt(self, content: ProcessedContent, target_length: int) -> str:
        """Construct optimized prompt for summarization"""

        # Determine content type specific instructions
        type_instructions = self._get_content_type_instructions(content.content_type)

        # Construct the main prompt
        prompt = f"""
Please create an accurate, informative summary of the following {content.content_type.value.replace('_', ' ')}.

REQUIREMENTS:
- Target length: approximately {target_length} words
- Maintain factual accuracy - do not add information not present in the source
- Preserve key entities, dates, and numbers exactly as stated
- Use neutral, objective tone
- Include the most important information first
- Ensure the summary is coherent and well-structured
- {type_instructions}

ARTICLE METADATA:
Title: {content.title or 'Not specified'}
Author: {content.author or 'Not specified'}
Publication: {content.source or 'Not specified'}
Published: {content.published_at.strftime('%Y-%m-%d') if content.published_at else 'Not specified'}

ARTICLE CONTENT:
{content.text[:4000]}  # Limit input to avoid token limits

SUMMARY:
"""

        return prompt.strip()

    def _get_content_type_instructions(self, content_type) -> str:
        """Get content type specific instructions"""
        instructions = {
            "news_article": "Focus on the 5 W's (who, what, when, where, why). Include key quotes and statistics.",
            "blog_post": "Capture the main argument or key insights. Include practical takeaways if applicable.",
            "academic_paper": "Focus on methodology, key findings, and conclusions. Use formal academic language.",
            "social_media": "Capture the main sentiment and key points. Keep it concise and engaging.",
            "product_description": "Focus on key features, benefits, and specifications. Highlight unique selling points.",
            "general": "Focus on the main ideas and key information. Maintain clarity and coherence.",
        }

        return instructions.get(content_type.value, instructions["general"])

    async def _generate_with_retry(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 500,
        top_p: float = 0.8,
        top_k: int = 40,
        max_retries: int = 3,
    ) -> str:
        """Generate text with retry logic"""

        for attempt in range(max_retries):
            try:
                response = self.model.predict(
                    prompt=prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    top_p=top_p,
                    top_k=top_k,
                )

                return response.text

            except gcp_exceptions.ResourceExhausted:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                    await asyncio.sleep(wait_time)
                else:
                    raise

            except gcp_exceptions.ServiceUnavailable:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Service unavailable, waiting {wait_time}s before retry {attempt + 1}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise

            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)

        raise RuntimeError("All generation attempts failed")

    def _post_process_summary(
        self, summary: str, content: ProcessedContent, target_length: int
    ) -> str:
        """Post-process and validate generated summary"""
        try:
            # Clean up the summary
            summary = summary.strip()

            # Remove any remaining prompt text
            if "SUMMARY:" in summary:
                summary = summary.split("SUMMARY:")[-1].strip()

            # Ensure proper sentence structure
            if not summary.endswith((".", "!", "?")):
                summary += "."

            # Validate length
            word_count = len(summary.split())
            if word_count > target_length * 1.5:  # Allow some flexibility
                # Truncate to target length
                words = summary.split()
                summary = " ".join(words[:target_length])
                if not summary.endswith("."):
                    summary += "."

            # Basic quality checks
            if len(summary.split()) < 10:
                raise ValueError("Generated summary too short")

            if summary.lower() == content.text.lower()[: len(summary)]:
                raise ValueError("Summary too similar to original text")

            return summary

        except Exception as e:
            logger.error(f"Summary post-processing failed: {str(e)}")
            raise

    async def generate_variants(
        self, content: ProcessedContent, target_lengths: List[int], style_variants: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate multiple summary variants with different styles"""

        if style_variants is None:
            style_variants = ["standard", "concise", "detailed"]

        variants = []

        for target_length in target_lengths:
            for style in style_variants:
                try:
                    # Adjust parameters based on style
                    temperature, max_tokens = self._get_style_parameters(style, target_length)

                    # Generate variant
                    summary = await self.summarize(
                        content=content,
                        target_length=target_length,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    variants.append(
                        {
                            "text": summary,
                            "style": style,
                            "target_length": target_length,
                            "actual_length": len(summary.split()),
                            "method": SummarizationMethod.ABSTRACTIVE,
                        }
                    )

                except Exception as e:
                    logger.error(f"Failed to generate {style} variant: {str(e)}")
                    continue

        return variants

    def _get_style_parameters(self, style: str, target_length: int) -> tuple:
        """Get temperature and max_tokens for different styles"""

        style_configs = {
            "standard": (0.2, target_length * 2),
            "concise": (0.1, target_length * 1.5),
            "detailed": (0.3, target_length * 2.5),
            "creative": (0.7, target_length * 2),
            "formal": (0.1, target_length * 2),
            "casual": (0.4, target_length * 2),
        }

        return style_configs.get(style, (0.2, target_length * 2))

    async def generate_with_custom_prompt(
        self, content: ProcessedContent, custom_prompt: str, target_length: int = 120
    ) -> str:
        """Generate summary with custom prompt"""
        try:
            # Use custom prompt
            full_prompt = f"{custom_prompt}\n\nContent: {content.text[:3000]}\n\nSummary:"

            response = await self._generate_with_retry(
                prompt=full_prompt, temperature=0.2, max_output_tokens=target_length * 2
            )

            return self._post_process_summary(response, content, target_length)

        except Exception as e:
            logger.error(f"Custom prompt summarization failed: {str(e)}")
            raise

    async def batch_summarize(
        self, contents: List[ProcessedContent], target_length: int = 120
    ) -> List[str]:
        """Batch summarization for multiple contents"""
        try:
            # Process in parallel with controlled concurrency
            tasks = []
            for content in contents:
                task = self.summarize(content, target_length)
                tasks.append(task)

            # Execute with semaphore to control concurrency
            semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

            async def limited_summarize(content):
                async with semaphore:
                    return await self.summarize(content, target_length)

            limited_tasks = [limited_summarize(content) for content in contents]
            summaries = await asyncio.gather(*limited_tasks, return_exceptions=True)

            # Handle exceptions
            results = []
            for i, summary in enumerate(summaries):
                if isinstance(summary, Exception):
                    logger.error(f"Batch item {i} failed: {str(summary)}")
                    results.append(f"Error: {str(summary)}")
                else:
                    results.append(summary)

            return results

        except Exception as e:
            logger.error(f"Batch summarization failed: {str(e)}")
            raise

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities"""
        return {
            "model_name": self.model_name,
            "project_id": self.project_id,
            "location": self.location,
            "max_tokens": 8192,
            "supported_languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
            "capabilities": [
                "abstractive_summarization",
                "multi_style_generation",
                "custom_prompts",
                "batch_processing",
                "multi_language",
            ],
        }

    async def get_status(self) -> Dict[str, Any]:
        """Get summarizer status"""
        return {
            "initialized": self._initialized,
            "model": self.model_name,
            "project": self.project_id,
            "location": self.location,
            "endpoint": self.endpoint,
        }
