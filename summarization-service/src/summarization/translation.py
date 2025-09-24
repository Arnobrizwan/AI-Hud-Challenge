"""
Multi-language Translation Service
Advanced translation capabilities for multi-language summarization
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import langdetect
import requests
from googletrans import Translator
from langdetect import DetectorFactory
from transformers import MarianMTModel, MarianTokenizer

from config.settings import settings

from .models import Language, ProcessedContent

logger = logging.getLogger(__name__)

# Set seed for consistent language detection
DetectorFactory.seed = 0


class TranslationService:
    """Advanced translation service for multi-language support"""

    def __init__(self):
        """Initialize the translation service"""
        self.google_translator = None
        self.marian_models = {}
        self.supported_languages = settings.SUPPORTED_LANGUAGES
        self._initialized = False

        # Language code mapping
        self.language_codes = {
            Language.ENGLISH: "en",
            Language.SPANISH: "es",
            Language.FRENCH: "fr",
            Language.GERMAN: "de",
            Language.ITALIAN: "it",
            Language.PORTUGUESE: "pt",
            Language.RUSSIAN: "ru",
            Language.CHINESE: "zh",
            Language.JAPANESE: "ja",
            Language.KOREAN: "ko",
        }

        # Reverse mapping
        self.code_to_language = {v: k for k, v in self.language_codes.items()}

    async def initialize(self) -> Dict[str, Any]:
        """Initialize translation models and services"""
        try:
            logger.info("Initializing translation service...")

            # Initialize Google Translate
            self.google_translator = Translator()

            # Initialize MarianMT models for key language pairs
            await self._initialize_marian_models()

            self._initialized = True
            logger.info("Translation service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize translation service: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
        """Clean up resources"""
        try:
            # Clean up MarianMT models
            for model_name, (model, tokenizer) in self.marian_models.items():
                del model
                del tokenizer

            self.marian_models.clear()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def detect_language(self, text: str) -> str:
        """Detect the language of the input text"""
        try:
            if not text or len(text.strip()) < 10:
                return "en"  # Default to English for very short text

            # Use langdetect for language detection
            detected_lang = langdetect.detect(text)

            # Validate detected language
            if detected_lang in self.supported_languages:
                return detected_lang
            else:
                logger.warning(
                    f"Unsupported language detected: {detected_lang}, defaulting to English"
                )
                return "en"

        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            return "en"  # Default to English on error

    async def translate(
            self,
            text: str,
            source_lang: str,
            target_lang: str) -> str:
        """Translate text from source language to target language"""

        if not self._initialized:
            raise RuntimeError("Translation service not initialized")

        try:
            # Check if translation is needed
            if source_lang == target_lang:
                return text

            # Validate languages
            if source_lang not in self.supported_languages:
                raise ValueError(f"Unsupported source language: {source_lang}")
            if target_lang not in self.supported_languages:
                raise ValueError(f"Unsupported target language: {target_lang}")

            # Try MarianMT first for better quality
            if f"{source_lang}-{target_lang}" in self.marian_models:
                return await self._translate_with_marian(text, source_lang, target_lang)
            else:
                return await self._translate_with_google(text, source_lang, target_lang)

        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return text  # Return original text on error

    async def _translate_with_marian(
            self,
            text: str,
            source_lang: str,
            target_lang: str) -> str:
        """Translate using MarianMT model"""
        try:
            model_key = f"{source_lang}-{target_lang}"
            model, tokenizer = self.marian_models[model_key]

            # Tokenize input
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512)

            # Generate translation
            with tokenizer.as_target_tokenizer():
                translated = model.generate(
                    **inputs, max_length=512, num_beams=4, early_stopping=True
                )

            # Decode translation
            translation = tokenizer.decode(
                translated[0], skip_special_tokens=True)

            return translation

        except Exception as e:
            logger.error(f"MarianMT translation failed: {str(e)}")
            # Fallback to Google Translate
            return await self._translate_with_google(text, source_lang, target_lang)

    async def _translate_with_google(
            self,
            text: str,
            source_lang: str,
            target_lang: str) -> str:
        """Translate using Google Translate"""
        try:
            # Use Google Translate
            result = self.google_translator.translate(
                text, src=source_lang, dest=target_lang)

            return result.text

        except Exception as e:
            logger.error(f"Google Translate failed: {str(e)}")
            return text  # Return original text on error

    async def _initialize_marian_models(self) -> Dict[str, Any]:
        """Initialize MarianMT models for key language pairs"""
        try:
            # Key language pairs for summarization
            language_pairs = [
                ("en", "es"),  # English to Spanish
                ("en", "fr"),  # English to French
                ("en", "de"),  # English to German
                ("es", "en"),  # Spanish to English
                ("fr", "en"),  # French to English
                ("de", "en"),  # German to English
            ]

            for source_lang, target_lang in language_pairs:
                try:
                    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

                    # Load model and tokenizer
                    tokenizer = MarianTokenizer.from_pretrained(model_name)
                    model = MarianMTModel.from_pretrained(model_name)

                    # Store model
                    model_key = f"{source_lang}-{target_lang}"
                    self.marian_models[model_key] = (model, tokenizer)

                    logger.info(f"Loaded MarianMT model: {model_name}")

                except Exception as e:
                    logger.warning(
                        f"Failed to load MarianMT model {model_name}: {str(e)}")
                    continue

            logger.info(
                f"Initialized {len(self.marian_models)} MarianMT models")

        except Exception as e:
            logger.error(f"Failed to initialize MarianMT models: {str(e)}")

    async def translate_content(
        self, content: ProcessedContent, target_language: Language
    ) -> ProcessedContent:
        """Translate entire content object to target language"""
        try:
            # Detect current language
            current_lang_code = await self.detect_language(content.text)
            target_lang_code = self.language_codes[target_language]

            # Translate if needed
            if current_lang_code != target_lang_code:
                # Translate main text
                translated_text = await self.translate(
                    content.text, current_lang_code, target_lang_code
                )

                # Translate title if present
                translated_title = None
                if content.title:
                    translated_title = await self.translate(
                        content.title, current_lang_code, target_lang_code
                    )

                # Create translated content
                translated_content = ProcessedContent(
                    text=translated_text,
                    title=translated_title,
                    author=content.author,  # Keep author name as is
                    source=content.source,  # Keep source as is
                    published_at=content.published_at,
                    language=target_language,
                    content_type=content.content_type,
                    metadata=content.metadata.copy(),
                )

                # Add translation metadata
                translated_content.metadata.update(
                    {
                        "original_language": current_lang_code,
                        "translated": True,
                        "translation_method": (
                            "marian"
                            if f"{current_lang_code}-{target_lang_code}" in self.marian_models
                            else "google"
                        ),
                    }
                )

                return translated_content
            else:
                # No translation needed
                return content

        except Exception as e:
            logger.error(f"Content translation failed: {str(e)}")
            return content  # Return original content on error

    async def batch_translate(
        self, texts: List[str], source_lang: str, target_lang: str
    ) -> List[str]:
        """Translate multiple texts in batch"""
        try:
            if source_lang == target_lang:
                return texts

            # Process in parallel with controlled concurrency
            semaphore = asyncio.Semaphore(5)  # Limit concurrent translations

            async def translate_single(text) -> Dict[str, Any]:
                async with semaphore:
                    return await self.translate(text, source_lang, target_lang)

            # Execute translations
            tasks = [translate_single(text) for text in texts]
            translations = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            results = []
            for i, translation in enumerate(translations):
                if isinstance(translation, Exception):
                    logger.error(f"Translation {i} failed: {str(translation)}")
                    results.append(texts[i])  # Return original text
                else:
                    results.append(translation)

            return results

        except Exception as e:
            logger.error(f"Batch translation failed: {str(e)}")
            return texts  # Return original texts on error

    async def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return self.supported_languages.copy()

    async def get_translation_quality(
            self,
            original: str,
            translation: str,
            source_lang: str,
            target_lang: str) -> float:
        """Estimate translation quality"""
        try:
            # Simple quality estimation based on length preservation
            original_length = len(original.split())
            translation_length = len(translation.split())

            if original_length == 0:
                return 0.0

            # Length ratio (should be close to 1.0 for good translation)
            length_ratio = translation_length / original_length

            # Quality score based on length preservation
            if 0.7 <= length_ratio <= 1.3:
                quality_score = 1.0
            elif 0.5 <= length_ratio <= 1.5:
                quality_score = 0.8
            elif 0.3 <= length_ratio <= 2.0:
                quality_score = 0.6
            else:
                quality_score = 0.4

            return quality_score

        except Exception as e:
            logger.error(f"Translation quality assessment failed: {str(e)}")
            return 0.5

    async def get_status(self) -> Dict[str, Any]:
        """Get translation service status"""
        return {
            "initialized": self._initialized,
            "google_translator_available": self.google_translator is not None,
            "marian_models_loaded": len(self.marian_models),
            "supported_languages": self.supported_languages,
            "available_models": list(self.marian_models.keys()),
        }
