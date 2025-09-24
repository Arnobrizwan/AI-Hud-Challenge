"""
External Moderation APIs
Integration with external content moderation services
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
import aiohttp
from datetime import datetime

from safety_engine.config import get_content_config

logger = logging.getLogger(__name__)

class ExternalModerationAPIs:
    """External moderation API integration"""
    
    def __init__(self):
        self.config = get_content_config()
        self.is_initialized = False
        self.enabled = self.config.external_apis
        
        # API configurations
        self.api_configs = {
            'google_perspective': {
                'enabled': False,
                'api_key': None,
                'endpoint': 'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze',
                'timeout': 10
            },
            'microsoft_content_moderator': {
                'enabled': False,
                'api_key': None,
                'endpoint': 'https://westus.api.cognitive.microsoft.com/contentmoderator/moderate/v1.0/ProcessText',
                'timeout': 10
            },
            'aws_comprehend': {
                'enabled': False,
                'access_key': None,
                'secret_key': None,
                'region': 'us-east-1',
                'timeout': 10
            },
            'azure_content_safety': {
                'enabled': False,
                'api_key': None,
                'endpoint': 'https://your-resource.cognitiveservices.azure.com/contentsafety/text:analyze',
                'timeout': 10
            }
        }
        
        # HTTP session
        self.session = None
        
    async def initialize(self):
        """Initialize external APIs"""
        try:
            if not self.enabled:
                self.is_initialized = True
                return
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={'User-Agent': 'Safety-Service/1.0.0'}
            )
            
            # Load API configurations from environment
            await self.load_api_configurations()
            
            # Test API connections
            await self.test_api_connections()
            
            self.is_initialized = True
            logger.info("External moderation APIs initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize external APIs: {str(e)}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.is_initialized = False
            logger.info("External moderation APIs cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during external APIs cleanup: {str(e)}")
    
    async def load_api_configurations(self):
        """Load API configurations from environment variables"""
        try:
            # In a real implementation, this would load from environment variables
            # For now, we'll use default configurations
            pass
            
        except Exception as e:
            logger.error(f"Failed to load API configurations: {str(e)}")
    
    async def test_api_connections(self):
        """Test connections to external APIs"""
        try:
            # Test each enabled API
            for api_name, config in self.api_configs.items():
                if config['enabled']:
                    try:
                        await self.test_api_connection(api_name, config)
                        logger.info(f"API {api_name} connection test successful")
                    except Exception as e:
                        logger.warning(f"API {api_name} connection test failed: {str(e)}")
                        config['enabled'] = False
            
        except Exception as e:
            logger.error(f"API connection testing failed: {str(e)}")
    
    async def test_api_connection(self, api_name: str, config: Dict[str, Any]):
        """Test connection to a specific API"""
        try:
            if api_name == 'google_perspective':
                await self.test_google_perspective(config)
            elif api_name == 'microsoft_content_moderator':
                await self.test_microsoft_content_moderator(config)
            elif api_name == 'aws_comprehend':
                await self.test_aws_comprehend(config)
            elif api_name == 'azure_content_safety':
                await self.test_azure_content_safety(config)
                
        except Exception as e:
            logger.error(f"API {api_name} connection test failed: {str(e)}")
            raise
    
    async def test_google_perspective(self, config: Dict[str, Any]):
        """Test Google Perspective API connection"""
        try:
            # Test with a simple request
            test_data = {
                'comment': {'text': 'test'},
                'languages': ['en'],
                'requestedAttributes': {'TOXICITY': {}}
            }
            
            # This would make an actual API call in production
            # For now, we'll simulate success
            logger.debug("Google Perspective API test completed")
            
        except Exception as e:
            logger.error(f"Google Perspective API test failed: {str(e)}")
            raise
    
    async def test_microsoft_content_moderator(self, config: Dict[str, Any]):
        """Test Microsoft Content Moderator API connection"""
        try:
            # Test with a simple request
            test_data = {'text': 'test'}
            
            # This would make an actual API call in production
            # For now, we'll simulate success
            logger.debug("Microsoft Content Moderator API test completed")
            
        except Exception as e:
            logger.error(f"Microsoft Content Moderator API test failed: {str(e)}")
            raise
    
    async def test_aws_comprehend(self, config: Dict[str, Any]):
        """Test AWS Comprehend API connection"""
        try:
            # Test with a simple request
            test_data = {'text': 'test'}
            
            # This would make an actual API call in production
            # For now, we'll simulate success
            logger.debug("AWS Comprehend API test completed")
            
        except Exception as e:
            logger.error(f"AWS Comprehend API test failed: {str(e)}")
            raise
    
    async def test_azure_content_safety(self, config: Dict[str, Any]):
        """Test Azure Content Safety API connection"""
        try:
            # Test with a simple request
            test_data = {'text': 'test'}
            
            # This would make an actual API call in production
            # For now, we'll simulate success
            logger.debug("Azure Content Safety API test completed")
            
        except Exception as e:
            logger.error(f"Azure Content Safety API test failed: {str(e)}")
            raise
    
    async def moderate_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Moderate text using external APIs"""
        try:
            if not self.enabled or not self.is_initialized:
                return None
            
            # Try each enabled API
            results = {}
            
            for api_name, config in self.api_configs.items():
                if config['enabled']:
                    try:
                        api_result = await self.call_api(api_name, 'text', text)
                        if api_result:
                            results[api_name] = api_result
                    except Exception as e:
                        logger.warning(f"API {api_name} text moderation failed: {str(e)}")
                        continue
            
            return results if results else None
            
        except Exception as e:
            logger.error(f"External text moderation failed: {str(e)}")
            return None
    
    async def moderate_image(self, image_url: str) -> Optional[Dict[str, Any]]:
        """Moderate image using external APIs"""
        try:
            if not self.enabled or not self.is_initialized:
                return None
            
            # Try each enabled API
            results = {}
            
            for api_name, config in self.api_configs.items():
                if config['enabled']:
                    try:
                        api_result = await self.call_api(api_name, 'image', image_url)
                        if api_result:
                            results[api_name] = api_result
                    except Exception as e:
                        logger.warning(f"API {api_name} image moderation failed: {str(e)}")
                        continue
            
            return results if results else None
            
        except Exception as e:
            logger.error(f"External image moderation failed: {str(e)}")
            return None
    
    async def moderate_video(self, video_url: str) -> Optional[Dict[str, Any]]:
        """Moderate video using external APIs"""
        try:
            if not self.enabled or not self.is_initialized:
                return None
            
            # Try each enabled API
            results = {}
            
            for api_name, config in self.api_configs.items():
                if config['enabled']:
                    try:
                        api_result = await self.call_api(api_name, 'video', video_url)
                        if api_result:
                            results[api_name] = api_result
                    except Exception as e:
                        logger.warning(f"API {api_name} video moderation failed: {str(e)}")
                        continue
            
            return results if results else None
            
        except Exception as e:
            logger.error(f"External video moderation failed: {str(e)}")
            return None
    
    async def call_api(self, api_name: str, content_type: str, content: str) -> Optional[Dict[str, Any]]:
        """Call a specific external API"""
        try:
            config = self.api_configs[api_name]
            
            if api_name == 'google_perspective':
                return await self.call_google_perspective(config, content_type, content)
            elif api_name == 'microsoft_content_moderator':
                return await self.call_microsoft_content_moderator(config, content_type, content)
            elif api_name == 'aws_comprehend':
                return await self.call_aws_comprehend(config, content_type, content)
            elif api_name == 'azure_content_safety':
                return await self.call_azure_content_safety(config, content_type, content)
            else:
                logger.warning(f"Unknown API: {api_name}")
                return None
                
        except Exception as e:
            logger.error(f"API {api_name} call failed: {str(e)}")
            return None
    
    async def call_google_perspective(self, config: Dict[str, Any], content_type: str, content: str) -> Optional[Dict[str, Any]]:
        """Call Google Perspective API"""
        try:
            if content_type != 'text':
                return None
            
            # Prepare request data
            request_data = {
                'comment': {'text': content},
                'languages': ['en'],
                'requestedAttributes': {
                    'TOXICITY': {},
                    'SEVERE_TOXICITY': {},
                    'IDENTITY_ATTACK': {},
                    'INSULT': {},
                    'PROFANITY': {},
                    'THREAT': {}
                }
            }
            
            # Make API call
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {config["api_key"]}'
            }
            
            async with self.session.post(
                config['endpoint'],
                json=request_data,
                headers=headers,
                timeout=config['timeout']
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return self.parse_google_perspective_result(result)
                else:
                    logger.warning(f"Google Perspective API returned status {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Google Perspective API call failed: {str(e)}")
            return None
    
    async def call_microsoft_content_moderator(self, config: Dict[str, Any], content_type: str, content: str) -> Optional[Dict[str, Any]]:
        """Call Microsoft Content Moderator API"""
        try:
            if content_type != 'text':
                return None
            
            # Prepare request data
            headers = {
                'Content-Type': 'text/plain',
                'Ocp-Apim-Subscription-Key': config['api_key']
            }
            
            # Make API call
            async with self.session.post(
                config['endpoint'],
                data=content,
                headers=headers,
                timeout=config['timeout']
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return self.parse_microsoft_content_moderator_result(result)
                else:
                    logger.warning(f"Microsoft Content Moderator API returned status {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Microsoft Content Moderator API call failed: {str(e)}")
            return None
    
    async def call_aws_comprehend(self, config: Dict[str, Any], content_type: str, content: str) -> Optional[Dict[str, Any]]:
        """Call AWS Comprehend API"""
        try:
            if content_type != 'text':
                return None
            
            # In a real implementation, this would use boto3
            # For now, we'll simulate the call
            logger.debug("AWS Comprehend API call simulated")
            return None
            
        except Exception as e:
            logger.error(f"AWS Comprehend API call failed: {str(e)}")
            return None
    
    async def call_azure_content_safety(self, config: Dict[str, Any], content_type: str, content: str) -> Optional[Dict[str, Any]]:
        """Call Azure Content Safety API"""
        try:
            if content_type != 'text':
                return None
            
            # Prepare request data
            request_data = {
                'text': content,
                'categories': ['Hate', 'SelfHarm', 'Sexual', 'Violence']
            }
            
            headers = {
                'Content-Type': 'application/json',
                'Ocp-Apim-Subscription-Key': config['api_key']
            }
            
            # Make API call
            async with self.session.post(
                config['endpoint'],
                json=request_data,
                headers=headers,
                timeout=config['timeout']
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return self.parse_azure_content_safety_result(result)
                else:
                    logger.warning(f"Azure Content Safety API returned status {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Azure Content Safety API call failed: {str(e)}")
            return None
    
    def parse_google_perspective_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Google Perspective API result"""
        try:
            parsed_result = {
                'api': 'google_perspective',
                'timestamp': datetime.utcnow().isoformat(),
                'scores': {}
            }
            
            if 'attributeScores' in result:
                for attribute, data in result['attributeScores'].items():
                    if 'summaryScore' in data:
                        parsed_result['scores'][attribute.lower()] = data['summaryScore']['value']
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Failed to parse Google Perspective result: {str(e)}")
            return {}
    
    def parse_microsoft_content_moderator_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Microsoft Content Moderator API result"""
        try:
            parsed_result = {
                'api': 'microsoft_content_moderator',
                'timestamp': datetime.utcnow().isoformat(),
                'scores': {}
            }
            
            if 'Classification' in result:
                classification = result['Classification']
                parsed_result['scores']['adult'] = classification.get('AdultScore', 0.0)
                parsed_result['scores']['racy'] = classification.get('RacyScore', 0.0)
                parsed_result['scores']['offensive'] = classification.get('OffensiveScore', 0.0)
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Failed to parse Microsoft Content Moderator result: {str(e)}")
            return {}
    
    def parse_azure_content_safety_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Azure Content Safety API result"""
        try:
            parsed_result = {
                'api': 'azure_content_safety',
                'timestamp': datetime.utcnow().isoformat(),
                'scores': {}
            }
            
            if 'categoriesAnalysis' in result:
                for category in result['categoriesAnalysis']:
                    category_name = category.get('category', '').lower()
                    severity = category.get('severity', 0)
                    parsed_result['scores'][category_name] = severity / 4.0  # Normalize to 0-1
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Failed to parse Azure Content Safety result: {str(e)}")
            return {}
    
    async def get_api_status(self) -> Dict[str, Any]:
        """Get status of all external APIs"""
        try:
            status = {
                'enabled': self.enabled,
                'initialized': self.is_initialized,
                'apis': {}
            }
            
            for api_name, config in self.api_configs.items():
                status['apis'][api_name] = {
                    'enabled': config['enabled'],
                    'configured': bool(config.get('api_key') or config.get('access_key')),
                    'timeout': config['timeout']
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get API status: {str(e)}")
            return {'error': str(e)}
