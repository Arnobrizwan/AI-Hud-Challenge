"""
HTTP client utilities with connection pooling, retries, and rate limiting.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
import backoff
import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class HTTPResponse:
    """HTTP response wrapper."""

    status_code: int
    headers: Dict[str, str]
    content: bytes
    text: str
    url: str
    elapsed: float
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    content_type: Optional[str] = None
    content_length: Optional[int] = None


class RateLimiter:
    """Rate limiter for HTTP requests."""

    def __init__(self, rate_limit: int = 60, backoff_factor: float = 2.0):
        self.rate_limit = rate_limit  # requests per minute
        self.backoff_factor = backoff_factor
        self.requests: List[float] = []
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        async with self.lock:
            now = time.time()

            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]

            # If we're at the rate limit, wait
            if len(self.requests) >= self.rate_limit:
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    await asyncio.sleep(sleep_time)
                    # Clean up old requests after sleep
                    now = time.time()
                    self.requests = [req_time for req_time in self.requests if now - req_time < 60]

            # Record this request
            self.requests.append(now)


class HTTPClient:
    """Async HTTP client with connection pooling and retries."""

    def __init__(
        self,
        timeout: int = 30,
        max_connections: int = 100,
        rate_limit: int = 60,
        user_agent: str = None,
        default_headers: Dict[str, str] = None,
    ):
        self.timeout = timeout
        self.max_connections = max_connections
        self.rate_limiter = RateLimiter(rate_limit)
        self.user_agent = user_agent or settings.USER_AGENT
        self.default_headers = default_headers or {}

        # Connection limits
        connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=10,
            keepalive_timeout=settings.HTTP_KEEPALIVE_TIMEOUT,
            enable_cleanup_closed=True,
        )

        # Default headers
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            **self.default_headers,
        }

        # Create session
        self.session = aiohttp.ClientSession(
            connector=connector, timeout=aiohttp.ClientTimeout(total=timeout), headers=headers
        )

        # Domain-specific rate limiters
        self.domain_limiters: Dict[str, RateLimiter] = {}

    async def close(self):
        """Close the HTTP client session."""
        await self.session.close()

    def _get_domain_limiter(self, url: str) -> RateLimiter:
        """Get rate limiter for specific domain."""
        domain = urlparse(url).netloc
        if domain not in self.domain_limiters:
            self.domain_limiters[domain] = RateLimiter(
                rate_limit=settings.DEFAULT_RATE_LIMIT,
                backoff_factor=settings.RATE_LIMIT_BACKOFF_FACTOR,
            )
        return self.domain_limiters[domain]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        allow_redirects: bool = True,
        timeout: Optional[int] = None,
        etag: Optional[str] = None,
        last_modified: Optional[str] = None,
    ) -> HTTPResponse:
        """Make GET request with retries and rate limiting."""
        # Rate limiting
        domain_limiter = self._get_domain_limiter(url)
        await domain_limiter.acquire()

        # Prepare headers
        request_headers = {}
        if headers:
            request_headers.update(headers)

        # Add conditional request headers
        if etag:
            request_headers["If-None-Match"] = etag
        if last_modified:
            request_headers["If-Modified-Since"] = last_modified

        # Make request
        start_time = time.time()
        timeout_val = timeout or self.timeout

        try:
            async with self.session.get(
                url,
                headers=request_headers,
                params=params,
                allow_redirects=allow_redirects,
                timeout=aiohttp.ClientTimeout(total=timeout_val),
            ) as response:
                content = await response.read()
                text = content.decode("utf-8", errors="ignore")
                elapsed = time.time() - start_time

                # Extract headers
                response_headers = dict(response.headers)
                etag = response_headers.get("ETag")
                last_modified = response_headers.get("Last-Modified")
                content_type = response_headers.get("Content-Type")
                content_length = response_headers.get("Content-Length")

                if content_length:
                    content_length = int(content_length)

                return HTTPResponse(
                    status_code=response.status,
                    headers=response_headers,
                    content=content,
                    text=text,
                    url=str(response.url),
                    elapsed=elapsed,
                    etag=etag,
                    last_modified=last_modified,
                    content_type=content_type,
                    content_length=content_length,
                )

        except asyncio.TimeoutError:
            logger.warning(f"Request timeout for {url}")
            raise
        except aiohttp.ClientError as e:
            logger.warning(f"Request error for {url}: {e}")
            raise

    async def post(
        self,
        url: str,
        data: Optional[Any] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> HTTPResponse:
        """Make POST request."""
        # Rate limiting
        domain_limiter = self._get_domain_limiter(url)
        await domain_limiter.acquire()

        # Prepare headers
        request_headers = {}
        if headers:
            request_headers.update(headers)

        # Make request
        start_time = time.time()
        timeout_val = timeout or self.timeout

        try:
            async with self.session.post(
                url,
                data=data,
                json=json,
                headers=request_headers,
                timeout=aiohttp.ClientTimeout(total=timeout_val),
            ) as response:
                content = await response.read()
                text = content.decode("utf-8", errors="ignore")
                elapsed = time.time() - start_time

                # Extract headers
                response_headers = dict(response.headers)
                etag = response_headers.get("ETag")
                last_modified = response_headers.get("Last-Modified")
                content_type = response_headers.get("Content-Type")
                content_length = response_headers.get("Content-Length")

                if content_length:
                    content_length = int(content_length)

                return HTTPResponse(
                    status_code=response.status,
                    headers=response_headers,
                    content=content,
                    text=text,
                    url=str(response.url),
                    elapsed=elapsed,
                    etag=etag,
                    last_modified=last_modified,
                    content_type=content_type,
                    content_length=content_length,
                )

        except asyncio.TimeoutError:
            logger.warning(f"Request timeout for {url}")
            raise
        except aiohttp.ClientError as e:
            logger.warning(f"Request error for {url}: {e}")
            raise

    async def head(
        self, url: str, headers: Optional[Dict[str, str]] = None, timeout: Optional[int] = None
    ) -> HTTPResponse:
        """Make HEAD request to check resource without downloading."""
        # Rate limiting
        domain_limiter = self._get_domain_limiter(url)
        await domain_limiter.acquire()

        # Prepare headers
        request_headers = {}
        if headers:
            request_headers.update(headers)

        # Make request
        start_time = time.time()
        timeout_val = timeout or self.timeout

        try:
            async with self.session.head(
                url, headers=request_headers, timeout=aiohttp.ClientTimeout(total=timeout_val)
            ) as response:
                elapsed = time.time() - start_time

                # Extract headers
                response_headers = dict(response.headers)
                etag = response_headers.get("ETag")
                last_modified = response_headers.get("Last-Modified")
                content_type = response_headers.get("Content-Type")
                content_length = response_headers.get("Content-Length")

                if content_length:
                    content_length = int(content_length)

                return HTTPResponse(
                    status_code=response.status,
                    headers=response_headers,
                    content=b"",
                    text="",
                    url=str(response.url),
                    elapsed=elapsed,
                    etag=etag,
                    last_modified=last_modified,
                    content_type=content_type,
                    content_length=content_length,
                )

        except asyncio.TimeoutError:
            logger.warning(f"Request timeout for {url}")
            raise
        except aiohttp.ClientError as e:
            logger.warning(f"Request error for {url}: {e}")
            raise


class RobotsTxtChecker:
    """Check robots.txt compliance."""

    def __init__(self, http_client: HTTPClient):
        self.http_client = http_client
        self.robots_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry: Dict[str, float] = {}
        self.cache_duration = 3600  # 1 hour

    async def can_fetch(self, url: str, user_agent: str = None) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        from urllib.parse import urljoin, urlparse

        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        robots_url = urljoin(base_url, "/robots.txt")

        # Check cache
        if base_url in self.robots_cache:
            if time.time() < self.cache_expiry.get(base_url, 0):
                return self._check_robots_rules(
                    self.robots_cache[base_url], url, user_agent or settings.USER_AGENT
                )

        try:
            # Fetch robots.txt
            response = await self.http_client.get(robots_url, timeout=10)

            if response.status_code == 200:
                robots_content = response.text
                rules = self._parse_robots_txt(robots_content)

                # Cache the rules
                self.robots_cache[base_url] = rules
                self.cache_expiry[base_url] = time.time() + self.cache_duration

                return self._check_robots_rules(rules, url, user_agent or settings.USER_AGENT)
            else:
                # If robots.txt doesn't exist or is inaccessible, allow
                return True

        except Exception as e:
            logger.warning(f"Error checking robots.txt for {base_url}: {e}")
            # If we can't check robots.txt, allow the request
            return True

    def _parse_robots_txt(self, content: str) -> Dict[str, Any]:
        """Parse robots.txt content."""
        rules = {}
        current_user_agent = None

        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.lower().startswith("user-agent:"):
                current_user_agent = line.split(":", 1)[1].strip()
                if current_user_agent not in rules:
                    rules[current_user_agent] = {"allow": [], "disallow": []}
            elif line.lower().startswith("disallow:"):
                if current_user_agent:
                    path = line.split(":", 1)[1].strip()
                    rules[current_user_agent]["disallow"].append(path)
            elif line.lower().startswith("allow:"):
                if current_user_agent:
                    path = line.split(":", 1)[1].strip()
                    rules[current_user_agent]["allow"].append(path)

        return rules

    def _check_robots_rules(self, rules: Dict[str, Any], url: str, user_agent: str) -> bool:
        """Check if URL is allowed by robots.txt rules."""
        from urllib.parse import urlparse

        parsed_url = urlparse(url)
        path = parsed_url.path

        # Check specific user agent first, then wildcard
        for ua in [user_agent, "*"]:
            if ua in rules:
                ua_rules = rules[ua]

                # Check disallow rules
                for disallow_path in ua_rules.get("disallow", []):
                    if disallow_path and path.startswith(disallow_path):
                        # Check if there's a more specific allow rule
                        allowed = False
                        for allow_path in ua_rules.get("allow", []):
                            if allow_path and path.startswith(allow_path):
                                if len(allow_path) > len(disallow_path):
                                    allowed = True
                                    break

                        if not allowed:
                            return False

        return True


# Global HTTP client instance
http_client = HTTPClient(
    timeout=settings.HTTP_TIMEOUT,
    max_connections=settings.HTTP_MAX_CONNECTIONS,
    rate_limit=settings.DEFAULT_RATE_LIMIT,
    user_agent=settings.USER_AGENT,
)

# Global robots.txt checker
robots_checker = RobotsTxtChecker(http_client)
