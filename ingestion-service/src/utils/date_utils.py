"""
Date and time utilities for content processing.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import List, Optional

import pytz
from dateutil import parser as date_parser

logger = logging.getLogger(__name__)


class DateUtils:
    """Date and time processing utilities."""

    def __init__(self, default_timezone: str = "UTC"):
        self.default_timezone = pytz.timezone(default_timezone)
        self.common_timezones = {
            "UTC": pytz.UTC,
            "EST": pytz.timezone("US/Eastern"),
            "PST": pytz.timezone("US/Pacific"),
            "GMT": pytz.timezone("GMT"),
            "CET": pytz.timezone("Europe/Paris"),
            "JST": pytz.timezone("Asia/Tokyo"),
        }

        # Common date formats
        self.date_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y",
            "%B %d, %Y %H:%M:%S",
            "%B %d, %Y %H:%M",
            "%B %d, %Y",
            "%b %d, %Y %H:%M:%S",
            "%b %d, %Y %H:%M",
            "%b %d, %Y",
            "%d %B %Y %H:%M:%S",
            "%d %B %Y %H:%M",
            "%d %B %Y",
            "%d %b %Y %H:%M:%S",
            "%d %b %Y %H:%M",
            "%d %b %Y",
        ]

        # Relative time patterns
        self.relative_patterns = [
            (r"(\d+)\s+minutes?\s+ago", "minutes"),
            (r"(\d+)\s+hours?\s+ago", "hours"),
            (r"(\d+)\s+days?\s+ago", "days"),
            (r"(\d+)\s+weeks?\s+ago", "weeks"),
            (r"(\d+)\s+months?\s+ago", "months"),
            (r"(\d+)\s+years?\s+ago", "years"),
            (r"(\d+)\s+min\s+ago", "minutes"),
            (r"(\d+)\s+hr\s+ago", "hours"),
            (r"(\d+)\s+day\s+ago", "days"),
            (r"(\d+)\s+wk\s+ago", "weeks"),
            (r"(\d+)\s+mo\s+ago", "months"),
            (r"(\d+)\s+yr\s+ago", "years"),
            (r"(\d+)m\s+ago", "minutes"),
            (r"(\d+)h\s+ago", "hours"),
            (r"(\d+)d\s+ago", "days"),
            (r"(\d+)w\s+ago", "weeks"),
            (r"(\d+)mo\s+ago", "months"),
            (r"(\d+)y\s+ago", "years"),
        ]

    def parse_date(self, date_string: str, timezone: str = None) -> Optional[datetime]:
        """Parse date string with multiple fallback strategies."""
        if not date_string or not isinstance(date_string, str):
            return None

        date_string = date_string.strip()
        if not date_string:
            return None

        # Try relative time patterns first
        relative_date = self._parse_relative_date(date_string)
        if relative_date:
            return relative_date

        # Try dateutil parser
        try:
            parsed_date = date_parser.parse(date_string, fuzzy=True)
            if parsed_date:
                # Convert to UTC if timezone is specified
                if timezone:
                    tz = self._get_timezone(timezone)
                    if parsed_date.tzinfo is None:
                        parsed_date = tz.localize(parsed_date)
        else:
                        parsed_date = parsed_date.astimezone(tz)
        else:
                    # If no timezone info, assume UTC
                    if parsed_date.tzinfo is None:
                        parsed_date = self.default_timezone.localize(parsed_date)

                return parsed_date.astimezone(pytz.UTC)
        except Exception as e:
            logger.debug(f"Dateutil parsing failed for '{date_string}': {e}")

        # Try manual format parsing
        for fmt in self.date_formats:
            try:
                parsed_date = datetime.strptime(date_string, fmt)
                if timezone:
                    tz = self._get_timezone(timezone)
                    parsed_date = tz.localize(parsed_date)
            else:
                    parsed_date = self.default_timezone.localize(parsed_date)

                return parsed_date.astimezone(pytz.UTC)
            except ValueError:
                continue

        # Try cleaning the date string
        cleaned_date = self._clean_date_string(date_string)
        if cleaned_date != date_string:
            return self.parse_date(cleaned_date, timezone)

        logger.warning(f"Could not parse date: '{date_string}'")
        return None

    def _parse_relative_date(self, date_string: str) -> Optional[datetime]:
        """Parse relative date strings like '2 hours ago'."""
        now = datetime.now(pytz.UTC)
        date_string_lower = date_string.lower()

        for pattern, unit in self.relative_patterns:
            match = re.search(pattern, date_string_lower)
            if match:
                try:
                    value = int(match.group(1))

                    if unit == "minutes":
                        return now - timedelta(minutes=value)
                    elif unit == "hours":
                        return now - timedelta(hours=value)
                    elif unit == "days":
                        return now - timedelta(days=value)
                    elif unit == "weeks":
                        return now - timedelta(weeks=value)
                    elif unit == "months":
                        return now - timedelta(days=value * 30)  # Approximate
                    elif unit == "years":
                        return now - timedelta(days=value * 365)  # Approximate
                except ValueError:
                    continue

        return None

    def _clean_date_string(self, date_string: str) -> str:
        """Clean date string by removing common issues."""
        # Remove extra whitespace
        cleaned = re.sub(r"\s+", " ", date_string.strip())

        # Remove common prefixes
        prefixes = [
            "Published:",
            "Posted:",
            "Updated:",
            "Created:",
            "Date:",
            "Time:",
            "Posted on",
            "Published on",
            "Updated on",
            "Created on",
        ]

        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix) :].strip()
                break

        # Remove timezone abbreviations that might confuse parsers
        tz_abbrevs = ["EST", "PST", "CST", "MST", "EDT", "PDT", "CDT", "MDT", "GMT", "UTC"]
        for tz in tz_abbrevs:
            cleaned = re.sub(rf"\b{tz}\b", "", cleaned, flags=re.IGNORECASE)

        # Clean up extra spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    def _get_timezone(self, timezone_str: str) -> pytz.BaseTzInfo:
        """Get timezone object from string."""
        if timezone_str in self.common_timezones:
            return self.common_timezones[timezone_str]

        try:
            return pytz.timezone(timezone_str)
        except pytz.UnknownTimeZoneError:
            logger.warning(f"Unknown timezone: {timezone_str}, using UTC")
            return pytz.UTC

    def normalize_date(self, date_obj: datetime, timezone: str = None) -> datetime:
        """Normalize datetime object to UTC."""
        if not isinstance(date_obj, datetime):
            return None

        # If no timezone info, assume the specified timezone or default
        if date_obj.tzinfo is None:
            if timezone:
                tz = self._get_timezone(timezone)
                date_obj = tz.localize(date_obj)
            else:
                date_obj = self.default_timezone.localize(date_obj)

        # Convert to UTC
        return date_obj.astimezone(pytz.UTC)

    def is_valid_date(self, date_obj: datetime) -> bool:
        """Check if date is valid and reasonable."""
        if not isinstance(date_obj, datetime):
            return False

        now = datetime.now(pytz.UTC)

        # Check if date is too far in the past (more than 10 years)
        if date_obj < now - timedelta(days=3650):
            return False

        # Check if date is too far in the future (more than 1 year)
        if date_obj > now + timedelta(days=365):
            return False

        return True

    def format_date(self, date_obj: datetime, format_str: str = None) -> str:
        """Format datetime object to string."""
        if not isinstance(date_obj, datetime):
            return ""

        if format_str is None:
            format_str = "%Y-%m-%d %H:%M:%S UTC"

        # Convert to UTC for consistent formatting
        utc_date = date_obj.astimezone(pytz.UTC)
        return utc_date.strftime(format_str)

    def get_date_range(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Get list of dates between start and end date."""
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            return []

        dates = []
        current = start_date.date()
        end = end_date.date()

        while current <= end:
            dates.append(datetime.combine(current, datetime.min.time()).replace(tzinfo=pytz.UTC))
            current += timedelta(days=1)

        return dates

    def get_relative_time(self, date_obj: datetime) -> str:
        """Get human-readable relative time."""
        if not isinstance(date_obj, datetime):
            return "Unknown"

        now = datetime.now(pytz.UTC)
        if date_obj.tzinfo is None:
            date_obj = self.default_timezone.localize(date_obj)

        date_obj = date_obj.astimezone(pytz.UTC)
        diff = now - date_obj

        if diff.days > 0:
            if diff.days == 1:
                return "1 day ago"
            elif diff.days < 7:
                return f"{diff.days} days ago"
            elif diff.days < 30:
                weeks = diff.days // 7
                return f"{weeks} week{'s' if weeks > 1 else ''} ago"
            elif diff.days < 365:
                months = diff.days // 30
                return f"{months} month{'s' if months > 1 else ''} ago"
            else:
                years = diff.days // 365
                return f"{years} year{'s' if years > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "Just now"

    def extract_dates_from_text(self, text: str) -> List[datetime]:
        """Extract all dates from text content."""
        if not text:
            return []

        dates = []

        # Common date patterns
        date_patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",  # YYYY-MM-DD
            r"\b\d{2}/\d{2}/\d{4}\b",  # MM/DD/YYYY
            r"\b\d{2}-\d{2}-\d{4}\b",  # MM-DD-YYYY
            # DD Mon YYYY
            r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b",
            # Mon DD, YYYY
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                parsed_date = self.parse_date(match)
                if parsed_date and self.is_valid_date(parsed_date):
                    dates.append(parsed_date)

        return sorted(set(dates))  # Remove duplicates and sort

    def get_timezone_offset(self, timezone_str: str) -> int:
        """Get timezone offset in minutes from UTC."""
        try:
            tz = self._get_timezone(timezone_str)
            now = datetime.now(tz)
            offset = now.utcoffset()
            return int(offset.total_seconds() / 60)
        except Exception:
            return 0


# Global date utils instance
date_utils = DateUtils()
