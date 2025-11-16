# File: src/scraping/scraper_utils.py
"""
Utilities for web scraping, including polite session management and parsing.
"""

import time
import logging
from typing import Optional
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup, element

from src.utils import get_logger

logger = get_logger(__name__)

# Standard headers to mimic a real browser
DEFAULT_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/91.0.4472.124 Safari/537.36'),
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept': ('text/html,application/xhtml+xml,application/xml;q=0.9,'
               'image/webp,image/apng,*/*;q=0.8'),
    'Connection': 'keep-alive',
}

def create_polite_session(
    retries: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: tuple = (500, 502, 503, 504),
) -> requests.Session:
    """
    Creates a requests.Session with automatic retries and default headers.
    """
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    
    retry_strategy = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    return session

def polite_get(
    session: requests.Session,
    url: str,
    delay_seconds: float = 1.0,
    timeout: int = 15,
) -> Optional[BeautifulSoup]:
    """
    Performs a GET request using the provided session, with a delay.
    Returns a BeautifulSoup object or None on failure.
    """
    try:
        # Respectful delay
        time.sleep(delay_seconds)
        
        response = session.get(url, timeout=timeout)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        
        soup = BeautifulSoup(response.text, 'lxml')
        return soup
    
    except requests.exceptions.HTTPError as e:
        logger.warning(f"HTTP Error {e.response.status_code} for {url}: {e}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection Error for {url}: {e}")
    except requests.exceptions.Timeout as e:
        logger.warning(f"Timeout for {url}: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred during request for {url}: {e}")
    
    return None

def clean_text(text: Optional[str]) -> Optional[str]:
    """Strips whitespace and special characters from text."""
    if text is None:
        return None
    return text.strip().replace('\n', ' ').replace('\r', ' ')

def safe_find_text(
    container: element.Tag,
    selector: str,
    default: Optional[str] = None
) -> Optional[str]:
    """
    Safely finds an element by CSS selector and returns its cleaned text.
    Returns 'default' if not found.
    """
    element = container.select_one(selector)
    if element:
        return clean_text(element.get_text())
    return default

def safe_find_attr(
    container: element.Tag,
    selector: str,
    attribute: str,
    default: Optional[str] = None
) -> Optional[str]:
    """
    Safely finds an element by CSS selector and returns a specific attribute.
    Returns 'default' if not found.
    """
    element = container.select_one(selector)
    if element and element.has_attr(attribute):
        return element[attribute]
    return default