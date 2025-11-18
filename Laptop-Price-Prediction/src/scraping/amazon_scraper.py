"""
Scraper for Amazon.in laptop listings.

*** WARNING: AMAZON IS EXTREMELY DIFFICULT TO SCRAPE. ***
These selectors are placeholders and WILL FAIL. They must be
updated by inspecting the live website. You will likely be
blocked by a CAPTCHA.

Run as module to manually scrape data:
python -m src.scraping.amazon_scraper --pages 1 --query "dell laptop"
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin
import sys

# Ensure src is in path for utils import
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.scraping.scraper_utils import (
    create_polite_session, 
    polite_get, 
    safe_find_text, 
    safe_find_attr,
    clean_text
)
from src.utils import get_logger

logger = get_logger(__name__)

AMAZON_BASE_URL = "https://www.amazon.in"
AMAZON_SEARCH_URL = AMAZON_BASE_URL + "/s?k={query}&page={page}"
DATA_DIR = Path("data/raw")

# --- !! PLACEHOLDER SELECTORS (THESE WILL NOT WORK) !! ---
AMAZON_LISTING_SELECTOR = "div[data-component-type='s-search-result']"
AMAZON_PRODUCT_LINK_SELECTOR = "a.a-link-normal.s-underline-text"
AMAZON_PRODUCT_NAME_SELECTOR = "span.a-size-medium.a-color-base"
AMAZON_PRICE_SELECTOR = "span.a-price-whole"
AMAZON_RATING_SELECTOR = "span.a-icon-alt"
# --- End Selectors ---

def parse_product_listing_amazon(item: "bs4.element.Tag") -> Optional[Dict[str, Any]]:
    """
    Parses a single product *listing* from the search results page.
    This is highly simplified. Amazon's specs are often not on the search page.
    """
    data = {}
    
    # Get URL
    link_tag = item.select_one(AMAZON_PRODUCT_LINK_SELECTOR)
    if not link_tag:
        return None
    
    href = link_tag.get('href', '')
    if not href.startswith('/'):
        return None # Skip ads or external links
        
    data['url'] = urljoin(AMAZON_BASE_URL, href)
    
    # Get basic info
    data['model'] = safe_find_text(item, AMAZON_PRODUCT_NAME_SELECTOR)
    data['price_raw'] = safe_find_text(item, AMAZON_PRICE_SELECTOR)
    data['user_ratings'] = safe_find_text(item, AMAZON_RATING_SELECTOR, "0.0").split()[0] # e.g., "4.5 out of 5 stars"

    if not data['model'] or not data['price_raw']:
        return None

    # --- Amazon Spec Parsing (Difficult) ---
    # Unlike Flipkart, Amazon hides most specs. We must guess from the title.
    # A real version would need to visit the product page (data['url'])
    # and parse the complex spec table.
    
    title_low = data['model'].lower()
    specs = {}
    
    # Guess from title
    if 'gb ram' in title_low:
        specs['ram'] = title_low
    if 'ssd' in title_low or 'hdd' in title_low:
        specs['storage'] = title_low
    if 'intel' in title_low or 'amd' in title_low or 'ryzen' in title_low:
        specs['cpu'] = title_low
    if 'nvidia' in title_low or 'rtx' in title_low or 'gtx' in title_low:
        specs['gpu'] = title_low
    if 'windows' in title_low:
        specs['os'] = 'Windows'
    if 'macbook' in title_low:
        specs['os'] = 'macOS'
        
    # We are missing: display_size, weight
    
    data.update(specs)
    
    # Fill missing keys
    for key in ['cpu', 'ram', 'storage', 'display_size', 'os', 'gpu', 'weight']:
        if key not in data:
            data[key] = None
            
    return data

def scrape_search_page_amazon(session, page: int, query: str) -> List[Dict[str, Any]]:
    """Scrapes a single Amazon search results page."""
    url = AMAZON_SEARCH_URL.format(query=query, page=page)
    logger.info(f"Scraping Amazon search page: {url}")
    
    # Use a longer delay for Amazon
    soup = polite_get(session, url, delay_seconds=2.5)
    if not soup:
        logger.error(f"Failed to fetch search page {url}. Possible CAPTCHA.")
        return []

    listings = soup.select(AMAZON_LISTING_SELECTOR)
    if not listings:
        logger.warning(f"No listings found on page {page}. Selectors may be outdated or page is blocked.")
        return []

    logger.info(f"Found {len(listings)} listings on page {page}.")
    scraped_data = []

    for item in listings:
        try:
            product_data = parse_product_listing_amazon(item)
            if product_data:
                scraped_data.append(product_data)
        except Exception as e:
            logger.error(f"Error parsing item: {e}", exc_info=True)

    return scraped_data

def save_data(data: List[Dict[str, Any]], filename: str):
    """Saves the scraped data to a JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Successfully saved {len(data)} items to {filepath}")

def main(num_pages: int, query: str):
    """Main scraping pipeline for Amazon."""
    logger.info(f"Starting Amazon scraper for query='{query}'...")
    session = create_polite_session()
    all_data = []

    for page in range(1, num_pages + 1):
        data = scrape_search_page_amazon(session, page, query)
        all_data.extend(data)
        if not data:
            logger.warning(f"Stopping early, no data from page {page}.")
            break
        
        time.sleep(5) # Long delay between pages
        
    if all_data:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"amazon_{query.replace(' ','_')}_{timestamp}.json"
        save_data(all_data, filename)
    else:
        logger.warning("No data scraped. Check selectors and network.")
        
    logger.info("Amazon scraper finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Amazon.in laptop data.")
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Number of search result pages to scrape."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="laptops",
        help="Search query to use (e.g., 'gaming laptops')."
    )
    args = parser.parse_args()
    main(args.pages, args.query)