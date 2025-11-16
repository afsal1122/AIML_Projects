# File: src/scraping/amazon_scraper.py
"""
Scraper for Amazon.in laptop listings.

NOTE: CSS selectors are illustrative and *will* break over time.
Amazon is notoriously difficult to scrape.

Run as module:
python -m src.scraping.amazon_scraper --pages 2
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urljoin

from src.scraping.scraper_utils import (
    create_polite_session, 
    polite_get, 
    safe_find_text, 
    safe_find_attr
)
from src.utils import get_logger

logger = get_logger(__name__)

BASE_URL = "https://www.amazon.in"
SEARCH_URL = BASE_URL + "/s?k=laptops&page={page}"
DATA_DIR = Path("data/raw")

# --- Illustrative Selectors (WILL NEED UPDATING) ---
LISTING_SELECTOR = "div[data-component-type='s-search-result']"
PRODUCT_LINK_SELECTOR = "a.a-link-normal.s-underline-text.s-underline-link-text.s-link-style"
PRODUCT_NAME_SELECTOR = "span.a-size-medium.a-color-base.a-text-normal"
PRODUCT_PRICE_SELECTOR = "span.a-price-whole"
PRODUCT_RATING_SELECTOR = "span.a-icon-alt"
# --- End Selectors ---

def parse_product_page_amazon(soup, url: str) -> Dict[str, Any]:
    """
    Parses the individual product page for detailed specs.
    This is extremely simplified. Amazon's layout varies.
    """
    data = {"url": url}
    
    data["model"] = safe_find_text(soup, "span#productTitle")
    data["price_raw"] = safe_find_text(soup, "span.a-price-whole")
    data["user_ratings"] = safe_find_text(soup, "span#acrCustomerReviewText")

    # Specs are in a large table
    specs = {}
    spec_rows = soup.select("table#productDetails_techSpec_section_1 tr")
    
    for row in spec_rows:
        key = safe_find_text(row, "th")
        value = safe_find_text(row, "td")
        if not key or not value:
            continue
            
        key = key.lower()
        if "brand" in key:
            specs["brand"] = value
        elif "model number" in key:
            specs["model_number"] = value
        elif "processor" in key:
            specs["cpu"] = value
        elif "ram" in key:
            specs["ram"] = value
        elif "hard drive" in key:
            specs["storage"] = value
        elif "graphics" in key:
            specs["gpu"] = value
        elif "display size" in key:
            specs["display_size"] = value
        elif "resolution" in key:
            specs["resolution"] = value
        elif "operating system" in key:
            specs["os"] = value
        elif "item weight" in key:
            specs["weight"] = value

    data.update(specs)
    return data

def scrape_search_page_amazon(session, page: int) -> List[Dict[str, Any]]:
    """Scrapes a single Amazon search results page."""
    url = SEARCH_URL.format(page=page)
    logger.info(f"Scraping Amazon search page: {url}")
    
    soup = polite_get(session, url, delay_seconds=2.5)
    if not soup:
        logger.error(f"Failed to fetch search page {url}. Possible CAPTCHA.")
        return []

    listings = soup.select(LISTING_SELECTOR)
    if not listings:
        logger.warning(f"No listings found on page {page}. Selectors may be outdated.")
        return []

    logger.info(f"Found {len(listings)} listings on page {page}.")
    scraped_data = []

    for item in listings:
        try:
            link = safe_find_attr(item, PRODUCT_LINK_SELECTOR, 'href')
            if not link or not link.startswith('/'):
                continue
            
            product_url = urljoin(BASE_URL, link)
            
            # Limit product page scraping to avoid blocks
            logger.info(f"Scraping product: {product_url}")
            product_soup = polite_get(session, product_url, delay_seconds=3.0)
            
            if product_soup:
                product_data = parse_product_page_amazon(product_soup, product_url)
                scraped_data.append(product_data)
            
        except Exception as e:
            logger.error(f"Error parsing item: {e}", exc_info=True)

    return scraped_data

def save_data_amazon(data: List[Dict[str, Any]], filename: str):
    """Saves the scraped data to a JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Successfully saved {len(data)} items to {filepath}")

def main(num_pages: int):
    """Main scraping pipeline for Amazon."""
    logger.info("Starting Amazon scraper...")
    session = create_polite_session()
    all_data = []

    for page in range(1, num_pages + 1):
        data = scrape_search_page_amazon(session, page)
        all_data.extend(data)
        if not data:
            logger.warning(f"Stopping early, no data from page {page}. Possible block.")
            break
        
        time.sleep(5) # Longer delay for Amazon
        
    if all_data:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"amazon_raw_{timestamp}.json"
        save_data_amazon(all_data, filename)
    else:
        logger.warning("No data scraped from Amazon.")
        
    logger.info("Amazon scraper finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Amazon.in laptop data.")
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Number of search result pages to scrape."
    )
    args = parser.parse_args()
    main(args.pages)