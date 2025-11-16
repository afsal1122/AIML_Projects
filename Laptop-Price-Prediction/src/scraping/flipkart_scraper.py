# File: src/scraping/flipkart_scraper.py
"""
Scraper for Flipkart laptop listings.

NOTE: CSS selectors are illustrative and *will* break over time.
They must be updated by inspecting the live website.

Run as module:
python -m src.scraping.flipkart_scraper --pages 3
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
    safe_find_attr,
    clean_text
)
from src.utils import get_logger

logger = get_logger(__name__)

BASE_URL = "https://www.flipkart.com"
SEARCH_URL = BASE_URL + "/search?q=laptops&page={page}"
DATA_DIR = Path("data/raw")

# --- Illustrative Selectors (WILL NEED UPDATING) ---
# These selectors are based on a snapshot of the site and are not stable.
LISTING_SELECTOR = "div._1AtVbE"
PRODUCT_LINK_SELECTOR = "a._1fQZEK"
PRODUCT_NAME_SELECTOR = "div._4rR01T"
PRODUCT_PRICE_SELECTOR = "div._30jeq3._1_WHN1"
PRODUCT_RATING_SELECTOR = "div._3LWZlK"
SPEC_LIST_SELECTOR = "ul._1xgFaf li.rgWa7D"
# --- End Selectors ---

def parse_product_page(soup, url: str) -> Dict[str, Any]:
    """
    Parses the individual product page for detailed specs.
    This is highly simplified; a real implementation would be more complex.
    """
    data = {"url": url}
    
    data["brand"] = safe_find_text(soup, "span._16sV6o", "").split()[0]
    data["model"] = safe_find_text(soup, "span.B_NuCI")
    data["price_raw"] = safe_find_text(soup, "div._30jeq3._1_WHN1")
    data["user_ratings"] = safe_find_text(soup, "div._3LWZlK")
    
    # Specs are often in tables or lists
    specs = {}
    spec_rows = soup.select("div._2418kt > ul > li._21lJbe")
    if not spec_rows:
        spec_rows = soup.select(SPEC_LIST_SELECTOR) # Fallback

    for item in spec_rows:
        text = clean_text(item.get_text())
        if "Processor" in text:
            specs["cpu"] = text
        elif "RAM" in text:
            specs["ram"] = text
        elif "Storage" in text:
            specs["storage"] = text
        elif "Display" in text:
            specs["display_size"] = text
        elif "Operating System" in text:
            specs["os"] = text
        elif "Graphics" in text:
            specs["gpu"] = text
        elif "Weight" in text:
            specs["weight"] = text

    data.update(specs)
    return data

def scrape_search_page(session, page: int) -> List[Dict[str, Any]]:
    """Scrapes a single search results page for laptop listings."""
    url = SEARCH_URL.format(page=page)
    logger.info(f"Scraping search page: {url}")
    
    soup = polite_get(session, url, delay_seconds=1.5)
    if not soup:
        logger.error(f"Failed to fetch search page {url}")
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
            if not link:
                continue
            
            product_url = urljoin(BASE_URL, link)
            
            # For this example, we'll scrape details from the product page.
            # A faster (but less detailed) scrape could just take list-page data.
            logger.info(f"Scraping product: {product_url}")
            product_soup = polite_get(session, product_url, delay_seconds=2.0)
            
            if product_soup:
                product_data = parse_product_page(product_soup, product_url)
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

def main(num_pages: int):
    """Main scraping pipeline for Flipkart."""
    logger.info("Starting Flipkart scraper...")
    session = create_polite_session()
    all_data = []

    for page in range(1, num_pages + 1):
        data = scrape_search_page(session, page)
        all_data.extend(data)
        if not data:
            logger.warning(f"Stopping early, no data from page {page}.")
            break
        
        # Add a longer delay between pages
        time.sleep(3) 
        
    if all_data:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"flipkart_raw_{timestamp}.json"
        save_data(all_data, filename)
    else:
        logger.warning("No data scraped. Check selectors and network.")
        
    logger.info("Flipkart scraper finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Flipkart laptop data.")
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Number of search result pages to scrape."
    )
    args = parser.parse_args()
    main(args.pages)