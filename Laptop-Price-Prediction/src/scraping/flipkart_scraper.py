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

FLIPKART_BASE_URL = "https://www.flipkart.com"
FLIPKART_SEARCH_URL = FLIPKART_BASE_URL + "/search?q={query}&page={page}"
DATA_DIR = Path("data/raw")

# --- Updated Selectors (Must be updated periodically) ---
FLIPKART_LISTING_SELECTOR = "div._1AtVbE" # Main container for each product
FLIPKART_PRODUCT_LINK_SELECTOR = "a._1fQZEK" # Link to product page
FLIPKART_PRODUCT_NAME_SELECTOR = "div._4rR01T" # Product title
FLIPKART_PRICE_SELECTOR = "div._30jeq3._1_WHN1" # Sale price
FLIPKART_RATING_SELECTOR = "div._3LWZlK" # Rating e.g., "4.5"
FLIPKART_SPEC_LIST_SELECTOR = "ul._1xgFaf li.rgWa7D" # Spec list items

def parse_product_page_flipkart(item: "bs4.element.Tag") -> Optional[Dict[str, Any]]:
    """
    Parses a single product *listing* from the search results page.
    This is faster than visiting each product page.
    """
    data = {}
    
    # Get URL
    link_tag = item.select_one(FLIPKART_PRODUCT_LINK_SELECTOR)
    if not link_tag:
        return None # Not a valid product item
    
    data['url'] = urljoin(FLIPKART_BASE_URL, link_tag.get('href', ''))
    
    # Get basic info
    data['model'] = safe_find_text(item, FLIPKART_PRODUCT_NAME_SELECTOR)
    data['price_raw'] = safe_find_text(item, FLIPKART_PRICE_SELECTOR)
    data['user_ratings'] = safe_find_text(item, FLIPKART_RATING_SELECTOR)

    # If no price or model, it's not a useful listing
    if not data['model'] or not data['price_raw']:
        return None

    # Parse specs from the <ul> list
    specs_list = item.select(FLIPKART_SPEC_LIST_SELECTOR)
    specs = {}
    for spec_item in specs_list:
        text = clean_text(spec_item.get_text())
        if not text:
            continue
            
        if "Processor" in text and 'cpu' not in specs:
            specs["cpu"] = text
        elif "RAM" in text and 'ram' not in specs:
            specs["ram"] = text
        elif ("SSD" in text or "HDD" in text) and 'storage' not in specs:
            specs["storage"] = text
        elif "Display" in text and 'display_size' not in specs:
            specs["display_size"] = text
        elif "Operating System" in text and 'os' not in specs:
            specs["os"] = text
        elif "Graphics" in text and 'gpu' not in specs:
            specs["gpu"] = text
    
    data.update(specs)
    
    # Fill missing keys for robustness so all dicts have the same shape
    for key in ['cpu', 'ram', 'storage', 'display_size', 'os', 'gpu', 'weight']:
        if key not in data:
            data[key] = None
            
    return data

def scrape_search_page(session, page: int, query: str) -> List[Dict[str, Any]]:
    """Scrapes a single search results page for laptop listings."""
    url = FLIPKART_SEARCH_URL.format(query=query, page=page)
    logger.info(f"Scraping search page: {url}")
    
    soup = polite_get(session, url, delay_seconds=1.5)
    if not soup:
        logger.error(f"Failed to fetch search page {url}")
        return []

    listings = soup.select(FLIPKART_LISTING_SELECTOR)
    if not listings:
        logger.warning(f"No listings found on page {page}. Selectors may be outdated.")
        return []

    logger.info(f"Found {len(listings)} listings on page {page}.")
    scraped_data = []

    for item in listings:
        try:
            product_data = parse_product_page_flipkart(item)
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
    """Main scraping pipeline for Flipkart."""
    logger.info(f"Starting Flipkart scraper for query='{query}'...")
    session = create_polite_session()
    all_data = []

    for page in range(1, num_pages + 1):
        data = scrape_search_page(session, page, query)
        all_data.extend(data)
        if not data:
            logger.warning(f"Stopping early, no data from page {page}.")
            break
        
        # Add a longer delay between pages
        time.sleep(3) 
        
    if all_data:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"flipkart_{query.replace(' ','_')}_{timestamp}.json"
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
    parser.add_argument(
        "--query",
        type=str,
        default="laptops",
        help="Search query to use (e.g., 'gaming laptops')."
    )
    args = parser.parse_args()
    main(args.pages, args.query)