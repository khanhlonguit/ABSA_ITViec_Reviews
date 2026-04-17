"""
crawl_reviews.py
----------------
Web scraper for congty.review - Vietnamese company review platform.
Crawls all companies and their reviews, exports to Excel (.xlsx) or CSV.

Usage:
    python src/crawl_reviews.py [--output PATH] [--test N] [--resume] [--max-reviews N]
    
Examples:
    python src/crawl_reviews.py                                         # Full crawl → data/raw/reviews_congty.xlsx
    python src/crawl_reviews.py --test 5                                # Test with 5 companies
    python src/crawl_reviews.py --max-reviews 20000                     # Stop at 20,000 review rows total
    python src/crawl_reviews.py --output data/raw/reviews.csv --resume  # CSV + resume interrupted crawl
"""

import argparse
import csv
import logging
import re
import sys
import time
from pathlib import Path
from log_setup import setup_logging
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
MISSING = []
try:
    import requests
except ImportError:
    MISSING.append("requests")
try:
    from bs4 import BeautifulSoup
except ImportError:
    MISSING.append("beautifulsoup4")
try:
    import pandas as pd
except ImportError:
    MISSING.append("pandas")
try:
    import openpyxl  # noqa: F401 – required for .xlsx output (pandas)
except ImportError:
    MISSING.append("openpyxl")
try:
    from tqdm import tqdm
except ImportError:
    MISSING.append("tqdm")

if MISSING:
    print("Thiếu các thư viện sau. Hãy cài bằng lệnh:")
    print(f"  pip install {' '.join(MISSING)}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://congty.review"
HOMEPAGE_URL = f"{BASE_URL}/"
TOTAL_PAGES = 272  # Based on website pagination
REQUEST_DELAY = 1.5  # Delay between requests in seconds
REQUEST_TIMEOUT = 10  # Request timeout in seconds
MAX_RETRIES = 3
CHECKPOINT_INTERVAL = 50  # Save checkpoint every N companies

REVIEW_FIELDNAMES = [
    'reviewer_id', 'timestamp', 'review_content',
    'company_name', 'company_type', 'company_size', 'company_address',
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
setup_logging()
logger = logging.getLogger("crawl_reviews")

# ---------------------------------------------------------------------------
# HTTP Session setup
# ---------------------------------------------------------------------------
session = requests.Session()
session.headers.update({'User-Agent': USER_AGENT})


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def fetch_page(url: str, retries: int = MAX_RETRIES) -> Optional[str]:
    """Fetch HTML content from URL with retry logic."""
    for attempt in range(retries):
        try:
            time.sleep(REQUEST_DELAY)
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            response.encoding = 'utf-8'
            return response.text
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt+1}/{retries} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to fetch {url} after {retries} attempts")
                return None
    return None


def clean_text(text: str) -> str:
    """Clean extracted text - remove extra whitespace."""
    if not text:
        return ""
    # Remove multiple spaces, newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_company_card(card_element) -> Optional[Dict[str, str]]:
    """Parse company information from a card element on homepage."""
    try:
        # Find company link and name
        link_elem = card_element.find('a', href=re.compile(r'/companies/'))
        if not link_elem:
            return None
            
        company_url = urljoin(BASE_URL, link_elem.get('href', ''))
        company_slug = company_url.split('/companies/')[-1].split('?')[0]
        
        # Company name - from link text or heading
        name_elem = card_element.find(['h3', 'h2'])
        if name_elem:
            company_name = clean_text(name_elem.get_text())
        else:
            company_name = clean_text(link_elem.get_text())
        
        # Review count - usually in parentheses like (24)
        review_count = 0
        review_match = re.search(r'\((\d+)\)', card_element.get_text())
        if review_match:
            review_count = int(review_match.group(1))
        
        # Company type, size, address - usually in same card
        meta_text = card_element.get_text()
        company_type = ""
        company_size = ""
        company_address = ""
        
        # Try to extract structured metadata
        meta_parts = [clean_text(s) for s in meta_text.split('\n') if clean_text(s)]
        for part in meta_parts:
            # Type indicators: product, outsource, etc.
            if part.lower() in ['product', 'outsource', 'agency marketing', 'imc', 'it', 'kinh doanh']:
                company_type = part
            # Size indicators: number ranges or keywords
            elif re.match(r'\d+-\d+|\d+\+|1-50|51-150|151-500|501-1000|1000\+', part, re.IGNORECASE):
                company_size = part
            # Address: usually longer text without numbers-only pattern
            elif len(part) > 15 and not re.match(r'^\d+$', part):
                if 'District' in part or 'Quận' in part or 'Phường' in part or 'Ho Chi Minh' in part or 'Ha Noi' in part or 'Hà Nội' in part:
                    company_address = part
        
        return {
            'company_name': company_name,
            'company_slug': company_slug,
            'company_url': company_url,
            'review_count': review_count,
            'company_type': company_type,
            'company_size': company_size,
            'company_address': company_address
        }
    except Exception as e:
        logger.warning(f"Error parsing company card: {e}")
        return None


def crawl_companies_list(test_limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Crawl all companies from homepage pagination."""
    logger.info("Starting company list crawl...")
    companies = []
    
    pages_to_crawl = range(1, TOTAL_PAGES + 1)
    if test_limit:
        pages_to_crawl = range(1, min(5, TOTAL_PAGES + 1))  # Limit pages for testing
    
    for page_num in tqdm(pages_to_crawl, desc="Crawling company pages"):
        url = f"{HOMEPAGE_URL}?page={page_num}"
        html = fetch_page(url)
        
        if not html:
            logger.error(f"Skipping page {page_num} - failed to fetch")
            continue
        
        soup = BeautifulSoup(html, 'lxml')
        
        # Find company cards - need to identify the correct selector
        # Based on the HTML structure, companies are in links with pattern /companies/
        company_links = soup.find_all('a', href=re.compile(r'/companies/[^/]+$'))
        
        page_companies = []
        for link in company_links:
            # Get parent container that has all company info
            card = link.find_parent(['div', 'article', 'section'])
            if card:
                company_data = parse_company_card(card)
                if company_data and company_data['company_slug']:
                    # Avoid duplicates
                    if not any(c['company_slug'] == company_data['company_slug'] for c in companies):
                        page_companies.append(company_data)
        
        companies.extend(page_companies)
        logger.info(f"Page {page_num}: found {len(page_companies)} companies (total: {len(companies)})")
        
        # Test limit check
        if test_limit and len(companies) >= test_limit:
            companies = companies[:test_limit]
            logger.info(f"Test limit reached: {test_limit} companies")
            break
    
    logger.info(f"Company list crawl complete: {len(companies)} companies found")
    return companies


def parse_review_element(review_elem) -> Optional[Dict[str, str]]:
    """Parse a single review element and extract data."""
    try:
        review_text = review_elem.get_text()
        
        # Reviewer ID - hex pattern like "369e5a", "7858e1"
        reviewer_id = ""
        reviewer_link = review_elem.find('a', href=re.compile(r'/review/'))
        if reviewer_link:
            reviewer_id = clean_text(reviewer_link.get_text())
        
        # If not found in link, look for hex pattern in text
        if not reviewer_id or len(reviewer_id) != 6:
            hex_match = re.search(r'\b([a-f0-9]{6})\b', review_text)
            if hex_match:
                reviewer_id = hex_match.group(1)
        
        # Timestamp - look for relative time patterns (Vietnamese)
        timestamp = ""
        time_patterns = [
            r'(\d+\s*(giờ|ngày|tháng|năm)\s*trước)',
            r'(Một giờ trước|Vài giờ trước)',
        ]
        for pattern in time_patterns:
            time_match = re.search(pattern, review_text, re.IGNORECASE)
            if time_match:
                timestamp = time_match.group(1)
                break
        
        # Extract review content - the main paragraph after timestamp
        review_content = ""
        
        # Strategy: Find all paragraphs, skip UI elements
        paragraphs = review_elem.find_all(['p', 'div'])
        content_parts = []
        
        for para in paragraphs:
            para_text = clean_text(para.get_text())
            # Skip empty, very short, or UI elements
            if len(para_text) < 20:
                continue
            if any(keyword in para_text.lower() for keyword in ['reply', 'báo cáo', 'xóa']):
                continue
            content_parts.append(para_text)
        
        if content_parts:
            review_content = ' '.join(content_parts)
        else:
            # Fallback: get all text and clean it
            all_text = review_text
            # Remove UI elements and metadata
            all_text = re.sub(r'\s*Reply\s*', ' ', all_text, flags=re.IGNORECASE)
            all_text = re.sub(r'\s*Báo cáo\s*', ' ', all_text, flags=re.IGNORECASE)
            all_text = re.sub(r'\s*\d+\s*(giờ|ngày|tháng|năm)\s*trước\s*', ' ', all_text)
            # Remove reviewer ID from content
            if reviewer_id:
                all_text = all_text.replace(reviewer_id, '')
            review_content = clean_text(all_text)
        
        # Only return if we have meaningful content
        if review_content and len(review_content) > 20:
            return {
                'reviewer_id': reviewer_id or 'unknown',
                'timestamp': timestamp or 'unknown',
                'review_content': review_content
            }
        return None
        
    except Exception as e:
        logger.warning(f"Error parsing review element: {e}")
        return None


def crawl_company_reviews(company: Dict[str, str]) -> List[Dict[str, str]]:
    """Crawl all reviews for a specific company."""
    company_slug = company['company_slug']
    company_url = company['company_url']
    reviews = []
    
    page_num = 1
    has_more_pages = True
    
    while has_more_pages:
        # Construct URL for pagination
        if page_num == 1:
            url = company_url
        else:
            url = f"{company_url}?page={page_num}"
        
        html = fetch_page(url)
        if not html:
            logger.error(f"Failed to fetch reviews for {company_slug} page {page_num}")
            break
        
        soup = BeautifulSoup(html, 'lxml')
        
        # Find the main review section
        review_section = soup.find(['div', 'section'], string=re.compile(r'Review công ty|reviews'))
        if not review_section:
            # Fallback: find by looking for multiple review links
            review_section = soup
        
        # Find all review links to identify boundaries
        pattern = rf'/companies/{company_slug}/review/[a-f0-9-]+'
        review_links = review_section.find_all('a', href=re.compile(pattern))
        
        if not review_links:
            logger.warning(f"No review links found for {company_slug} page {page_num}")
            break
        
        page_reviews = []
        
        # Process each review link
        for idx, link in enumerate(review_links):
            try:
                reviewer_id = clean_text(link.get_text())
                if not reviewer_id or len(reviewer_id) != 6:
                    continue
                
                # Get parent that contains review content (go up 4 levels)
                parent = link
                for _ in range(4):
                    if parent and parent.parent:
                        parent = parent.parent
                
                if not parent:
                    continue
                
                # Get all text and find this review's segment
                full_text = parent.get_text(separator='\n')
                
                # Find position of this reviewer ID in text
                reviewer_pattern = rf'\b{re.escape(reviewer_id)}\b'
                match = re.search(reviewer_pattern, full_text)
                if not match:
                    continue
                
                start_pos = match.end()
                
                # Find end position (next reviewer ID or "Reply")
                end_pos = len(full_text)
                
                # Look for next reviewer (another hex ID)
                if idx + 1 < len(review_links):
                    next_reviewer_id = clean_text(review_links[idx + 1].get_text())
                    if next_reviewer_id and len(next_reviewer_id) == 6:
                        next_match = re.search(rf'\b{re.escape(next_reviewer_id)}\b', full_text[start_pos:])
                        if next_match:
                            end_pos = start_pos + next_match.start()
                
                # Also look for "Reply" or "Báo cáo" as boundaries
                reply_match = re.search(r'\s+(Reply|Báo cáo)\s+', full_text[start_pos:end_pos])
                if reply_match:
                    end_pos = start_pos + reply_match.start()
                
                # Extract review text
                review_text = full_text[start_pos:end_pos]
                
                # Clean up whitespace
                review_text = re.sub(r'\s+', ' ', review_text).strip()
                
                # Extract timestamp from review text
                timestamp = ""
                time_patterns = [
                    r'\d+\s*giờ\s*trước',
                    r'\d+\s*ngày\s*trước',
                    r'\d+\s*tháng\s*trước',
                    r'\d+\s*năm\s*trước',
                    r'Một giờ trước',
                ]
                for pattern_str in time_patterns:
                    time_match = re.search(pattern_str, review_text, re.IGNORECASE)
                    if time_match:
                        timestamp = time_match.group(0)
                        # Remove timestamp from content
                        review_text = review_text.replace(timestamp, '', 1).strip()
                        break
                
                # Validate content length
                if len(review_text) > 20:
                    review_data = {
                        'reviewer_id': reviewer_id,
                        'timestamp': timestamp or 'unknown',
                        'review_content': review_text,
                        'company_name': company['company_name'],
                        'company_type': company['company_type'],
                        'company_size': company['company_size'],
                        'company_address': company['company_address']
                    }
                    page_reviews.append(review_data)
            
            except Exception as e:
                logger.warning(f"Error parsing review {idx} for {company_slug}: {e}")
                continue
        
        if page_reviews:
            reviews.extend(page_reviews)
            logger.info(f"{company_slug} page {page_num}: {len(page_reviews)} reviews")
        
        # Check for next page
        next_page_link = soup.find('a', href=re.compile(rf'\?page={page_num+1}'))
        if not next_page_link:
            all_page_links = soup.find_all('a', href=re.compile(r'\?page=\d+'))
            if all_page_links:
                page_numbers = []
                for a in all_page_links:
                    match = re.search(r'page=(\d+)', a.get('href', ''))
                    if match:
                        page_numbers.append(int(match.group(1)))
                if page_numbers and max(page_numbers) > page_num:
                    next_page_link = True
        
        if next_page_link and page_reviews:
            page_num += 1
        else:
            has_more_pages = False
    
    logger.info(f"{company['company_name']}: crawled {len(reviews)} reviews")
    return reviews


def save_checkpoint(companies: List[Dict[str, str]], output_path: Path, processed_count: int):
    """Save list of companies as checkpoint."""
    checkpoint_path = output_path.parent / f"{output_path.stem}_companies.csv"
    try:
        df = pd.DataFrame(companies)
        df['processed'] = False
        df.loc[:processed_count-1, 'processed'] = True
        df.to_csv(checkpoint_path, index=False, encoding='utf-8')
        logger.info(f"Checkpoint saved: {processed_count}/{len(companies)} companies processed")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def load_checkpoint(output_path: Path) -> Optional[Tuple[List[Dict[str, str]], int]]:
    """Load checkpoint of company list and processed count."""
    checkpoint_path = output_path.parent / f"{output_path.stem}_companies.csv"
    if not checkpoint_path.exists():
        return None
    
    try:
        df = pd.read_csv(checkpoint_path, encoding='utf-8')
        companies = df.to_dict('records')
        processed_count = df['processed'].sum() if 'processed' in df.columns else 0
        logger.info(f"Checkpoint loaded: {processed_count}/{len(companies)} companies already processed")
        return companies, processed_count
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None


def count_reviews_in_output(output_path: Path) -> int:
    """Count data rows in existing review file (CSV or Excel)."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        return 0
    suffix = output_path.suffix.lower()
    try:
        if suffix == '.xlsx':
            df = pd.read_excel(output_path, engine='openpyxl')
            return len(df)
        if suffix == '.csv':
            with open(output_path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                return sum(1 for _ in reader)
        logger.warning(f"Unsupported output format: {output_path}")
        return 0
    except Exception as e:
        logger.warning(f"Could not count rows in {output_path}: {e}")
        return 0


def save_reviews_to_csv(reviews: List[Dict[str, str]], output_path: Path, mode: str = 'a') -> None:
    """Append reviews to a CSV file."""
    if not reviews:
        return

    file_exists = output_path.exists() and output_path.stat().st_size > 0

    try:
        with open(output_path, mode=mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDNAMES)

            if not file_exists or mode == 'w':
                writer.writeheader()

            for review in reviews:
                row = {field: review.get(field, '') for field in REVIEW_FIELDNAMES}
                writer.writerow(row)

    except Exception as e:
        logger.error(f"Failed to save reviews to CSV: {e}")


def save_reviews_to_xlsx(reviews: List[Dict[str, str]], output_path: Path) -> None:
    """Append reviews to an Excel file (read + concat + overwrite)."""
    if not reviews:
        return
    try:
        new_df = pd.DataFrame(
            [{field: review.get(field, '') for field in REVIEW_FIELDNAMES} for review in reviews]
        )
        if output_path.exists() and output_path.stat().st_size > 0:
            old_df = pd.read_excel(output_path, engine='openpyxl')
            out_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            out_df = new_df
        out_df.to_excel(output_path, index=False, engine='openpyxl')
    except Exception as e:
        logger.error(f"Failed to save reviews to Excel: {e}")


def append_reviews(reviews: List[Dict[str, str]], output_path: Path) -> None:
    """Append review rows; format is chosen from file extension (.xlsx or .csv)."""
    if not reviews:
        return
    suffix = output_path.suffix.lower()
    if suffix == '.xlsx':
        save_reviews_to_xlsx(reviews, output_path)
    elif suffix == '.csv':
        save_reviews_to_csv(reviews, output_path, mode='a')
    else:
        logger.error(
            f"Unsupported --output extension {suffix!r}; use .xlsx or .csv"
        )


# ---------------------------------------------------------------------------
# Main crawling logic
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Crawl company reviews from congty.review',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'raw' / 'reviews_congty.xlsx'),
        help='Output file path: .xlsx (default) or .csv',
    )
    parser.add_argument(
        '--test',
        type=int,
        metavar='N',
        help='Test mode: crawl only N companies'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if available'
    )
    parser.add_argument(
        '--max-reviews',
        type=int,
        default=None,
        metavar='N',
        help='Stop when total review rows in the output file reaches N (e.g. 20000). '
             'Existing rows in the file are counted when resuming.',
    )

    args = parser.parse_args()

    if args.max_reviews is not None and args.max_reviews < 1:
        parser.error('--max-reviews must be a positive integer')
    output_path = Path(args.output)
    out_suffix = output_path.suffix.lower()
    if out_suffix not in ('.xlsx', '.csv'):
        parser.error('--output must end with .xlsx or .csv')
    
    logger.info("=" * 60)
    logger.info("Starting congty.review crawler")
    logger.info(f"Output file: {output_path}")
    if args.test:
        logger.info(f"Test mode: {args.test} companies")
    if args.max_reviews:
        logger.info(f"Target: stop at {args.max_reviews} review rows (total in output file)")
    logger.info("=" * 60)
    
    # Step 1: Get company list
    companies = []
    start_index = 0
    
    if args.resume:
        checkpoint = load_checkpoint(output_path)
        if checkpoint:
            companies, start_index = checkpoint
            logger.info(f"Resuming from checkpoint: {start_index} companies already processed")
    
    if not companies:
        companies = crawl_companies_list(test_limit=args.test)
        if not companies:
            logger.error("No companies found. Exiting.")
            return
        
        # Save initial checkpoint
        save_checkpoint(companies, output_path, 0)
    
    # Step 2: Crawl reviews for each company
    total_reviews = count_reviews_in_output(output_path) if args.max_reviews else 0
    if args.max_reviews and total_reviews > 0:
        logger.info(f"Existing review rows in output file: {total_reviews}")

    for idx, company in enumerate(tqdm(companies[start_index:], desc="Crawling reviews", initial=start_index, total=len(companies))):
        actual_idx = start_index + idx

        if args.max_reviews and total_reviews >= args.max_reviews:
            logger.info(f"Already at or above target ({args.max_reviews} reviews). Stopping.")
            break

        # Skip if no reviews
        if company.get('review_count', 0) == 0:
            logger.debug(f"Skipping {company['company_name']} (no reviews)")
            continue

        # Crawl reviews
        reviews = crawl_company_reviews(company)

        if not reviews:
            continue

        if args.max_reviews:
            remaining = args.max_reviews - total_reviews
            if remaining <= 0:
                break
            if len(reviews) > remaining:
                reviews = reviews[:remaining]

        append_reviews(reviews, output_path)
        total_reviews += len(reviews)

        # Periodic checkpoint
        if (actual_idx + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(companies, output_path, actual_idx + 1)

        if args.max_reviews and total_reviews >= args.max_reviews:
            logger.info(f"Reached target of {args.max_reviews} review rows. Stopping crawl.")
            save_checkpoint(companies, output_path, actual_idx + 1)
            break

    # Mark every company processed only if we did not stop early due to --max-reviews
    if args.max_reviews is None or total_reviews < args.max_reviews:
        save_checkpoint(companies, output_path, len(companies))

    logger.info("=" * 60)
    logger.info("Crawling complete!")
    logger.info(f"Total companies: {len(companies)}")
    logger.info(f"Total reviews: {total_reviews}")
    logger.info(f"Output file: {output_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
