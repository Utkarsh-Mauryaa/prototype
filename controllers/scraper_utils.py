"""
Scraper and URL Validation Utilities
"""

import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse
from fastapi import HTTPException

# Browser TLS Impersonation (bypasses Cloudflare/Akamai/WAF 403 blocks)
try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    cffi_requests = None
    HAS_CURL_CFFI = False

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Encoding': 'gzip, deflate',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Boilerplate phrases commonly found in UI, cookie banners, accessibility helpers
BOILERPLATE_PHRASES = [
    'press enter to open the list',
    'use the arrow keys to choose',
    'arrow keys to choose',
    'cookie policy',
    'privacy policy',
    'terms of service',
    'all rights reserved',
    'subscribe now',
    'sign in to continue',
    'share on whatsapp',
    'share on twitter',
    'share on facebook',
    'javascript is disabled',
    'please enable javascript',
]

def is_valid_url(url: str) -> bool:
    """Checks if a string is a well-formed HTTP/HTTPS URL."""
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
    except Exception:
        return False

def normalize_news_url(url: str) -> str:
    """
    Normalizes known government and news portals to their static/server-rendered endpoints.
    For example, PIB (Press Information Bureau) PressReleaseDetail.aspx loads via client-side JS,
    while PressReleasePage.aspx renders full static HTML.
    """
    clean_url = url.strip()
    try:
        parsed = urlparse(clean_url)
        # Transform PIB client-side page to full static server-side page
        if 'pib.gov.in' in parsed.netloc.lower():
            if 'PressReleaseDetail.aspx' in parsed.path:
                new_path = parsed.path.replace('PressReleaseDetail.aspx', 'PressReleasePage.aspx')
                return urlunparse((parsed.scheme, parsed.netloc, new_path, parsed.params, parsed.query, parsed.fragment))
    except Exception:
        pass
    return clean_url

def clean_headline(title: str) -> str:
    """Cleans brand suffixes like '| Press Information Bureau' or '- NDTV' from titles."""
    if not title:
        return ""
    # Strip common site delimiters
    for delim in [' | ', ' - ', ' :: ', ' — ']:
        if delim in title:
            parts = title.split(delim)
            # Take the longest part or the first part if it's descriptive
            if len(parts[0].strip()) > 15:
                title = parts[0].strip()
            elif len(parts) > 1 and len(parts[1].strip()) > 15:
                title = parts[1].strip()
    return re.sub(r'\s+', ' ', title).strip()

def scrape_url_content(url: str) -> tuple[str, str]:
    """
    Fetches and extracts the headline and body text from a webpage URL.
    Returns:
        tuple (headline, body_text)
    Raises:
        HTTPException if the URL cannot be fetched or contains no text.
    """
    clean_url = url.strip()
    if not is_valid_url(clean_url):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid URL format: '{clean_url}'. Must start with http:// or https://"
        )

    # Normalize special URLs (e.g. PIB SSR endpoint)
    fetch_url = normalize_news_url(clean_url)

    resp = None
    last_error_detail = None

    # Step A: Primary Attempt with curl_cffi (mimics genuine Chrome TLS & bypasses Cloudflare/Akamai WAF)
    if HAS_CURL_CFFI:
        try:
            resp = cffi_requests.get(
                fetch_url,
                impersonate="chrome120",
                timeout=15,
                allow_redirects=True
            )
        except Exception as cffi_err:
            last_error_detail = f"Browser impersonation error: {str(cffi_err)}"

    # Step B: Fallback to standard requests if curl_cffi wasn't available or failed
    if resp is None or resp.status_code >= 400:
        for headers in [HEADERS, {**HEADERS, 'Accept-Encoding': 'identity'}]:
            try:
                fallback_resp = requests.get(fetch_url, headers=headers, timeout=15, allow_redirects=True)
                if fallback_resp.status_code == 200:
                    resp = fallback_resp
                    break
                elif resp is None:
                    resp = fallback_resp
            except requests.exceptions.SSLError:
                try:
                    fallback_resp = requests.get(fetch_url, headers=headers, timeout=15, verify=False, allow_redirects=True)
                    if fallback_resp.status_code == 200:
                        resp = fallback_resp
                        break
                except Exception as ssl_err:
                    last_error_detail = f"SSL Certificate Verification Error: {str(ssl_err)}"
            except requests.exceptions.Timeout:
                last_error_detail = "Request timed out while trying to reach the provided URL."
            except requests.exceptions.ConnectionError as conn_err:
                last_error_detail = f"Connection failed: Unable to reach host ({str(conn_err)})"
            except Exception as err:
                last_error_detail = f"Network error while fetching URL: {str(err)}"

    if resp is None:
        raise HTTPException(
            status_code=400,
            detail=last_error_detail or "Failed to connect to the provided URL."
        )

    if resp.status_code == 403:
        raise HTTPException(
            status_code=403,
            detail="Failed to access URL (HTTP 403 Forbidden). This website has strict anti-bot firewall protection. Please copy and paste the article headline or body text directly."
        )

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=resp.status_code if resp.status_code in (401, 404, 408, 429) else 400,
            detail=f"Failed to access URL. Server responded with HTTP status {resp.status_code}."
        )

    soup = BeautifulSoup(resp.text, 'html.parser')

    # 1. Extract Headline BEFORE tag decomposition so meta tags are intact
    headline = ""
    # Check OpenGraph / Twitter meta titles first
    og_title = soup.find('meta', property='og:title') or soup.find('meta', attrs={'name': 'og:title'})
    tw_title = soup.find('meta', property='twitter:title') or soup.find('meta', attrs={'name': 'twitter:title'})
    
    if og_title and og_title.get('content') and len(og_title['content'].strip()) > 10:
        headline = clean_headline(og_title['content'])
    elif tw_title and tw_title.get('content') and len(tw_title['content'].strip()) > 10:
        headline = clean_headline(tw_title['content'])
    
    # If no meta title, check h1
    if not headline:
        h1 = soup.find('h1')
        if h1 and len(h1.get_text(strip=True)) > 10:
            headline = clean_headline(h1.get_text(strip=True))
            
    # Check h2 with title-related ID/class (e.g. PIB <h2 id="Titleh2">)
    if not headline:
        for h2 in soup.find_all('h2'):
            h2_id = (h2.get('id') or '').lower()
            h2_cls = " ".join(h2.get('class') or []).lower()
            if any(k in h2_id or k in h2_cls for k in ['title', 'head', 'release']):
                t = h2.get_text(strip=True)
                if len(t) > 15:
                    headline = clean_headline(t)
                    break
                    
    # Fallback to <title>
    if not headline and soup.find('title'):
        headline = clean_headline(soup.find('title').get_text(strip=True))

    # Remove script, style, and navigation tags
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript', 'aside', 'svg']):
        tag.decompose()

    # 2. Extract Body Text
    # Check for dedicated article container
    article_container = (
        soup.find('article') or
        soup.find(class_=lambda c: c and any(k in c.lower() for k in [
            'article-body', 'story-content', 'story__content', 'release-text',
            'inner-page-main', 'entry-content', 'post-content'
        ]))
    )
    
    search_context = article_container if article_container else soup
    raw_paragraphs = [p.get_text(separator=' ', strip=True) for p in search_context.find_all('p')]

    # Clean and filter paragraphs
    clean_paragraphs = []
    for p in raw_paragraphs:
        p_clean = re.sub(r'\s+', ' ', p).strip()
        p_lower = p_clean.lower()
        # Discard if too short or matches boilerplate/accessibility noise
        if len(p_clean) < 30:
            continue
        if any(phrase in p_lower for phrase in BOILERPLATE_PHRASES):
            continue
        clean_paragraphs.append(p_clean)

    body_text = " ".join(clean_paragraphs)

    # 3. Fallback to meta descriptions if body text is still empty
    if not body_text:
        meta_desc = (
            soup.find('meta', property='og:description') or
            soup.find('meta', attrs={'name': 'description'}) or
            soup.find('meta', attrs={'name': 'twitter:description'})
        )
        if meta_desc and meta_desc.get('content'):
            body_text = meta_desc['content'].strip()

    if not body_text and not headline:
        raise HTTPException(
            status_code=422,
            detail="No readable article text or headline could be extracted from this webpage."
        )

    # If body is still empty, fallback to headline
    if not body_text:
        body_text = headline

    return headline, body_text
