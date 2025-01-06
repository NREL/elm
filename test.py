import json
import logging
import requests
import re
from typing import Dict, List, Union

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def clean_escape_sequences(text: str) -> str:
    """Clean problematic escape sequences in text"""
    # Replace known problematic escape sequences
    text = text.replace(r'\"', '"')
    text = text.replace(r'\/', '/')
    text = text.replace(r"\'", "'")

    # Remove any remaining invalid escapes
    text = re.sub(r'\\(?!["\\/bfnrt])', '', text)

    return text

def parse_json_safely(text: str) -> List[Dict]:
    """Safely parse JSON with escape sequence handling"""
    try:
        # Pre-process the text to handle escape sequences
        cleaned_text = clean_escape_sequences(text)

        # Parse JSON
        result = json.loads(cleaned_text)
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {str(e)}\nFirst 500 chars: {text[:500]}")
        # One more attempt with minimal cleaning
        try:
            minimal_clean = text.replace(r'\"', '"').replace(r"\'", "'")
            return json.loads(minimal_clean)
        except json.JSONDecodeError:
            raise

def get_osti_records(url: str) -> List[Dict[str, Union[str, List]]]:
    """Get OSTI records from a given URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Get raw text without any decoding tricks
        raw_text = response.text

        # Parse the records
        records = parse_json_safely(raw_text)

        # Get pagination info
        n_pages = 1
        if 'last' in response.links:
            last_url = response.links['last']['url']
            n_pages = int(last_url.split('page=')[-1])

        logger.debug(f'Found approximately {n_pages * len(records)} records')
        return records

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

def main():
    base_url = "https://www.osti.gov/api/v1/records"

    try:
        records = get_osti_records(base_url)
        if records:
            print("\nFirst record:")
            print(json.dumps(records[0], indent=2))
            print(f"\nTotal records in first page: {len(records)}")
            return 0
    except Exception as e:
        logger.error(f"Failed to fetch OSTI records: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())