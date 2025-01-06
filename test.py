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
    # First fix any invalid escape sequences
    text = re.sub(r'\\([^"\\/bfnrtu])', r'\1', text)

    # Handle proper escape sequences
    text = text.replace(r'\"', '"')
    text = text.replace(r'\/', '/')
    text = text.replace(r"\'", "'")
    text = text.replace(r'\b', '')
    text = text.replace(r'\f', '')
    text = text.replace(r'\n', '\n')
    text = text.replace(r'\r', '\r')
    text = text.replace(r'\t', '\t')

    # Cleanup array structure
    text = re.sub(r'\}\s*\r?\n\s*\]$', '}]', text)
    text = re.sub(r'\]\s*\}\s*\r?\n\s*\]$', ']}]', text)

    # Clean newlines between objects
    text = re.sub(r'},\s*\r?\n\s*{', '},{', text)

    return text.strip()

def parse_json_safely(text: str) -> List[Dict]:
    """Safely parse JSON with multiple fallback strategies"""
    try:
        # First attempt: direct parse after cleaning
        cleaned_text = clean_escape_sequences(text)
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e1:
        logger.debug(f"First parse attempt failed: {e1}")
        try:
            # Second attempt: more aggressive cleaning
            text = re.sub(r'[\x00-\x1F]+', '', text)  # Remove control characters
            text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)  # Remove unicode escapes
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            return json.loads(text)
        except json.JSONDecodeError as e2:
            logger.debug(f"Second parse attempt failed: {e2}")
            try:
                # Final attempt: extract what we can
                matches = re.findall(r'{[^{}]*}', text)
                if matches:
                    # Reconstruct array with valid objects
                    valid_json = f"[{','.join(matches)}]"
                    return json.loads(valid_json)
                raise e2
            except json.JSONDecodeError as e3:
                logger.error(f"All parsing attempts failed. Final error: {e3}")
                raise

def get_osti_records(url: str) -> List[Dict[str, Union[str, List]]]:
    """Get OSTI records from a given URL with extended error handling"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        raw_text = response.text
        logger.debug(f"Received response length: {len(raw_text)}")

        records = parse_json_safely(raw_text)

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