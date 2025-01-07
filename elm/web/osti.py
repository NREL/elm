# -*- coding: utf-8 -*-
"""
Utilities for retrieving data from OSTI.
"""
import re
import copy
import requests
import json
from typing import Dict, List
import os
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class OstiRecord(dict):
    """Class to handle a single OSTI record as dictionary data"""

    def __init__(self, record):
        """
        Parameters
        ----------
        record : dict
            OSTI record in dict form, typically a response from OSTI API.
        """
        assert isinstance(record, dict)
        super().__init__(**record)

    @staticmethod
    def strip_nested_brackets(text):
        """
        Remove text between brackets/parentheses for cleaning OSTI text
        """
        ret = ''
        skip1c = 0
        skip2c = 0
        for i in text:
            if i == '[':
                skip1c += 1
            elif i == '(':
                skip2c += 1
            elif i == ']' and skip1c > 0:
                skip1c -= 1
            elif i == ')' and skip2c > 0:
                skip2c -= 1
            elif skip1c == 0 and skip2c == 0:
                ret += i
        return ret

    @property
    def authors(self):
        """Get the list of authors of this record.

        Returns
        -------
        str
        """
        au = copy.deepcopy(self.get('authors', None))
        if au is not None:
            for i, name in enumerate(au):
                name = self.strip_nested_brackets(name)
                if name.count(',') == 1:
                    second, first = name.split(',')
                    name = f'{first.strip()} {second.strip()}'
                au[i] = name
            au = ', '.join(au)
        return au

    @property
    def title(self):
        """Get the title of this record

        Returns
        -------
        str | None
        """
        return self.get('title', None)

    @property
    def year(self):
        """Get the year of publication of this record

        Returns
        -------
        str | None
        """
        year = self.get('publication_date', None)
        if year is not None:
            year = year.split('-')[0]
            year = str(year)
        return year

    @property
    def date(self):
        """Get the date of publication of this record

        Returns
        -------
        str | None
        """
        date = self.get('publication_date', None)
        if date is not None:
            date = date.split('T')[0]
            date = str(date)
        return date

    @property
    def doi(self):
        """Get the DOI of this record

        Returns
        -------
        str | None
        """
        return self.get('doi', None)

    @property
    def osti_id(self):
        """Get the OSTI ID of this record which is typically a 7 digit number

        Returns
        -------
        str | None
        """
        return self.get('osti_id', None)

    @property
    def url(self):
        """Get the download URL of this record

        Returns
        -------
        str | None
        """
        url = None
        for link in self['links']:
            if link.get('rel', None) == 'fulltext':
                url = link.get('href', None)
                break
        return url

    def download(self, fp):
        """Download the PDF of this record

        Parameters
        ----------
        fp : str
            Filepath to download this record to, typically a .pdf
        """
        session = requests.Session()
        response = session.get(self.url)
        response = session.get(self.url)
        with open(fp, 'wb') as f_pdf:
            f_pdf.write(response.content)


class OstiList(list):
    """Class to retrieve and handle multiple OSTI records from an API URL."""

    BASE_URL = 'https://www.osti.gov/api/v1/records'
    """Base OSTI API URL. This can be appended with search parameters"""

    def __init__(self, url, n_pages=1):
        """
        Parameters
        ----------
        url : str
            OSTI API URL to request, see this for details:
                https://www.osti.gov/api/v1/docs
        n_pages : int
            Number of pages to get from the API. Typical response has 20
            entries per page. Default of 1 ensures that this class doesnt
            hang on a million responses.
        """

        self.url = url
        self._session = requests.Session()
        self._response = None
        self._n_pages = 0
        self._iter = 0

        records = self._get_all(n_pages)
        records = [OstiRecord(single) for single in records]
        super().__init__(records)

    def clean_escape_sequences(self, text: str) -> str:
        """Clean problematic escape sequences and formatting in JSON text.

        Parameters
        ----------
        text : str
            Raw JSON text to be cleaned

        Returns
        -------
        str
            Cleaned JSON text with proper escape sequences and formatting
        """
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

    def parse_json_safely(self, text: str) -> List[Dict]:
        """Safely parse JSON with multiple fallback strategies"""
        try:
            cleaned_text = self.clean_escape_sequences(text)
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e1:
            logger.debug(f"First parse attempt failed: {e1}")
            try:
                text = re.sub(r'[\x00-\x1F]+', '', text)
                text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
                text = re.sub(r'\s+', ' ', text)
                return json.loads(text)
            except json.JSONDecodeError as e2:
                logger.debug(f"Second parse attempt failed: {e2}")
                try:
                    matches = re.findall(r'{[^{}]*}', text)
                    if matches:
                        valid_json = f"[{','.join(matches)}]"
                        return json.loads(valid_json)
                    raise e2
                except json.JSONDecodeError as e3:
                    logger.error(f"""All parsing attempts
                                 failed. Final error: {e3}""")
                    raise

    def _get_first(self):
        """Get the first page of OSTI records"""
        self._response = self._session.get(self.url)

        if not self._response.ok:
            msg = f'''OSTI API Request got error
            {self._response.status_code}:
            "{self._response.reason}"'''
            raise RuntimeError(msg)

        try:
            raw_text = self._response.text
            first_page = self.parse_json_safely(raw_text)
        except (json.JSONDecodeError, UnicodeError) as e:
            logger.error(f"""JSON decode error:
                        {str(e)}\nRaw text: {raw_text[:500]}...""")
            raise
        self._n_pages = 1
        if 'last' in self._response.links:
            url = self._response.links['last']['url']
            self._n_pages = int(url.split('page=')[-1])
        return first_page

    def _get_pages(self, n_pages):
        """Get response pages up to n_pages from OSTI.

        Parameters
        ----------
        n_pages : int
            Number of pages to retrieve

        Returns
        -------
        next_pages : list
            Generator of next pages, each a list of OSTI records
        """
        if n_pages <= 1:
            return
        for page in range(2, self._n_pages + 1):
            if page > n_pages:
                break

            try:
                response = self._session.get(
                    self.url,
                    params={'page': page}
                )
                if not response.ok:
                    logger.error(f"""Failed to get page {page}:
                                 {response.status_code}""")
                    continue
                page_records = self.parse_json_safely(response.text)

                yield page_records

            except Exception as e:
                logger.error(f"Error processing page {page}: {str(e)}")
                continue

    def _get_all(self, n_pages):
        """Get all pages of records up to n_pages.

        Parameters
        ----------
        n_pages : int
            Number of pages to retrieve

        Returns
        -------
        all_records : list
            List of all records.
        """
        first_page = self._get_first()
        records = first_page

        for page in self._get_pages(n_pages):
            records.extend(page)

        return records

    def download(self, out_dir):
        """Download all PDFs from the records in this OSTI object into a
        directory. PDFs will be given file names based on their OSTI record
        ID

        Parameters
        ----------
        out_dir : str
            Directory to download PDFs to. This directory will be created if
            it does not already exist.
        """
        logger.info(f'Downloading {len(self)} records to: {out_dir}')
        os.makedirs(out_dir, exist_ok=True)
        for record in self:
            fp_out = os.path.join(out_dir, record.osti_id + '.pdf')
            if not os.path.exists(fp_out):
                try:
                    record.download(fp_out)
                except Exception as e:
                    msg = (f'Could not download OSTI ID {record.osti_id} '
                           f'"{record.title}": {e}')
                    logger.exception(msg)

        logger.info('Finished download!')

    @property
    def meta(self):
        """Get a meta dataframe with details on all of the OSTI records.

        Returns
        -------
        pd.DataFrame
        """
        i = 0
        attrs = ('authors', 'title', 'year', 'date', 'doi', 'osti_id', 'url')
        df = pd.DataFrame(columns=attrs)
        for record in self:
            for attr in attrs:
                out = getattr(record, attr)
                if not isinstance(out, str):
                    out = json.dumps(out)
                df.at[i, attr] = out
            df.at[i, 'fn'] = f'{record.osti_id}.pdf'
            i += 1
        return df

    @classmethod
    def from_osti_ids(cls, oids):
        """Initialize OSTI records from one or more numerical IDS

        Parameters
        ----------
        oids : list
            List of string or integer OSTI IDs which are typically 7 digit
            numbers

        Returns
        -------
        out : OstiList
            OstiList object with entries for each oid input.
        """
        if not isinstance(oids, (list, tuple)):
            oids = [oids]
        oids = [str(oid) for oid in oids]
        out = None
        for oid in oids:
            iout = cls(cls.BASE_URL + '/' + oid)
            if out is None:
                out = iout
            else:
                out += iout
        return out
