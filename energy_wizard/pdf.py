# -*- coding: utf-8 -*-
"""
Energy Wizard PDF to text parser
"""
import requests
import copy
from PyPDF2 import PdfReader
import logging

from energy_wizard.abs import ApiBase


logger = logging.getLogger(__name__)


class PDFtoTXT(ApiBase):
    """Class to parse text from a PDF document."""

    def __init__(self, fp, page_range=None, model=None):
        """
        Parameters
        ----------
        fp : str
            Filepath to .pdf file to extract.
        page_range : None | list
            Optional 2-entry list/tuple to set starting and ending pages
            (python indexing)
        model : None | str
            Optional specification of OpenAI model to use. Default is
            cls.DEFAULT_MODEL
        """
        super().__init__(model)
        self.fp = fp
        self.raw_text = self.load_pdf(page_range)
        self.text = None
        self.full = None

    def load_pdf(self, page_range):
        """Basic load of pdf to text strings

        Parameters
        ----------
        page_range : None | list
            Optional 2-entry list/tuple to set starting and ending pages
            (python indexing)

        Returns
        -------
        out : list
            List of strings where each entry is a page. This is the raw PDF
            text before GPT cleaning
        """

        logger.info('Loading PDF: {}'.format(self.fp))
        out = []
        pdf = PdfReader(self.fp)

        if page_range is not None:
            assert len(page_range) == 2
            page_range = slice(*page_range)
        else:
            page_range = slice(0, None)

        for i, page in enumerate(pdf.pages[page_range]):
            page_text = page.extract_text()
            if len(page_text.strip()) == 0:
                logger.debug('Skipping empty page {} out of {}'
                             .format(i + 1 + page_range.start, len(pdf.pages)))
            else:
                out.append(page_text)
                logger.debug('Loaded page {} out of {}'
                             .format(i + 1 + page_range.start, len(pdf.pages)))

        logger.info('Finished loading PDF.')
        return out

    @staticmethod
    def make_gpt_messages(pdf_raw_text):
        """Make the chat completion messages list for input to GPT

        Parameters
        ----------
        pdf_raw_text : str
            Raw PDF text to be cleaned

        Returns
        -------
        messages : list
            Messages for OpenAI chat completion model. Typically this looks
            like this:
                [{"role": "system", "content": "You do this..."},
                 {"role": "user", "content": "Please do this: {}"}]
        """
        query = ('Text extracted from a PDF: '
                 '\"\"\"\n{}\"\"\"\n\n'
                 'The text above was extracted from a PDF document. '
                 'Can you make it nicely formatted? '
                 'Please only return the formatted text, nothing else.'
                 .format(pdf_raw_text))

        role_str = ('You clean up poorly formatted text '
                    'extracted from PDF documents.')
        messages = [{"role": "system", "content": role_str},
                    {"role": "user", "content": query}]

        return messages

    def clean_txt(self):
        """Use GPT to clean raw pdf text in serial calls to the OpenAI API.

        Returns
        -------
        clean_pages : list
            List of clean text strings where each list entry is a page from the
            PDF
        """

        logger.info('Cleaning PDF text...')
        clean_pages = []

        for i, raw_page in enumerate(self.raw_text):
            msg = self.make_gpt_messages(copy.deepcopy(raw_page))
            req = {"model": self.model, "messages": msg, "temperature": 0.0}

            kwargs = dict(url=self.URL, headers=self.HEADERS, json=req)

            try:
                response = requests.post(**kwargs)
                response = response.json()
            except Exception as e:
                msg = 'Error in OpenAI API call!'
                logger.exception(msg)
                response = {'error': str(e)}

            choice = response.get('choices', [{'message': {'content': ''}}])[0]
            message = choice.get('message', {'content': ''})
            content = message.get('content', '')
            clean_pages.append(content)
            logger.debug('Cleaned page {} out of {}'
                         .format(i + 1, len(self.raw_text)))

        logger.info('Finished cleaning PDF.')

        self.text = clean_pages
        self.full = self.combine_pages(self.text)
        self.validate_clean()

        return clean_pages

    async def clean_txt_async(self, rate_limit=40e3):
        """Use GPT to clean raw pdf text in parallel calls to the OpenAI API.

        NOTE: you need to call this using the await command in ipython or
        jupyter, e.g.: `out = await PDFtoTXT.clean_txt_async()`

        Parameters
        ----------
        rate_limit : float
            OpenAI API rate limit (tokens / minute). Note that the
            gpt-3.5-turbo limit is 90k as of 4/2023, but we're using a large
            factor of safety (~1/2) because we can only count the tokens on the
            input side and assume the output is about the same count.

        Returns
        -------
        clean_pages : list
            List of clean text strings where each list entry is a page from the
            PDF
        """

        logger.info('Cleaning PDF text asyncronously...')

        all_request_jsons = []
        for page in self.raw_text:
            msg = self.make_gpt_messages(page)
            req = {"model": self.model, "messages": msg, "temperature": 0.0}
            all_request_jsons.append(req)

        clean_pages = await self.call_api_async(self.URL, self.HEADERS,
                                                all_request_jsons,
                                                rate_limit=rate_limit)

        for i, page in enumerate(clean_pages):
            choice = page.get('choices', [{'message': {'content': ''}}])[0]
            message = choice.get('message', {'content': ''})
            content = message.get('content', '')
            clean_pages[i] = content

        logger.info('Finished cleaning PDF.')

        self.text = clean_pages
        self.full = self.combine_pages(self.text)
        self.validate_clean()

        return clean_pages

    def validate_clean(self):
        """Run some basic checks on the GPT cleaned text vs. the raw text"""
        repl = ('\n', '.', ',', '-', '/', ':')

        if not any(self.full.replace('\n', '').strip()):
            msg = 'Didnt get ANY clean output text!'
            logger.error(msg)
            raise RuntimeError(msg)

        def replace_chars_for_clean(text):
            for char in repl:
                text = text.replace(char, ' ')
            return text

        for i, (raw, clean) in enumerate(zip(self.raw_text, self.text)):
            raw_words = replace_chars_for_clean(raw).split(' ')
            clean_words = replace_chars_for_clean(clean).split(' ')

            raw_words = set([x for x in raw_words if len(x) > 2])
            clean_words = set([x for x in clean_words if len(x) > 2])

            isin = sum(x in clean_words for x in raw_words)

            perc = 100
            if isin > 0 and len(raw_words) > 0:
                perc = 100 * isin / len(raw_words)

            if perc < 70:
                logger.warning('Page {} of {} has a {:.2f}% match with {} '
                               'unique words in the raw text.'
                               .format(i + 1, len(self.raw_text), perc,
                                       len(raw_words)))
            else:
                logger.info('Page {} of {} has a {:.2f}% match with {} '
                            'unique words in the raw text.'
                            .format(i + 1, len(self.raw_text), perc,
                                    len(raw_words)))

    @staticmethod
    def combine_pages(pages):
        """Combine pages of GPT cleaned text into a single string.

        Parameters
        ----------
        pages : list
            List of clean text strings where each list entry is a page from the
            PDF

        Returns
        -------
        full : str
            Single multi-page string
        """
        full = '\n'.join(pages)
        full = full.replace('\n•', '-')
        full = full.replace('•', '-')
        return full
