# -*- coding: utf-8 -*-
"""
Energy Wizard PDF to text parser
"""
import asyncio
import aiohttp
import openai
import copy
import numpy as np
import tiktoken
import time
from PyPDF2 import PdfReader
import logging


logger = logging.getLogger(__name__)


class PDFtoTXT:
    """Class to parse text from a PDF document."""

    DEFAULT_MODEL = 'gpt-3.5-turbo'
    """Default model to do pdf text cleaning."""

    URL = 'https://api.openai.com/v1/chat/completions'
    """OpenAI API URL"""

    HEADERS = {"Content-Type": "application/json",
               "Authorization": f"Bearer {openai.api_key}"}
    """OpenAI API Headers"""

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
        self.fp = fp
        self.raw_text = self.load_pdf(page_range)
        self.text = None
        self.full = None
        self.model = model or self.DEFAULT_MODEL

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
        pages = pdf.pages

        if page_range is not None:
            pages = pages[page_range[0]:page_range[1]]

        for i, page in enumerate(pages):
            out.append(page.extract_text())
            logger.debug('Loaded page {} out of {}'
                         .format(i + 1, len(pages)))

        logger.info('Finished loading PDF.')
        return out

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
            messages = self.make_gpt_messages(copy.deepcopy(raw_page))
            response = openai.ChatCompletion.create(model=self.model,
                                                    messages=messages,
                                                    temperature=0)
            response_message = response["choices"][0]["message"]["content"]
            clean_pages.append(response_message)
            logger.debug('Cleaned page {} out of {}'
                         .format(i + 1, len(self.raw_text)))

        logger.info('Finished cleaning PDF.')

        self.text = clean_pages
        self.validate_clean()
        self.full = self.combine_pages(self.text)

        return clean_pages

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

    async def call_api(self, messages, temperature=0):
        """Make an asyncronous OpenAI API call.

        Parameters
        ----------
        messages : list
            Messages for OpenAI chat completion model. Typically this looks
            like this:
                [{"role": "system", "content": "You do this..."},
                 {"role": "user", "content": "Please do this: {}"}]
        temperature : float
            Model temperature (0 is near deterministic, 1 will be more
            flexible)

        Returns
        -------
        out : str
            Chat completion response string.
        """
        request_json = {"model": self.model,
                        "messages": messages,
                        "temperature": temperature}
        kwargs = dict(url=self.URL, headers=self.HEADERS, json=request_json)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(**kwargs) as response:
                    out = await response.json()

        except Exception as e:
            msg = 'Error in OpenAI API call!'
            logger.exception(msg)
            raise RuntimeError(msg) from e

        if 'error' in out:
            msg = ('Received API error {}'.format(out))
            logger.error(msg)
            raise RuntimeError(msg)

        try:
            out = out['choices'][0]['message']['content']
        except Exception as e:
            msg = 'Could not parse API output json: {}'.format(out.json())
            logger.exception(msg)
            raise RuntimeError(msg) from e

        return out

    async def clean_txt_async(self, rate_limit=40e3):
        """Use GPT to clean raw pdf text in parallel calls to the OpenAI API.

        Returns
        -------
        clean_pages : list
            List of clean text strings where each list entry is a page from the
            PDF
        rate_limit : float
            OpenAI API rate limit (tokens / minute). Note that the
            gpt-3.5-turbo limit is 90k as of 4/2023, but we're using a large
            factor of safety (~1/2)
        """

        tasks = {}
        to_do = list(range(len(self.raw_text)))
        clean_pages = [None] * len(self.raw_text)
        logger.info('Cleaning PDF text asyncronously...')

        while True:
            t0 = time.time()
            token_count = 0
            for i in to_do:
                txt = self.raw_text[i]
                token = self.num_tokens(txt)
                token_count += token
                logger.debug('Submitting {} out of {}, token count is at {} '
                             '(rate limit is {})'
                             .format(i + 1, len(self.raw_text),
                                     token_count, rate_limit))

                if token_count > rate_limit:
                    token_count = 0
                    break
                else:
                    messages = self.make_gpt_messages(txt)
                    task = asyncio.create_task(self.call_api(messages))
                    tasks[i] = task

            to_retrieve = [j for j in to_do if j in tasks]
            for j in to_retrieve:
                clean_pages[j] = await tasks[j]
                complete = sum(x is not None for x in clean_pages)
                logger.debug('Finished cleaning page {} out of {}.'
                             .format(complete, len(self.raw_text)))
                to_do.remove(j)

            logger.debug('Finished {}, still have {} left'
                         .format(complete, len(to_do)))
            if token_count == 0:
                tsleep = np.maximum(0, 60 - (time.time() - t0))
                time.sleep(tsleep)
            if not any(to_do):
                break

        logger.info('Finished cleaning PDF.')

        self.text = clean_pages
        self.validate_clean()
        self.full = self.combine_pages(self.text)

        return clean_pages

    def num_tokens(self, text):
        """Return the number of tokens in a string.

        Parameters
        ----------
        text : str
            Text string to get number of tokens for

        Returns
        -------
        n : int
            Number of tokens in text
        """
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(text))

    def validate_clean(self):
        """Run some basic checks on the GPT cleaned text vs. the raw text"""
        repl = ('\n', '.', ',', '-', '/', ':')

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
