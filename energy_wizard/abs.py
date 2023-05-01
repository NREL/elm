# -*- coding: utf-8 -*-
"""
Energy Wizard abstract class for API calls
"""
from abc import ABC
import asyncio
import aiohttp
import openai
import numpy as np
import tiktoken
import time
import logging


logger = logging.getLogger(__name__)


class ApiBase(ABC):
    """Class to parse text from a PDF document."""

    DEFAULT_MODEL = 'gpt-3.5-turbo'
    """Default model to do pdf text cleaning."""

    EMBEDDING_MODEL = 'text-embedding-ada-002'
    """Default model to do text embeddings."""

    URL = 'https://api.openai.com/v1/chat/completions'
    """OpenAI API URL"""

    HEADERS = {"Content-Type": "application/json",
               "Authorization": f"Bearer {openai.api_key}"}
    """OpenAI API Headers"""

    def __init__(self, model=None):
        """
        Parameters
        ----------
        model : None | str
            Optional specification of OpenAI model to use. Default is
            cls.DEFAULT_MODEL
        """
        self.model = model or self.DEFAULT_MODEL

    @staticmethod
    async def call_api(url, headers, request_json):
        """Make an asyncronous OpenAI API call.

        Parameters
        ----------
        url : str
            OpenAI API url, typically either:
                https://api.openai.com/v1/embeddings
                https://api.openai.com/v1/chat/completions
        headers : dict
            OpenAI API headers, typically:
                {"Content-Type": "application/json",
                 "Authorization": f"Bearer {openai.api_key}"}
        request_json : dict
            API data input, typically looks like this for chat completion:
                {"model": "gpt-3.5-turbo",
                 "messages": [{"role": "system", "content": "You do this..."},
                              {"role": "user", "content": "Do this: {}"}],
                 "temperature": 0.0}

        Returns
        -------
        out : str
            Chat completion response string.
        """

        kwargs = dict(url=url, headers=headers, json=request_json)
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

        return out

    async def call_api_async(self, url, headers, all_request_jsons,
                             rate_limit=40e3):
        """Use GPT to clean raw pdf text in parallel calls to the OpenAI API.

        NOTE: you need to call this using the await command in ipython or
        jupyter, e.g.: `out = await PDFtoTXT.clean_txt_async()`

        Parameters
        ----------
        url : str
            OpenAI API url, typically either:
                https://api.openai.com/v1/embeddings
                https://api.openai.com/v1/chat/completions
        headers : dict
            OpenAI API headers, typically:
                {"Content-Type": "application/json",
                 "Authorization": f"Bearer {openai.api_key}"}
        all_request_jsons : list
            List of API data input, one entry typically looks like this for
            chat completion:
                {"model": "gpt-3.5-turbo",
                 "messages": [{"role": "system", "content": "You do this..."},
                              {"role": "user", "content": "Do this: {}"}],
                 "temperature": 0.0}
        rate_limit : float
            OpenAI API rate limit (tokens / minute). Note that the
            gpt-3.5-turbo limit is 90k as of 4/2023, but we're using a large
            factor of safety (~1/2) because we can only count the tokens on the
            input side and assume the output is about the same count.

        Returns
        -------
        out : list
            List of API outputs where each list entry is a GPT answer from the
            corresponding message in the all_request_jsons input.
        """

        tasks = {}
        to_do = list(range(len(all_request_jsons)))
        out = [None] * len(all_request_jsons)
        logger.info('Submitting async API calls...')

        while True:
            t0 = time.time()
            token_count = 0
            for i in to_do:
                request = all_request_jsons[i]
                token = self.num_tokens(str(request))
                token_count += token
                logger.debug('Submitting {} out of {}, token count is at {} '
                             '(rate limit is {})'
                             .format(i + 1, len(all_request_jsons),
                                     token_count, rate_limit))

                if token_count > rate_limit:
                    token_count = 0
                    break
                else:
                    task = asyncio.create_task(self.call_api(url, headers,
                                                             request))
                    tasks[i] = task

            to_retrieve = [j for j in to_do if j in tasks]
            for j in to_retrieve:
                out[j] = await tasks[j]
                complete = sum(x is not None for x in out)
                logger.debug('Finished API call {} out of {}.'
                             .format(complete, len(all_request_jsons)))
                to_do.remove(j)

            logger.debug('Finished {} API calls, still have {} left'
                         .format(complete, len(to_do)))
            if token_count == 0:
                tsleep = np.maximum(0, 60 - (time.time() - t0))
                time.sleep(tsleep)
            if not any(to_do):
                break

        return out

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

    @classmethod
    def get_embedding(cls, text):
        """Get the 1D array (list) embedding of a text string.

        Parameters
        ----------
        text : str
            Text to embed

        Returns
        -------
        embedding : list
            List of float that represents the numerical embedding of the text
        """
        embedding = openai.Embedding.create(model=cls.EMBEDDING_MODEL,
                                            input=text)
        embedding = embedding["data"][0]["embedding"]
        return embedding
