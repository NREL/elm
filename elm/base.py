# -*- coding: utf-8 -*-
"""
ELM abstract class for API calls
"""
from abc import ABC
import os
import numpy as np
import asyncio
import aiohttp
import openai
import requests
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

    EMBEDDING_URL = 'https://api.openai.com/v1/embeddings'
    """OpenAI embedding API URL"""

    URL = 'https://api.openai.com/v1/chat/completions'
    """OpenAI API URL to be used with environment variable OPENAI_API_KEY. Use
    an Azure API endpoint to trigger Azure usage along with environment
    variables AZURE_OPENAI_KEY, AZURE_OPENAI_VERSION, and
    AZURE_OPENAI_ENDPOINT"""

    HEADERS = {"Content-Type": "application/json",
               "Authorization": f"Bearer {openai.api_key}",
               "api-key": f"{openai.api_key}",
               }
    """OpenAI API Headers"""

    MODEL_ROLE = "You are a research assistant that answers questions."
    """High level model role"""

    TOKENIZER_ALIASES = {'gpt-35-turbo': 'gpt-3.5-turbo',
                         'gpt-4-32k': 'gpt-4-32k-0314'
                         }
    """Optional mappings for unusual Azure names to tiktoken/openai names."""

    def __init__(self, model=None):
        """
        Parameters
        ----------
        model : None | str
            Optional specification of OpenAI model to use. Default is
            cls.DEFAULT_MODEL
        """
        self.model = model or self.DEFAULT_MODEL
        self.api_queue = None
        self.messages = []
        self.clear()

        if 'openai.azure.com' in self.URL.lower():
            key = os.environ.get("AZURE_OPENAI_KEY")
            version = os.environ.get("AZURE_OPENAI_VERSION")
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            assert key is not None, "Must set AZURE_OPENAI_KEY!"
            assert version is not None, "Must set AZURE_OPENAI_VERSION!"
            assert endpoint is not None, "Must set AZURE_OPENAI_ENDPOINT!"
            self._client = openai.AzureOpenAI(api_key=key,
                                              api_version=version,
                                              azure_endpoint=endpoint)
        else:
            key = os.environ.get("OPENAI_API_KEY")
            assert key is not None, "Must set OPENAI_API_KEY!"
            self._client = openai.OpenAI(api_key=key)

    @property
    def all_messages_txt(self):
        """Get a string printout of the full conversation with the LLM

        Returns
        -------
        str
        """
        messages = [f"{msg['role'].upper()}: {msg['content']}"
                    for msg in self.messages]
        messages = '\n\n'.join(messages)
        return messages

    def clear(self):
        """Clear chat history and reduce messages to just the initial model
        role message."""
        self.messages = [{"role": "system", "content": self.MODEL_ROLE}]

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
        out : dict
            API response in json format
        """

        out = None
        kwargs = dict(url=url, headers=headers, json=request_json)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(**kwargs) as response:
                    out = await response.json()

        except Exception as e:
            logger.debug(f'Error in OpenAI API call from '
                         f'`aiohttp.ClientSession().post(**kwargs)` with '
                         f'kwargs: {kwargs}')
            logger.exception('Error in OpenAI API call! Turn on debug logging '
                             'to see full query that caused error.')
            out = {'error': str(e)}

        return out

    async def call_api_async(self, url, headers, all_request_jsons,
                             ignore_error=None, rate_limit=40e3):
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
        ignore_error : None | callable
            Optional callable to parse API error string. If the callable
            returns True, the error will be ignored, the API call will not be
            tried again, and the output will be an empty string.
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
        self.api_queue = ApiQueue(url, headers, all_request_jsons,
                                  ignore_error=ignore_error,
                                  rate_limit=rate_limit)
        out = await self.api_queue.run()
        return out

    def chat(self, query, temperature=0):
        """Have a continuous chat with the LLM including context from previous
        chat() calls stored as attributes in this class.

        Parameters
        ----------
        query : str
            Question to ask ChatGPT
        temperature : float
            GPT model temperature, a measure of response entropy from 0 to 1. 0
            is more reliable and nearly deterministic; 1 will give the model
            more creative freedom and may not return as factual of results.

        Returns
        -------
        response : str
            Model response
        """

        self.messages.append({"role": "user", "content": query})

        kwargs = dict(model=self.model,
                      messages=self.messages,
                      temperature=temperature,
                      stream=False)

        response = self._client.chat.completions.create(**kwargs)
        response = response.choices[0].message.content
        self.messages.append({'role': 'assistant', 'content': response})

        return response

    def generic_query(self, query, model_role=None, temperature=0):
        """Ask a generic single query without conversation

        Parameters
        ----------
        query : str
            Question to ask ChatGPT
        model_role : str | None
            Role for the model to take, e.g.: "You are a research assistant".
            This defaults to self.MODEL_ROLE
        temperature : float
            GPT model temperature, a measure of response entropy from 0 to 1. 0
            is more reliable and nearly deterministic; 1 will give the model
            more creative freedom and may not return as factual of results.

        Returns
        -------
        response : str
            Model response
        """

        model_role = model_role or self.MODEL_ROLE
        messages = [{"role": "system", "content": model_role},
                    {"role": "user", "content": query}]
        kwargs = dict(model=self.model,
                      messages=messages,
                      temperature=temperature,
                      stream=False)

        response = self._client.chat.completions.create(**kwargs)
        response = response.choices[0].message.content
        return response

    async def generic_async_query(self, queries, model_role=None,
                                  temperature=0, ignore_error=None,
                                  rate_limit=40e3):
        """Run a number of generic single queries asynchronously
        (not conversational)

        NOTE: you need to call this using the await command in ipython or
        jupyter, e.g.: `out = await Summary.run_async()`

        Parameters
        ----------
        query : list
            Questions to ask ChatGPT (list of strings)
        model_role : str | None
            Role for the model to take, e.g.: "You are a research assistant".
            This defaults to self.MODEL_ROLE
        temperature : float
            GPT model temperature, a measure of response entropy from 0 to 1. 0
            is more reliable and nearly deterministic; 1 will give the model
            more creative freedom and may not return as factual of results.
        ignore_error : None | callable
            Optional callable to parse API error string. If the callable
            returns True, the error will be ignored, the API call will not be
            tried again, and the output will be an empty string.
        rate_limit : float
            OpenAI API rate limit (tokens / minute). Note that the
            gpt-3.5-turbo limit is 90k as of 4/2023, but we're using a large
            factor of safety (~1/2) because we can only count the tokens on the
            input side and assume the output is about the same count.

        Returns
        -------
        response : list
            Model responses with same length as query input.
        """

        model_role = model_role or self.MODEL_ROLE
        all_request_jsons = []
        for msg in queries:
            msg = [{'role': 'system', 'content': self.MODEL_ROLE},
                   {'role': 'user', 'content': msg}]
            req = {"model": self.model, "messages": msg,
                   "temperature": temperature}
            all_request_jsons.append(req)

        self.api_queue = ApiQueue(self.URL, self.HEADERS, all_request_jsons,
                                  ignore_error=ignore_error,
                                  rate_limit=rate_limit)
        out = await self.api_queue.run()

        for i, response in enumerate(out):
            choice = response.get('choices', [{'message': {'content': ''}}])[0]
            message = choice.get('message', {'content': ''})
            content = message.get('content', '')
            if not any(content):
                logger.error(f'Received no output for query {i + 1}!')
            else:
                out[i] = content

        return out

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
        kwargs = dict(url=cls.EMBEDDING_URL,
                      headers=cls.HEADERS,
                      json={'model': cls.EMBEDDING_MODEL,
                            'input': text})

        out = requests.post(**kwargs)
        embedding = out.json()

        try:
            embedding = embedding["data"][0]["embedding"]
        except Exception as exc:
            msg = ('Embedding request failed: {} {}'
                   .format(out.reason, embedding))
            logger.error(msg)
            raise RuntimeError(msg) from exc

        return embedding

    @classmethod
    def count_tokens(cls, text, model):
        """Return the number of tokens in a string.

        Parameters
        ----------
        text : str
            Text string to get number of tokens for
        model : str
            specification of OpenAI model to use (e.g., "gpt-3.5-turbo")

        Returns
        -------
        n : int
            Number of tokens in text
        """

        token_model = cls.TOKENIZER_ALIASES.get(model, model)
        encoding = tiktoken.encoding_for_model(token_model)

        return len(encoding.encode(text))


class ApiQueue:
    """Class to manage the parallel API queue and submission"""

    def __init__(self, url, headers, request_jsons, ignore_error=None,
                 rate_limit=40e3, max_retries=10):
        """
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
        ignore_error : None | callable
            Optional callable to parse API error string. If the callable
            returns True, the error will be ignored, the API call will not be
            tried again, and the output will be an empty string.
        rate_limit : float
            OpenAI API rate limit (tokens / minute). Note that the
            gpt-3.5-turbo limit is 90k as of 4/2023, but we're using a large
            factor of safety (~1/2) because we can only count the tokens on the
            input side and assume the output is about the same count.
        max_retries : int
            Number of times to retry an API call with an error response before
            raising an error.
        """

        self.url = url
        self.headers = headers
        self.request_jsons = request_jsons
        self.ignore_error = ignore_error
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.api_jobs = None
        self.todo = None
        self.out = None
        self.errors = None
        self.tries = None
        self._retry = False
        self._tsub = 0
        self._reset()
        self.job_names = [f'job_{str(ijob).zfill(4)}'
                          for ijob in range(len(request_jsons))]

    def _reset(self):
        self.api_jobs = {}
        self.todo = [True] * len(self)
        self.out = [None] * len(self)
        self.errors = [None] * len(self)
        self.tries = np.zeros(len(self), dtype=int)
        self._retry = False
        self._tsub = 0

    def __len__(self):
        """Number of API calls to submit"""
        return len(self.request_jsons)

    @property
    def waiting_on(self):
        """Get a list of async jobs that are being waited on."""
        return [job for ijob, job in self.api_jobs.items() if self.todo[ijob]]

    def submit_jobs(self):
        """Submit a subset jobs asynchronously and hold jobs in the `api_jobs`
        attribute. Break when the `rate_limit` is exceeded."""

        token_count = 0
        t_elap = (time.time() - self._tsub) / 60
        avail_tokens = self.rate_limit * t_elap
        avail_tokens = min(self.rate_limit, avail_tokens)

        for ijob, itodo in enumerate(self.todo):
            if (ijob not in self.api_jobs
                    and itodo
                    and token_count < avail_tokens):
                request = self.request_jsons[ijob]
                model = request['model']
                tokens = ApiBase.count_tokens(str(request), model)

                if tokens > self.rate_limit:
                    msg = ('Job index #{} with has {} tokens which '
                           'is greater than the rate limit of {}!'
                           .format(ijob, tokens, self.rate_limit))
                    logger.error(msg)
                    raise RuntimeError(msg)

                elif tokens < avail_tokens:
                    token_count += tokens
                    task = asyncio.create_task(ApiBase.call_api(self.url,
                                                                self.headers,
                                                                request),
                                               name=self.job_names[ijob])
                    self.api_jobs[ijob] = task
                    self.tries[ijob] += 1
                    self._tsub = time.time()

                    logger.debug('Submitted "{}" ({} out of {}). '
                                 'Token count: {} '
                                 '(rate limit is {}). '
                                 'Attempts: {}'
                                 .format(self.job_names[ijob],
                                         ijob + 1, len(self), token_count,
                                         self.rate_limit,
                                         self.tries[ijob]))

            elif token_count >= avail_tokens:
                token_count = 0
                break

    async def collect_jobs(self):
        """Collect asyncronous API calls and API outputs. Store outputs in the
        `out` attribute."""

        if not any(self.waiting_on):
            return

        complete, _ = await asyncio.wait(self.waiting_on,
                                         return_when=asyncio.FIRST_COMPLETED)

        for job in complete:
            job_name = job.get_name()
            ijob = self.job_names.index(job_name)
            task_out = job.result()

            if 'error' in task_out:
                msg = ('Received API error for task #{0} '
                       '(see `ApiQueue.errors[{1}]` and '
                       '`ApiQueue.request_jsons[{1}]` for more details). '
                       'Error message: {2}'.format(ijob + 1, ijob, task_out))
                self.errors[ijob] = 'Error: {}'.format(task_out)

                if (self.ignore_error is not None
                        and self.ignore_error(str(task_out))):
                    msg += ' Ignoring error and moving on.'
                    dummy = {'choices': [{'message': {'content': ''}}]}
                    self.out[ijob] = dummy
                    self.todo[ijob] = False
                else:
                    del self.api_jobs[ijob]
                    msg += ' Retrying query.'
                    self._retry = True
                logger.error(msg)

            else:
                self.out[ijob] = task_out
                self.todo[ijob] = False

        n_complete = len(self) - sum(self.todo)
        logger.debug('Finished {} API calls, {} left'
                     .format(n_complete, sum(self.todo)))

    async def run(self):
        """Run all asyncronous API calls.

        Returns
        -------
        out : list
            List of API call outputs with same ordering as `request_jsons`
            input.
        """

        self._reset()
        logger.debug('Submitting async API calls...')

        i = 0
        while any(self.todo):
            i += 1
            self._retry = False
            self.submit_jobs()
            await self.collect_jobs()

            if any(self.tries > self.max_retries):
                msg = (f'Hit {self.max_retries} retries on API queries. '
                       'Stopping. See `ApiQueue.errors` for more '
                       'details on error response')
                logger.error(msg)
                raise RuntimeError(msg)
            elif self._retry:
                time.sleep(10)
            elif i > 1e4:
                raise RuntimeError('Hit 1e4 iterations. What are you doing?')
            elif any(self.todo):
                time.sleep(5)

        return self.out
