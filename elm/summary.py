# -*- coding: utf-8 -*-
"""
Research Summarization and Distillation with LLMs
"""
import logging
import os

from elm.base import ApiBase
from elm.chunk import Chunker


logger = logging.getLogger(__name__)


class Summary(ApiBase):
    """Interface to perform Recursive Summarization and Distillation of
    research text"""

    MODEL_ROLE = "You are an energy scientist summarizing prior research"
    """High level model role, somewhat redundant to MODEL_INSTRUCTION"""

    MODEL_INSTRUCTION = ('Can you please summarize the text quoted above '
                         'in {n_words} words?\n\n"""\n{text_chunk}\n"""')
    """Prefix to the engineered prompt. The format args `text_chunk` and
    `n_words` will be formatted by the Summary class at runtime. `text_chunk`
    will be provided by the Summary text chunks, `n_words` is an initialization
    argument for the Summary class."""

    def __init__(self, text, model=None, n_words=500, **chunk_kwargs):
        """
        Parameters
        ----------
        text : str | list
            Single body of text to chunk up using elm.Chunker or a pre-chunked
            list of strings. Works well if this is a single document with empty
            lines between paragraphs.
        model : str
            GPT model name, default is the DEFAULT_MODEL global var
        n_words : int
            Desired length of the output text. Note that this is never perfect
            but helps guide the LLM to an approximate desired output length.
            400-600 words seems to work quite well with GPT-4. This gets
            formatted into the MODEL_INSTRUCTION attribute.
        chunk_kwargs : dict | None
            kwargs for initialization of :class:`elm.chunk.Chunker`
        """

        super().__init__(model)

        self.text = text
        self.n_words = n_words

        assert isinstance(self.text, (str, list, tuple))

        if isinstance(self.text, str):
            if os.path.isfile(text):
                logger.info('Loading text file: {}'.format(text))
                with open(text, 'r') as f:
                    self.text = f.read()
            self.text_chunks = Chunker(self.text, **chunk_kwargs)
        else:
            self.text_chunks = self.text

        self.summary_chunks = []

    def combine(self, text_summary):
        """Combine separate chunk summaries into one more comprehensive
        narrative

        Parameters
        ----------
        summary : str
            Summary of text. May be several disjointed paragraphs

        Returns
        -------
        summary : str
            Summary of text. Paragraphs will be more cohesive.
        """
        role = 'You provide editorial services for technical writing.'
        query = ('Can you combine the following paragraphs and '
                 'ease the transitions between them? '
                 f'\n\n"""{text_summary}"""')
        text_summary = self.generic_query(query, model_role=role)
        return text_summary

    def run(self, temperature=0, fancy_combine=True):
        """Use GPT to do a summary of input text.

        Parameters
        ----------
        temperature : float
            GPT model temperature, a measure of response entropy from 0 to 1. 0
            is more reliable and nearly deterministic; 1 will give the model
            more creative freedom and may not return as factual of results.
        fancy_combine : bool
            Flag to use the GPT model to combine the separate outputs into a
            cohesive summary.

        Returns
        -------
        summary : str
            Summary of text.
        """

        logger.info('Summarizing {} text chunks in serial...'
                    .format(len(self.text_chunks)))
        summary = ''

        for i, chunk in enumerate(self.text_chunks):
            logger.debug('Summarizing text chunk {} out of {}'
                         .format(i + 1, len(self.text_chunks)))

            msg = self.MODEL_INSTRUCTION.format(text_chunk=chunk,
                                                n_words=self.n_words)
            response = self.generic_query(msg, model_role=self.MODEL_ROLE,
                                          temperature=temperature)
            self.summary_chunks.append(response)
            summary += f'\n\n{response}'

        if fancy_combine:
            summary = self.combine(summary)

        logger.info('Finished all summaries.')

        return summary

    async def run_async(self, temperature=0, ignore_error=None,
                        rate_limit=40e3, fancy_combine=True):
        """Run text summary asynchronously for all text chunks

        NOTE: you need to call this using the await command in ipython or
        jupyter, e.g.: `out = await Summary.run_async()`

        Parameters
        ----------
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
        fancy_combine : bool
            Flag to use the GPT model to combine the separate outputs into a
            cohesive summary.

        Returns
        -------
        summary : str
            Summary of text.
        """

        logger.info('Summarizing {} text chunks asynchronously...'
                    .format(len(self.text_chunks)))

        queries = []
        for chunk in self.text_chunks:
            msg = self.MODEL_INSTRUCTION.format(text_chunk=chunk,
                                                n_words=self.n_words)
            queries.append(msg)

        summaries = await self.generic_async_query(queries,
                                                   model_role=self.MODEL_ROLE,
                                                   temperature=temperature,
                                                   ignore_error=ignore_error,
                                                   rate_limit=rate_limit)

        self.summary_chunks = summaries
        summary = '\n\n'.join(summaries)

        if fancy_combine:
            summary = self.combine(summary)

        logger.info('Finished all summaries.')

        return summary
