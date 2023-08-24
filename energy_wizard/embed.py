# -*- coding: utf-8 -*-
"""
Energy Wizard text embedding
"""
import openai
import re
import os
import logging

from energy_wizard.base import ApiBase
from energy_wizard.chunk import Chunker


logger = logging.getLogger(__name__)


class ChunkAndEmbed(ApiBase):
    """Class to chunk text data and create embeddings"""

    DEFAULT_MODEL = 'text-embedding-ada-002'
    """Default model to do embeddings."""

    def __init__(self, text, tag=None, model=None, tokens_per_chunk=500,
                 overlap=1):
        """
        Parameters
        ----------
        text : str
            Single continuous piece of text to chunk up by paragraph and embed
            or filepath to .txt file containing one piece of text.
        tag : None | str
            Optional reference tag to include at the beginning of each text
            chunk
        model : None | str
            Optional specification of OpenAI model to use. Default is
            cls.DEFAULT_MODEL
        tokens_per_chunk : float
            Nominal token count per text chunk
        overlap : int
            Number of paragraphs to overlap between chunks
        """
        super().__init__(model)

        self.text = text
        self.tag = tag

        if os.path.isfile(text):
            logger.info('Loading text file: {}'.format(text))
            with open(text, 'r') as f:
                self.text = f.read()

        assert isinstance(self.text, str)
        self.text = self.clean_tables(self.text)

        self.text_chunks = Chunker(self.text, tag=tag,
                                   tokens_per_chunk=tokens_per_chunk,
                                   overlap=overlap)

    @staticmethod
    def clean_tables(text):
        """Make sure that table headers are in the same paragraph as the table
        itself. Typically, tables are looked for with pipes and hyphens, which
        is how GPT cleans tables in text."""

        # looks for "Table N.", should expand to other formats with additional
        # regex patterns later
        table_regex = r"^Table [0-9]+."

        lines = text.split('\n')
        for idx, line in enumerate(lines[:-2]):
            next_line_1 = lines[idx + 1]
            next_line_2 = lines[idx + 2]
            match = re.search(table_regex, line)
            cond1 = match is not None
            cond2 = next_line_1.strip() == ''
            cond3 = next_line_2.startswith('|')

            if all([cond1, cond2, cond3]):
                lines[idx + 1] = line
                lines[idx] = ''

        return '\n'.join(lines)

    async def run_async(self, rate_limit=175e3):
        """Run text embedding

        NOTE: you need to call this using the await command in ipython or
        jupyter, e.g.: `out = await ChunkAndEmbed.run_async()`

        Parameters
        ----------
        rate_limit : float
            OpenAI API rate limit (tokens / minute). Note that the
            embedding limit is 350k as of 4/2023, but we're using a large
            factor of safety (~1/2) because we can only count the tokens on the
            input side and assume the output is about the same count.

        Returns
        -------
        embedding : list
            List of 1D arrays representing the embeddings for all text chunks
        """

        if not isinstance(self.text_chunks, Chunker):
            msg = ('You must init a Chunker obj with the text before '
                   'running async embeddings!')
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info('Embedding {} text chunks...'
                    .format(len(self.text_chunks)))

        all_request_jsons = []
        for chunk in self.text_chunks:
            req = {"input": chunk, "model": self.model}

            if 'azure' in str(openai.api_type).lower():
                req['engine'] = self.model

            all_request_jsons.append(req)

        embeddings = await self.call_api_async(self.EMBEDDING_URL,
                                               self.HEADERS,
                                               all_request_jsons,
                                               rate_limit=rate_limit)

        for i, chunk in enumerate(embeddings):
            try:
                embeddings[i] = chunk['data'][0]['embedding']
            except Exception:
                msg = ('Could not get embeddings for chunk {}, '
                       'received API response: {}'.format(i + 1, chunk))
                logger.error(msg)
                embeddings[i] = None

        logger.info('Finished all embeddings.')

        return embeddings
