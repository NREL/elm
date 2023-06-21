# -*- coding: utf-8 -*-
"""
Energy Wizard text embedding
"""
import openai
import re
import os
import logging

from energy_wizard.abs import ApiBase


logger = logging.getLogger(__name__)


class ChunkAndEmbed(ApiBase):
    """Class to chunk text data and create embeddings"""

    DEFAULT_MODEL = 'text-embedding-ada-002'
    """Default model to do embeddings."""

    def __init__(self, text, tag=None, model=None):
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
        """
        super().__init__(model)

        self.text = text
        self.tag = tag
        self.text_chunks = None

        if os.path.isfile(text):
            logger.info('Loading text file: {}'.format(text))
            with open(text, 'r') as f:
                self.text = f.read()

        assert isinstance(self.text, str)
        self.text = self.clean_tables(self.text)
        paragraphs = self.text.split('\n\n')
        paragraphs = [p for p in paragraphs if self.is_good_paragraph(p)]
        self.paragraphs = paragraphs

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

    @staticmethod
    def is_good_paragraph(paragraph):
        """Basic tests to make sure the paragraph is useful text."""
        if '.....' in paragraph:
            return False
        elif paragraph.strip().isnumeric():
            return False
        else:
            return True

    def chunk_text(self, tokens_per_chunk=500, overlap=1, token_limit=8191):
        """Chunk a large text string into multiple small chunks with overlap

        NOTE: this will currently fail if a single paragraph is greater than
        the token limit. Need to add a piece to split large paragraphs.

        Parameters
        ----------
        tokens_per_chunk : float
            Nominal token count per text chunk
        overlap : int
            Number of paragraphs to overlap between chunks
        token_limit : float
            Hard limit on the maximum number of tokens that can be embedded at
            once

        Returns
        -------
        text_chunks : list
            List of strings where each string is an overlapping chunk of text
        """

        tokens = [self.count_tokens(p) for p in self.paragraphs]

        chunks = []
        current = [0]
        tcount = 0

        for i, token in enumerate(tokens):
            tcount += token
            if tcount < tokens_per_chunk:
                current.append(i)
            else:
                tcount = 0

                if len(current) > 0:
                    chunks.append(current)
                    current = []

                if i > 0:
                    current = [i]

        if len(current) > 0:
            chunks.append(current)

        text_chunks = []
        for chunk in chunks:
            current_text_chunk = [self.paragraphs[i] for i in chunk]
            current_text_chunk = '\n\n'.join(current_text_chunk)
            if chunk[-1] < len(self.paragraphs) - overlap:
                for k in range(1, overlap + 1):
                    overlap_text = self.paragraphs[chunk[-1] + k]
                    new = current_text_chunk + '\n\n' + overlap_text
                    if self.count_tokens(new) < token_limit:
                        current_text_chunk = new

            text_chunks.append(current_text_chunk)

        if self.tag is not None:
            text_chunks = [self.tag + '\n\n' + chunk for chunk in text_chunks]

        self.text_chunks = text_chunks

        return text_chunks

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

        if not isinstance(self.text_chunks, (list, tuple)):
            msg = ('You must chunk the text before running async embeddings!')
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
