# -*- coding: utf-8 -*-
"""
Energy Wizard text embedding
"""
import re
import os
import logging

from energy_wizard.abs import ApiBase


logger = logging.getLogger(__name__)


class ChunkAndEmbed(ApiBase):
    """Class to chunk text data and create embeddings"""

    URL = 'https://api.openai.com/v1/embeddings'
    """URL for embeddings"""

    DEFAULT_MODEL = 'text-embedding-ada-002'
    """Default model to do embeddings."""

    def __init__(self, text, tag=None, model=None):
        """
        Parameters
        ----------
        text : str | list
            One or more pieces of text to embed or filepath to .txt file
            containing one piece of text.
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
        self.paragraphs = None
        self.text_chunks = None

        if os.path.isfile(text):
            logger.info('Loading text file: {}'.format(text))
            with open(text, 'r') as f:
                self.text = f.read()

        if isinstance(self.text, (list, tuple)):
            self.text_chunks = self.text

    @staticmethod
    def clean_tables(text):
        """Make sure that table headers are in the same paragraph as the table
        itself. Typically, tables are looked for with pipes and hyphens, which
        is how GPT cleans tables in text."""

        # looks for "Table N.", should expand later
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

    def chunk_text(self, text, tokens_per_chunk=500, overlap=1):
        """Chunk a large text string into multiple small chunks with overlap

        Parameters
        ----------
        text : str
            Large body of text to chunk by paragraphs (looks for double
            new-line chars)
        tokens_per_chunk : float
            Nominal token count per text chunk
        overlap : int
            Number of paragraphs to overlap between chunks

        Returns
        -------
        text_chunks : list
            List of strings where each string is an overlapping chunk of text
        """

        assert isinstance(text, str)
        text = self.clean_tables(text)
        paragraphs = text.split('\n\n')
        paragraphs = [p for p in paragraphs if self.is_good_paragraph(p)]
        self.paragraphs = paragraphs
        tokens = [self.num_tokens(p) for p in paragraphs]

        chunks = []
        current = []
        tcount = 0

        for i, (par, token) in enumerate(zip(paragraphs, tokens)):
            tcount += token
            if tcount < tokens_per_chunk:
                current.append(i)
            else:
                tcount = 0
                chunks.append(current)
                current = [i]

        if any(current):
            chunks.append(current)

        for i, chunk in enumerate(chunks):
            if chunk[-1] < len(paragraphs) - overlap:
                chunk.append(chunk[-1] + overlap)
            chunks[i] = chunk

        text_chunks = ['\n\n'.join([paragraphs[i] for i in chunk])
                       for chunk in chunks]
        if self.tag is not None:
            text_chunks = [self.tag + '\n' + chunk for chunk in text_chunks]

        self.text_chunks = text_chunks

        return text_chunks

    async def run_async(self, rate_limit=175e3):
        """Run text embedding

        NOTE: you need to call this using the await command in ipython or
        jupyter, e.g.: `out = await Embed.run_async()`

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
            all_request_jsons.append(req)

        embeddings = await self.call_api_async(self.URL, self.HEADERS,
                                               all_request_jsons,
                                               rate_limit=rate_limit)

        for i, chunk in enumerate(embeddings):
            embeddings[i] = chunk['data'][0]['embedding']

        logger.info('Finished all embeddings.')

        return embeddings
