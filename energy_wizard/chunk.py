# -*- coding: utf-8 -*-
"""
Utility to break text up into overlapping chunks.
"""

from energy_wizard.abs import ApiBase


class Chunker(ApiBase):
    """
    Class to break text up into overlapping chunks

    NOTE: this will currently fail if a single paragraph is greater than
    the token limit. Need to add logic to split large paragraphs.
    """

    def __init__(self, text, tag=None, tokens_per_chunk=500, overlap=1,
                 token_limit=8191):
        """
        Parameters
        ----------
        text : str
            Single body of text to break up. Works well if this is a single
            document with empty lines between paragraphs.
        tag : None | str
            Optional reference tag to include at the beginning of each text
            chunk
        tokens_per_chunk : float
            Nominal token count per text chunk
        overlap : int
            Number of paragraphs to overlap between chunks
        token_limit : float
            Hard limit on the maximum number of tokens that can be embedded at
            once
        """

        super().__init__()

        self._idc = 0  # iter index for chunk
        self.text = text
        self.tag = tag
        self.tokens_per_chunk = tokens_per_chunk
        self.overlap = overlap
        self.token_limit = token_limit
        self._chunks = self.chunk_text()

    def __getitem__(self, i):
        """Get a chunk index

        Returns
        -------
        str
        """
        return self.chunks[i]

    def __iter__(self):
        self._idc = 0
        return self

    def __next__(self):
        """Iterator returns one of the text chunks at a time

        Returns
        -------
        str
        """

        if self._idc >= len(self):
            raise StopIteration

        out = self.chunks[self._idc]
        self._idc += 1
        return out

    def __len__(self):
        """Number of text chunks

        Return
        ------
        int
        """
        return len(self.chunks)

    @property
    def chunks(self):
        """List of overlapping text chunks (strings).

        Returns
        -------
        list
        """
        return self._chunks

    @property
    def paragraphs(self):
        paragraphs = self.text.split('\n\n')
        paragraphs = [p for p in paragraphs if self.is_good_paragraph(p)]
        return paragraphs

    @staticmethod
    def is_good_paragraph(paragraph):
        """Basic tests to make sure the paragraph is useful text."""
        if '.....' in paragraph:
            return False
        elif paragraph.strip().isnumeric():
            return False
        else:
            return True

    def chunk_text(self):
        """Perform the text chunking operation

        Returns
        -------
        chunks : list
            List of strings where each string is an overlapping chunk of text
        """

        tokens = [self.count_tokens(p) for p in self.paragraphs]

        chunks = []
        current = [0]
        tcount = 0

        for i, token in enumerate(tokens):
            tcount += token
            if tcount < self.tokens_per_chunk:
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
            if chunk[-1] < len(self.paragraphs) - self.overlap:
                for k in range(1, self.overlap + 1):
                    overlap_text = self.paragraphs[chunk[-1] + k]
                    new = current_text_chunk + '\n\n' + overlap_text
                    if self.count_tokens(new) < self.token_limit:
                        current_text_chunk = new

            text_chunks.append(current_text_chunk)

        if self.tag is not None:
            text_chunks = [self.tag + '\n\n' + chunk for chunk in text_chunks]

        return text_chunks
