# -*- coding: utf-8 -*-
"""
Utility to break text up into overlapping chunks.
"""
import copy
from elm.base import ApiBase


class Chunker(ApiBase):
    """
    Class to break text up into overlapping chunks

    NOTE: very large paragraphs that exceed the tokens per chunk will not be
    split up and will still be padded with overlap.
    """

    def __init__(self, text, tag=None, tokens_per_chunk=500, overlap=1,
                 split_on='\n\n'):
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
            Nominal token count per text chunk. Overlap paragraphs will exceed
            this.
        overlap : int
            Number of paragraphs to overlap between chunks
        split_on : str
            Sub string to split text into paragraphs.
        """

        super().__init__()

        self._split_on = split_on
        self._idc = 0  # iter index for chunk
        self.text = self.clean_paragraphs(text)
        self.tag = tag
        self.tokens_per_chunk = tokens_per_chunk
        self.overlap = overlap
        self._paragraphs = None
        self._ptokens = None
        self._ctokens = None
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
        """Get a list of paragraphs in the text demarkated by an empty line.

        Returns
        -------
        list
        """
        if self._paragraphs is None:
            self._paragraphs = self.text.split(self._split_on)
            self._paragraphs = [p for p in self._paragraphs
                                if self.is_good_paragraph(p)]
        return self._paragraphs

    @staticmethod
    def clean_paragraphs(text):
        """Clean up double line breaks to make sure paragraphs can be detected
        in the text."""
        previous_len = len(text)
        while True:
            text = text.replace('\n ', '\n')
            if len(text) == previous_len:
                break
            else:
                previous_len = len(text)
        return text

    @staticmethod
    def is_good_paragraph(paragraph):
        """Basic tests to make sure the paragraph is useful text."""
        if '.....' in paragraph:
            return False
        elif paragraph.strip().isnumeric():
            return False
        else:
            return True

    @property
    def paragraph_tokens(self):
        """Number of tokens per paragraph.

        Returns
        -------
        list
        """
        if self._ptokens is None:
            self._ptokens = [self.count_tokens(p, self.model)
                             for p in self.paragraphs]
        return self._ptokens

    @property
    def chunk_tokens(self):
        """Number of tokens per chunk.

        Returns
        -------
        list
        """
        if self._ctokens is None:
            self._ctokens = [self.count_tokens(c, self.model)
                             for c in self.chunks]
        return self._ctokens

    def merge_chunks(self, chunks_input):
        """Merge chunks until they reach the token limit per chunk.

        Parameters
        ----------
        chunks_input : list
            List of list of integers: [[0, 1], [2], [3, 4]] where nested lists
            are chunks and the integers are paragraph indices

        Returns
        -------
        chunks : list
            List of list of integers: [[0, 1], [2], [3, 4]] where nested lists
            are chunks and the integers are paragraph indices
        """

        chunks = copy.deepcopy(chunks_input)

        for i in range(len(chunks) - 1):
            chunk0 = chunks[i]
            chunk1 = chunks[i + 1]
            if chunk0 is not None and chunk1 is not None:
                tcount0 = sum(self.paragraph_tokens[j] for j in chunk0)
                tcount1 = sum(self.paragraph_tokens[j] for j in chunk1)
                if tcount0 + tcount1 < self.tokens_per_chunk:
                    chunk0 += chunk1
                    chunks[i] = chunk0
                    chunks[i + 1] = None

        chunks = [c for c in chunks if c is not None]
        flat_chunks = [a for b in chunks for a in b]

        assert all(c in list(range(len(self.paragraphs))) for c in flat_chunks)

        return chunks

    def add_overlap(self, chunks_input):
        """Add overlap on either side of a text chunk. This ignores token
        limit.

        Parameters
        ----------
        chunks_input : list
            List of list of integers: [[0, 1], [2], [3, 4]] where nested lists
            are chunks and the integers are paragraph indices

        Returns
        -------
        chunks : list
            List of list of integers: [[0, 1], [2], [3, 4]] where nested lists
            are chunks and the integers are paragraph indices
        """

        if len(chunks_input) == 1 or self.overlap == 0:
            return chunks_input

        chunks = copy.deepcopy(chunks_input)

        for i, chunk1 in enumerate(chunks_input):

            if i == 0:
                chunk2 = chunks_input[i + 1]
                chunk1 = chunk1 + chunk2[:self.overlap]

            elif i == len(chunks) - 1:
                chunk0 = chunks_input[i - 1]
                chunk1 = chunk0[-self.overlap:] + chunk1

            else:
                chunk0 = chunks_input[i - 1]
                chunk2 = chunks_input[i + 1]
                chunk1 = (chunk0[-self.overlap:]
                          + chunk1
                          + chunk2[:self.overlap])

            chunks[i] = chunk1

        return chunks

    def chunk_text(self):
        """Perform the text chunking operation

        Returns
        -------
        chunks : list
            List of strings where each string is an overlapping chunk of text
        """

        chunks_input = [[i] for i in range(len(self.paragraphs))]
        while True:
            chunks = self.merge_chunks(chunks_input)
            if chunks == chunks_input:
                break
            else:
                chunks_input = copy.deepcopy(chunks)

        chunks = self.add_overlap(chunks)
        text_chunks = []
        for chunk in chunks:
            paragraphs = [self.paragraphs[c] for c in chunk]
            text_chunks.append(self._split_on.join(paragraphs))

        if self.tag is not None:
            text_chunks = [self.tag + '\n\n' + chunk for chunk in text_chunks]

        return text_chunks
