# -*- coding: utf-8 -*-
"""ELM Ords Temporary file cache"""
import os
import asyncio
from functools import partial
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor

from elm.ords.services.base import Service
from elm.web.utilities import write_url_doc_to_file


class TempFileCache(Service):
    """Service that locally caches files downloaded from the internet"""

    def __init__(self, td_kwargs=None, tpe_kwargs=None):
        """

        Parameters
        ----------
        td_kwargs : dict, optional
            Keyword-value argument pairs to pass to
            :class:`tempfile.TemporaryDirectory`. By default, ``None``.
        tpe_kwargs : dict, optional
            Keyword-value argument pairs to pass to
            :class:`concurrent.futures.ThreadPoolExecutor`.
            By default, ``None``.
        """
        self.td_kwargs = td_kwargs or {}
        self.tpe_kwargs = tpe_kwargs or {}
        self._pool = None
        self._td = None

    @property
    def can_process(self):
        """bool: Always ``True`` (limiting is handled by asyncio)"""
        return True

    def acquire_resources(self):
        """Open thread pool and temp directory"""
        self._pool = ThreadPoolExecutor(**self.tpe_kwargs)
        self._td = TemporaryDirectory(**self.td_kwargs)

    def release_resources(self):
        """Shutdown thread pool and cleanup temp directory"""
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._td.cleanup()

    async def process(self, doc, file_content, make_name_unique=False):
        """Write URL doc to file asynchronously.

        Parameters
        ----------
        doc : elm.web.document.Document
            Document containing meta information about the file. Must
            have a "source" key in the `metadata` dict containing the
            URL, which will be converted to a file name using
            :func:`compute_fn_from_url`.
        file_content : str | bytes
            File content, typically string text for HTML files and bytes
            for PDF file.
        make_name_unique : bool, optional
            Option to make file name unique by adding a UUID at the end
            of the file name. By default, ``False``.

        Returns
        -------
        Path
            Path to output file.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._pool,
            partial(
                write_url_doc_to_file,
                doc,
                file_content,
                self._td.name,
                make_name_unique=make_name_unique,
            ),
        )
        return result
