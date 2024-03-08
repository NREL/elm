# -*- coding: utf-8 -*-
"""ELM Ordinance Threaded services"""
import shutil
import asyncio
from pathlib import Path
from functools import partial
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor

from elm.ords.services.base import Service
from elm.web.utilities import write_url_doc_to_file


class ThreadedService(Service):
    """Service that contains a ThreadPoolExecutor instance"""

    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        **kwargs
            Keyword-value argument pairs to pass to
            :class:`concurrent.futures.ThreadPoolExecutor`.
            By default, ``None``.
        """
        self._tpe_kwargs = kwargs or {}
        self.pool = None

    def acquire_resources(self):
        """Open thread pool and temp directory"""
        self.pool = ThreadPoolExecutor(**self._tpe_kwargs)

    def release_resources(self):
        """Shutdown thread pool and cleanup temp directory"""
        self.pool.shutdown(wait=True, cancel_futures=True)


class TempFileCache(ThreadedService):
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
        super().__init__(**(tpe_kwargs or {}))
        self._td_kwargs = td_kwargs or {}
        self._td = None

    @property
    def can_process(self):
        """bool: Always ``True`` (limiting is handled by asyncio)"""
        return True

    def acquire_resources(self):
        """Open thread pool and temp directory"""
        super().acquire_resources()
        self._td = TemporaryDirectory(**self._td_kwargs)

    def release_resources(self):
        """Shutdown thread pool and cleanup temp directory"""
        self._td.cleanup()
        super().release_resources()

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
            self.pool,
            partial(
                write_url_doc_to_file,
                doc,
                file_content,
                self._td.name,
                make_name_unique=make_name_unique,
            ),
        )
        return result


class FileMover(ThreadedService):
    """Service that moves files to an output directory"""

    def __init__(self, out_dir, tpe_kwargs=None):
        """

        Parameters
        ----------
        out_dir : path-like
            Path to output directory that files should be moved to.
        tpe_kwargs : dict, optional
            Keyword-value argument pairs to pass to
            :class:`concurrent.futures.ThreadPoolExecutor`.
            By default, ``None``.
        """
        super().__init__(**(tpe_kwargs or {}))
        self.out_dir = out_dir

    @property
    def can_process(self):
        """bool: Always ``True`` (limiting is handled by asyncio)"""
        return True

    async def process(self, doc):
        """Move doc contents from temp directory to out directory.

        Parameters
        ----------
        doc : elm.web.document.Document
            Document containing meta information about the file. Must
            have a "cache_fn" key in the `metadata` dict containing the
            temp file path, otherwise no file will be moved.

        Returns
        -------
        Path | None
            Path to output file, or `None` if no file was moved.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self.pool, partial(_move_file, doc, self.out_dir)
        )
        return result


def _move_file(doc, out_dir):
    """Move a file from a temp directory to an output directory."""
    cached_fp = doc.metadata.get("cache_fn")
    if cached_fp is None:
        return

    cached_fp = Path(cached_fp)
    out_fp = Path(out_dir) / cached_fp.name
    shutil.move(cached_fp, Path(out_dir) / cached_fp.name)
    return out_fp
