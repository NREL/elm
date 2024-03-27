# -*- coding: utf-8 -*-
# pylint: disable=consider-using-with
"""ELM Ordinance Threaded services"""
import json
import shutil
import asyncio
from pathlib import Path
from functools import partial
from abc import abstractmethod
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor

from elm.ords.services.base import Service
from elm.web.utilities import write_url_doc_to_file


def _move_file(doc, out_dir):
    """Move a file from a temp directory to an output directory."""
    cached_fp = doc.metadata.get("cache_fn")
    if cached_fp is None:
        return

    cached_fp = Path(cached_fp)
    out_fn = doc.metadata.get("location_name", cached_fp.name)
    if not out_fn.endswith(cached_fp.suffix):
        out_fn = f"{out_fn}{cached_fp.suffix}"

    out_fp = Path(out_dir) / out_fn
    shutil.move(cached_fp, out_fp)
    return out_fp


def _write_cleaned_file(doc, out_dir):
    """Write cleaned ordinance text to directory."""
    cleaned_text = doc.metadata.get("cleaned_ordinance_text")
    location_name = doc.metadata.get("location_name")

    if cleaned_text is None or location_name is None:
        return

    out_fp = Path(out_dir) / f"{location_name} Summary.txt"
    with open(out_fp, "w", encoding="utf-8") as fh:
        fh.write(cleaned_text)
    return out_fp


def _write_ord_db(doc, out_dir):
    """Write parsed ordinance database to directory."""
    ord_db = doc.metadata.get("ordinance_values")
    location_name = doc.metadata.get("location_name")

    if ord_db is None or location_name is None:
        return

    out_fp = Path(out_dir) / f"{location_name} Ordinances.csv"
    ord_db.to_csv(out_fp, index=False)
    return out_fp


_PROCESSING_FUNCTIONS = {
    "move": _move_file,
    "write_clean": _write_cleaned_file,
    "write_db": _write_ord_db,
}


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


class StoreFileOnDisk(ThreadedService):
    """Abstract service that manages the storage of a file on disk.

    Storage can occur due to creation or a move of a file.
    """

    def __init__(self, out_dir, tpe_kwargs=None):
        """

        Parameters
        ----------
        out_dir : path-like
            Path to output directory where file should be stored.
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
        """Store file in out directory.

        Parameters
        ----------
        doc : elm.web.document.Document
            Document containing meta information about the file. Must
            have relevant processing keys in the `metadata` dict,
            otherwise the file may not be stored in the output
            directory.

        Returns
        -------
        Path | None
            Path to output file, or `None` if no file was stored.
        """
        return await _run_func_in_pool(
            self.pool,
            partial(_PROCESSING_FUNCTIONS[self._PROCESS], doc, self.out_dir),
        )

    @property
    @abstractmethod
    def _PROCESS(self):
        """str: Key in `_PROCESSING_FUNCTIONS` that defines the doc func."""
        raise NotImplementedError


class FileMover(StoreFileOnDisk):
    """Service that moves files to an output directory"""

    _PROCESS = "move"


class CleanedFileWriter(StoreFileOnDisk):
    """Service that writes cleaned text to a file"""

    _PROCESS = "write_clean"


class OrdDBFileWriter(StoreFileOnDisk):
    """Service that writes cleaned text to a file"""

    _PROCESS = "write_db"


class UsageUpdater(ThreadedService):
    """Service that updates usage info from a tracker into a file."""

    def __init__(self, usage_fp, tpe_kwargs=None):
        """

        Parameters
        ----------
        usage_fp : path-like
            Path to JSON file where usage should be tracked.
        tpe_kwargs : dict, optional
            Keyword-value argument pairs to pass to
            :class:`concurrent.futures.ThreadPoolExecutor`.
            By default, ``None``.
        """
        super().__init__(**(tpe_kwargs or {}))
        self.usage_fp = usage_fp
        self._is_processing = False

    @property
    def can_process(self):
        """bool: ``True`` if file not currently being written to.``"""
        return not self._is_processing

    async def process(self, tracker):
        """Add usage from tracker to file.

        Any existing usage info in the file will remain unchanged
        EXCEPT for anything under the label of the input `tracker`,
        all of which will be replaced with info from the tracker itself.

        Parameters
        ----------
        tracker : elm.ods.services.usage.UsageTracker
            A usage tracker instance that contains usage info to be
            added to output file.
        """
        self._is_processing = True
        await _run_func_in_pool(
            self.pool, partial(_dump_usage, self.usage_fp, tracker)
        )
        self._is_processing = False


async def _run_func_in_pool(pool, callable_fn):
    """Run a callable in process pool"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(pool, callable_fn)


def _dump_usage(fp, tracker):
    """Dump usage to an existing file."""
    if not Path(fp).exists():
        usage_info = {}
    else:
        with open(fp, "r") as fh:
            usage_info = json.load(fh)

    tracker.add_to(usage_info)
    with open(fp, "w") as fh:
        json.dump(usage_info, fh, indent=4)
