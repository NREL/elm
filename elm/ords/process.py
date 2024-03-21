# -*- coding: utf-8 -*-
"""ELM Ordinance full processing logic"""
import os
import time
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from functools import partial

import openai
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

from elm import ApiBase
from elm.ords.download import download_county_ordinance
from elm.ords.extraction import (
    extract_ordinance_text_with_ngram_validation,
    extract_ordinance_values,
)
from elm.ords.services.usage import UsageTracker
from elm.ords.services.openai import OpenAIService, usage_from_response
from elm.ords.services.provider import RunningAsyncServices
from elm.ords.services.threaded import (
    TempFileCache,
    FileMover,
    CleanedFileWriter,
    UsageUpdater,
)
from elm.ords.services.cpu import PDFLoader, read_pdf_doc, read_pdf_doc_ocr
from elm.ords.utilities import (
    RTS_SEPARATORS,
    load_all_county_info,
    load_counties_from_fp,
)
from elm.ords.utilities.location import County
from elm.ords.utilities.queued_logging import LocationFileLog, LogListener


logger = logging.getLogger(__name__)


OUT_COLS = [
    "county",
    "state",
    "FIPS",
    "feature",
    "fixed_value",
    "mult_value",
    "mult_type",
    "adder",
    "min_dist",
    "max_dist",
    "value",
    "units",
    "ord_year",
    "last_updated",
    "section",
    "source",
    "comment",
]


async def process_counties_with_openai(
    out_dir,
    county_fp=None,
    model="gpt-4",
    azure_api_key=None,
    azure_version=None,
    azure_endpoint=None,
    llm_call_kwargs=None,
    llm_service_rate_limit=4000,
    text_splitter_chunk_size=3000,
    text_splitter_chunk_overlap=300,
    num_urls_to_check_per_county=5,
    file_loader_kwargs=None,
    pytesseract_exe_fp=None,
    td_kwargs=None,
    tpe_kwargs=None,
    ppe_kwargs=None,
    log_dir=None,
    clean_dir=None,
    log_level="INFO",
):
    """Download and extract ordinances for a list of counties.

    Parameters
    ----------
    out_dir : path-like
        Path to output directory. This directory will be created if it
        does not exist. This directory will contain the structured
        ordinance output CSV as well as all of the scraped ordinance
        documents (PDFs and HTML text files). Usage information and
        default options for log/clean directories will also be stored
        here.
    county_fp : path-like, optional
        Path to CSV file containing a list of counties to extract
        ordinance information for. This CSV should have "County" and
        "State" columns that contains the county and state names.
        By default, ``None``, which runs the extraction for all known
        counties (this is untested and not currently recommended).
    model : str, optional
        Name of LLM model to perform scraping. By default, ``"gpt-4"``.
    azure_api_key : str, optional
        Azure OpenAI API key. By default, ``None``, which pulls the key
        from the environment variable ``AZURE_OPENAI_API_KEY`` instead.
    azure_version : str, optional
        Azure OpenAI API version. By default, ``None``, which pulls the
        version from the environment variable ``AZURE_OPENAI_VERSION``
        instead.
    azure_endpoint : str, optional
        Azure OpenAI API endpoint. By default, ``None``, which pulls the
        endpoint from the environment variable ``AZURE_OPENAI_ENDPOINT``
        instead.
    llm_call_kwargs : dict, optional
        Keyword-value pairs used to initialize an
        `elm.ords.llm.LLMCaller` instance. By default, ``None``.
    llm_service_rate_limit : int, optional
        Token rate limit of LLm service being used (OpenAI).
        By default, ``4000``.
    text_splitter_chunk_size : int, optional
        Chunk size input to
        `langchain.text_splitter.RecursiveCharacterTextSplitter`.
        By default, ``3000``.
    text_splitter_chunk_overlap : int, optional
        Chunk overlap input to
        `langchain.text_splitter.RecursiveCharacterTextSplitter`.
        By default, ``300``.
    num_urls_to_check_per_county : int, optional
        Number of unique Google search result URL's to check for
        ordinance document. By default, ``5``.
    pytesseract_exe_fp : path-like, optional
        Path to pytesseract executable. If this option is specified, OCR
        parsing for PDf files will be enabled via pytesseract.
        By default, ``None``.
    td_kwargs : dict, optional
        Keyword-value argument pairs to pass to
        :class:`tempfile.TemporaryDirectory`. The temporary directory is
        used to store files downloaded from the web that are still being
        parsed for ordinance information. By default, ``None``.
    tpe_kwargs : dict, optional
        Keyword-value argument pairs to pass to
        :class:`concurrent.futures.ThreadPoolExecutor`. The thread pool
        executor is used to run I/O intensive tasks like writing to a
        log file. By default, ``None``.
    ppe_kwargs : dict, optional
        Keyword-value argument pairs to pass to
        :class:`concurrent.futures.ProcessPoolExecutor`. The process
        pool executor is used to run CPU intensive tasks like loading
        a PDF file. By default, ``None``.
    log_dir : path-like, optional
        Path to directory for log files. This directory will be created
        if it does not exist. By default, ``None``, which
        creates a ``logs`` folder in the output directory for the
        county-specific log files.
    clean_dir : path-like, optional
        Path to directory for cleaned ordinance text output. This
        directory will be created if it does not exist. By default,
        ``None``, which creates a ``clean`` folder in the output
        directory for the cleaned ordinance text files.
    log_level : str, optional
        Log level to set for county retrieval and parsing loggers.
        By default, ``"INFO"``.

    Returns
    -------
    pd.DataFrame
        DataFrame of parsed ordinance information. This file will also
        be stored in the output directory under "wind_db.csv".
    """
    start_time = time.time()
    out_dir, log_dir, clean_dir = _setup_folders(
        out_dir, log_dir=log_dir, clean_dir=clean_dir
    )
    counties = _load_counties_to_process(county_fp)
    azure_api_key, azure_version, azure_endpoint = _validate_api_params(
        azure_api_key, azure_version, azure_endpoint
    )

    tpe_kwargs = _configure_thread_pool_kwargs(tpe_kwargs)
    file_loader_kwargs = _configure_file_loader_kwargs(file_loader_kwargs)
    if pytesseract_exe_fp is not None:
        _setup_pytesseract(pytesseract_exe_fp)
        file_loader_kwargs.update({"pdf_ocr_read_coroutine": read_pdf_doc_ocr})

    text_splitter = RecursiveCharacterTextSplitter(
        RTS_SEPARATORS,
        chunk_size=text_splitter_chunk_size,
        chunk_overlap=text_splitter_chunk_overlap,
        length_function=partial(ApiBase.count_tokens, model=model),
    )
    client = openai.AsyncAzureOpenAI(
        api_key=azure_api_key,
        api_version=azure_version,
        azure_endpoint=azure_endpoint,
    )
    log_listener = LogListener(["elm"], level=log_level)

    services = [
        OpenAIService(client, rate_limit=llm_service_rate_limit),
        TempFileCache(td_kwargs=td_kwargs, tpe_kwargs=tpe_kwargs),
        FileMover(out_dir, tpe_kwargs=tpe_kwargs),
        CleanedFileWriter(clean_dir, tpe_kwargs=tpe_kwargs),
        UsageUpdater(out_dir / "usage.json", tpe_kwargs=tpe_kwargs),
        PDFLoader(**(ppe_kwargs or {})),
    ]

    async with log_listener as ll, RunningAsyncServices(services):
        tasks = []
        trackers = []
        for __, row in counties.iterrows():
            county, state, fips = row[["County", "State", "FIPS"]]
            location = County(county.strip(), state=state.strip(), fips=fips)
            usage_tracker = UsageTracker(
                location.full_name, usage_from_response
            )
            trackers.append(usage_tracker)
            task = asyncio.create_task(
                download_docs_for_county_with_logging(
                    ll,
                    log_dir,
                    location,
                    text_splitter,
                    num_urls=num_urls_to_check_per_county,
                    file_loader_kwargs=file_loader_kwargs,
                    level=log_level,
                    llm_service=OpenAIService,
                    usage_tracker=usage_tracker,
                    model=model,
                    **(llm_call_kwargs or {}),
                ),
                name=location.full_name,
            )
            tasks.append(task)
        docs = await asyncio.gather(*tasks)

    db = _docs_to_db(docs)
    db.to_csv(out_dir / "wind_db.csv", index=False)

    _record_total_time(out_dir / "usage.json", time.time() - start_time)
    return db


def _setup_folders(out_dir, log_dir=None, clean_dir=None):
    """Setup output directory folders."""
    out_dir = Path(out_dir)
    log_dir = Path(log_dir) if log_dir else out_dir / "logs"
    clean_dir = Path(clean_dir) if clean_dir else out_dir / "clean"

    for folder in [out_dir, log_dir, clean_dir]:
        folder.mkdir(exist_ok=True, parents=True)
    return out_dir, log_dir, clean_dir


def _load_counties_to_process(county_fp):
    """Load the counties to retrieve documents for."""
    if county_fp is None:
        logger.info("No `county_fp` input! Loading all counties")
        return load_all_county_info()
    return load_counties_from_fp(county_fp)


def _validate_api_params(azure_api_key, azure_version, azure_endpoint):
    """Validate OpenAI API parameters."""
    azure_api_key = azure_api_key or os.environ.get("AZURE_OPENAI_API_KEY")
    azure_version = azure_version or os.environ.get("AZURE_OPENAI_VERSION")
    azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
    assert azure_api_key is not None, "Must set AZURE_OPENAI_API_KEY!"
    assert azure_version is not None, "Must set AZURE_OPENAI_VERSION!"
    assert azure_endpoint is not None, "Must set AZURE_OPENAI_ENDPOINT!"
    return azure_api_key, azure_version, azure_endpoint


def _configure_thread_pool_kwargs(tpe_kwargs):
    """Set thread pool workers to 5 if user didn't specify."""
    tpe_kwargs = tpe_kwargs or {}
    tpe_kwargs.setdefault("max_workers", 5)
    return tpe_kwargs


def _configure_file_loader_kwargs(file_loader_kwargs):
    """Add PDF reading coroutine to kwargs."""
    file_loader_kwargs = file_loader_kwargs or {}
    file_loader_kwargs.update({"pdf_read_coroutine": read_pdf_doc})
    return file_loader_kwargs


async def download_docs_for_county_with_logging(
    listener,
    log_dir,
    county,
    text_splitter,
    num_urls=5,
    file_loader_kwargs=None,
    level="INFO",
    **kwargs,
):
    """Retrieve ordinance document for a single county with async logs.

    Parameters
    ----------
    listener : elm.ords.utilities.queued_logging.LogListener
        Active ``LogListener`` instance that can be passed to
        :cls:`elm.ords.utilities.queued_logging.LocationFileLog`.
    log_dir : path-like
        Path to output directory to contain log file.
    county : elm.ords.utilities.location.Location
        County to retrieve ordinance document for.
    text_splitter : obj, optional
        Instance of an object that implements a `split_text` method.
        The method should take text as input (str) and return a list
        of text chunks. Langchain's text splitters should work for this
        input.
    num_urls : int, optional
        Number of unique Google search result URL's to check for
        ordinance document. By default, ``5``.
    file_loader_kwargs : dict, optional
        Dictionary of keyword-argument pairs to initialize
        :cls:`elm.web.file_loader.AsyncFileLoader` with. The
        "pw_launch_kwargs" key in these will also be used to initialize
        the :cls:`elm.web.google_search.PlaywrightGoogleLinkSearch` used
        for the google URL search. By default, ``None``.
    level : str, optional
        Log level to set for retrieval logger. By default, ``"INFO"``.
    **kwargs
        Keyword-value pairs used to initialize an
        `elm.ords.llm.LLMCaller` instance.

    Returns
    -------
    elm.web.document.BaseDocument | None
        Document instance for the ordinance document, or ``None`` if no
        document was found. Extracted ordinance information is stored in
        the document's ``metadata`` attribute.
    """
    with LocationFileLog(
        listener, log_dir, county=county.full_name, level=level
    ):
        task = asyncio.create_task(
            download_doc_for_county(
                county,
                text_splitter,
                num_urls=num_urls,
                file_loader_kwargs=file_loader_kwargs,
                **kwargs,
            ),
            name=county.full_name,
        )
        doc, *__ = await asyncio.gather(task)
        return doc


async def download_doc_for_county(
    county, text_splitter, num_urls=5, file_loader_kwargs=None, **kwargs
):
    """Download and parse ordinance document for a single county.

    Parameters
    ----------
    county : elm.ords.utilities.location.Location
        County to retrieve ordinance document for.
    text_splitter : obj, optional
        Instance of an object that implements a `split_text` method.
        The method should take text as input (str) and return a list
        of text chunks. Langchain's text splitters should work for this
        input.
    num_urls : int, optional
        Number of unique Google search result URL's to check for
        ordinance document. By default, ``5``.
    file_loader_kwargs : dict, optional
        Dictionary of keyword-argument pairs to initialize
        :cls:`elm.web.file_loader.AsyncFileLoader` with. The
        "pw_launch_kwargs" key in these will also be used to initialize
        the :cls:`elm.web.google_search.PlaywrightGoogleLinkSearch` used
        for the google URL search. By default, ``None``.
    **kwargs
        Keyword-value pairs used to initialize an
        `elm.ords.llm.LLMCaller` instance.

    Returns
    -------
    elm.web.document.BaseDocument | None
        Document instance for the ordinance document, or ``None`` if no
        document was found. Extracted ordinance information is stored in
        the document's ``metadata`` attribute.
    """
    start_time = time.time()
    doc = await download_county_ordinance(
        county,
        text_splitter,
        num_urls=num_urls,
        file_loader_kwargs=file_loader_kwargs,
        **kwargs,
    )
    if doc is None:
        await _record_time_and_usage(start_time, **kwargs)
        return None

    doc.metadata["location"] = county
    doc.metadata["location_name"] = county.full_name
    await _record_usage(**kwargs)

    doc = await _move_file_to_out_dir(doc)
    doc = await extract_ordinance_text_with_ngram_validation(
        doc, text_splitter, **kwargs
    )
    await _record_usage(**kwargs)

    doc = await _write_cleaned_text(doc)
    doc = await extract_ordinance_values(doc, **kwargs)
    await _record_time_and_usage(start_time, **kwargs)
    return doc


async def _record_usage(**kwargs):
    """Dump usage to file if tracker found in kwargs."""
    usage_tracker = kwargs.get("usage_tracker")
    if usage_tracker:
        await UsageUpdater.call(usage_tracker)


async def _record_time_and_usage(start_time, **kwargs):
    """Add elapsed time before updating usage to file."""
    seconds_elapsed = time.time() - start_time
    usage_tracker = kwargs.get("usage_tracker")
    if usage_tracker:
        usage_tracker["total_time_seconds"] = seconds_elapsed
        usage_tracker["total_time"] = str(timedelta(seconds=seconds_elapsed))
        await UsageUpdater.call(usage_tracker)


async def _move_file_to_out_dir(doc):
    """Move PDF or HTML text file to output directory."""
    out_fp = await FileMover.call(doc)
    doc.metadata["out_fp"] = out_fp
    return doc


async def _write_cleaned_text(doc):
    """Write cleaned text to `clean_dir`."""
    out_fp = await CleanedFileWriter.call(doc)
    doc.metadata["cleaned_fp"] = out_fp
    return doc


def _setup_pytesseract(exe_fp):
    """Set the pytesseract command."""
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = exe_fp


def _record_total_time(fp, seconds_elapsed):
    """Dump usage to an existing file."""
    if not Path(fp).exists():
        usage_info = {}
    else:
        with open(fp, "r") as fh:
            usage_info = json.load(fh)

    usage_info["total_time_seconds"] = seconds_elapsed
    usage_info["total_time"] = str(timedelta(seconds=seconds_elapsed))

    with open(fp, "w") as fh:
        json.dump(usage_info, fh, indent=4)


def _docs_to_db(docs):
    """Convert list of docs to output database."""
    db = []
    for doc in docs:
        if doc is None:
            continue

        if "ordinance_values" not in doc.metadata:
            continue

        results = _db_results(doc)
        results = _formatted_db(results)
        db.append(results)

    if not db:
        return pd.DataFrame(columns=OUT_COLS)

    db = pd.concat(db)
    db = _empirical_adjustments(db)
    return _formatted_db(db)


def _db_results(doc):
    """Extract results from doc metadata to DataFrame."""
    results = doc.metadata.get("ordinance_values")
    if results is None:
        return None

    results["source"] = doc.metadata.get("sources")
    year = doc.metadata.get("date", (None, None, None))[0]
    results["ord_year"] = year if year is not None and year > 0 else None
    results["last_updated"] = datetime.now().strftime("%m/%d/%Y")

    location = doc.metadata["location"]
    results["FIPS"] = location.fips
    results["county"] = location.name
    results["state"] = location.state
    return results


def _empirical_adjustments(db):
    """Post-processing adjustments based on empirical observations.

    Current adjustments include:

        - Limit adder to max of 250 ft.
            - Chat GPT likes to report large values here, but in
            practice all values manually observed in ordinance documents
            are below 250 ft.

    """
    if "adder" in db.columns:
        db.loc[db["adder"] > 250, "adder"] = None
    return db


def _formatted_db(db):
    """Format DataFrame for output."""
    out_cols = [col for col in OUT_COLS if col in db.columns]
    return db[out_cols]
