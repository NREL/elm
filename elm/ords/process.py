# -*- coding: utf-8 -*-
"""ELM Ordinance full processing logic"""
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
from elm.utilities import validate_azure_api_params
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
    OrdDBFileWriter,
    UsageUpdater,
)
from elm.ords.services.cpu import PDFLoader, read_pdf_doc, read_pdf_doc_ocr
from elm.ords.utilities import (
    RTS_SEPARATORS,
    load_all_county_info,
    load_counties_from_fp,
)
from elm.ords.utilities.location import County
from elm.ords.utilities.queued_logging import (
    LocationFileLog,
    LogListener,
    NoLocationFilter,
)

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

CHECK_COLS = [
    "fixed_value",
    "mult_value",
    "adder",
    "min_dist",
    "max_dist",
    "value",
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
    max_num_concurrent_browsers=10,
    file_loader_kwargs=None,
    pytesseract_exe_fp=None,
    td_kwargs=None,
    tpe_kwargs=None,
    ppe_kwargs=None,
    log_dir=None,
    clean_dir=None,
    county_ords_dir=None,
    county_dbs_dir=None,
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
    max_num_concurrent_browsers : int, optional
        Number of unique concurrent browser instances to open when
        performing Google search. Setting this number too high on a
        machine with limited processing can lead to increased timeouts
        and therefore decreased quality of Google search results.
        By default, ``10``.
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
    county_ords_dir : path-like, optional
        Path to directory for individual county ordinance file outputs.
        This directory will be created if it does not exist.
        By default, ``None``, which creates a ``county_ord_files``
        folder in the output directory.
    county_dbs_dir : path-like, optional
        Path to directory for individual county ordinance database
        outputs. This directory will be created if it does not exist.
        By default, ``None``, which creates a ``county_dbs`` folder in
        the output directory.
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
    log_listener = LogListener(["elm"], level=log_level)
    dirs = _setup_folders(
        out_dir,
        log_dir=log_dir,
        clean_dir=clean_dir,
        cod=county_ords_dir,
        cdd=county_dbs_dir,
    )
    out_dir, log_dir, clean_dir, county_ords_dir, county_dbs_dir = dirs
    async with log_listener as ll:
        _setup_main_logging(log_dir, log_level, ll)
        db = await _process_with_logs(
            out_dir,
            log_dir,
            clean_dir,
            county_ords_dir,
            county_dbs_dir,
            ll,
            county_fp=county_fp,
            model=model,
            azure_api_key=azure_api_key,
            azure_version=azure_version,
            azure_endpoint=azure_endpoint,
            llm_call_kwargs=llm_call_kwargs,
            llm_service_rate_limit=llm_service_rate_limit,
            text_splitter_chunk_size=text_splitter_chunk_size,
            text_splitter_chunk_overlap=text_splitter_chunk_overlap,
            num_urls_to_check_per_county=num_urls_to_check_per_county,
            max_num_concurrent_browsers=max_num_concurrent_browsers,
            file_loader_kwargs=file_loader_kwargs,
            pytesseract_exe_fp=pytesseract_exe_fp,
            td_kwargs=td_kwargs,
            tpe_kwargs=tpe_kwargs,
            ppe_kwargs=ppe_kwargs,
            log_level=log_level,
        )
    _record_total_time(out_dir / "usage.json", time.time() - start_time)
    return db


async def _process_with_logs(
    out_dir,
    log_dir,
    clean_dir,
    county_ords_dir,
    county_dbs_dir,
    log_listener,
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
    max_num_concurrent_browsers=10,
    file_loader_kwargs=None,
    pytesseract_exe_fp=None,
    td_kwargs=None,
    tpe_kwargs=None,
    ppe_kwargs=None,
    log_level="INFO",
):
    """Process counties with logging enabled."""
    counties = _load_counties_to_process(county_fp)
    azure_api_key, azure_version, azure_endpoint = validate_azure_api_params(
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

    services = [
        OpenAIService(client, rate_limit=llm_service_rate_limit),
        TempFileCache(td_kwargs=td_kwargs, tpe_kwargs=tpe_kwargs),
        FileMover(county_ords_dir, tpe_kwargs=tpe_kwargs),
        CleanedFileWriter(clean_dir, tpe_kwargs=tpe_kwargs),
        OrdDBFileWriter(county_dbs_dir, tpe_kwargs=tpe_kwargs),
        UsageUpdater(out_dir / "usage.json", tpe_kwargs=tpe_kwargs),
        PDFLoader(**(ppe_kwargs or {})),
    ]

    browser_semaphore = (
        asyncio.Semaphore(max_num_concurrent_browsers)
        if max_num_concurrent_browsers
        else None
    )

    async with RunningAsyncServices(services):
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
                process_county_with_logging(
                    log_listener,
                    log_dir,
                    location,
                    text_splitter,
                    num_urls=num_urls_to_check_per_county,
                    file_loader_kwargs=file_loader_kwargs,
                    browser_semaphore=browser_semaphore,
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
    return db


def _setup_main_logging(log_dir, level, listener):
    """Setup main logger for catching exceptions during execution."""
    handler = logging.FileHandler(log_dir / "main.log", encoding="utf-8")
    handler.setLevel(level)
    handler.addFilter(NoLocationFilter())
    listener.addHandler(handler)


def _setup_folders(
    out_dir,
    log_dir=None,
    clean_dir=None,
    cod=None,
    cdd=None,
):
    """Setup output directory folders."""
    out_dir = Path(out_dir)
    out_folders = [
        out_dir,
        Path(log_dir) if log_dir else out_dir / "logs",
        Path(clean_dir) if clean_dir else out_dir / "clean",
        Path(cod) if cod else out_dir / "county_ord_files",
        Path(cdd) if cdd else out_dir / "county_dbs",
    ]
    for folder in out_folders:
        folder.mkdir(exist_ok=True, parents=True)
    return out_folders


def _load_counties_to_process(county_fp):
    """Load the counties to retrieve documents for."""
    if county_fp is None:
        logger.info("No `county_fp` input! Loading all counties")
        return load_all_county_info()
    return load_counties_from_fp(county_fp)


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


async def process_county_with_logging(
    listener,
    log_dir,
    county,
    text_splitter,
    num_urls=5,
    file_loader_kwargs=None,
    browser_semaphore=None,
    level="INFO",
    **kwargs,
):
    """Retrieve ordinance document for a single county with async logs.

    Parameters
    ----------
    listener : elm.ords.utilities.queued_logging.LogListener
        Active ``LogListener`` instance that can be passed to
        :class:`elm.ords.utilities.queued_logging.LocationFileLog`.
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
        :class:`elm.web.file_loader.AsyncFileLoader` with. The
        "pw_launch_kwargs" key in these will also be used to initialize
        the :class:`elm.web.google_search.PlaywrightGoogleLinkSearch`
        used for the google URL search. By default, ``None``.
    browser_semaphore : asyncio.Semaphore, optional
        Semaphore instance that can be used to limit the number of
        playwright browsers open concurrently. If ``None``, no limits
        are applied. By default, ``None``.
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
        listener, log_dir, location=county.full_name, level=level
    ):
        task = asyncio.create_task(
            process_county(
                county,
                text_splitter,
                num_urls=num_urls,
                file_loader_kwargs=file_loader_kwargs,
                browser_semaphore=browser_semaphore,
                **kwargs,
            ),
            name=county.full_name,
        )
        try:
            doc, *__ = await asyncio.gather(task)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(
                "Encountered error while processing %s:", county.full_name
            )
            logger.exception(e)
            doc = None

        return doc


async def process_county(
    county,
    text_splitter,
    num_urls=5,
    file_loader_kwargs=None,
    browser_semaphore=None,
    **kwargs,
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
        :class:`elm.web.file_loader.AsyncFileLoader` with. The
        "pw_launch_kwargs" key in these will also be used to initialize
        the :class:`elm.web.google_search.PlaywrightGoogleLinkSearch`
        used for the google URL search. By default, ``None``.
    browser_semaphore : asyncio.Semaphore, optional
        Semaphore instance that can be used to limit the number of
        playwright browsers open concurrently. If ``None``, no limits
        are applied. By default, ``None``.
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
        browser_semaphore=browser_semaphore,
        **kwargs,
    )
    if doc is None:
        await _record_time_and_usage(start_time, **kwargs)
        return None

    doc.attrs["location"] = county
    doc.attrs["location_name"] = county.full_name
    await _record_usage(**kwargs)

    doc = await extract_ordinance_text_with_ngram_validation(
        doc, text_splitter, **kwargs
    )
    await _record_usage(**kwargs)

    doc = await _write_cleaned_text(doc)
    doc = await extract_ordinance_values(doc, **kwargs)

    ord_count = _num_ords_in_doc(doc)
    if ord_count > 0:
        doc = await _move_file_to_out_dir(doc)
        doc = await _write_ord_db(doc)
        logger.info(
            "%d ordinance value(s) found for %s. Outputs are here: '%s'",
            ord_count,
            county.full_name,
            doc.attrs["ord_db_fp"],
        )
    else:
        logger.info("No ordinances found for %s.", county.full_name)

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
    doc.attrs["out_fp"] = out_fp
    return doc


async def _write_cleaned_text(doc):
    """Write cleaned text to `clean_dir`."""
    out_fp = await CleanedFileWriter.call(doc)
    doc.attrs["cleaned_fp"] = out_fp
    return doc


async def _write_ord_db(doc):
    """Write cleaned text to `county_dbs_dir`."""
    out_fp = await OrdDBFileWriter.call(doc)
    doc.attrs["ord_db_fp"] = out_fp
    return doc


def _setup_pytesseract(exe_fp):
    """Set the pytesseract command."""
    import pytesseract

    logger.debug("Setting `tesseract_cmd` to %s", exe_fp)
    pytesseract.pytesseract.tesseract_cmd = exe_fp


def _record_total_time(fp, seconds_elapsed):
    """Dump usage to an existing file."""
    if not Path(fp).exists():
        usage_info = {}
    else:
        with open(fp, "r") as fh:
            usage_info = json.load(fh)

    total_time_str = str(timedelta(seconds=seconds_elapsed))
    usage_info["total_time_seconds"] = seconds_elapsed
    usage_info["total_time"] = total_time_str

    with open(fp, "w") as fh:
        json.dump(usage_info, fh, indent=4)

    logger.info("Total processing time: %s", total_time_str)


def _num_ords_in_doc(doc):
    """Check if doc contains any scraped ordinance values."""
    if doc is None:
        return 0

    if "ordinance_values" not in doc.attrs:
        return 0

    ord_vals = doc.attrs["ordinance_values"]
    if ord_vals.empty:
        return 0

    check_cols = [col for col in CHECK_COLS if col in ord_vals]
    if not check_cols:
        return 0

    return (~ord_vals[check_cols].isna()).values.sum(axis=1).sum()


def _docs_to_db(docs):
    """Convert list of docs to output database."""
    db = []
    for doc in docs:
        if doc is None or isinstance(doc, Exception):
            continue

        if _num_ords_in_doc(doc) == 0:
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
    results = doc.attrs.get("ordinance_values")
    if results is None:
        return None

    results["source"] = doc.attrs.get("source")
    year = doc.attrs.get("date", (None, None, None))[0]
    results["ord_year"] = year if year is not None and year > 0 else None
    results["last_updated"] = datetime.now().strftime("%m/%d/%Y")

    location = doc.attrs["location"]
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
