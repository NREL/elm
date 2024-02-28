# -*- coding: utf-8 -*-
"""Test ELM Ordinance location validation tests. """
import os
from pathlib import Path
from functools import partial

import pytest
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter

from elm import TEST_DATA_DIR, ApiBase
from elm.web.document import PDFDocument, HTMLDocument
from elm.utilities.parse import read_pdf
from elm.ords.llm import StructuredLLMCaller
from elm.ords.services.openai import OpenAIService
from elm.ords.services.provider import RunningAsyncServices
from elm.ords.utilities import RTS_SEPARATORS
from elm.ords.validation.location import (
    CountyValidator,
    CountyNameValidator,
    CountyJurisdictionValidator,
    URLValidator,
    _validator_check_for_doc,
)


SHOULD_SKIP = os.getenv("AZURE_OPENAI_API_KEY") is None
TESTING_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    RTS_SEPARATORS,
    chunk_size=3000,
    chunk_overlap=300,
    length_function=partial(ApiBase.count_tokens, model="gpt-4"),
)


@pytest.fixture(scope="module")
def oai_async_azure_client():
    """OpenAi Azure client to use for tests"""
    return openai.AsyncAzureOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_VERSION"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    )


@pytest.fixture()
def structured_llm_caller():
    """StructuredLLMCaller instance for testing"""
    return StructuredLLMCaller(
        llm_service=OpenAIService,
        model="gpt-4",
        temperature=0,
        seed=42,
        timeout=30,
    )


@pytest.mark.skipif(SHOULD_SKIP, reason="requires Azure OpenAI key")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "county,state,url,truth",
    [
        (
            "El Paso",
            "Indiana",
            "https://programs.dsireusa.org/system/program/detail/4332/"
            "madison-county-wind-energy-systems-ordinance",
            False,
        ),
        (
            "Madison",
            "Indiana",
            "https://programs.dsireusa.org/system/program/detail/4332/"
            "madison-county-wind-energy-systems-ordinance",
            False,
        ),
        (
            "Madison",
            "North Carolina",
            "https://programs.dsireusa.org/system/program/detail/4332/"
            "madison-county-wind-energy-systems-ordinance",
            False,
        ),
        (
            "Decatur",
            "Indiana",
            "http://www.decaturcounty.in.gov/doc/area-plan-commission/updates/"
            "zoning_ordinance_-_article_13_wind_energy_conversion_system_"
            "(WECS).pdf",
            True,
        ),
        (
            "Decatur",
            "Colorado",
            "http://www.decaturcounty.in.gov/doc/area-plan-commission/updates/"
            "zoning_ordinance_-_article_13_wind_energy_conversion_system_"
            "(WECS).pdf",
            False,
        ),
        (
            "El Paso",
            "Indiana",
            "http://www.decaturcounty.in.gov/doc/area-plan-commission/updates/"
            "zoning_ordinance_-_article_13_wind_energy_conversion_system_"
            "(WECS).pdf",
            False,
        ),
    ],
)
async def test_url_matches_county(
    oai_async_azure_client, structured_llm_caller, county, state, url, truth
):
    """Test the URL validator class (basic execution)"""
    url_validator = URLValidator(structured_llm_caller)
    services = [OpenAIService(oai_async_azure_client, rate_limit=50_000)]
    async with RunningAsyncServices(services):
        out = await url_validator.check(url, county=county, state=state)
        assert out == truth


@pytest.mark.skipif(SHOULD_SKIP, reason="requires Azure OpenAI key")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "county,doc_fp,truth",
    [
        (
            "Decatur",
            Path(TEST_DATA_DIR) / "indiana_general_ord.pdf",
            False,
        ),
        (
            "Decatur",
            Path(TEST_DATA_DIR) / "Decatur Indiana.pdf",
            True,
        ),
        (
            "Hamlin",
            Path(TEST_DATA_DIR) / "Hamlin South Dakota.pdf",
            True,
        ),
        (
            "Atlantic",
            Path(TEST_DATA_DIR) / "Atlantic New Jersey.txt",
            False,
        ),
        (
            "Barber",
            Path(TEST_DATA_DIR) / "Barber Kansas.pdf",
            False,
        ),
    ],
)
async def test_doc_matches_county_jurisdiction(
    oai_async_azure_client, structured_llm_caller, county, doc_fp, truth
):
    """Test the `CountyJurisdictionValidator` class (basic execution)"""
    if doc_fp.suffix == ".pdf":
        with open(doc_fp, "rb") as fh:
            pages = read_pdf(fh.read())
            doc = PDFDocument(pages)
    else:
        with open(doc_fp, "r", encoding="utf-8") as fh:
            text = fh.read()
            doc = HTMLDocument([text], text_splitter=TESTING_TEXT_SPLITTER)

    cj_validator = CountyJurisdictionValidator(structured_llm_caller)
    services = [OpenAIService(oai_async_azure_client, rate_limit=100_000)]
    async with RunningAsyncServices(services):
        out = await _validator_check_for_doc(
            doc=doc, validator=cj_validator, county=county
        )
        assert out == truth


@pytest.mark.skipif(SHOULD_SKIP, reason="requires Azure OpenAI key")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "county,state,doc_fp,truth",
    [
        (
            "Decatur",
            "Indiana",
            Path(TEST_DATA_DIR) / "Decatur Indiana.pdf",
            True,
        ),
        (
            "Hamlin",
            "South Dakota",
            Path(TEST_DATA_DIR) / "Hamlin South Dakota.pdf",
            True,
        ),
        (
            "Anoka",
            "Minnesota",
            Path(TEST_DATA_DIR) / "Anoka Minnesota.txt",
            True,
        ),
    ],
)
async def test_doc_matches_county_name(
    oai_async_azure_client, structured_llm_caller, county, state, doc_fp, truth
):
    """Test the `CountyNameValidator` class (basic execution)"""
    if doc_fp.suffix == ".pdf":
        with open(doc_fp, "rb") as fh:
            pages = read_pdf(fh.read())
            doc = PDFDocument(pages)
    else:
        with open(doc_fp, "r", encoding="utf-8") as fh:
            text = fh.read()
            doc = HTMLDocument([text], text_splitter=TESTING_TEXT_SPLITTER)

    cn_validator = CountyNameValidator(structured_llm_caller)
    services = [OpenAIService(oai_async_azure_client, rate_limit=100_000)]
    async with RunningAsyncServices(services):
        out = await _validator_check_for_doc(
            doc=doc, validator=cn_validator, county=county, state=state
        )
        assert out == truth


@pytest.mark.skipif(SHOULD_SKIP, reason="requires Azure OpenAI key")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "county,state,doc_fp,url,truth",
    [
        (
            "Decatur",
            "Indiana",
            Path(TEST_DATA_DIR) / "Decatur Indiana.pdf",
            "http://www.decaturcounty.in.gov/doc/area-plan-commission/z.pdf",
            True,
        ),
        (
            "Hamlin",
            "South Dakota",
            Path(TEST_DATA_DIR) / "Hamlin South Dakota.pdf",
            "http://www.test.gov",
            True,
        ),
        (
            "Anoka",
            "Minnesota",
            Path(TEST_DATA_DIR) / "Anoka Minnesota.txt",
            "http://www.test.gov",
            False,
        ),
        (
            "Atlantic",
            "New Jersey",
            Path(TEST_DATA_DIR) / "Atlantic New Jersey.txt",
            "http://www.test.gov",
            False,
        ),
    ],
)
async def test_doc_matches_county(
    oai_async_azure_client,
    structured_llm_caller,
    county,
    state,
    doc_fp,
    url,
    truth,
):
    """Test the `CountyValidator` class (basic execution)"""
    if doc_fp.suffix == ".pdf":
        with open(doc_fp, "rb") as fh:
            pages = read_pdf(fh.read())
            doc = PDFDocument(pages)
    else:
        with open(doc_fp, "r", encoding="utf-8") as fh:
            text = fh.read()
            doc = HTMLDocument([text], text_splitter=TESTING_TEXT_SPLITTER)

    doc.metadata["source"] = url

    county_validator = CountyValidator(structured_llm_caller)
    services = [OpenAIService(oai_async_azure_client, rate_limit=100_000)]
    async with RunningAsyncServices(services):
        out = await county_validator.check(doc=doc, county=county, state=state)
        assert out == truth


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
