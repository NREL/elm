"""Scrape water conservation district well permit requirements"""
from elm.web.search import web_search_links_as_docs
import asyncio


QUERIES = ['panola county groundwater conservation district',
           "panola county groundwater conservation district well permits",
           "panola county groundwater conservation district well requirements"]

loop = asyncio.get_event_loop()
docs = asyncio.run(web_search_links_as_docs(
    QUERIES,
    pdf_read_kwargs={"verbose": False},
    ))

breakpoint()