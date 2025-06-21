# -*- coding: utf-8 -*-
"""ELM Web crawling tests"""
from random import choice
from pathlib import Path

import pytest

from elm.ords.validation.content import possibly_mentions_wind
from elm.web.website_crawl import ELMLinkScorer, ELMWebsiteCrawler


@pytest.mark.asyncio
async def test_basic_website_crawl():
    """Test crawling a website gives back results"""

    kw = {"pdf": 4608, "wecs": 2304, "wind": 1152, "zoning": 576,
          "ordinance": 288, "planning": 144, "plan": 72, "government": 36,
          "code": 12, "area": 12, "land development": 6, "land": 3,
          "municipal": 1, "department": 1}

    async def validation(doc):
        return possibly_mentions_wind(doc.text.lower())

    async def found_enough_test_docs(out_docs):
        return len(out_docs) >= 1

    crawler = ELMWebsiteCrawler(validator=validation,
                                url_scorer=ELMLinkScorer(kw).score)

    website = choice(["https://www.renocountyks.gov",
                      "https://www.blackhawkcounty.iowa.gov",
                      "https://www.elpasoco.com"])
    out_docs = await crawler.run(website,
                                 termination_callback=found_enough_test_docs)
    assert out_docs


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
