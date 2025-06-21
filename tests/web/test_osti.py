# -*- coding: utf-8 -*-
"""
Test
"""
import os
import time
import random
import platform
import tempfile

from flaky import flaky
import pytest

from elm import OstiList


def _random_delay(*__):
    """Randomly sleep for a short time; used for flaky reruns"""
    time.sleep(random.uniform(0.5, 3.0))
    return True


@flaky(max_runs=5, min_passes=1)
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Too flaky on windows")
def test_osti_from_url():
    """Test osti list, make sure we can find LA100 documents"""
    url = ('https://www.osti.gov/api/v1/records?'
           'research_org=NREL'
           '&sort=publication_date%20desc'
           '&product_type=Technical%20Report'
           '&has_fulltext=true'
           '&publication_date_start=03/01/2021'
           '&publication_date_end=03/01/2021')

    osti = OstiList(url, n_pages=1e6)
    docs = []
    for orec in osti:
        if orec.title is not None and isinstance(orec.title, str):
            if 'la100' in orec.title.lower():
                docs.append(orec.title)

    assert len(docs) == 12


@flaky(max_runs=10, min_passes=1, rerun_filter=_random_delay)
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Too flaky on windows")
def test_osti_from_oids():
    """Test osti list, make sure we can find specific oids from storage futures
    study"""
    oids = ['1832215', '1811650', 1785959, '1785688', 1763974]
    osti = OstiList.from_osti_ids(oids)
    assert len(osti) == len(oids)


@flaky(max_runs=5, min_passes=1)
@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Too flaky on windows")
def test_osti_download():
    """Test osti download"""
    oids = 1962806  # single small report
    osti = OstiList.from_osti_ids(oids)
    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, 'out')
        osti.download(out_dir)
        fps = os.listdir(out_dir)
        assert len(fps) == 1
        assert all(fp.endswith('.pdf') for fp in fps)
