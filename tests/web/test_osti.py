# -*- coding: utf-8 -*-
"""
Test
"""
import os
import tempfile
from elm import OstiList


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
    docs = [orec.title for orec in osti if 'la100' in orec.title.lower()]
    assert len(docs) == 12


def test_osti_from_oids():
    """Test osti list, make sure we can find specific oids from storage futures
    study"""
    oids = ['1832215', '1811650', 1785959, '1785688', 1763974]
    osti = OstiList.from_osti_ids(oids)
    assert len(osti) == len(oids)


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
