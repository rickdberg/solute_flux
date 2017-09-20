# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:39:14 2017

@author: rickdberg

Combine calculated metadata, site_info, and summary_all tables for sites with
calculated fluxes
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def comp(database, metadata, site_info, hole_info):
    engine = create_engine(database)

    # Load metadata
    sql = "SELECT * FROM {};".format(metadata)
    metadata = pd.read_sql(sql, engine)

    # Load site data
    sql = "SELECT * FROM {};".format(site_info)
    sitedata = pd.read_sql(sql, engine)

    # Load hole data
    sql = "SELECT * FROM {};".format(hole_info)
    holedata = pd.read_sql(sql, engine)
    # Group and average hole data for sites
    hole_grouped = holedata.groupby("site_key").mean().reset_index()

    # Combine all tables
    site_meta_data = pd.merge(metadata, sitedata, how='outer', on=('site_key', 'leg', 'site'))
    data = pd.merge(site_meta_data, hole_grouped, how='outer', on=('site_key')).fillna(np.nan)
    site_metadata = data.dropna(subset = ['interface_flux']).reset_index(drop=True)
    return site_metadata

# eof
