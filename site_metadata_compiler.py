# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:39:14 2017
Author: Rick Berg, University of Washington School of Oceanography, Seattle WA

Module with one compiler function to get all available site metadata.

"""

import numpy as np
import pandas as pd

def metadata_compiler(engine, metadata, site_info, hole_info, Leg, Site):
    """
    Combine site information, hole information, and
    existing metadata (if available) for particular drilling sites.
    Hole data must be quantitative, as they are averaged for the site.

    Inputs:
    engine:     SQLAlchemy engine
    metadata:   name of MySQL metadata table
    site_info:  name of MySQL site information table
    hole_info:  name of MySQL hole information table
    Leg:        drilling leg/expedition number
    Site:       drilling site number

    Outputs:
    site_metadata: combined table with all available data from metadata,
                   site_info, and (averaged) hole_info for a particular site
    """
    # Load site data
    sql = """SELECT *
             FROM {}
             where leg = '{}' and site = '{}';
             """.format(site_info, Leg, Site)
    sitedata = pd.read_sql(sql, engine)

    # Load hole data
    sql = """SELECT *
             FROM {}
             where leg = '{}' and site = '{}';
             """.format(hole_info, Leg, Site)
    holedata = pd.read_sql(sql, engine)

    # Group and average hole data for sites
    hole_grouped = holedata.groupby("site_key").mean().reset_index()

    # Load existing metadata, if available
    try:
        sql = """SELECT *
                 FROM {}
                 where leg = '{}' and site = '{}';
                 """.format(metadata, Leg, Site)
        ran_metadata = pd.read_sql(sql, engine)
    except:
        ran_metadata = pd.DataFrame(sitedata.loc[:,['site_key','leg','site']])

    # Combine all tables
    site_meta_data = pd.merge(ran_metadata,
                              sitedata,
                              how='outer',
                              on=('site_key', 'leg', 'site'))
    site_metadata = pd.merge(site_meta_data,
                             hole_grouped,
                             how='outer',
                             on=('site_key')).fillna(np.nan)
    return site_metadata

# eof
