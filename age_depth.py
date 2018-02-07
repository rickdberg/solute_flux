# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:49:51 2016
Author: Rick Berg, University of Washington School of Oceanography, Seattle WA

Module for building age-depth profile from biostratigraphy or
magnetostratigraphy data of an ocean drilling site.
Boundaries between different sedimentation rate regimes are manually input
based on visual inspection of data. This module does not specify particular
holes at drilling site, uses all data from site.

Note: Ages in database must be in years and depths in meters.

Inputs:
leg:                  drilling leg/expedition number
site:                 drilling site number
holes:                drilling hole IDs
bottom_boundary:      lower depth boundary of data to fit (mbsf)
age_depth_boundaries: indices of age-depth data bounds for piece-wise line fit
top_age:              sediment age at sediment-water interface (default = 0 y)
user:                 MySQL database username
passwd:               MySQL database password
host:                 MySQL database IP address
db:                   MYSQL database name
age_table:            MYSQL database age-depth data table name

Outputs:
sedrate_ages:    ages at boundaries of sedimentation regime sections (years)
sedrate_depths:  depths at boundaries of sedimentation regime sections (mbsf)
datapoints:      number of datapoints to bottom_boundary
script:          name of this module
date:            date that age-depth module was run on particular site

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import os
import datetime

from user_parameters import (engine, age_table)

script = os.path.basename(__file__)
date = datetime.datetime.now()

# Site Information
leg = '199'
site = '1219'
holes = "('','A','B')"

# Model Parameters
bottom_boundary = 'none' # 'none', or an integer depth
age_depth_boundaries = [0,4,15,22,32] # Index when sorted by depth
top_age = 0  # years

###############################################################################
###############################################################################
# Create database cursor
cur = engine.connect()

# Load age-depth data
sql = """SELECT depth, age
         FROM {}
         where leg = '{}' and site = '{}' order by 1 ;
         """.format(age_table, leg, site)
sed = pd.read_sql(sql, engine)
sed = sed.as_matrix()

# If input data don't include time at depth=0, insert top_age
if min(sed[:,0]) > 0:
    sed = np.vstack((sed, np.array([0,top_age])))

# Sort data by depth
sed = sed[np.argsort(sed[:,0])]
if bottom_boundary == 'none':
    sed = sed
else:
    deepest_sed_idx = np.searchsorted(sed[:,0], bottom_boundary)
    sed = sed[:deepest_sed_idx, :]

# Print number of age-depth data available
datapoints = len(sed)
print('dp:', datapoints)

# Linear age-depth curve
def age_curve(z, p):
    return p*z

# Plot age-depth data used for fit
plt.plot(sed[:,1]/1000000, sed[:,0], "o")
plt.ylabel('Depth (mbsf)')
plt.xlabel('Age (Ma)')

# Piece-wise Linear regressions for as many sections
# as specified by age_depth_boundaries
cut_depths = sed[age_depth_boundaries, 0]
age_upper = sed[age_depth_boundaries[0], 1]
sedrate_ages = [age_upper]
sedrate_depths = [cut_depths[0]]
for n in np.arange(len(cut_depths)-1):
    depth_upper = cut_depths[n]
    depth_lower = cut_depths[n+1]
    sed_isolated = (
        np.column_stack((sed[age_depth_boundaries[n]:
                             age_depth_boundaries[n+1]+1,0]
                         -depth_upper,
                         sed[age_depth_boundaries[n]:
                             age_depth_boundaries[n+1]+1,1]
                         -age_upper)))
    p , e = optimize.curve_fit(age_curve, sed_isolated[:,0], sed_isolated[:,1])
    z = np.linspace(0, depth_lower-depth_upper, 100)
    age = age_curve(z, p)+age_upper
    plt.plot(age/1000000, z+depth_upper, '--', linewidth=2)
    age_upper = age_curve(z[-1], *p)+age_upper
    sedrate_ages.append(age_upper)
    sedrate_depths.append(depth_lower)
plt.show()

# Formatting for input into SQL database metadata table
hole = ''.join(filter(str.isalpha, holes))

# Load metadata in database
site_key = cur.execute("""select site_key
               from site_info
               where leg = '{}' and site = '{}' ;
               """.format(leg, site))
site_key = site_key.fetchone()[0]
cur.execute(
    """insert into metadata_sed_rate (site_key, leg, site, hole,
                                      bottom_boundary, age_depth_boundaries,
                                      sedrate_ages, sedrate_depths,
                                      datapoints,script, run_date)
       VALUES ({}, '{}', '{}', '{}', '{}', '{}', '{}', '{}', {}, '{}', '{}')
       ON DUPLICATE KEY UPDATE hole='{}',bottom_boundary='{}',
       age_depth_boundaries='{}', sedrate_ages='{}', sedrate_depths='{}',
       datapoints={}, script='{}', run_date='{}' ;
       """.format(site_key, leg, site, hole, bottom_boundary,
                  age_depth_boundaries, sedrate_ages, sedrate_depths,
                  datapoints, script, date,
                  hole, bottom_boundary, age_depth_boundaries, sedrate_ages,
                  sedrate_depths, datapoints, script, date))

# eof
