# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:49:51 2016

@author: rickdberg
Script for building age-depth profile from biostratigraphy or
magnetostratigraphy data.

Boundaries between different sedimentation rates are manually input based on
visual inspection of data.

Inputs: Leg, Site, Hole(s), bottom_boundary, age_depth_boundaries
"""

import numpy as np
import pandas as pd
import MySQLdb
import matplotlib.pyplot as plt
from scipy import optimize
import os
import datetime

Script = os.path.basename(__file__)
Date = datetime.datetime.now()

# Site ID
Leg = '64'
Site = '474'
Holes = "('','A')"
Bottom_boundary = 'none' # 'none', or an integer depth
# age_depth_boundaries = [0, 4, 7] # Index when sorted by age
age_depth_boundaries = [0,2,7,15] # Index when sorted by depth
top_age = 0  # years

###############################################################################
###############################################################################
###############################################################################
# Load data from database

# Connect to database
user = 'root'
passwd = 'neogene227'
host = '127.0.0.1'
db = 'iodp_compiled'
con = MySQLdb.connect(user=user, passwd=passwd, host=host, db=db)
cur = con.cursor()

# Sedimentation rate profile (m/y)
# Note: Input data must have time at depth=0
sql = """SELECT depth, age FROM age_depth where leg = '{}' and site = '{}' order by 1 ;""".format(Leg, Site, Holes)
sed = pd.read_sql(sql, con)
sed = sed.as_matrix()
if min(sed[:,0]) > 0:
    sed = np.vstack((sed, np.array([0,top_age])))
sed = sed[np.argsort(sed[:,0])]
if Bottom_boundary == 'none':
    sed = sed
else:
    deepest_sed_idx = np.searchsorted(sed[:,0], Bottom_boundary)
    sed = sed[:deepest_sed_idx, :]

# Sort by age
# sed = sed[np.argsort(sed[:,1])]
datapoints = len(sed)
print('dp:', datapoints)

# Put in for loop to run linear regressions for as many sections as each site has

def age_curve(z, p):
    return p*z

plt.plot(sed[:,1]/1000000, sed[:,0], "o")
plt.ylabel('Depth (mbsf)')
plt.xlabel('Age (Ma)')

cut_depths = sed[age_depth_boundaries, 0]
age_upper = sed[age_depth_boundaries[0], 1]
sedrate_ages = [age_upper]
sedrate_depths = [cut_depths[0]]
for n in np.arange(len(cut_depths)-1):
    depth_upper = cut_depths[n]
    depth_lower = cut_depths[n+1]
    sed_isolated = np.column_stack((sed[age_depth_boundaries[n]:age_depth_boundaries[n+1]+1,0]-depth_upper, sed[age_depth_boundaries[n]:age_depth_boundaries[n+1]+1,1]-age_upper))
    p , e = optimize.curve_fit(age_curve, sed_isolated[:,0], sed_isolated[:,1])
    z = np.linspace(0, depth_lower-depth_upper, 100)
    age = age_curve(z, p)+age_upper
    plt.plot(age/1000000, z+depth_upper, '--', linewidth=2)
    age_upper = age_curve(z[-1], *p)+age_upper
    sedrate_ages.append(age_upper)
    sedrate_depths.append(depth_lower)
plt.show()

'''
cut_depths = sed[age_depth_boundaries, 0]
last_depth = 0
last_age = 0
sedrate_ages = [0]
sedrate_depths = [0]
for n in np.arange(len(cut_depths)-1):
    next_depth = cut_depths[n+1]
    sed_alt = np.column_stack((sed[:,0]-last_depth, sed[:,1]-last_age))
    p , e = optimize.curve_fit(age_curve, sed_alt[age_depth_boundaries[n]:age_depth_boundaries[n+1],0], sed_alt[age_depth_boundaries[n]:age_depth_boundaries[n+1],1])
    z = np.linspace(0, next_depth-last_depth, 100)
    age = age_curve(z, p)+last_age
    plt.plot(age/1000000, z+last_depth, '-x')
    last_age = age_curve(z[-1], *p)+last_age
    last_depth = cut_depths[n+1]
    sedrate_ages.append(last_age)
    sedrate_depths.append(last_depth)
plt.show()
'''

# Formatting for input into SQL database metadata table
Hole = ''.join(filter(str.isalpha, Holes))

# Save metadata in database
cur.execute("""select site_key from site_info where leg = '{}' and site = '{}' ;""".format(Leg, Site))
site_key = cur.fetchone()[0]
cur.execute("""insert into metadata_sed_rate (site_key, leg, site, hole,
bottom_boundary, age_depth_boundaries, sedrate_ages, sedrate_depths, datapoints, script, run_date)
VALUES ({}, '{}', '{}', '{}', '{}', '{}', '{}', '{}', {}, '{}', '{}')  ON DUPLICATE KEY UPDATE hole='{}',
bottom_boundary='{}', age_depth_boundaries='{}', sedrate_ages='{}', sedrate_depths='{}', datapoints={},
script='{}', run_date='{}' ;""".format(site_key, Leg, Site, Hole, Bottom_boundary, age_depth_boundaries, sedrate_ages, sedrate_depths, datapoints, Script, Date,
Hole, Bottom_boundary, age_depth_boundaries, sedrate_ages, sedrate_depths, datapoints, Script, Date))
con.commit()

# eof
