# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:22:16 2017
Author: Rick Berg, University of Washington School of Oceanography, Seattle WA

Module to assign a cutoff depth for the porosity data based on site
lithology and porosity profile structure. This cutoff depth is then used in
modules that calculate fluxes of solutes. This allows a better
compaction curve fit and subsequently better pore water burial flux estimate.

Inputs:
Leg:              drilling leg/expedition number
Site:             drilling site number
Holes:            drilling hole IDs
por_cutoff_depth: bottom boundary of compaction regime (mbsf)
Ocean:            concentration of conservative solute in the ocean (mM)
Solute_db:        solute name for inserting into database
Solute:           solute name in database
engine:           SQLAlchemy engine
conctable:        name of MySQL solute concentration table
portable:         name of MySQL porosity (MAD) table
metadata_table:   name of MySQL metadata table
site_info:        name of MySQL site information table
hole_info:        name of MySQL hole information table

Outputs:
por_cutoff_depth: bottom boundary of compaction regime (mbsf)

"""
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import matplotlib.gridspec as gridspec

import flux_functions as ff
from site_metadata_compiler import metadata_compiler

# Site information
Leg = '315'
Site = 'C0001'
Holes = "('E','F','H') or hole is null"
por_cutoff_depth = 195  # Integer depth, otherwise np.nan

# Species parameters
Solute = 'Mg'
Ocean = 54
Solute_db = 'Mg'

# Connect to database
engine = create_engine("mysql://root:neogene227@localhost/iodp_compiled")
conctable = 'iw_all'
portable = 'mad_all'
metadata_table = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"

###############################################################################
###############################################################################
con = engine.connect()

# Load site data
site_metadata = metadata_compiler(engine, metadata_table, site_info,
                                  hole_info, Leg, Site)
concunique, temp_gradient, bottom_conc, bottom_temp, bottom_temp_est, pordata, sedtimes, seddepths, sedrate, picks, age_depth_boundaries, advection = ff.load_and_prep(Leg, Site, Holes, Solute, Ocean, engine, conctable, portable, site_metadata)

# Average duplicates and apply porosity cutoff depth
por = ff.averages(pordata[:, 0],
                  pordata[:, 1])
por_all = por
if ~np.isnan(por_cutoff_depth):
    por = por[por[:,0] <= por_cutoff_depth]
else:
    por_cutoff_depth = "NULL"

# Fit porosity curve
por_fit = ff.porosity_fit(por)
por_error = ff.rmse(ff.por_curve(por[:,0], por, por_fit), por[:,1])

# Plot input data
plt.ioff()
figure_1, (ax1) = plt.subplots(1, 1, figsize=(4, 7))
grid = gridspec.GridSpec(3, 8, wspace=0.7)
ax1 = plt.subplot(grid[0:3, :8])
ax1.grid()
figure_1.suptitle(r"$Expedition\ {},\ Site\ {}$".format(Leg, Site),
                  fontsize=20)
ax1.plot(por_all[:, 1],
         por_all[:, 0],
         'mo',
         label='Measured')
ax1.plot(ff.por_curve(por[:,0], por, *por_fit),
         por[:,0],
         'k-',
         label='Curve fit',
         linewidth=3)
ax1.legend(loc='best', fontsize='small')
ax1.set_ylabel('Depth (mbsf)')
ax1.set_xlabel('Porosity')
ax1.locator_params(axis='x', nbins=4)
ax1.invert_yaxis()
figure_1.show()

# Retrieve Site key
site_key = con.execute("""select site_key
                from site_info
                where leg = '{}' and site = '{}'
                ;""".format(Leg, Site))
site_key = site_key.fetchone()[0]

# Send metadata to database
sql= """insert into metadata_{}_flux (site_key,leg,site,por_cutoff)
                   VALUES ({}, '{}', '{}', {})
                   ON DUPLICATE KEY UPDATE por_cutoff={}
                   ;""".format(Solute_db, site_key,Leg,Site,
                               por_cutoff_depth,por_cutoff_depth)
con.execute(sql)

# eof
