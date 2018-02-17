# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:22:16 2017
Author: Rick Berg, University of Washington School of Oceanography, Seattle WA

Module to assign a cutoff depth for the porosity data based on site
lithology and porosity profile structure. This cutoff depth is then used in
modules that calculate fluxes of solutes. This allows a better
compaction curve fit and subsequently better pore water burial flux estimate.

Inputs:
leg:              drilling leg/expedition number
site:             drilling site number
holes:            drilling hole IDs
por_cutoff_depth: bottom boundary of compaction regime (mbsf)
ocean:            concentration of conservative solute in the ocean (mM)
solute:           solute name in database

Outputs:
por_cutoff_depth: bottom boundary of compaction regime (mbsf)

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import flux_functions as ff
from user_parameters import (engine, conctable, portable, metadata_table,
                             site_info, hole_info)

# Site information
leg = '315'
site = 'C0001'
holes = "('E','F','H') or hole is null"
por_cutoff_depth = 195  # Integer depth, otherwise np.nan

# Species parameters
solute = 'Mg'
solute_units = 'mM'
ocean = 54

###############################################################################
###############################################################################
con = engine.connect()

# Load site data
site_metadata = ff.metadata_compiler(engine, metadata_table, site_info,
                                  hole_info, leg, site)
(concunique, temp_gradient, bottom_conc, bottom_temp, bottom_temp_est,
 pordata, sedtimes, seddepths, sedrate, picks, age_depth_boundaries,
 advection) = ff.load_and_prep(leg, site, holes, solute, ocean, engine,
                               conctable, portable, site_metadata,
                               solute_units)

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
figure_1.suptitle(r"$Expedition\ {0},\ Site\ {1}$".format(leg, site),
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
                where leg = '{0}' and site = '{1}'
                ;""".format(leg, site))
site_key = site_key.fetchone()[0]

# Send metadata to database
sql= """insert into porosity_cutoff (site_key,site,por_cutoff)
                   VALUES ({0}, '{1}', {2})
                   ON DUPLICATE KEY UPDATE por_cutoff={2}
                   ;""".format(site_key,site,por_cutoff_depth)
con.execute(sql)

# eof
