# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:13:57 2017
Author: Rick Berg, University of Washington School of Oceanography, Seattle WA

Module to calculate the total flux of a solute across the sediment-water
interface at ocean drilling locations, accounting for diffusion,
advection, and pore water burial with compaction.
Module is designed to be paired with a MySQL database of ocean drilling data.
See flux_functions.py for detailed descriptions of each function used.
Positive flux value is downward (into the sediment)

Intended primarily for data exploration purposes and parameter fitting.
Can also be used to adjust parameters before running flux_rerun.py.

********Differences to interface_flux.py*********
Does not perform Monte Carlo simulation to get flux errors
Has optional functionality to only send some information to database
Does not save figures or output data

Inputs:
Leg:            drilling leg/expedition number
Site:           drilling site number
Holes:          drilling hole IDs
Comments:       comments for this location or dataset
Complete:       If "partial", sends outputs to database, otherwise does not
Solute:         solute name in database
Ds:             diffusion coefficient at reference temperature (m^2 y^-1)
TempD:          reference temperature of diffusion coefficient (C)
Precision:      relative standard deviation of concentration measurements
Ocean:          concentration of conservative solute in the ocean (mM)
Solute_db:      solute name for inserting into database
z:              depth at which flux is calculated (mbsf)
line_fit:       "linear" or "exponential" line fit to concentration profile
dp:             concentration datapoints below seafloor used for line fit
engine:         SQLAlchemy engine
conctable:      name of MySQL solute concentration table
portable:       name of MySQL porosity (MAD) table
metadata_table: name of MySQL metadata table
site_info:      name of MySQL site information table
hole_info:      name of MySQL hole information table

Outputs:
flux:                 solute flux at z (mol m^-2 y^-1)
burial_flux:          solute flux due to pore water burial at z (mol m^-2 y^-1)
gradient:             pore water solute concentration gradient at z (mol m^-1)
porosity:             porosity at z
bottom_conc:          ocean bottom water solute concentration (mM)
conc_fit:             parameters of solute concentration curve (see conc_curve)
r_squared:            R-squared of regression between model and measurements
age_depth_boundaries: boundaries between discrete sedimentation rate regimes
sedrate:              modern sediment accumulation rate (solid volume per year)
bottom_temp:          ocean bottom water temperature (C)
bottom_temp_est:      ocean bottom water temperature parameter for estimation

Note: all other ouputs are taken directly from database site information and
hole information tables.

"""
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

import flux_functions as ff
from site_metadata_compiler import metadata_compiler

plt.close('all')
###############################################################################
###############################################################################
###############################################################################
# Site Information
Leg = '199'
Site = '1219'
Holes = "('A','B') or hole is null"
Comments = ''
Complete = 'no'

# Species parameters
Solute = 'Mg'  # Change X to X_ic if needed, based on availability in database
Ds = 1.875*10**-2  # m^2 per year (Li & Gregory 1971)
TempD = 18  # degrees C
Precision = 0.02
Ocean = 54  # mM
Solute_db = 'Mg'

# Model parameters
z = 0
line_fit = 'exponential'
dp = 7

# Connect to database
engine = create_engine("mysql://root:neogene227@localhost/iodp_compiled")
con = engine.connect()
conctable = 'iw_all'
portable = 'mad_all'
metadata_table = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"
# Formatting for saving in metadata
Hole = ''.join(filter(str.isupper, filter(str.isalpha, Holes)))

###############################################################################
###############################################################################
# Load and prepare all input data
site_metadata = metadata_compiler(engine, metadata_table, site_info,
                                  hole_info, Leg, Site)

# dp = int(site_metadata.datapoints[0])
# Comments = site_metadata.comments[0]

concunique, temp_gradient, bottom_conc, bottom_temp, bottom_temp_est, pordata, sedtimes, seddepths, sedrate, picks, age_depth_boundaries, advection = ff.load_and_prep(Leg, Site, Holes, Solute, Ocean, engine, conctable, portable, site_metadata)
concunique[1:,1] = concunique[1:,1]

# Fit pore water concentration curve
conc_fit = ff.concentration_fit(concunique, dp, line_fit)
conc_interp_fit_plot = ff.conc_curve(line_fit)(np.linspace(concunique[0,0],
                                                            concunique[dp-1,0],
                                                            num=50), *conc_fit)

# R-squared function
r_squared = ff.rsq(ff.conc_curve(line_fit)(concunique[:dp,0], *conc_fit),
                   concunique[:dp,1])

# Fit porosity curve
por = ff.averages(pordata[:, 0], pordata[:, 1])
por_all = por
if site_metadata.por_cutoff.notnull().all():
    por = por[por[:,0] <= float(site_metadata.por_cutoff[0])]

por_fit = ff.porosity_fit(por)
por_error = ff.rmse(ff.por_curve(por[:,0], por, por_fit), por[:,1])

# Sediment properties at flux depth
porosity = ff.por_curve(z, por, por_fit)[0]
tortuosity = 1-np.log(porosity**2)

# Calculate effective diffusion coefficient
D_in_situ = ff.d_stp(TempD, bottom_temp, Ds)
Dsed = D_in_situ/tortuosity  # Effective diffusion coefficient

# Calculate pore water burial flux rate
pwburialflux = ff.pw_burial(seddepths, sedtimes, por_fit, por)

# Calculate solute flux
flux, burial_flux, gradient = ff.flux_model(conc_fit, concunique, z,
                                            pwburialflux, porosity, Dsed,
                                            advection, dp, Site, line_fit)

# Plot input data
plt.ioff()
figure_1 = ff.flux_plots(concunique, conc_interp_fit_plot, por, por_all,
                         por_fit, bottom_temp, picks, sedtimes, seddepths,
                         Leg, Site, Solute_db, flux, dp, temp_gradient)
figure_1.show()

# Date and time
Date = datetime.datetime.now()

# Retrieve Site key
site_key = con.execute("""select site_key
                from site_info
                where leg = '{}' and site = '{}'
                ;""".format(Leg, Site))
site_key = site_key.fetchone()[0]

# Send metadata to database
if Complete == 'partial':
    ff.flux_only_to_sql(con, Solute_db, site_key,Leg,Site,Hole,Solute,flux,
                        burial_flux,gradient,porosity,z,dp,bottom_conc,
                        conc_fit,r_squared,age_depth_boundaries,sedrate,
                        advection,Precision,Ds,TempD,bottom_temp,
                        bottom_temp_est,Date,Comments,Complete)

# eof
