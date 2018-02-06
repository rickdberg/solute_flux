# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:13:57 2017
Author: Rick Berg, University of Washington School of Oceanography, Seattle WA

Module to calculate the total flux of a solute across the sediment-water
interface at ocean drilling locations, accounting for diffusion,
advection, and pore water burial with compaction. Although meant for the
sediment-water interface, the module is generalized to calculate the flux
across any depth horizon.
Module is designed to be paired with a MySQL database of ocean drilling data.
See flux_functions.py for detailed descriptions of each function used.

Inputs:
Leg:            drilling leg/expedition number
Site:           drilling site number
Holes:          drilling hole IDs
Comments:       comments for this location or dataset
Solute:         solute name in database
Ds:             diffusion coefficient at reference temperature (m^2 y^-1)
TempD:          reference temperature of diffusion coefficient (C)
Precision:      relative standard deviation of concentration measurements
Ocean:          concentration of conservative solute in the ocean (mM)
Solute_db:      solute name for inserting into database
z:              depth at which flux is calculated (mbsf)
cycles:         number of monte carlo simulations to run
line_fit:       "linear" or "exponential" line fit to concentration profile
dp:             concentration datapoints below seafloor used for line fit
engine:         SQLAlchemy engine
conctable:      name of MySQL solute concentration table
portable:       name of MySQL porosity (MAD) table
metadata_table: name of MySQL metadata table
site_info:      name of MySQL site information table
hole_info:      name of MySQL hole information table

In addition, filepaths to directories where the figures and output data are to
be stored need to be specified in the script.

Outputs:
flux:                 solute flux at z (mol m^-2 y^-1). Positive flux value is
                      downward (into the sediment)
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
por_error:            relative standard deviation of porosity fit
mean_flux:            average solute flux from monte carlo simulation
median_flux:          median solute flux from monte carlo simulation
stdev_flux:           standard deviation of solute flux from monte carlo sim.
skewness:             skewness of distribution of fluxes from monte carlo sim.
p_value:              Kolmogorov-Smirvov p-value of distribution of fluxes
mean_flux_log:        Log-normal average of fluxes from monte carlo sim.
median_flux_log:      Log-normal median of fluxes from monte carlo sim.
stdev_flux_log:       Log-normal standard deviation of fluxes from monte carlo sim.
stdev_flux_lower:     Upper log-normal value for 1 standard deviation
stdev_flux_upper:     Lower log-normal value for 1 standard deviation
skewness_log:         skewness of distribution of log-normal fluxes
p_value_log:          Kolmogorov-Smirvov p-value of distribution of log-normal fluxes
runtime_errors:       number of runtime errors resulting from line-fitting procedure
Date:                 date the site was modeled

Note: all other ouputs are taken directly from database site information and
hole information tables.

"""
import numpy as np
import datetime
from pylab import savefig
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

import flux_functions as ff
from site_metadata_compiler import metadata_compiler

# Site Information
Leg = '190'
Site = '1178'
Holes = "('A','B') or hole is null"
Comments = ''

# Species parameters
Solute = 'Mg'  # Change X to X_ic if needed, based on availability in database
Ds = 1.875*10**-2  # m^2 per year (Li & Gregory 1971)
TempD = 18  # degrees C
Precision = 0.02
Ocean = 54  # mM
Solute_db = 'Mg'

# Model parameters
z = 0
cycles = 5000
line_fit = 'exponential'
dp = 8

# Connect to database
engine = create_engine("mysql://root:neogene227@localhost/iodp_compiled")
conctable = 'iw_all'
portable = 'mad_all'
metadata_table = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"

# File Paths
flux_fig_path = r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output flux figures\\"
mc_fig_path = r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output monte carlo distributions\\"
mc_text_path = r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output monte carlo distributions\\"

###############################################################################
###############################################################################
con = engine.connect()
Complete = 'yes'
runtime_errors = 0
plt.close('all')

# Load and prepare all input data
site_metadata = metadata_compiler(engine, metadata_table, site_info,
                                  hole_info, Leg, Site)
concunique, temp_gradient, bottom_conc, bottom_temp, bottom_temp_est, pordata, sedtimes, seddepths, sedrate, picks, age_depth_boundaries, advection = ff.load_and_prep(Leg, Site, Holes, Solute, Ocean, engine, conctable, portable, site_metadata)
# concunique = concunique[1:,:]  # If you don't use bottom water concentration

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

# Save Figure
savefig(flux_fig_path + "interface_{}_flux_{}_{}.png".format(Solute_db, Leg, Site))

# Monte Carlo Simulation
interface_fluxes, interface_fluxes_log, cycles, por_error, mean_flux, median_flux, stdev_flux, skewness, p_value, mean_flux_log, median_flux_log, stdev_flux_log, stdev_flux_lower, stdev_flux_upper, skewness_log, p_value_log, runtime_errors, conc_fits = ff.monte_carlo(cycles, Precision, concunique, bottom_temp_est, dp, por, por_fit, seddepths, sedtimes, TempD, bottom_temp, z, advection, Leg, Site, Solute_db, Ds, por_error, conc_fit, runtime_errors, line_fit)

# Plot Monte Carlo Distributions
mc_figure =ff.monte_carlo_plot(interface_fluxes, median_flux, stdev_flux,
                               skewness, p_value, interface_fluxes_log,
                               median_flux_log, stdev_flux_log, skewness_log,
                               p_value_log)
mc_figure.show()

# Save figure and fluxes from each run
savefig(mc_fig_path + "monte carlo_{}_{}_{}.png".format(Solute_db, Leg, Site))
np.savetxt(mc_text_path + "monte carlo_{}_{}_{}.csv".format(Solute_db, Leg, Site), interface_fluxes, delimiter=",")

# Retrieve misc metadata
Hole = ''.join(filter(str.isupper, filter(str.isalpha, Holes)))
Date = datetime.datetime.now()
site_key = con.execute("""select site_key
                from site_info
                where leg = '{}' and site = '{}'
                ;""".format(Leg, Site))
site_key = site_key.fetchone()[0]

# Send metadata to database
ff.flux_to_sql(con, Solute_db, site_key,Leg,Site,Hole,Solute,flux,
               burial_flux,gradient,porosity,z,dp,bottom_conc,conc_fit,
               r_squared,age_depth_boundaries,sedrate,advection,Precision,Ds,
               TempD,bottom_temp,bottom_temp_est,cycles,por_error,mean_flux,
               median_flux,stdev_flux,skewness,p_value,mean_flux_log,
               median_flux_log,stdev_flux_log,stdev_flux_lower,
               stdev_flux_upper,skewness_log,p_value_log,runtime_errors,
               Date,Comments,Complete)

# eof
