# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:13:57 2017

@author: rickdberg

Module to calculate sediment-water interface fluxes of a dissolved constituent
Uses an exponential fit to the upper sediment section for concentration gradient

Does not do Monte Carlo analysis

Positive flux is downward (into the sediment)

"""
import numpy as np
import datetime
from pylab import savefig
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

import flux_modules
import data_handling
from site_metadata_compiler import metadata_compiler


plt.close('all')
###############################################################################
###############################################################################
###############################################################################
# Site Information
Leg = '323'
Site = 'U1345'
Holes = "('A') or hole is null"
Comments = ''
Complete = 'yes'

# Species parameters
Solute = 'Mg_ic'  # Change to Mg_ic if needed, based on what's available in database
Ds = 1.875*10**-2  # m^2 per year free diffusion coefficient at 18C (ref?)
TempD = 18  # Temperature at which diffusion coefficient is known
Precision = 0.02  # measurement precision
Ocean = 54  # Concentration in modern ocean (mM)
Solute_db = 'Mg' # Solute label to send to the database

# Model parameters
dp = 4 # Number of concentration datapoints to use for exponential curve fit
z = 0  # Depth (meters below seafloor) at which to calculate flux
cycles = 5000  # Monte Carlo simulations
runtime_errors = 0

# Connect to database
engine = create_engine("mysql://root:neogene227@localhost/iodp_compiled")
con = engine.connect()
conctable = 'iw_all'
portable = 'mad_all'
metadata_table = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"
Hole = ''.join(filter(str.isupper, filter(str.isalpha, Holes)))  # Formatting for saving in metadata

###############################################################################
###############################################################################
# Load and prepare all input data
site_metadata = metadata_compiler(engine, metadata_table, site_info, hole_info, Leg, Site)

# dp = int(site_metadata.datapoints[0])
# Comments = site_metadata.comments[0]

concunique, temp_gradient, bottom_conc, bottom_temp, bottom_temp_est, pordata, sedtimes, seddepths, sedrate, picks, age_depth_boundaries, advection = flux_modules.load_and_prep(Leg, Site, Holes, Solute, Ocean, engine, conctable, portable, site_metadata)

# Fit pore water concentration curve
conc_fit = flux_modules.concentration_fit(concunique, dp)
conc_interp_fit_plot = flux_modules.conc_curve(np.linspace(concunique[0,0], concunique[dp-1,0], num=50), *conc_fit)

# R-squared function
r_squared = flux_modules.rsq(flux_modules.conc_curve(concunique[:dp,0], *conc_fit), concunique[:dp,1])

# Fit porosity curve
por = data_handling.averages(pordata[:, 0], pordata[:, 1])  # Average duplicates
por_all = por
if site_metadata.por_cutoff.notnull().all():
    por = por[por[:,0] <= float(site_metadata.por_cutoff[0])]

por_fit = flux_modules.porosity_fit(por)
por_error = data_handling.rmse(flux_modules.por_curve(por[:,0], por, por_fit), por[:,1])

# Sediment properties at flux depth
porosity = flux_modules.por_curve(z, por, por_fit)[0]
tortuosity = 1-np.log(porosity**2)

# Calculate effective diffusion coefficient
D_in_situ = flux_modules.d_stp(TempD, bottom_temp, Ds)
Dsed = D_in_situ/tortuosity  # Effective diffusion coefficient

# Calculate pore water burial flux rate
pwburialflux = flux_modules.pw_burial(seddepths, sedtimes, por_fit, por)

# Calculate solute flux
flux, burial_flux, gradient = flux_modules.flux_model(conc_fit, concunique, z, pwburialflux, porosity, Dsed, advection, dp, Site)

# Plot input data
plt.ioff()
figure_1 = flux_modules.flux_plots(concunique, conc_interp_fit_plot, por, por_all, por_fit, bottom_temp, picks, sedtimes, seddepths, Leg, Site, Solute_db, flux, dp, temp_gradient)
figure_1.show()


# Save Figure
savefig(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output flux figures\interface_flux_{}_{}.png".format(Leg, Site))


# Monte Carlo Simulation
interface_fluxes, interface_fluxes_log, cycles, por_error, mean_flux, median_flux, stdev_flux, skewness, z_score, mean_flux_log, median_flux_log, stdev_flux_log, stdev_flux_lower, stdev_flux_upper, skewness_log, z_score_log, runtime_errors = flux_modules.monte_carlo(cycles, Precision, concunique, bottom_temp_est, dp, por, por_fit, seddepths, sedtimes, TempD, bottom_temp, z, advection, Leg, Site, Solute_db, Ds, por_error, conc_fit, runtime_errors)

# Plot Monte Carlo Distributions
mc_figure =flux_modules.monte_carlo_plot(interface_fluxes, median_flux, stdev_flux, skewness, z_score, interface_fluxes_log, median_flux_log, stdev_flux_log, skewness_log, z_score_log)
mc_figure.show()

# Save figure and fluxes from each run
savefig(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output monte carlo distributions\montecarlo_{}_{}.png".format(Leg, Site))
np.savetxt(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output monte carlo distributions\monte carlo_{}_{}.csv".format(Leg, Site), interface_fluxes, delimiter=",")


# Date and time
Date = datetime.datetime.now()

# Retrieve Site key
site_key = con.execute("""select site_key
                from site_info
                where leg = '{}' and site = '{}'
                ;""".format(Leg, Site))
site_key = site_key.fetchone()[0]

# Send metadata to database
flux_modules.flux_to_sql(con, Solute_db, site_key,Leg,Site,Hole,Solute,flux,
                burial_flux,gradient,porosity,z,dp,bottom_conc,conc_fit,r_squared,
                           age_depth_boundaries,sedrate,advection,Precision,Ds,
                           TempD,bottom_temp,bottom_temp_est,cycles,
                           por_error,mean_flux,median_flux,stdev_flux,
                           skewness,z_score,mean_flux_log,median_flux_log,
                           stdev_flux_log,stdev_flux_lower,stdev_flux_upper,
                           skewness_log,z_score_log,runtime_errors,Date,Comments,Complete)



# eof
