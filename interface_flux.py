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
leg:            drilling leg/expedition number
site:           drilling site number
holes:          drilling hole Ids
comments:       comments for this location or dataset
solute:         solute name in database
ds:             diffusion coefficient at reference temperature (m^2 y^-1)
temp_d:          reference temperature of diffusion coefficient (C)
precision:      relative standard deviation of concentration measurements
ocean:          concentration of conservative solute in the ocean (millimolar)
solute_db:      solute name for inserting into database
z:              depth at which flux is calculated (mbsf)
cycles:         number of monte carlo simulations to run
line_fit:       "linear" or "exponential" line fit to concentration profile
dp:             concentration datapoints below seafloor used for line fit
optimized:      'yes' or 'no', whether model parameters and ready to run monte
                carlo simulation and save output to database for this site
top_seawater:   'yes' or 'no', whether to use ocean bottom water concentration
                 as a value at z=0
solute_units:   solute concetration units, 'mM', 'uM', or 'nM'

Outputs:
flux:                 solute flux at z (mol m^-2 y^-1). Positive flux value is
                      downward (into the sediment)
burial_flux:          solute flux due to pore water burial at z (mol m^-2 y^-1)
gradient:             pore water solute concentration gradient at z (mol m^-1)
porosity:             porosity at z
bottom_conc:          ocean bottom water solute concentration (millimolar)
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
date:                 date the site was modeled

Note: all other ouputs are taken directly from database site information and
hole information tables.

"""
import numpy as np
import datetime
from pylab import savefig
import matplotlib.pyplot as plt
import sys

import flux_functions as ff
from user_parameters import (engine, conctable, portable, metadata_table,
                             site_info, hole_info, flux_fig_path, mc_fig_path,
                             mc_text_path)

###############################################################################
###############################################################################
# User-specified parameters

# Site Information
leg = '198'
site = '1212'
holes = "('A','B') or hole is null"
comments = ''

# Species parameters
solute = 'Mg'  # Change X to X_ic if needed, based on availability in database
ds = 1.875*10**-2  # m^2 per year
temp_d = 18  # degrees C
precision = 0.02
ocean = 54
solute_db = 'Mg'
solute_units = 'mM'

# Model Parameters
z = 0
cycles = 1000
line_fit = 'exponential'
dp = 8
optimized = 'no'  # whether finished optimizing Model Parameters for this site
top_seawater = 'yes'  # whether to use ocean bottom water as top boundary

###############################################################################
###############################################################################
con = engine.connect()
complete = 'yes'
runtime_errors = 0
plt.close('all')

# Load and prepare all input data
site_metadata = ff.metadata_compiler(engine, metadata_table, site_info,
                                     hole_info, leg, site)
(concunique, temp_gradient, bottom_conc, bottom_temp, bottom_temp_est,
 pordata, sedtimes, seddepths, sedrate, picks, age_depth_boundaries,
 advection) = ff.load_and_prep(leg, site, holes, solute, ocean, engine,
                               conctable, portable, site_metadata,
                               solute_units)
if top_seawater != 'yes':
    concunique = concunique[1:,:]

if 'por_cutoff' not in  site_metadata:
    sys.exit('Run porosity_cutoff.py for site before calculating flux.')

# Fit pore water concentration curve
try:
    conc_fit = ff.concentration_fit(concunique, dp, line_fit)
except:
    sys.exit("Concentration fit failed, try fitting a different number of"
        "datapoints")
conc_interp_fit_plot = ff.conc_curve(line_fit)(np.linspace(concunique[0,0],
                                                concunique[dp-1,0],
                                                num=50), *conc_fit)

# R-squared of line fit to concentration profile
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
d_in_situ = ff.d_stp(temp_d, bottom_temp, ds)
dsed = d_in_situ/tortuosity

# Calculate pore water burial flux rate
pwburialflux = ff.pw_burial(seddepths, sedtimes, por_fit, por)

# Calculate solute flux
flux, burial_flux, gradient = ff.flux_model(conc_fit, concunique, z,
                                            pwburialflux, porosity, dsed,
                                            advection, dp, site, line_fit)

# Plot input data
plt.ioff()
figure_1 = ff.flux_plots(concunique, conc_interp_fit_plot, por, por_all,
                         por_fit, bottom_temp, picks, sedtimes, seddepths,
                         leg, site, solute_db, flux, dp, temp_gradient)
figure_1.show()
if optimized == 'yes':
    # Save Figure
    if flux_fig_path:
        savefig(flux_fig_path +
               "interface_{}_flux_{}_{}.png".format(solute_db, leg, site))

    # Monte Carlo Simulation
    (interface_fluxes, interface_fluxes_log, cycles, por_error, mean_flux,
     median_flux, stdev_flux, skewness, p_value, mean_flux_log,
     median_flux_log, stdev_flux_log, stdev_flux_lower, stdev_flux_upper,
     skewness_log, p_value_log, runtime_errors,
     conc_fits) = ff.monte_carlo(cycles, precision, concunique,
                                bottom_temp_est, dp, por, por_fit, seddepths,
                                sedtimes, temp_d, bottom_temp, z, advection,
                                leg, site, solute_db, ds, por_error, conc_fit,
                                runtime_errors, line_fit)

    # Plot Monte Carlo Distributions
    mc_figure =ff.monte_carlo_plot(interface_fluxes, median_flux, stdev_flux,
                                   skewness, p_value, interface_fluxes_log,
                                   median_flux_log, stdev_flux_log,
                                   skewness_log, p_value_log)
    mc_figure.show()

    # Save figure and fluxes from each run
    if mc_fig_path:
        savefig(mc_fig_path +
            "monte carlo_{}_{}_{}.png".format(solute_db, leg, site))
    if mc_text_path:
        np.savetxt(mc_text_path +
            "monte carlo_{}_{}_{}.csv".format(solute_db, leg, site),
            interface_fluxes, delimiter=",")

    # Retrieve misc metadata
    hole = ''.join(filter(str.isupper, filter(str.isalpha, holes)))
    date = datetime.datetime.now()
    site_key = con.execute("""select site_key
                    from site_info
                    where leg = '{}' and site = '{}'
                    ;""".format(leg, site))
    site_key = site_key.fetchone()[0]

    # Send metadata to database
    ff.flux_to_sql(con, solute_db, site_key,leg,site,hole,solute,flux,
                   burial_flux,gradient,porosity,z,dp,bottom_conc,conc_fit,
                   r_squared,age_depth_boundaries,sedrate,advection,precision,ds,
                   temp_d,bottom_temp,bottom_temp_est,cycles,por_error,mean_flux,
                   median_flux,stdev_flux,skewness,p_value,mean_flux_log,
                   median_flux_log,stdev_flux_log,stdev_flux_lower,
                   stdev_flux_upper,skewness_log,p_value_log,runtime_errors,
                   date,comments,complete)
    print('Results sent to database')
else:
    print('Continue optimization: Results NOT sent to database')

# eof
