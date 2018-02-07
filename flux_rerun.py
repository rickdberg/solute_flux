# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:22:16 2017
Author: Rick Berg, University of Washington School of Oceanography, Seattle WA

Module to re-calculate fluxes that have already been run and stored in
the MySQL database using interface_flux.py.
For use if any substantive changes are made in flux_functions.py.

Module is designed to be paired with a MySQL database of ocean drilling data.
See flux_functions.py for detailed descriptions of each function used.

Inputs:
cycles:         number of monte carlo simulations to run
line_fit:       "linear" or "exponential" line fit to concentration profile
ocean:          concentration of conservative solute in the ocean (mM)
solute_db:      solute name for inserting into database

Outputs are the same as interface_flux.py
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
mean_flux_log:        log-normal average of fluxes from monte carlo sim.
median_flux_log:      log-normal median of fluxes from monte carlo sim.
stdev_flux_log:       log-normal standard deviation of fluxes from monte carlo sim.
stdev_flux_lower:     upper log-normal value for 1 standard deviation
stdev_flux_upper:     lower log-normal value for 1 standard deviation
skewness_log:         skewness of distribution of log-normal fluxes
p_value_log:          Kolmogorov-Smirvov p-value of distribution of log-normal fluxes
runtime_errors:       number of runtime errors resulting from line-fitting procedure
date:                 date the site was modeled

Note: all other ouputs are taken directly from database site information and
hole information tables.

"""
import numpy as np
import pandas as pd
import datetime
from pylab import savefig
import matplotlib.pyplot as plt

import flux_functions as ff
from user_parameters import (engine, conctable, portable, metadata_table,
                             site_info, hole_info, flux_fig_path, mc_fig_path,
                             mc_text_path)

# Simulation parameters
cycles = 5000
line_fit = 'exponential'
top_seawater = 'yes'  # whether to use ocean bottom water as top boundary

# Species parameters
ocean = 54
solute_db = 'Mg'

###############################################################################
###############################################################################
# Create sqlalchemy cursor
con = engine.connect()

sql = """select *
from {}
;""".format(metadata_table)
metadata = pd.read_sql(sql, engine)

for i in np.arange(np.size(metadata, axis=0))[:]:  # If script errors out on specific site, can type in index here
    cycles = 5000
    complete = 'yes'
    comments = metadata.comments[i]
    leg = metadata.leg[i]
    site = metadata.site[i]
    holes = "('{}') or hole is null".format('\',\''.join(filter(str.isalpha,
                                                                metadata.hole[i])))
    hole = metadata.hole[i]
    solute = metadata.solute[i]
    ds = float(metadata.ds[i])
    temp_d = float(metadata.ds_reference_temp[i])
    precision = float(metadata.measurement_precision[i])
    dp = int(metadata.datapoints[i])
    z = float(metadata.flux_depth[i])
    runtime_errors = 0
    print(i)

    # Load and prepare all input data
    site_metadata = ff.metadata_compiler(engine, metadata_table, site_info,
                                      hole_info, leg, site)
    concunique, temp_gradient, bottom_conc, bottom_temp, bottom_temp_est, pordata, sedtimes, seddepths, sedrate, picks, age_depth_boundaries, advection = ff.load_and_prep(leg, site, holes, solute, ocean, engine, conctable, portable, site_metadata)
    if top_seawater != 'yes':
        concunique = concunique[1:,:]

    # Fit pore water concentration curve
    try:
        conc_fit = ff.concentration_fit(concunique, dp, line_fit)
    except:
        print('Failed initial concentration fit: Site', site)
        complete = 'no'
        continue
    conc_interp_fit_plot = (
        ff.conc_curve(line_fit)(np.linspace(concunique[0,0],
                                            concunique[dp-1,0],
                                            num=50), *conc_fit))

    # R-squared function
    r_squared = ff.rsq(ff.conc_curve(line_fit)(concunique[:dp,0], *conc_fit),
                       concunique[:dp,1])

    # Fit porosity curve
    por = ff.averages(pordata[:, 0], pordata[:, 1])
    por_all = por
    if site_metadata.por_cutoff.notnull().all():
        por = por[por[:,0] <= float(site_metadata.por_cutoff[0])]
    try:
        por_fit = ff.porosity_fit(por)
    except:
        print('Failed initial porosity fit: Site', site)
        complete = 'no'
        continue
    por_error = ff.rmse(ff.por_curve(por[:,0], por, por_fit), por[:,1])

    # Sediment properties at flux depth
    porosity = ff.por_curve(z, por, por_fit)[0]
    tortuosity = 1-np.log(porosity**2)

    # Calculate effective diffusion coefficient
    d_in_situ = ff.d_stp(temp_d, bottom_temp, ds)
    dsed = d_in_situ/tortuosity  # Effective diffusion coefficient

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

    # Save Figure
    if flux_fig_path:
        savefig(flux_fig_path +
            "interface_{}_flux_{}_{}.png".format(solute_db, leg, site))
    figure_1.clf()
    plt.close('all')

    # Monte Carlo Simulation
    fluxes_real, fluxes_log_real, cycles, por_error, mean_flux, median_flux, stdev_flux, skewness, p_value, mean_flux_log, median_flux_log, stdev_flux_log, stdev_flux_lower, stdev_flux_upper, skewness_log, p_value_log, runtime_errors, conc_fits = ff.monte_carlo(cycles, precision, concunique, bottom_temp_est, dp, por, por_fit, seddepths, sedtimes, temp_d, bottom_temp, z, advection, leg, site, solute_db, ds, por_error, conc_fit, runtime_errors, line_fit)

    # Plot Monte Carlo Distributions
    mc_figure = ff.monte_carlo_plot(fluxes_real, median_flux, stdev_flux,
                                    skewness, p_value, fluxes_log_real,
                                    median_flux_log, stdev_flux_log,
                                    skewness_log, p_value_log)

    # Save figure and fluxes from each run
    if mc_fig_path:
        savefig(mc_fig_path +
            "interface_{}_flux_{}_{}.png".format(solute_db, leg, site))
    mc_figure.clf()
    if mc_text_path:
        np.savetxt(mc_text_path +
            "monte carlo_{}_{}_{}.csv".format(solute_db, leg, site),
            fluxes_real, delimiter=",")

    # Date and time
    date = datetime.datetime.now()

    # Retrieve Site key
    site_key = con.execute("""select site_key
                    from site_info
                    where leg = '{}' and site = '{}'
                    ;""".format(leg, site))
    site_key = site_key.fetchone()[0]

    # Send metadata to database
    ff.flux_to_sql(con, solute_db, site_key,leg,site,hole,solute,flux,
                   burial_flux,gradient,porosity,z,dp,bottom_conc,conc_fit,
                   r_squared,age_depth_boundaries,sedrate,advection,precision,
                   ds,temp_d,bottom_temp,bottom_temp_est,cycles,por_error,
                   mean_flux,median_flux,stdev_flux,skewness,p_value,
                   mean_flux_log,median_flux_log,stdev_flux_log,
                   stdev_flux_lower,stdev_flux_upper,skewness_log,p_value_log,
                   runtime_errors,date,comments,complete)
    print('Cycles:', cycles)
    print('Errors:', runtime_errors)

# eof
