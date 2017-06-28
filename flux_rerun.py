# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:22:16 2017

@author: rickdberg
"""
import numpy as np
import pandas as pd
import datetime
from pylab import savefig
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

import flux_functions
import data_handling
from site_metadata_compiler import metadata_compiler

Complete = 'yes'

# Simulation parameters
cycles = 5000

# Species parameters
Ocean = 54  # Concentration in modern ocean (mM)
Solute_db = 'Mg' # Label for the database loading

# Connect to database
engine = create_engine("mysql://root:neogene227@localhost/iodp_compiled")
con = engine.connect()
conctable = 'iw_all'
portable = 'mad_all'
metadata_table = "metadata_mg_flux"
metadata_table_write = "metadata_mg_flux_flow"
site_info = "site_info"
hole_info = "summary_all"

sql = """select distinct mmf.*, sa.sed_thickness_combined
from {} sa right join {} mmf
on sa.leg = mmf.leg and sa.site = mmf.site
where sa.sed_thickness_combined <=250
;
;""".format(hole_info, metadata_table)
metadata = pd.read_sql(sql, engine)

for i in np.arange(np.size(metadata, axis=0)):
    cycles = 5000
    Comments = metadata.comments[i]
    Leg = metadata.leg[i]
    Site = metadata.site[i]
    Holes = "('{}') or hole is null".format('\',\''.join(filter(str.isalpha, metadata.hole[i])))
    Hole = metadata.hole[i]
    Solute = metadata.solute[i]
    Ds = float(metadata.ds[i])
    TempD = float(metadata.ds_reference_temp[i])
    Precision = float(metadata.measurement_precision[i])
    dp = int(metadata.datapoints[i])
    z = float(metadata.flux_depth[i])
    runtime_errors = 0
    print(i)

    # Load and prepare all input data
    site_metadata = metadata_compiler(engine, metadata_table, site_info, hole_info, Leg, Site)
    concunique, temp_gradient, bottom_conc, bottom_temp, bottom_temp_est, pordata, sedtimes, seddepths, sedrate, picks, age_depth_boundaries, advection, sed_thickness, lithology = flux_functions.load_and_prep(Leg, Site, Holes, Solute, Ocean, engine, conctable, portable, site_metadata)
    advection = flux_functions.adv_rate(sed_thickness, lithology, advection)

    # Fit pore water concentration curve
    conc_fit = flux_functions.concentration_fit(concunique, dp)
    conc_interp_fit_plot = flux_functions.conc_curve(np.linspace(concunique[0,0], concunique[dp-1,0], num=50), *conc_fit)

    # R-squared function
    r_squared = flux_functions.rsq(flux_functions.conc_curve(concunique[:dp,0], *conc_fit), concunique[:dp,1])

    # Fit porosity curve
    por = data_handling.averages(pordata[:, 0], pordata[:, 1])  # Average duplicates
    por_all = por
    if site_metadata.por_cutoff.notnull().all():
        por = por[por[:,0] <= float(site_metadata.por_cutoff[0])]


    por_fit = flux_functions.porosity_fit(por)
    por_error = data_handling.rmse(flux_functions.por_curve(por[:,0], por, por_fit), por[:,1])

    # Sediment properties at flux depth
    porosity = flux_functions.por_curve(z, por, por_fit)[0]
    tortuosity = 1-np.log(porosity**2)

    # Calculate effective diffusion coefficient
    D_in_situ = flux_functions.d_stp(TempD, bottom_temp, Ds)
    Dsed = D_in_situ/tortuosity  # Effective diffusion coefficient

    # Calculate pore water burial flux rate
    pwburialflux = flux_functions.pw_burial(seddepths, sedtimes, por_fit, por)

    # Calculate solute flux
    flux, burial_flux, gradient = flux_functions.flux_model(conc_fit, concunique, z, pwburialflux, porosity, Dsed, advection, dp, Site)

    # Plot input data
    plt.ioff()
    figure_1 = flux_functions.flux_plots(concunique, conc_interp_fit_plot, por, por_all, por_fit, bottom_temp, picks, sedtimes, seddepths, Leg, Site, Solute_db, flux, dp, temp_gradient)
    # figure_1.show()

    # Save Figure
    savefig(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output flux figures\interface_flux_{}_{}_flow.png".format(Leg, Site))
    figure_1.clf()
    plt.close('all')

    # Monte Carlo Simulation
    interface_fluxes, interface_fluxes_log, cycles, por_error, mean_flux, median_flux, stdev_flux, skewness, z_score, mean_flux_log, median_flux_log, stdev_flux_log, stdev_flux_lower, stdev_flux_upper, skewness_log, z_score_log, runtime_errors = flux_functions.monte_carlo(cycles, Precision, concunique, bottom_temp_est, dp, por, por_fit, seddepths, sedtimes, TempD, bottom_temp, z, advection, Leg, Site, Solute_db, Ds, por_error, conc_fit, runtime_errors)

    # Plot Monte Carlo Distributions
    mc_figure =flux_functions.monte_carlo_plot(interface_fluxes, median_flux, stdev_flux, skewness, z_score, interface_fluxes_log, median_flux_log, stdev_flux_log, skewness_log, z_score_log)
    # mc_figure.show()

    # Save figure and fluxes from each run
    savefig(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output monte carlo distributions\montecarlo_{}_{}_flow.png".format(Leg, Site))
    mc_figure.clf()
    np.savetxt(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output monte carlo distributions\monte carlo_{}_{}_flow.csv".format(Leg, Site), interface_fluxes, delimiter=",")


    # Date and time
    Date = datetime.datetime.now()

    # Retrieve Site key
    site_key = con.execute("""select site_key
                    from site_info
                    where leg = '{}' and site = '{}'
                    ;""".format(Leg, Site))
    site_key = site_key.fetchone()[0]

    # Send metadata to database
    flux_functions.flux_to_sql(con, metadata_table_write, Solute_db, site_key,Leg,Site,Hole,Solute,flux,
                    burial_flux,gradient,porosity,z,dp,bottom_conc,conc_fit,r_squared,
                               age_depth_boundaries,sedrate,advection,Precision,Ds,
                               TempD,bottom_temp,bottom_temp_est,cycles,
                               por_error,mean_flux,median_flux,stdev_flux,
                               skewness,z_score,mean_flux_log,median_flux_log,
                               stdev_flux_log,stdev_flux_lower,stdev_flux_upper,
                               skewness_log,z_score_log,runtime_errors,Date,Comments,Complete)
    print('Cycles:', cycles)
    print('Errors:', runtime_errors)
# eof
