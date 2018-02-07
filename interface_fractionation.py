# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:13:57 2017
Author: Rick Berg, University of Washington School of Oceanography, Seattle WA

Module to calculate sediment-water Mg isotope fractionations based on fluxes of
each isotope across the sediment-water interface, accounting for diffusion,
advection, and pore water burial with compaction. Although meant for the
sediment-water interface, the module is generalized to calculate the flux
across any depth horizon.
Module is designed to be paired with a MySQL database of ocean drilling data.
See flux_functions.py for detailed descriptions of each function used.

Inputs:
leg:            drilling leg/expedition number
site:           drilling site number
holes:          drilling hole IDs
solute:         solute name in database
ds:             diffusion coefficient at reference temperature (m^2 y^-1)
temp_d:          reference temperature of diffusion coefficient (C)
precision:      relative standard deviation of concentration measurements
ocean:          concentration of conservative solute in the ocean (mM)
solute_db:      solute name for referencing
z:              depth at which flux is calculated (mbsf)
cycles:         number of monte carlo simulations to run
line_fit:       "linear" or "exponential" line fit to concentration profile
top_boundary:   designate 'seawater' to use "ocean" value, otherwise uses the
                shallowest pore water measurement value for the upper boundary
dp:             concentration datapoints below seafloor used for line fit
top_seawater:   'yes' or 'no', whether to use ocean bottom water concentration
                 as a value at z=0

Outputs:
alpha:        fractionation factor from ocean to sediment column
alpha_mean:   average of fractionation factors from monte carlo simulation
alpha_median: median of fractionation factors from monte carlo simulation
alpha_stdev:  standard deviation of fractionation factors from monte carlo sim.
p_value:      Kolmogorov-Smirnov test p-value of Monte Carlo output

"""
import numpy as np
import pandas as pd
import datetime
from pylab import savefig
import matplotlib.pyplot as plt
import sys

import flux_functions as ff
from user_parameters import (engine, conctable, portable, metadata_table,
                             site_info, hole_info, flux_fig_path, mc_fig_path)

# site Information
leg = '344'
site = 'U1414'
holes = "('A') or hole is null"

# Species parameters
solute = 'Mg'  # Change X to X_ic if needed, based on availability in database
ds = 1.875*10**-2  # m^2 per year
temp_d = 18  # degrees C
precision = 0.02
ocean = 54  # mM
solute_db = 'Mg'
ct_d26 = -0.83
ct_d25 = -0.43

# Model parameters
z = 0
cycles = 5000
line_fit = 'exponential'
top_seawater = 'yes'  # whether to use ocean bottom water as top boundary
dp = 4

###############################################################################
###############################################################################
con = engine.connect()
runtime_errors = 0
plt.close('all')

# Load and prepare all input data
site_metadata = ff.metadata_compiler(engine, metadata_table, site_info,
                                  hole_info, leg, site)

concunique, temp_gradient, bottom_conc, bottom_temp, bottom_temp_est, pordata, sedtimes, seddepths, sedrate, picks, age_depth_boundaries, advection = ff.load_and_prep(leg, site, holes, solute, ocean, engine, conctable, portable, site_metadata)
concunique = pd.DataFrame(concunique)
concunique.columns = ['sample_depth', 'mg_conc']

if not sedtimes.size:
    sys.exit("Sedimentation Rates not yet calculated for this site."
        "Use age_depth.py to calculate.")

# Mg isotope data, combined with corresponding Mg concentrations
sql = """select sample_depth, d26mg, d25mg, d26mg_2sd
    from mg_isotopes
    where leg = '{}'
    and site = '{}'
    and hole in {}
    and d26mg is not NULL;""".format(leg, site, holes)
isotopedata = pd.read_sql(sql, engine)
isotopedata = isotopedata.sort_values(by='sample_depth')

# Calculate d25Mg for data from Higgins and Schrag 2010,
# based on d25Mg = 0.52*d26Mg, cited in Isotope Geochemistry,
# William White, pp.365
if isotopedata.d25mg.isnull().all():
    isotopedata.d25mg = 0.52 * isotopedata.d26mg
isotopedata = isotopedata[isotopedata.notnull().all(axis=1)]
isotopedata_26 = ff.averages(isotopedata['sample_depth'],
                                        isotopedata.iloc[:, 1])
isotopedata_25 = ff.averages(isotopedata['sample_depth'],
                                        isotopedata.iloc[:, 2])
isotopedata_26_2sd = ff.averages(isotopedata['sample_depth'],
                                        isotopedata.iloc[:, 3])
precision_iso = isotopedata_26_2sd[:,1]
isotopedata_26 = pd.DataFrame(isotopedata_26)
isotopedata_25 = pd.DataFrame(isotopedata_25)
isotopedata = pd.DataFrame(np.concatenate((isotopedata_26,
                                           isotopedata_25.iloc[:, 1][:, None]),
                                          axis=1),
                           columns=['sample_depth', 'd26Mg', 'd25Mg'])

isotopedata = pd.merge(isotopedata,
                       concunique,
                       how='left',
                       on='sample_depth')
isotopedata = isotopedata.as_matrix()

# Optionally use bottom water as upper bound (dirichlet)
ct_d26 = [ct_d26]
ct_d25 = [ct_d25]
mg26_24_ocean = ((ct_d26[0]/1000)+1)*0.13979
if top_seawater == 'yes':
    concunique_mg = concunique.reset_index(drop=True)
    isotopedata = np.concatenate((np.array(([0],
                                            ct_d26,
                                            ct_d25,
                                            [concunique_mg.iloc[0, 1]])).T,
                                  isotopedata),
                                 axis=0)
    # Standard deviation of seawater value of 0.025
    isotopedata_26_1sd = (
        0.5 * np.append([0.05],
                        isotopedata_26_2sd[:,1][~np.isnan(isotopedata_26_2sd[:,1])]))
    precision_iso = isotopedata_26_1sd
else:
    concunique.iloc[0,1] = concunique.iloc[1,1]

# Calculate Mg isotope concentrations
# Decimal numbers are isotopic ratios of standards.
# Source for 26/24std: Isotope Geochemistry, William White, pp.365
# 25Mg/24Mg calculated from from 26/24 value divided by 25/24 measured values.
mg26_24 = ((isotopedata[:, 1]/1000)+1)*0.13979
mg25_24 = ((isotopedata[:, 2]/1000)+1)*0.126635
concunique_mg24 = isotopedata[:, 3]/(mg26_24+mg25_24+1)
concunique_mg25 = concunique_mg24*mg25_24
concunique_mg26 = concunique_mg24*mg26_24
isotope_concs = [pd.DataFrame(np.array((isotopedata[:,0],concunique_mg24)).T),
                 pd.DataFrame(np.array((isotopedata[:,0],concunique_mg26)).T)]

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

# Model fluxes for each isotope
fluxes = []
for n in np.arange(2):
    concunique = np.array(isotope_concs[n])
    # Fit pore water concentration curve
    conc_fit = ff.concentration_fit(concunique, dp, line_fit)
    conc_interp_fit_plot = (
        ff.conc_curve(line_fit)(np.linspace(concunique[0,0],
                                            concunique[dp-1,0],
                                            num=50), *conc_fit))

    # R-squared function
    r_squared = (
        ff.rsq(ff.conc_curve(line_fit)(concunique[:dp,0],
                                       *conc_fit), concunique[:dp,1]))

    # Calculate solute flux
    flux, burial_flux, gradient = (
        ff.flux_model(conc_fit, concunique, z, pwburialflux, porosity,
                      dsed, advection, dp, site, line_fit))

    # Plot input data
    plt.ioff()
    figure_1 = ff.flux_plots(concunique, conc_interp_fit_plot, por, por_all,
                             por_fit, bottom_temp, picks, sedtimes,
                             seddepths, leg, site, solute_db, flux, dp,
                             temp_gradient)
    figure_1.show()

    # Save Figure, if folder specified in user_parameters
    if flux_fig_path:
        savefig(flux_fig_path +
            "interface_{}_fractionation_{}_{}_{}.png"
            .format(solute_db, leg, site, n))
    fluxes.append(flux)

# Calculate fractionation based on fluxes
mg26_24_flux = fluxes[1]/fluxes[0]
alpha = mg26_24_flux/mg26_24_ocean
epsilon = (alpha - 1) * 1000

# date and time for database
date = datetime.datetime.now()

# Retrieve site key from database
site_key = con.execute("""select site_key
                from site_info
                where leg = '{}' and site = '{}'
                ;""".format(leg, site))
site_key = site_key.fetchone()[0]

# Insert fractionation factor into database
hole = ''.join(filter(str.isupper, filter(str.isalpha, holes)))
sql= """insert into metadata_{}_flux (site_key,leg,site,hole,solute,
               alpha)
               VALUES ({}, '{}', '{}', '{}', '{}', {})
               ON DUPLICATE KEY UPDATE alpha={}
               ;""".format(solute_db, site_key,leg,site,hole,solute,
                           alpha,
                           alpha)
con.execute(sql)

# Monte Carlo Simulation
alpha, epsilon, cycles, por_error, alpha_mean, alpha_median, alpha_stdev, test_stat, p_value, runtime_errors, conc_fits, fluxes = ff.monte_carlo_fract(cycles, precision, precision_iso, concunique, bottom_temp_est, dp, por, por_fit, seddepths, sedtimes, temp_d, bottom_temp, z, advection, leg, site, solute_db, ds, por_error, conc_fit, runtime_errors, isotopedata, mg26_24_ocean, line_fit)

# Insert Monte Carlo simulation output into database
sql= """insert into metadata_{}_flux (site_key,leg,site,hole,solute,
               alpha_mean, alpha_median, alpha_stdev, alpha_p_value)
               VALUES ({}, '{}', '{}', '{}', '{}', {}, {}, {}, {})
               ON DUPLICATE KEY UPDATE alpha_mean={}, alpha_median={},
               alpha_stdev={}, alpha_p_value={}
               ;""".format(solute_db, site_key,leg,site,hole,solute,
                           alpha_mean, alpha_median, alpha_stdev, p_value,
                           alpha_mean, alpha_median, alpha_stdev, p_value)
con.execute(sql)

# Plot Monte Carlo distributions
mc_figure = ff.monte_carlo_plot_fract(alpha, alpha_median, alpha_stdev,
                                      alpha_mean, p_value)
mc_figure.show()
if mc_fig_path:
    savefig(mc_fig_path +
        "interface_{}_fractionation_distribution_{}_{}.png"
        .format(solute_db, leg, site))

print("Mean Epsilon:", round((alpha_mean-1)*1000, 4))
print("SD Epsilon:", round(alpha_stdev*1000, 4))

# eof
