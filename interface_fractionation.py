# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:13:57 2017
Author: Rick Berg, University of Washington School of Oceanography, Seattle WA

Module to calculate sediment-water isotope fractionations based on fluxes of
each isotope across the sediment-water interface, accounting for diffusion,
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
Complete:       designate 'yes' when model parameters are acceptable
Solute:         solute name in database
Ds:             diffusion coefficient at reference temperature (m^2 y^-1)
TempD:          reference temperature of diffusion coefficient (C)
Precision:      relative standard deviation of concentration measurements
Ocean:          concentration of conservative solute in the ocean (mM)
Solute_db:      solute name for inserting into database
z:              depth at which flux is calculated (mbsf)
cycles:         number of monte carlo simulations to run
line_fit:       "linear" or "exponential" line fit to concentration profile
top_boundary:   designate 'seawater' to use "Ocean" value, otherwise uses the
                shallowest pore water measurement value for the upper boundary
dp:             concentration datapoints below seafloor used for line fit
engine:         SQLAlchemy engine
conctable:      name of MySQL solute concentration table
portable:       name of MySQL porosity (MAD) table
metadata_table: name of MySQL metadata table, same as regular flux metadata
site_info:      name of MySQL site information table
hole_info:      name of MySQL hole information table

In addition, filepaths to directories where the figures and output data are to
be stored need to be specified in the script.

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
from sqlalchemy import create_engine

import flux_functions as ff
from site_metadata_compiler import metadata_compiler

# Site Information
Leg = '344'
Site = 'U1414'
Holes = "('A') or hole is null"
Comments = ''
Complete = 'yes'

# Species parameters
Solute = 'Mg'  # Change X to X_ic if needed, based on availability in database
Ds = 1.875*10**-2  # m^2 per year (Li & Gregory 1971)
TempD = 18  # degrees C
Precision = 0.02
Ocean = 54  # mM
Solute_db = 'Mg'
ct_d26 = [-0.82]
ct_d25 = [-0.41]

# Model parameters
z = 0
cycles = 5000
line_fit = 'exponential'
top_boundary = 'seawater'
dp = 4

# Connect to database
engine = create_engine("mysql://root:neogene227@localhost/iodp_compiled")
conctable = 'iw_all'
portable = 'mad_all'
metadata_table = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"

# File paths
flux_fig_path = r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output fractionation figures\\"
mc_fig_path = r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output fractionation figures\\"


###############################################################################
###############################################################################
con = engine.connect()
runtime_errors = 0
plt.close('all')

# Load and prepare all input data
site_metadata = metadata_compiler(engine, metadata_table, site_info,
                                  hole_info, Leg, Site)

concunique, temp_gradient, bottom_conc, bottom_temp, bottom_temp_est, pordata, sedtimes, seddepths, sedrate, picks, age_depth_boundaries, advection = ff.load_and_prep(Leg, Site, Holes, Solute, Ocean, engine, conctable, portable, site_metadata)
concunique = pd.DataFrame(concunique)
concunique.columns = ['sample_depth', 'mg_conc']
if top_boundary != 'seawater':
    concunique.iloc[0,1] = concunique.iloc[1,1]

# Mg isotope data, combined with corresponding Mg concentrations
sql = """select sample_depth, d26mg, d25mg, d26mg_2sd
    from mg_isotopes
    where leg = '{}'
    and site = '{}'
    and hole in {}
    and d26mg is not NULL;""".format(Leg, Site, Holes)
isotopedata = pd.read_sql(sql, engine)
isotopedata = isotopedata.sort_values(by='sample_depth')

# Calculate d25Mg for data from Higgins and Schrag 2010
if isotopedata.d25mg.isnull().all():
    isotopedata.d25mg = 0.527 * isotopedata.d26mg
isotopedata = isotopedata[isotopedata.notnull().all(axis=1)]
isotopedata_26 = ff.averages(isotopedata['sample_depth'],
                                        isotopedata.iloc[:, 1])
isotopedata_25 = ff.averages(isotopedata['sample_depth'],
                                        isotopedata.iloc[:, 2])
isotopedata_26_2sd = ff.averages(isotopedata['sample_depth'],
                                        isotopedata.iloc[:, 3])
# Standard deviation of seawater value of 0.025
isotopedata_26_1sd = (
    0.5 * np.append([0.05],
                    isotopedata_26_2sd[:,1][~np.isnan(isotopedata_26_2sd[:,1])]))
precision_iso = isotopedata_26_1sd
isotopedata_26 = pd.DataFrame(isotopedata_26)
isotopedata_25 = pd.DataFrame(isotopedata_25)
isotopedata = pd.DataFrame(np.concatenate((isotopedata_26,
                                           isotopedata_25.iloc[:, 1][:, None]),
                                          axis=1),
                           columns=['sample_depth', 'd26Mg', 'd25Mg'])

#concunique_mg = concunique_mg.reset_index(drop=True)
#concunique_mg.index = concunique_mg.index - 1
isotopedata = pd.merge(isotopedata,
                       concunique,
                       how='left',
                       on='sample_depth')
concunique_mg = concunique.reset_index(drop=True)
isotopedata = isotopedata.as_matrix()

# Optionally use bottom water as upper bound (dirichlet)
mg26_24_ocean = ((ct_d26[0]/1000)+1)*0.13979
if top_boundary == 'seawater':
    isotopedata = np.concatenate((np.array(([0],
                                            ct_d26,
                                            ct_d25,
                                            [concunique_mg.iloc[0, 1]])).T,
                                  isotopedata),
                                 axis=0)

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
D_in_situ = ff.d_stp(TempD, bottom_temp, Ds)
Dsed = D_in_situ/tortuosity

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
                      Dsed, advection, dp, Site, line_fit))

    # Plot input data
    plt.ioff()
    figure_1 = ff.flux_plots(concunique, conc_interp_fit_plot, por, por_all,
                             por_fit, bottom_temp, picks, sedtimes,
                             seddepths, Leg, Site, Solute_db, flux, dp,
                             temp_gradient)
    figure_1.show()

    # Save Figure
    savefig(flux_fig_path +
        "interface_{}_fractionation_{}_{}_{}.png"
        .format(Solute_db, Leg, Site, n))
    fluxes.append(flux)

# Calculate fractionation based on fluxes
mg26_24_flux = fluxes[1]/fluxes[0]
alpha = mg26_24_flux/mg26_24_ocean
epsilon = (alpha - 1) * 1000

# Date and time for database
Date = datetime.datetime.now()

# Retrieve Site key from database
site_key = con.execute("""select site_key
                from site_info
                where leg = '{}' and site = '{}'
                ;""".format(Leg, Site))
site_key = site_key.fetchone()[0]

# Insert fractionation factor into database
Hole = ''.join(filter(str.isupper, filter(str.isalpha, Holes)))
sql= """insert into metadata_{}_flux (site_key,leg,site,hole,solute,
               alpha)
               VALUES ({}, '{}', '{}', '{}', '{}', {})
               ON DUPLICATE KEY UPDATE alpha={}
               ;""".format(Solute_db, site_key,Leg,Site,Hole,Solute,
                           alpha,
                           alpha)
con.execute(sql)

# Monte Carlo Simulation
alpha, epsilon, cycles, por_error, alpha_mean, alpha_median, alpha_stdev, test_stat, p_value, runtime_errors, conc_fits, fluxes = ff.monte_carlo_fract(cycles, Precision, precision_iso, concunique, bottom_temp_est, dp, por, por_fit, seddepths, sedtimes, TempD, bottom_temp, z, advection, Leg, Site, Solute_db, Ds, por_error, conc_fit, runtime_errors, isotopedata, mg26_24_ocean, line_fit)

# Insert Monte Carlo simulation output into database
sql= """insert into metadata_{}_flux (site_key,leg,site,hole,solute,
               alpha_mean, alpha_median, alpha_stdev, alpha_p_value)
               VALUES ({}, '{}', '{}', '{}', '{}', {}, {}, {}, {})
               ON DUPLICATE KEY UPDATE alpha_mean={}, alpha_median={},
               alpha_stdev={}, alpha_p_value={}
               ;""".format(Solute_db, site_key,Leg,Site,Hole,Solute,
                           alpha_mean, alpha_median, alpha_stdev, p_value,
                           alpha_mean, alpha_median, alpha_stdev, p_value)
con.execute(sql)

# Plot Monte Carlo distributions
mc_figure = ff.monte_carlo_plot_fract(alpha, alpha_median, alpha_stdev,
                                      alpha_mean, p_value)
mc_figure.show()
savefig(mc_fig_path +
    "interface_{}_fractionation_distribution_{}_{}.png"
    .format(Solute_db, Leg, Site))

print("Mean Epsilon:", round((alpha_mean-1)*1000, 4))
print("SD Epsilon:", round(alpha_stdev*1000, 4))

# eof
