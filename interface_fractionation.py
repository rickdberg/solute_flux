# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:13:57 2017

@author: rickdberg

Module to calculate sediment-water isotope fractionations based on interface fluxes of a dissolved constituent and isotope delta values
Uses an exponential fit to the upper sediment section for concentration gradient

Positive flux is downward (into the sediment)

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


plt.close('all')
###############################################################################
###############################################################################
###############################################################################
# Site Information
Leg = '344'
Site = 'U1414'
Holes = "('A') or hole is null"
dp = 4 # Number of concentration datapoints to use for exponential curve fit

Comments = ''
Complete = 'yes'

# Species parameters
Solute = 'Mg'  # Change to Mg_ic if needed, based on what's available in database
Ds = 1.875*10**-2  # m^2 per year free diffusion coefficient at 18C (ref?)
TempD = 18  # Temperature at which diffusion coefficient is known
Precision = 0.02  # measurement precision
Ocean = 54  # Concentration in modern ocean
Solute_db = 'Mg' # Solute label to send to the database

# Model parameters
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

concunique, temp_gradient, bottom_conc, bottom_temp, bottom_temp_est, pordata, sedtimes, seddepths, sedrate, picks, age_depth_boundaries, advection = flux_functions.load_and_prep(Leg, Site, Holes, Solute, Ocean, engine, conctable, portable, site_metadata)
concunique = pd.DataFrame(concunique)
concunique.columns = ['sample_depth', 'mg_conc']

# For 1039
# concunique.iloc[0,1] = concunique.iloc[1,1]

# Mg isotope data, combined with corresponding Mg concentrations
sql = """select sample_depth, d26mg, d25mg
    from mg_isotopes
    where leg = '{}'
    and site = '{}'
    and hole in {}
    and d26mg is not NULL;""".format(Leg, Site, Holes)
isotopedata = pd.read_sql(sql, engine)
isotopedata = isotopedata.sort_values(by='sample_depth')
isotopedata_26 = data_handling.averages(isotopedata['sample_depth'],
                                        isotopedata.iloc[:, 1])
isotopedata_25 = data_handling.averages(isotopedata['sample_depth'],
                                        isotopedata.iloc[:, 2])
isotopedata_26 = pd.DataFrame(isotopedata_26)
isotopedata_25 = pd.DataFrame(isotopedata_25)
isotopedata = pd.DataFrame(np.concatenate((isotopedata_26,
                                           isotopedata_25.iloc[:, 1][:, None]),
                                          axis=1),
                           columns=['sample_depth', 'd26Mg', 'd25Mg'])
isotopedata = isotopedata[isotopedata.notnull().all(axis=1)]
#concunique_mg = concunique_mg.reset_index(drop=True)
#concunique_mg.index = concunique_mg.index - 1
isotopedata = pd.merge(isotopedata,
                       concunique,
                       how='left',
                       on='sample_depth')
concunique_mg = concunique.reset_index(drop=True)
# Add seawater values (upper boundary condition, dirichlet) to profiles
isotopedata = isotopedata.as_matrix()
#U1414 and U1039 don't use water column data as upper bound
#ct_d26 = [isotopedata[0,1]]  # [-0.83]
#ct_d25 = [isotopedata[0,2]]  # [-0.42]

# For U1414
# isotopedata[2,3] = concunique_mg.iloc[10,1]
# isotopedata[6,3] = concunique_mg.iloc[55,1]


ct_d26 = [-0.82]
ct_d25 = [-0.41]

mg26_24_ocean = ((ct_d26[0]/1000)+1)*0.13979
isotopedata = np.concatenate((np.array(([0],
                                        ct_d26,
                                        ct_d25,
                                        [concunique_mg.iloc[0, 1]])).T,
                              isotopedata),
                             axis=0)



# Calculate Mg isotope concentrations
# Source for 26/24std: Isotope Geochemistry, William White, pp.365
mg26_24 = ((isotopedata[:, 1]/1000)+1)*0.13979 # np.ones(len(isotopedata[:, 1]))*mg26_24_ocean #   # Decimal numbers are isotopic ratios of standards
mg25_24 = ((isotopedata[:, 2]/1000)+1)*0.126635  # Estimated for now (calc'd from 26/24 value divided by 25/24 measured values)
concunique_mg24 = isotopedata[:, 3]/(mg26_24+mg25_24+1)
concunique_mg25 = concunique_mg24*mg25_24
concunique_mg26 = concunique_mg24*mg26_24

conc_list = [pd.DataFrame(np.array((isotopedata[:,0],concunique_mg24)).T),
             pd.DataFrame(np.array((isotopedata[:,0],concunique_mg26)).T)]


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

fluxes = []
for n in np.arange(2):
    concunique = np.array(conc_list[n])
    # Fit pore water concentration curve
    conc_fit = flux_functions.concentration_fit(concunique, dp)
    conc_interp_fit_plot = flux_functions.conc_curve(np.linspace(concunique[0,0], concunique[dp-1,0], num=50), *conc_fit)

    # R-squared function
    r_squared = flux_functions.rsq(flux_functions.conc_curve(concunique[:dp,0], *conc_fit), concunique[:dp,1])

    # Calculate solute flux
    flux, burial_flux, gradient = flux_functions.flux_model(conc_fit, concunique, z, pwburialflux, porosity, Dsed, advection, dp, Site)

    # Plot input data
    plt.ioff()
    figure_1 = flux_functions.flux_plots(concunique, conc_interp_fit_plot, por, por_all, por_fit, bottom_temp, picks, sedtimes, seddepths, Leg, Site, Solute_db, flux, dp, temp_gradient)
    figure_1.show()

    # Save Figure
    savefig(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output flux figures\interface_{}_fractionation_{}_{}_{}.png".format(Solute_db, Leg, Site, n))
    fluxes.append(flux)



#######################################################################
# Calculate fractionation based on fluxes

mg26_24_flux = fluxes[1]/fluxes[0]
alpha = mg26_24_flux/mg26_24_ocean
epsilon = (alpha - 1) * 1000

# Date and time
Date = datetime.datetime.now()

# Retrieve Site key
site_key = con.execute("""select site_key
                from site_info
                where leg = '{}' and site = '{}'
                ;""".format(Leg, Site))
site_key = site_key.fetchone()[0]

sql= """insert into metadata_{}_flux (site_key,leg,site,hole,solute,
               alpha)
               VALUES ({}, '{}', '{}', '{}', '{}', {})
               ON DUPLICATE KEY UPDATE alpha={}
               ;""".format(Solute_db, site_key,Leg,Site,Hole,Solute,
                           alpha,
                           alpha)
con.execute(sql)






# eof