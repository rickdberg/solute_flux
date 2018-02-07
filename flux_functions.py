# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 14:39:00 2017
Author: Rick Berg, University of Washington School of Oceanography, Seattle WA

Module for calculating flux of solutes into/out of sediments.

Functions:
metadata_compiler:   Compiler function to get all available site metadata
averages:            average duplicate samples, grouped by depth
rmse:                calculate root mean squared error
load_and_prep:       load data
sedtemp:             geothermal gradient function
d_stp:               free diffusion coefficient at STP
rsq:                 r-squared of a regression
conc_curve:          concentration curve function definition
concentration_fit:   fit parameters of conc_curve to data
por_curve:           porosity curve function definition (Athy's Law)
porosity_fit:        fit parameters of por_curve to data
solid_curve:         compliment to por_curve
pw_burial:           calculate pore water volume flux
flux_model:          calculate flux of solute
monte_carlo:         monte carlo simulation for error of flux_model
monte_carlo_fract:   monte carlo simulation for error of isotopic fractionation
flux_plots:          plot flux_model inputs
monte_carlo_plot:    plot monte carlo simulation distributions
flux_to_sql:         send results to mysql database
flux_only_to_sql:    send limited dataset to mysql database

"""

import numpy as np
import pandas as pd
from scipy import integrate, stats
from sqlalchemy import text
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import mlab
from scipy import optimize
from collections import defaultdict
import sys

import seawater

def metadata_compiler(engine, metadata, site_info, hole_info, Leg, Site):
    """
    Combine site information, hole information, and
    existing metadata (if available) for particular drilling sites.
    Hole data must be quantitative, as they are averaged for the site.

    Inputs:
    engine:     SQLAlchemy engine
    metadata:   name of MySQL metadata table
    site_info:  name of MySQL site information table
    hole_info:  name of MySQL hole information table
    Leg:        drilling leg/expedition number
    Site:       drilling site number

    Outputs:
    site_metadata: combined table with all available data from metadata,
                   site_info, and (averaged) hole_info for a particular site
    """
    # Load site data
    sql = """SELECT *
             FROM {}
             where leg = '{}' and site = '{}';
             """.format(site_info, Leg, Site)
    sitedata = pd.read_sql(sql, engine)

    # Load hole data
    sql = """SELECT *
             FROM {}
             where leg = '{}' and site = '{}';
             """.format(hole_info, Leg, Site)
    holedata = pd.read_sql(sql, engine)

    # Group and average hole data for sites
    hole_grouped = holedata.groupby("site_key").mean().reset_index()

    # Load existing metadata, if available
    try:
        sql = """SELECT *
                 FROM {}
                 where leg = '{}' and site = '{}';
                 """.format(metadata, Leg, Site)
        ran_metadata = pd.read_sql(sql, engine)
    except:
        ran_metadata = pd.DataFrame(sitedata.loc[:,['site_key','leg','site']])

    # Load porosity cutoff depths, if available
    sql = """SELECT *
             FROM porosity_cutoff
             WHERE site = {}
             """.format(Site)
    por_cuts = pd.read_sql(sql, engine)
    if por_cuts.empty:
        sys.exit('Run porosity_cutoff.py for site before calculating flux.')

    # Combine all tables
    ran_and_site = pd.merge(ran_metadata,
                              sitedata,
                              how='outer',
                              on=('site_key', 'leg', 'site'))
    add_por_cut = pd.merge(ran_and_site,
                              por_cuts,
                              how='outer',
                              on=('site_key', 'site'))
    site_metadata = pd.merge(add_por_cut,
                             hole_grouped,
                             how='outer',
                             on=('site_key')).fillna(np.nan)
    return site_metadata


def averages(depths, values):
    """
     Average duplicates in concentration dataset, add seawater value,
     and make spline fit to first three values.

     names: depths of samples
     values: values of samples
     """
    # Group values by depth
    value_lists = defaultdict(list)
    for depth, value in zip(depths, values):
        value_lists[depth].append(value)

    # Take average of each list
    result = {}
    for depth, values in value_lists.items():
        result[depth] = sum(values) / float(len(values))

    # Make it a Numpy array and pull out values
    resultkeys = np.array(list(result.keys()))
    resultvalues = np.array(list(result.values()))
    sorted = np.column_stack((resultkeys[np.argsort(resultkeys)],
                                         resultvalues[np.argsort(resultkeys)]))
    return sorted


def rmse(model_values, measured_values):
    """
    Root mean squared error of curve fit to measured values.
    """
    return np.sqrt(((model_values-measured_values)**2).mean())


def load_and_prep(Leg, Site, Holes, Solute, Ocean, engine, conctable,
                    portable, site_metadata):
    """
    Imports data from MySQL database, prepares data for modeling, and ouputs
    relevant data for model.
    Units are listed in function comments.

    Function inputs:
    Leg:           Drilling leg/expedition number
    Site:          Drilling site number
    Holes:         Drilling hole designations
    Solute:        Solute name in database
    Ocean:         concentration of conservative solute in the ocean
    engine:        SQLAlchemy engine
    conctable:     Name of MySQL solute concentration table
    portable:      Name of MySQL porosity (MAD) table
    site_metadata: Site metadata from site_metadata_compiler.py

    Function outputs:
    concunique:           2D array of sample depth and solute concentration
    temp_gradient:        geothermal gradient
    bottom_conc:          ocean bottom water solute concentration (mM)
    bottom_temp:          ocean bottom water temperature (C)
    bottom_temp_est:      ocean bottom water temp parameter for estimation
    pordata:              2D array of depth below seafloor (m) and porosity
                           measurements from load_and_prep.
    seddepths:            sedimentation bounds from load_and_prep function
    sedtimes:             sedimentation bounds from load_and_prep function
    sedrate:              modern sedimentation rate (solid volume per year)
    picks:                biostratigraphic age-depth data
    age_depth_boundaries: indices between discrete sedimentation rate regimes
    advection:            external advection rate
    """
    # Temperature gradient (degrees C/m)
    temp_gradient = site_metadata['temp_gradient'][0]

    # Water depth for bottom water estimation (m)
    water_depth = site_metadata['water_depth'][0]

    # Bottom water temp (degrees C)
    if site_metadata['bottom_water_temp'].isnull().all():
        bottom_temp = site_metadata['woa_temp'][0]
        if water_depth >= 1500:
            bottom_temp_est = 'deep'
        else:
            bottom_temp_est = 'shallow'
    else:
        bottom_temp = site_metadata['bottom_water_temp'][0]
        bottom_temp_est = 'na'

    # Advection rate (m/y)
    advection = float(site_metadata['advection'][0])
    if pd.isnull(advection):
        advection = 0

    # Pore water chemistry data
    sql = """SELECT sample_depth, {}
        FROM {}
        where leg = '{}' and site = '{}' and (hole in {})
        and {} is not null and hydrate_affected is null
        ; """.format(Solute, conctable, Leg, Site, Holes, Solute)
    concdata = pd.read_sql(sql, engine)
    concdata = concdata.sort_values(by='sample_depth')
    concdata = concdata.as_matrix()

    # Bottom water concentration based on WOA bottom salinities.
    woa_salinity = site_metadata['woa_salinity'][0]
    density = seawater.eos80.dens0(woa_salinity, bottom_temp)
    # Muller 1999, In Grasshoff et al.
    woa_cl = (1000*(woa_salinity-0.03)*density/1000)/(1.805*35.45)
    bottom_conc=woa_cl/558*Ocean
    # mol per m^3 in modern seawater at specific site
    ct0 = [bottom_conc]

    # Porosity data
    sql = text("""SELECT sample_depth, porosity
    FROM {}
    where leg = '{}' and site = '{}' and (hole in {})
    and coalesce(method,'C') like '%C%'
    and {} is not null and {} > 0 AND {} < 1 and sample_depth is not null
    ;""".format(portable, Leg, Site, Holes,'porosity','porosity','porosity'))
    pordata = pd.read_sql(sql, engine)
    # If not enough data from method C, take all data
    if pordata.iloc[:,1].count() < 40:
        sql = """SELECT sample_depth, porosity
            FROM {}
            where leg = '{}' and site = '{}' and (hole in {})
            and {} is not null and {} > 0 AND {} < 1
            and sample_depth is not null
            ;""".format(portable, Leg, Site, Holes,
            'porosity', 'porosity', 'porosity')
        pordata_b = pd.read_sql(sql, engine)
        pordata = pd.concat((pordata, pordata_b), axis=0)
    pordata = pordata.as_matrix()

    # Sedimentation rate profile (m/y) (Calculated in age_depth.py)
    sql = """SELECT sedrate_ages, sedrate_depths
        FROM metadata_sed_rate
        where leg = '{}' and site = '{}'
        ; """.format(Leg, Site)
    sedratedata = pd.read_sql(sql, engine)
    sedratedata = sedratedata.sort_values(by='sedrate_depths')

    sedtimes = np.asarray(sedratedata.iloc[:,0][0][1:-1].split(","))
    seddepths = np.asarray(sedratedata.iloc[:,1][0][1:-1].split(","))
    sedtimes = sedtimes.astype(np.float)
    seddepths = seddepths.astype(np.float)
    sedrates = np.diff(seddepths, axis=0)/np.diff(sedtimes, axis=0)
    sedrate = sedrates[0]  # Modern sedimentation rate

    # Load age-depth data for plots
    sql = """SELECT depth, age
        FROM age_depth
        where leg = '{}' and site = '{}' order by 1
        ;""".format(Leg, Site)
    picks = pd.read_sql(sql, engine)
    picks = picks.as_matrix()
    picks = picks[np.argsort(picks[:,0])]

    # Age-Depth boundaries from database, indices sorted by age
    sql = """SELECT age_depth_boundaries
        FROM metadata_sed_rate
        where leg = '{}' and site = '{}' order by 1
        ;""".format(Leg, Site)
    age_depth_boundaries = pd.read_sql(sql, engine).iloc[0,0]

    # Concentration vector after averaging duplicates
    concunique = averages(concdata[:, 0], concdata[:, 1])

    # Add in seawater value as upper boundary if measurement > 5 cmbsf
    if concunique[0,0] > 0.05:
        concunique = np.concatenate((np.array(([0],ct0)).T, concunique),
                                    axis=0)
    return (concunique, temp_gradient, bottom_conc, bottom_temp,
            bottom_temp_est, pordata, sedtimes, seddepths, sedrate, picks,
            age_depth_boundaries, advection)


def sedtemp(z, bottom_temp, temp_gradient):
    """
    Geothermal gradient function (C/m)

    z:             depth below seafloor (m)
    bottom_temp:   ocean bottom water temperature (C)
    temp_gradient: temperature gradient (C/m)
    """
    if z.all() == 0:
        return bottom_temp*np.ones(len(z))
    else:
        return bottom_temp + np.multiply(z, temp_gradient)


def d_stp(Td, T, Ds):
    """
    Calculates viscosity from Mostafa H. Sharqawy 12-18-2009,
    MIT (mhamed@mit.edu) (Sharqawy M. H., Lienhard J. H., and Zubair, S. M.,
    Desalination and Water Treatment, 2009)
    Viscosity used as input into Stokes-Einstein equation

    Td:  reference temperature
    T:   in situ temperature
    Ds:  Diffusion coefficient at reference temperature
    """
    # Viscosity at reference temperature
    muwd = (4.2844324477E-05 +
            1 /
            (1.5700386464E-01 * (Td + 6.4992620050E+01)**2+-9.1296496657E+01))
    A = 1.5409136040E+00 + 1.9981117208E-02 * Td + -9.5203865864E-05 * Td**2
    B = 7.9739318223E+00 + -7.5614568881E-02 * Td + 4.7237011074E-04 * Td**2
    visd = muwd*(1 + A*0.035 + B*0.035**2)

    # Viscosity vector
    muw = (4.2844324477E-05 +
           1 /
           (1.5700386464E-01*(T + 6.4992620050E+01)**2+-9.1296496657E+01))
    C = 1.5409136040E+00 + 1.9981117208E-02 * T + -9.5203865864E-05 * T**2
    D = 7.9739318223E+00 + -7.5614568881E-02 * T + 4.7237011074E-04 * T**2
    vis = muw*(1 + C*0.035 + D*0.035**2)
    T = T + 273.15
    Td = Td + 273.15
    return T / vis * visd * Ds / Td  # Stokes-Einstein equation

def rsq(modeled, measured):
    """
    Calculate the coefficient of determination (R-squared) value
    of a regression.

    modeled:  model predictions of dependent variable
    measured: measured values of dependent variable
    """
    yresid = measured - modeled
    sse = sum(yresid**2)
    sstotal = (len(measured) - 1) * np.var(measured)
    return 1 - sse / sstotal


def conc_curve(line_fit):
    """
    Define the model to fit to concentration data.
    This function has linear or exponential options for "line_fit".
    For Athy's Law, a = (v - sqrt(v**2 * 4Dk))/2D.

    z:       depth below seafloor
    a, b, c: parameters to be fit
    """
    if line_fit == 'exponential':
        def conc_curve_specific(z,a,b,c):
            return b * np.exp(np.multiply(a,z)) + c
    elif line_fit == 'linear':
        def conc_curve_specific(z,a,b):
            return a * z + b
    return conc_curve_specific


def concentration_fit(concunique, dp, line_fit):
    """
    Find best-fitting parameters to conc_curve functions.

    concunique: concentration and depth array from load_and_prep function
    dp:         number of datapoints from sediment surface to use for fit
    line_fit:   "linear" or "exponential", as specified in conc_curve
    """
    if line_fit == 'exponential':
        a0 = -0.1
        b0 = concunique[0,1]/5
        c0 = concunique[0,1]
        if np.mean(concunique[1:dp,1]) <= concunique[0,1]:
            conc_fit, conc_cov = optimize.curve_fit(conc_curve(line_fit),
                                                    concunique[:dp,0],
                                                    concunique[:dp,1],
                                                    p0=[a0,b0,c0],
                                                    bounds=([-1,-1000,-1000],
                                                            [1,1000,1000]),
                                                    method='trf')
        else:
            conc_fit, conc_cov = optimize.curve_fit(conc_curve(line_fit),
                                                    concunique[:dp,0],
                                                    concunique[:dp,1],
                                                    p0=[-a0,-b0,c0],
                                                    bounds=([-1,-1000,-1000],
                                                            [1,1000,1000]),
                                                    method='trf')
    elif line_fit == 'linear':
        a0 = 1
        b0 = 20
        if np.mean(concunique[1:dp,1]) <= concunique[0,1]:
            conc_fit, conc_cov = optimize.curve_fit(conc_curve(line_fit),
                                                    concunique[:dp,0],
                                                    concunique[:dp,1],
                                                    p0=[-a0,b0],
                                                    bounds=([-10,-1000],
                                                            [10,1000]),
                                                    method='trf')
        else:
            conc_fit, conc_cov = optimize.curve_fit(conc_curve(line_fit),
                                                    concunique[:dp,0],
                                                    concunique[:dp,1],
                                                    p0=[a0,b0],
                                                    bounds=([-10,-1000],
                                                            [10,1000]),
                                                    method='trf')
    # To be used if Boudreau method for conc gradient is used
    # conc_interp_depths = np.arange(0,3,intervalthickness)
    # conc_interp_fit = conc_curve(conc_interp_depths, conc_fit)
    return conc_fit


def por_curve(z, por, a):
    """
    Porosity curve fit (Modified Athy's Law)
    Sets porosity at sed surface equal to greatest of first 3 measurements.

    z:   depth below seafloor (m)
    por: 2D array of depth below seafloor (m) and porosity measurements from
         load_and_prep after putting through averages function.
    a:   Athy's Law parameter to be fit
    """
    # Greatest of top 3 porosity measurements for upper porosity boundary
    # Least of bottom 3 porosity measurements as the lower boundary
    portop = np.max(por[:3,1])
    porbottom = np.mean(por[-3:,1])
    return (portop-porbottom) * np.exp(np.multiply(a, z)) + porbottom


def porosity_fit(por):
    """
    Find best-fitting parameters to por_curve function.

    por: 2D array of depth below seafloor (m) and porosity measurements from
         load_and_prep after putting through averages function.
    """
    porvalues = por[:, 1]
    pordepth = por[:, 0]
    por_fit, porcov = optimize.curve_fit(lambda z, a: por_curve(z,por,a),
                                         pordepth, porvalues, p0=-0.01)
    return por_fit


def solid_curve(z, por, a):
    """
    Solids curve function (Compliment of por_curve function)

    z:   depth below seafloor (m)
    por: 2D array of depth below seafloor (m) and porosity measurements from
         load_and_prep after putting through averages function.
    a:   Athy's Law parameter to be fit (same as por_curve Athy's parameter)
    """
    # Greatest of top 3 porosity measurements for upper porosity boundary
    # Least of bottom 3 porosity measurements as the lower boundary
    portop = np.max(por[:3,1])
    porbottom = np.mean(por[-3:,1])
    return 1-((portop-porbottom) * np.exp(np.multiply(a, z)) + porbottom)


def pw_burial(seddepths, sedtimes, por_fit, por):
    """
    Pore water volume (1-dimensional volume of water) accumulation
    rates for each age-depth section.
    Assumes constant sediment volume accumulation rates between
    age-depth measurements.

    seddepths: sediment accumulation bounds from load_and_prep function
    sedtimes:  sediment accumulation bounds from load_and_prep function
    por_fit:   Athy's Law parameter from porosity_fit function
    por:       2D array of depth below seafloor (m) and porosity measurements
               from load_and_prep after putting through averages function.
    """
    sectionmass = (integrate.quad(solid_curve,
                                  seddepths[0],
                                  seddepths[1],
                                  args=(por, por_fit)))[0]
    sedmassrate = (sectionmass/np.diff(sedtimes)[0])

    # Pore water burial flux calculation
    # Least of bottom 3 porosity measurements as the lower boundary
    deeppor = np.mean(por[-3:,1])
    deepsolid = 1 - deeppor
    pwburialflux = deeppor*sedmassrate/deepsolid
    return pwburialflux


def flux_model(conc_fit, concunique, z, pwburialflux, porosity, Dsed,
                advection, dp, Site, line_fit):
    """
    Solute flux calculation using 1D advection-diffusion model

    conc_fit:     parameters for conc_curve
    concunique:   2D numpy array of depth and concentration from load_and_prep
    z:            depth below seafloor at which flux is determined (m)
    pwburialflux: modern pore water burial rate
    porosity:     porosity at z
    Dsed:         diffusion coefficient at z
    advection:    advection rate at z
    dp:           concentration datapoints below seafloor used for line fit
    Site:         drilling site number
    line_fit:     "linear" or "exponential", as specified in conc_curve
    """
    if line_fit == 'exponential':
        # Derivative of conc_curve @ z
        gradient = (
            conc_fit[1] * conc_fit[0] * np.exp(np.multiply(conc_fit[0], z)))
        # Approximation according to Boudreau 1997
        # gradient = (-3*conc_interp_fit[0] +
        #             4*conc_interp_fit[1] -
        #             conc_interp_fit[2]) / (2*intervalthickness)
    elif line_fit == 'linear':
        # Derivative of conc_curve @ z
        gradient = conc_fit[0]
    burial_flux = pwburialflux * conc_curve(line_fit)(z, *conc_fit)
    flux = (porosity * Dsed * -gradient +
            (porosity * advection + pwburialflux)
            * conc_curve(line_fit)(z, *conc_fit))
    print('Site:', Site, 'Flux (mol/m^2 y^-1):', flux)
    return flux, burial_flux, gradient


def monte_carlo(cycles, Precision, concunique, bottom_temp_est, dp, por,
                por_fit, seddepths, sedtimes, TempD, bottom_temp, z,
                advection, Leg, Site, Solute_db, Ds, por_error, conc_fit,
                runtime_errors, line_fit):
    """
    Monte Carlo simulation of flux_model output to find total error in fluxes

    cycles:          number of simulations to run
    Precision:       relative standard deviation of concentration measurements
    concunique:      2D array of depth and concentration from load_and_prep
    bottom_temp_est: bottom water temperature
    dp:              concentration datapoints below seafloor used for line fit
    por:             2D array of depth below seafloor (m) and porosity
                     measurements from load_and_prep after putting through
                     averages function.
    por_fit:         Athy's Law parameter from porosity_fit function
    seddepths:       sediment accumulation bounds from load_and_prep function
    sedtimes:        sediment accumulation bounds from load_and_prep function
    TempD:           reference temperature of diffusion coefficient
    bottom_temp:     ocean bottom water temperature (C)
    z:               depth at which flux is calculated
    advection:       external advection rate
    Leg:             drilling leg/expedition number
    Site:            drilling site number
    Solute_db:       solute name for inserting into database
    Ds:              free diffusion coefficient at TempD
    por_error:       porosity measurement error
    conc_fit:        conc_curve parameter values
    runtime_errors:  number of runtime errors=0
    line_fit:        "linear" or "exponential", as specified in conc_curve
    """

    # Porosity offsets - using full gaussian probability
    portop = np.max(por[:3,1])
    if por_error == 0:
        por_error = 0.08 * portop
    por_offsets = np.random.normal(scale=por_error, size=(cycles,
                                                          len(por[:,1])))

    # Get randomized porosity matrix with realistic ranges between 30% and 90%
    por_rand = np.add(por_curve(por[:,0], por, *por_fit), por_offsets)
    por_rand[por_rand > 0.90] = 0.90
    por_rand[por_rand < 0.30] = 0.30

    portop_rand = np.add(portop, por_offsets[:,0])
    portop_rand = portop_rand[portop_rand > por_curve(por[-1,0],
                                                      por, *por_fit)]
    portop_rand = portop_rand[portop_rand < 0.90]

    cycles = len(portop_rand)
    por_rand = por_rand[:cycles,:]

    # Concentration offsets - using full gaussian probability
    relativeerror = Precision
    conc_offsets = np.random.normal(scale=relativeerror,
                                    size=(cycles, len(concunique[:dp,1])))

    # Bottom water temperature errors calc'd in bottom_temp_error_estimate.py
    if bottom_temp_est == 'deep':
        temp_error = 0.5
        temp_offsets = np.random.normal(scale=temp_error, size=cycles)
    elif bottom_temp_est == 'shallow':
        temp_error = 2.4
        temp_offsets = np.random.normal(scale=temp_error, size=cycles)
    else:
        temp_error = 0
        temp_offsets = np.zeros(cycles)

    '''
    # Concentration offsets - truncate at 2-sigma of gaussian distribution
    i=0
    offsets = []
    while i < cycles:
        offset = []
        j=0
        while j < len(concunique[:dp,1]):
            errors = np.random.normal(scale=relativeerror)
            if abs(errors) <= relativeerror:
                offset.append(errors)
                j = len(offset)
        offsets.append(offset)
        i = len(offsets)
    '''

    # Get randomized concentration matrix (within realistic ranges)
    conc_rand = np.add(concunique[:dp,1],
                       np.multiply(conc_offsets, concunique[:dp,1]))
    conc_rand[conc_rand < 0] = 0

    # Calculate fluxes for random profiles
    conc_fits = np.zeros((cycles, len(conc_fit))) * np.nan
    por_fits = np.zeros(cycles) * np.nan
    pwburialfluxes = np.zeros(cycles) * np.nan
    conc_rand_n = np.stack((concunique[:dp,0], np.zeros(dp) * np.nan), axis=1)
    por_rand_n = np.stack((por[:,0], np.zeros(len(por)) * np.nan), axis=1)
    for n in range(cycles):
        # Fit exponential curve to each randomized porosity profile
        # Fit exponential curve to each randomized concentration profile
        conc_rand_n[:,1] = conc_rand[n,:]
        por_rand_n[:,1] =  por_rand[n,:]
        try:
            por_fit = porosity_fit(por_rand_n)
            conc_fit2 = concentration_fit(conc_rand_n, dp, line_fit)
        except RuntimeError:
            runtime_errors += 1
            continue
        conc_fits[n,:] = conc_fit2
        por_fits[n] = por_fit
        # Pore water burial flux
        pwburialflux = pw_burial(seddepths, sedtimes, por_fit, por)
        pwburialfluxes[n] = pwburialflux

    # Dsed vector
    tortuosity_rand = 1-np.log(portop_rand**2)
    bottom_temp_rand = bottom_temp+temp_offsets
    bottom_temp_rand[bottom_temp_rand < -2] = -2
    Dsed_rand = d_stp(TempD, bottom_temp_rand, Ds)/tortuosity_rand

    """
    Plot all the monte carlo runs
    conc_interp_fit_plot = conc_curve(np.linspace(concunique[0,0],
                                                  concunique[dp-1,0],
                                                  num=50), conc_fits)
    por_interp_fit_plot = conc_curve(np.linspace(concunique[0,0],
                                                 concunique[dp-1,0],
                                                 num=50), conc_fits)
    """

    # Calculate randomized fluxes
    if line_fit == 'exponential':
        gradient = (conc_fits[:,1] * conc_fits[:,0]
                    * np.exp(np.multiply(conc_fits[:,0], z)))
        interface_fluxes = (portop_rand * Dsed_rand * -gradient
                            + (portop_rand * advection + pwburialfluxes)
                            * conc_curve(line_fit)(z,
                                                   conc_fits[:,0],
                                                   conc_fits[:,1],
                                                   conc_fits[:,2]))
    elif line_fit == 'linear':
        gradient = conc_fits[:,0]
        interface_fluxes = (portop_rand * Dsed_rand * -gradient
                            + (portop_rand * advection + pwburialfluxes)
                            * conc_curve(line_fit)(z,
                                                   conc_fits[:,0],
                                                   conc_fits[:,1]))

    # Distribution statistics
    fluxes_real = interface_fluxes[~np.isnan(interface_fluxes)]
    mean_flux = np.mean(fluxes_real)
    median_flux = np.median(fluxes_real)
    stdev_flux = np.std(fluxes_real)
    skewness = stats.skew(abs(fluxes_real))
    fluxes_norm = (fluxes_real - mean_flux) / stdev_flux
    test_stat, p_value = stats.kstest(fluxes_norm, 'norm')

    # Stats on lognormal distribution
    fluxes_log_real = -np.log(abs(fluxes_real))
    mean_flux_log = np.mean(fluxes_log_real)
    median_flux_log = np.median(fluxes_log_real)
    stdev_flux_log = np.std(fluxes_log_real)
    skewness_log = stats.skew(fluxes_log_real)
    fluxes_log_norm = (fluxes_log_real - mean_flux_log)/stdev_flux_log
    test_stat_log, p_value_log = stats.kstest(fluxes_log_norm, 'norm')
    stdev_flux_lower = np.exp(-(median_flux_log - stdev_flux_log))
    stdev_flux_upper = np.exp(-(median_flux_log + stdev_flux_log))
    return (fluxes_real, fluxes_log_real, cycles, por_error,
            mean_flux, median_flux, stdev_flux, skewness, p_value,
            mean_flux_log, median_flux_log, stdev_flux_log,
            stdev_flux_lower, stdev_flux_upper, skewness_log, p_value_log,
            runtime_errors, conc_fits)


def monte_carlo_fract(cycles, Precision, Precision_iso, concunique,
                      bottom_temp_est, dp, por, por_fit, seddepths, sedtimes,
                      TempD, bottom_temp, z, advection, Leg, Site, Solute_db,
                      Ds, por_error, conc_fit, runtime_errors, isotopedata,
                      mg26_24_ocean, line_fit):
    """
    Monte Carlo simulation of flux_model (applied to Mg fractionation) outputs
    to find total error in Mg fractionations.

    cycles:          number of simulations to run
    Precision:       relative standard deviation of concentration measurements
    Precision_iso:   standard deviation of isotope measurements
    concunique:      2D array of depth and concentration from load_and_prep
    bottom_temp_est: bottom water temperature
    dp:              concentration datapoints below seafloor used for line fit
    por:             2D array of depth below seafloor (m) and porosity
                     measurements from load_and_prep after putting through
                     averages function.
    por_fit:         Athy's Law parameter from porosity_fit function
    seddepths:       sediment accumulation bounds from load_and_prep function
    sedtimes:        sediment accumulation bounds from load_and_prep function
    TempD:           reference temperature of diffusion coefficient
    bottom_temp:     ocean bottom water temperature (C)
    z:               depth at which flux is calculated
    advection:       external advection rate
    Leg:             drilling leg/expedition number
    Site:            drilling site number
    Solute_db:       solute name for inserting into database
    Ds:              free diffusion coefficient at TempD
    por_error:       porosity measurement error
    conc_fit:        conc_curve parameter values
    runtime_errors:  number of runtime errors=0
    isotopedata:     2D numpy array of depth and isotopic delta values
    mg26_24_ocean:   delta value of the ocean
    line_fit:        "linear" or "exponential", as specified in conc_curve
    """
    # Porosity offsets - using full gaussian probability
    por_offsets = np.random.normal(scale=por_error,
                                   size=(cycles, len(por[:,1])))

    # Randomized porosity matrix (within realistic ranges between 30% and 90%)
    por_rand = np.add(por_curve(por[:,0], por, *por_fit), por_offsets)
    por_rand[por_rand > 0.90] = 0.90
    por_rand[por_rand < 0.30] = 0.30

    portop = np.max(por[:3,1])
    portop_rand = np.add(portop, por_offsets[:,0])
    portop_rand = portop_rand[portop_rand > por_curve(por[-1,0],
                                                      por,
                                                      *por_fit)]
    portop_rand = portop_rand[portop_rand < 0.90]

    cycles = len(portop_rand)
    por_rand = por_rand[:cycles,:]

    # Concentration offsets - using full gaussian probability
    # relativeerror = Precision
    # conc_offsets = np.random.normal(scale=relativeerror,
    #                                 size=(cycles, len(isotopedata[:, 3])))
    # Get randomized concentration matrix (within realistic ranges)
    # conc_rand = np.add(isotopedata[:, 3], np.multiply(conc_offsets,
    #                                                    isotopedata[:, 3]))
    # conc_rand[conc_rand < 0] = 0

    # Isotopic ratio (in delta notation) offsets - full gaussian probability
    delta_offsets = Precision_iso[:,None]*np.random.normal(scale=1,
                                            size=(len(Precision_iso), cycles))

    # Calculate randomized Mg isotope concentration profiles
    # Decimal numbers are isotopic ratios of standards.
    # Source for 26/24std: Isotope Geochemistry, William White, pp.365
    # 25Mg/24Mg calc'd from from 26/24 value divided by 25/24 measured values.
    mg26_24 = (((isotopedata[:, 1][:,None]+delta_offsets)/1000)+1)*0.13979
    mg25_24 = (((isotopedata[:, 2][:,None]+delta_offsets)/1000)+1)*0.126635
    concunique_mg24 = isotopedata[:, 3][:,None]/(mg26_24+mg25_24+1)
    # concunique_mg24 = conc_rand.T/(mg26_24+mg25_24+1)
    concunique_mg26 = concunique_mg24*mg26_24
    isotope_concs = [concunique_mg24, concunique_mg26]

    # Bottom water temp errors calculated in bottom_temp_error_estimate.py
    if bottom_temp_est == 'deep':
        temp_error = 0.5
        temp_offsets = np.random.normal(scale=temp_error, size=cycles)
    elif bottom_temp_est == 'shallow':
        temp_error = 2.4
        temp_offsets = np.random.normal(scale=temp_error, size=cycles)
    else:
        temp_error = 0
        temp_offsets = np.zeros(cycles)

    # Calculate fluxes for random profiles
    conc_fits = np.zeros((cycles, len(conc_fit), 2)) * np.nan
    por_fits = np.zeros(cycles) * np.nan
    pwburialfluxes = np.zeros(cycles) *np.nan
    conc_rand_n = np.stack((isotopedata[:dp,0], np.zeros(dp) * np.nan), axis=1)
    por_rand_n = np.stack((por[:,0], np.zeros(len(por)) * np.nan), axis=1)
    for n in range(cycles):
        for i in np.arange(2):
            conc_rand_n[:,1] = isotope_concs[i][:dp,n]
            # Fit pore water concentration curve
            try:
                conc_fit = concentration_fit(conc_rand_n, dp, line_fit)
                conc_fits[n,:,i] = conc_fit
            except RuntimeError:
                runtime_errors += 1
                continue
        # Fit exponential curve to each randomized porosity profile
        por_rand_n[:,1] =  por_rand[n,:]
        por_fit = porosity_fit(por_rand_n)
        por_fits[n] = por_fit
        # Pore water burial mass flux
        pwburialflux = pw_burial(seddepths, sedtimes, por_fit, por)
        pwburialfluxes[n] = pwburialflux

    # Filter out profiles where 26Mg or 24Mg were erroneously fit
    if line_fit == 'exponential':
        conc_fits[abs(conc_fits[:,0,0] - conc_fits[:,0,1]) >
                  0.05 * abs(conc_fits[:,0,0] + conc_fits[:,0,1])/2] = np.nan

    # Effective diffusion coeffiecient vector
    tortuosity_rand = 1-np.log(portop_rand**2)
    bottom_temp_rand = bottom_temp+temp_offsets
    bottom_temp_rand[bottom_temp_rand < -2] = -2
    Dsed_rand = d_stp(TempD, bottom_temp_rand, Ds)/tortuosity_rand

    """
    Plot all the monte carlo runs
    conc_interp_fit_plot = conc_curve(np.linspace(concunique[0,0],
                                                  concunique[dp-1,0],
                                                  num=50), conc_fits)
    por_interp_fit_plot = conc_curve(np.linspace(concunique[0,0],
                                                 concunique[dp-1,0],
                                                 num=50), conc_fits)
    """

    # Calculate fluxes
    fluxes = []
    for i in np.arange(2):
        if line_fit == 'exponential':
            gradient = (conc_fits[:,1,i] *
                        conc_fits[:,0,i] *
                        np.exp(np.multiply(conc_fits[:,0,i], z)))
            interface_fluxes = (portop_rand * Dsed_rand * -gradient +
                               (portop_rand * advection + pwburialfluxes) *
                               conc_curve(line_fit)(z,
                                                    conc_fits[:,0,i],
                                                    conc_fits[:,1,i],
                                                    conc_fits[:,2,i]))
        elif line_fit == 'linear':
            gradient = conc_fits[:,0,i]
            interface_fluxes = (portop_rand * Dsed_rand * -gradient +
                               (portop_rand * advection + pwburialfluxes) *
                               conc_curve(line_fit)(z,
                                                    conc_fits[:,0,i],
                                                    conc_fits[:,1,i]))
        fluxes.append(interface_fluxes)
    mg26_24_flux = fluxes[1]/fluxes[0]
    alpha = mg26_24_flux/mg26_24_ocean
    epsilon = (alpha - 1) * 1000

    # Distribution statistics
    alpha_mean = np.nanmean(alpha)
    alpha_median = np.nanmedian(alpha)
    alpha_stdev = np.nanstd(alpha)
    alpha_norm = (alpha[~np.isnan(alpha)]-alpha_mean)/alpha_stdev
    test_stat, p_value = stats.kstest(alpha_norm, 'norm')
    return (alpha, epsilon, cycles, por_error, alpha_mean, alpha_median,
            alpha_stdev, test_stat, p_value, runtime_errors, conc_fits, fluxes)


def flux_plots(concunique, conc_interp_fit_plot, por, por_all, porfit,
               bottom_temp, picks, sedtimes, seddepths, Leg, Site, Solute_db,
               flux, dp, temp_gradient):
    """
    Plot the input data and line fits used in the flux_model.

    concunique:        2D array of depth and concentration from load_and_prep
    conc_interp_fit_plot: 2D array of concentration model
    por:               2D array of depth below seafloor (m) and porosity
                       measurements from load_and_prep after putting through
                       averages function.
    por_all:           2D array of all porosity data
    porfit:            por_curve parameters
    bottom_temp:       ocean bottom water temperature (C)
    picks:             biostratigraphic age-depth data
    seddepths:         sediment accumulation bounds from load_and_prep function
    sedtimes:          sediment accumulation bounds from load_and_prep function
    Leg:               drilling leg/expedition number
    Site:              drilling site number
    Solute_db:         solute name for inserting into database
    flux:              solute flux calculated at this site (mol m^-2 y^-1)
    dp:                datapoints used for solute concentration curve fit
    temp_gradient:     geothermal gradient (C/m)

    """

    # Set up axes and subplot grid
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rc('font',family='serif')
    figure_1, (ax1, ax2, ax3) = plt.subplots(1, 3,
                                             figsize=(10, 6),
                                             facecolor='none')
    grid = gridspec.GridSpec(3, 6, wspace=0.7)
    ax1 = plt.subplot(grid[0:3, :2])
    ax1.grid()
    ax2 = plt.subplot(grid[0:3, 2:4], sharey=ax1)
    ax2.grid()
    ax3 = plt.subplot(grid[0:3, 4:6], sharey=ax1)
    ax3.grid()

    # Figure title
    figure_1.suptitle(
        r"$Expedition\ {},\ Site\ {}\ \ \ \ \ \ \ \ \ \ {}\ flux={}\ mol/m^2y$"
        .format(Leg, Site, Solute_db, round(flux,4)), fontsize=20)

    # Plot input data
    ax1.plot(concunique[:,1],
             concunique[:,0],
             'go')
    ax1.plot(concunique[0:dp,1],
             concunique[0:dp,0],
             'bo', label="Used for curve fit")
    ax1.plot(conc_interp_fit_plot,
             np.linspace(concunique[0,0], concunique[dp-1,0], num=50),
            'k-', label="Curve fit", linewidth=2)
    ax2.plot(por_all[:, 1],
             por_all[:, 0],
             'mo',
             label='Measured')
    ax2.plot(por_curve(por[:,0], por, *porfit),
             por[:,0],
            'k-',
            label='Curve fit',
            linewidth=3)
    ax3.plot(picks[:,1]/1000000,
             picks[:,0],
             'ro',
             label='Biostrat picks')
    ax3.plot(sedtimes/1000000,
             seddepths,
             'k-',
             label='Curve fit',
             linewidth=2)

    # Inset in concentration plot
    y2 = np.ceil(concunique[dp-1,0])
    x2 = max(concunique[:dp,1])+0.01
    x1 = min(concunique[:dp,1])-0.01
    axins1 = inset_axes(ax1, width="50%", height="30%", loc=5)
    axins1.plot(concunique[:,1],
                concunique[:,0],
                'go')
    axins1.plot(concunique[0:dp,1],
                concunique[0:dp,0],
                'bo',
                label="Used for curve fit")
    axins1.plot(conc_interp_fit_plot,
                np.linspace(concunique[0,0],
                            concunique[dp-1,0],
                            num=50),
                            'k-')
    axins1.set_xlim(x1-0.01, x2+0.01)
    axins1.set_ylim(0, y2)
    mark_inset(ax1, axins1, loc1=1, loc2=2, fc="none", ec="0.5")

    # Additional formatting
    ax1.legend(loc='lower left', fontsize='small')
    ax2.legend(loc='best', fontsize='small')
    ax3.legend(loc='best', fontsize='small')
    ax1.set_ylabel('$Depth\ (mbsf)$', fontsize=18)
    ax1.set_xlabel('${}\ (mM)$'.format(Solute_db), fontsize=18)
    ax2.set_xlabel('$Porosity$', fontsize=18)
    ax3.set_xlabel('$Age\ (Ma)$', fontsize=18)
    #ax4.set_xlabel('Temperature (\u00b0C)')
    ax1.locator_params(axis='x', nbins=4)
    ax2.locator_params(axis='x', nbins=4)
    ax3.locator_params(axis='x', nbins=4)
    #ax4.locator_params(axis='x', nbins=4)
    axins1.locator_params(axis='x', nbins=3)
    ax1.invert_yaxis()
    axins1.invert_yaxis()
    return figure_1


def monte_carlo_plot_fract(alpha, alpha_median, alpha_stdev,
                              alpha_mean, p_value):
    """
    Plot the distribution of values from the Monte Carlo simulation of
    fractionation factors.

    alpha:        fractionation factor from ocean to sediment column
    alpha_median: median of fractionation factors from monte carlo simulation
    alpha_stdev:  standard deviation of fractionation factors from m.c. sim.
    alpha_mean:   average of fractionation factors from monte carlo simulation
    p_value:      Kolmogorov-Smirnov test p-value of Monte Carlo output
    """

    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rc('font',family='serif')
    mc_figure, (ax5) = plt.subplots(1, 1, figsize=(6, 5),gridspec_kw={
                                                        'wspace':0.2,
                                                        'top':0.92,
                                                        'bottom':0.13,
                                                        'left':0.1,
                                                        'right':0.90})

    # Plot histogram of results
    n_1, bins_1, patches_1 = ax5.hist((alpha[~np.isnan(alpha)]-1)*1000,
                                      normed=1,
                                      bins=50,
                                      facecolor='blue')

    # Best fit normal distribution line to results
    bf_line_1 = mlab.normpdf(bins_1, (alpha_mean-1)*1000, alpha_stdev*1000)
    ax5.plot(bins_1, bf_line_1, 'k--', linewidth=2)
    ax5.set_xlabel("$Epsilon$", fontsize=20)
    ax5.locator_params(axis='x', nbins=5)
    [left_raw, right_raw] = ax5.get_xlim()
    [bottom_raw, top_raw] = ax5.get_ylim()
    ax5.text((left_raw+(right_raw-left_raw)/20),
             (top_raw-(top_raw-bottom_raw)/10),
             """$K-S\ p-value\ =\ {}$""".format(np.round(p_value, 2)))
    return mc_figure


def monte_carlo_plot(fluxes_real, mean_flux, stdev_flux, skewness,
                       p_value, fluxes_log_real, median_flux_log,
                       stdev_flux_log, skewness_log, p_value_log):

    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rc('font',family='serif')
    mc_figure, (ax5, ax6) = plt.subplots(1, 2,
                                         figsize=(12, 5),
                                         gridspec_kw={'wspace':0.2,
                                                      'top':0.92,
                                                      'bottom':0.13,
                                                      'left':0.1,
                                                      'right':0.90})

    # Plot histogram of results
    n_1, bins_1, patches_1 = ax5.hist(fluxes_real * 1000,
                                      normed=1,
                                      bins=30,
                                      facecolor='orange')

    # Best fit normal distribution line to results
    bf_line_1 = mlab.normpdf(bins_1, mean_flux*1000, stdev_flux*1000)
    ax5.plot(bins_1, bf_line_1, 'k--', linewidth=2)
    ax5.set_xlabel("$Interface\ flux\ (mmol\ m^{-2}\ y^{-1})$", fontsize=20)

    [left_raw, right_raw] = ax5.get_xlim()
    [bottom_raw, top_raw] = ax5.get_ylim()
    ax5.text((left_raw+(right_raw-left_raw)/20),
             (top_raw-(top_raw-bottom_raw)/20),
             '$sk\ =\ {}$'.format(np.round(skewness, 2)))
    ax5.text((left_raw+(right_raw-left_raw)/20),
             (top_raw-(top_raw-bottom_raw)/10),
             "$K-S\ p-value\ =\ {}$".format(np.round(p_value, 2)))

    # Plot histogram of ln(results)
    n_2, bins_2, patches_2 = plt.hist(fluxes_log_real,
                                      normed=1,
                                      bins=30,
                                      facecolor='g')

    # Best fit normal distribution line to ln(results)
    bf_line_2 = mlab.normpdf(bins_2, median_flux_log, stdev_flux_log)
    ax6.plot(bins_2, bf_line_2, 'k--', linewidth=2)
    ax6.set_xlabel("$ln(abs(Interface flux)$", fontsize=20)

    [left_log, right_log] = ax6.get_xlim()
    [bottom_log, top_log] = ax6.get_ylim()
    ax6.text((left_log+(right_log-left_log)/20),
             (top_log-(top_log-bottom_log)/20),
             '$sk\ =\ {}$'.format(np.round(skewness_log, 2)))
    ax6.text((left_log+(right_log-left_log)/20),
             (top_log-(top_log-bottom_log)/10),
             "$K-S\ p-value\ =\ {}$".format(np.round(p_value_log, 2)))
    return mc_figure


def flux_to_sql(con, Solute_db, site_key,Leg,Site,Hole,Solute,flux,
                 burial_flux,gradient,porosity,z,dp,bottom_conc,conc_fit,
                 r_squared,age_depth_boundaries,sedrate,advection,Precision,Ds,
                 TempD,bottom_temp,bottom_temp_est,cycles,por_error,mean_flux,
                 median_flux,stdev_flux,skewness,p_value,mean_flux_log,
                 median_flux_log,stdev_flux_log,stdev_flux_lower,
                 stdev_flux_upper,skewness_log,p_value_log,runtime_errors,Date,
                 Comments,Complete):
    """
    Send the output of interface_flux.py or flux_rerun.py to the
    MySQL database. If data for a particular site already exists, replaces
    old data with new data.

    con:              database engine connection
    Solute_db:        solute name for inserting into database
    site_key:         MySQL database site key
    Leg:              drilling leg/expedition number
    Site:             drilling site number
    Hole:             drilling hole IDs
    Solute:           solute name in database
    flux:             solute flux at z (mol m^-2 y^-1). Positive flux value is
                      downward (into the sediment)
    burial_flux:      solute flux due to pore water burial at z (mol m^-2 y^-1)
    gradient:         pore water solute concentration gradient at z (mol m^-1)
    porosity:         porosity at z
    z:                depth at which flux is calculated (mbsf)
    dp:               concentration datapoints below seafloor used for line fit
    bottom_conc:      ocean bottom water solute concentration (mM)
    conc_fit:         parameters of solute concentration curve (see conc_curve)
    r_squared:        R-squared of regression between model and measurements
    age_depth_boundaries: bounds between discrete sedimentation rate regimes
    sedrate:          modern sediment accumulation rate (solid volume per year)
    advection:        external advection rate (m/y)
    Precision:        relative standard deviation of concentration measurements
    Ds:               diffusion coefficient at reference temperature (m^2 y^-1)
    TempD:            reference temperature of diffusion coefficient (C)
    bottom_temp:      ocean bottom water temperature (C)
    bottom_temp_est:  ocean bottom water temperature parameter for estimation
    cycles:           number of monte carlo simulations to run
    por_error:        relative standard deviation of porosity fit
    mean_flux:        average solute flux from monte carlo simulation
    median_flux:      median solute flux from monte carlo simulation
    stdev_flux:       standard deviation of solute flux from monte carlo sim.
    skewness:         skewness of distribution of fluxes from monte carlo sim.
    p_value:          Kolmogorov-Smirvov p-value of distribution of fluxes
    mean_flux_log:    log-normal average of fluxes from monte carlo sim.
    median_flux_log:  log-normal median of fluxes from monte carlo sim.
    stdev_flux_log:   log-normal standard deviation of fluxes from m.c. sim.
    stdev_flux_lower: upper log-normal value for 1 standard deviation
    stdev_flux_upper: lower log-normal value for 1 standard deviation
    skewness_log:     skewness of distribution of log-normal fluxes
    p_value_log:      Kolmogorov-Smirvov p-value of log-normal fluxes
    runtime_errors:   number of errors from line-fitting procedure
    Date:             date the site was modeled
    Complete:         if "yes", modeling is complete for this site
    Comments:         comments for this location or dataset
    """

    # Send metadata to database
    sql= """insert into metadata_{}_flux (site_key,leg,site,hole,solute,
                                          interface_flux,burial_flux,gradient,
                                          top_por,flux_depth,datapoints,
                                          bottom_conc,conc_fit,r_squared,
                                          age_depth_boundaries,sed_rate,
                                          advection,measurement_precision,ds,
                                          ds_reference_temp,bottom_temp,
                                          bottom_temp_est,mc_cycles,
                                          porosity_error,mean_flux,median_flux,
                                          stdev_flux,skewness,p_value,
                                          mean_flux_log,median_flux_log,
                                          stdev_flux_log,stdev_flux_lower,
                                          stdev_flux_upper,skewness_log,
                                          p_value_log,runtime_errors,run_date,
                                          comments,complete)
            VALUES ({}, '{}', '{}', '{}', '{}', {}, {}, {}, {}, {}, {}, {},
                    '{}', {},'{}', {}, {}, {}, {}, {}, {}, '{}', {}, {}, {},
                    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},'{}','{}',
                    '{}')
            ON DUPLICATE KEY UPDATE hole='{}', solute='{}', interface_flux={},
            burial_flux={}, gradient={}, top_por={}, flux_depth={},
            datapoints={}, bottom_conc={}, conc_fit='{}', r_squared={},
            age_depth_boundaries='{}', sed_rate={}, advection={},
            measurement_precision={}, ds={}, ds_reference_temp={},
            bottom_temp={}, bottom_temp_est='{}', mc_cycles={},
            porosity_error={}, mean_flux={}, median_flux={},
            stdev_flux={}, skewness={}, p_value={}, mean_flux_log={},
            median_flux_log={}, stdev_flux_log={}, stdev_flux_lower={},
            stdev_flux_upper={}, skewness_log={}, p_value_log={},
            runtime_errors={}, run_date='{}', comments='{}', complete='{}'
            ;""".format(Solute_db, site_key,Leg,Site,Hole,Solute,
                        flux,burial_flux,gradient,porosity,
                        z,dp,bottom_conc,conc_fit,r_squared,
                        age_depth_boundaries,sedrate,advection,
                        Precision,Ds,TempD,bottom_temp,bottom_temp_est,cycles,
                        por_error,mean_flux,median_flux,stdev_flux,skewness,
                        p_value,mean_flux_log,median_flux_log,stdev_flux_log,
                        stdev_flux_lower,stdev_flux_upper,skewness_log,
                        p_value_log,runtime_errors,Date,Comments,Complete,
                        Hole,Solute,flux,burial_flux,gradient,porosity,
                        z,dp,bottom_conc,conc_fit,r_squared,
                        age_depth_boundaries,sedrate,advection,Precision,Ds,
                        TempD,bottom_temp,bottom_temp_est,cycles,
                        por_error,mean_flux,median_flux,stdev_flux,
                        skewness,p_value,mean_flux_log,median_flux_log,
                        stdev_flux_log,stdev_flux_lower,stdev_flux_upper,
                        skewness_log,p_value_log,runtime_errors,Date,Comments,
                        Complete)
    con.execute(sql)


def flux_only_to_sql(con, Solute_db, site_key,Leg,Site,Hole,Solute,flux,
                       burial_flux,gradient,porosity,z,dp,bottom_conc,conc_fit,
                       r_squared,age_depth_boundaries,sedrate,advection,
                       Precision,Ds,TempD,bottom_temp,bottom_temp_est,Date,
                       Comments,Complete):
    """
    Send the output of initial_flux.py to the MySQL database. Does not include
    any data from Monte Carlo simulation. If data for a particular site
    already exists, replaces old data with new data.

        con:              database engine connection
    Solute_db:        solute name for inserting into database
    site_key:         MySQL database site key
    Leg:              drilling leg/expedition number
    Site:             drilling site number
    Hole:             drilling hole IDs
    Solute:           solute name in database
    flux:             solute flux at z (mol m^-2 y^-1). Positive flux value is
                      downward (into the sediment)
    burial_flux:      solute flux due to pore water burial at z (mol m^-2 y^-1)
    gradient:         pore water solute concentration gradient at z (mol m^-1)
    porosity:         porosity at z
    z:                depth at which flux is calculated (mbsf)
    dp:               concentration datapoints below seafloor used for line fit
    bottom_conc:      ocean bottom water solute concentration (mM)
    conc_fit:         parameters of solute concentration curve (see conc_curve)
    r_squared:        R-squared of regression between model and measurements
    age_depth_boundaries: bounds between discrete sedimentation rate regimes
    sedrate:          modern sediment accumulation rate (solid volume per year)
    advection:        external advection rate (m/y)
    Precision:        relative standard deviation of concentration measurements
    Ds:               diffusion coefficient at reference temperature (m^2 y^-1)
    TempD:            reference temperature of diffusion coefficient (C)
    bottom_temp:      ocean bottom water temperature (C)
    bottom_temp_est:  ocean bottom water temperature parameter for estimation
    Date:             date the site was modeled
    Complete:         if "yes", modeling is complete for this site
    Comments:         comments for this location or dataset
    """

    # Send metadata to database
    sql= """insert into metadata_{}_flux (site_key,leg,site,hole,solute,
                                          interface_flux,burial_flux,gradient,
                                          top_por,flux_depth,datapoints,
                                          bottom_conc,conc_fit,r_squared,
                                          age_depth_boundaries,sed_rate,
                                          advection,measurement_precision,ds,
                                          ds_reference_temp,bottom_temp,
                                          bottom_temp_est,run_date,comments,
                                          complete)
            VALUES ({}, '{}', '{}', '{}', '{}', {}, {}, {}, {}, {}, {}, {},
                    '{}', {},'{}', {}, {}, {}, {}, {}, {}, '{}','{}','{}','{}')
            ON DUPLICATE KEY UPDATE hole='{}', solute='{}', interface_flux={},
            burial_flux={}, gradient={}, top_por={}, flux_depth={},
            datapoints={}, bottom_conc={}, conc_fit='{}', r_squared={},
            age_depth_boundaries='{}', sed_rate={}, advection={},
            measurement_precision={}, ds={}, ds_reference_temp={},
            bottom_temp={}, bottom_temp_est='{}', run_date='{}',
            comments='{}', complete='{}'
            ;""".format(Solute_db, site_key,Leg,Site,Hole,Solute,
                        flux,burial_flux,gradient,porosity,z,dp,bottom_conc,
                        conc_fit,r_squared,age_depth_boundaries,sedrate,
                        advection, Precision,Ds,TempD,bottom_temp,
                        bottom_temp_est,Date,Comments,Complete,
                        Hole,Solute,flux,burial_flux,gradient,porosity,
                        z,dp,bottom_conc,conc_fit,r_squared,
                        age_depth_boundaries,sedrate,advection,
                        Precision,Ds,TempD,bottom_temp,bottom_temp_est,
                        Date,Comments,Complete)
    con.execute(sql)

# eof
