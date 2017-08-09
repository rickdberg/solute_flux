# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 14:39:00 2017

@author: rickdberg

Modules for calculating flux of solutes into/out of sediments

"""

import numpy as np
import pandas as pd
from scipy import integrate, stats, optimize
from sqlalchemy import text
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import mlab

import seawater

from data_handling import averages

# Assign advection rate to site based on sediment thickness and lithology
# Sediment types: 1: gravel, 2: sand, 3:silt, 4: clay, 5:Calcareous ooze,
# 6:Radiolarian ooze, 7:Diatom ooze, 8:Sponge spicules, 9:Mixed calc/sil ooze,
# 10: shell/coral fragments, 11:Tephra, 12:Siliceous mud,
# 13:Fine-grained calcareous sediment
# Equations from Davis & Elderfield 2004
# Hydrogeology of the Oceanic Lithosphere Table 6.2
def adv_rate(sed_thickness, lithology, advection, pressure=10000):
    if sed_thickness < 250:
        if lithology in (1,2):
            def k(z):
                p = 0.84 * z**(-0.125)
                return 1/(3.7*10**(-18) * np.exp(1.7*(p/(1-p))))
        elif lithology in (5,9,10,13):
            def k(z):
                p = 0.72 - 0.987*(z/10**3) - 0.83*(z/10**3)**2
                return 1/(5.6*10**(-21)*np.exp(18*p))
        elif lithology in (3,11,12):
            def k(z):
                p = 0.909*z**(-0.073)
                return 1/(1.1*10**(-18)*np.exp(2.2*(p/(1-p))))
        elif lithology == 4:
            def k(z):
                p = 0.814 - 0.813*(z/10**3) - 0.164*(z/10**3)**2
                return 1/((10**(1.15*(p/(1-p)) - 13.8))/10**5)
        elif lithology in (6,7,8):
            def k(z):
                p = 0.9 - 0.016*(z/10**3) - 3.854*(z/10**3)**2
                return 1/(4.6 * 10**(-23)*np.exp(23*p))
        impedance = integrate.fixed_quad(k,0.001,sed_thickness)
        advection = (sed_thickness/impedance[0])*pressure/sed_thickness/(1.5*10**-3)*(60*60*24*365.25)
    else:
        advection = -advection
    return -advection

# Get site data and prepare for modeling
def load_and_prep(Leg, Site, Holes, Solute, Ocean, engine, conctable,
                  portable, site_metadata):

    # Temperature gradient (degrees C/m)
    temp_gradient = site_metadata['temp_gradient'][0]

    # Water depth for bottom water estimation
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

    # Lithology (from Dutkeiwitz)
    lithology = int(site_metadata['lithology'][0])

    # Sediment thickness (From Sediment_thickness_combined)
    sed_thickness = float(site_metadata['sed_thickness_combined'][0])

    # Pore water chemistry data
    sql = """SELECT sample_depth, {}
        FROM {}
        where leg = '{}' and site = '{}' and (hole in {})
        and {} is not null and hydrate_affected is null
        ; """.format(Solute, conctable, Leg, Site, Holes, Solute)
    concdata = pd.read_sql(sql, engine)
    concdata = concdata.sort_values(by='sample_depth')
    concdata = concdata.as_matrix()

    # Bottom water concentration of Mg based on WOA bottom salinities.
    woa_salinity = site_metadata['woa_salinity'][0]
    density = seawater.eos80.dens0(woa_salinity, bottom_temp)
    woa_cl = (1000*(woa_salinity-0.03)*density/1000)/(1.805*35.45)  # Bowles et al reference
    bottom_conc=woa_cl/558*Ocean
    ct0 = [bottom_conc]  # mol per m^3 in modern seawater at specific site

    # Porosity data
    sql = text("""SELECT sample_depth, porosity
    FROM {}
    where leg = '{}' and site = '{}' and (hole in {})
    and coalesce(method,'C') like '%C%'
    and {} is not null and {} > 0 AND {} < 1 and sample_depth is not null
    ;""".format(portable, Leg, Site, Holes, 'porosity', 'porosity', 'porosity'))
    pordata = pd.read_sql(sql, engine)
    if pordata.iloc[:,1].count() < 40:  # If not enough data from method C, take all data
        sql = """SELECT sample_depth, porosity
            FROM {}
            where leg = '{}' and site = '{}' and (hole in {})
            and {} is not null and {} > 0 AND {} < 1 and sample_depth is not null
            ;""".format(portable, Leg, Site, Holes, 'porosity', 'porosity', 'porosity')
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
    sedrates = np.diff(seddepths, axis=0)/np.diff(sedtimes, axis=0)  # m/y
    sedrate = sedrates[0]  # Using modern sedimentation rate
    # print('Modern sed rate (cm/ky):', np.round(sedrate*100000, decimals=3))

    # Load age-depth data for plots
    sql = """SELECT depth, age
        FROM age_depth
        where leg = '{}' and site = '{}' order by 1
        ;""".format(Leg, Site)
    picks = pd.read_sql(sql, engine)
    picks = picks.as_matrix()
    picks = picks[np.argsort(picks[:,0])]

    # Age-Depth boundaries from database used in this run
    sql = """SELECT age_depth_boundaries
        FROM metadata_sed_rate
        where leg = '{}' and site = '{}' order by 1
        ;""".format(Leg, Site)
    age_depth_boundaries = pd.read_sql(sql, engine).iloc[0,0] # Indices when sorted by age

    ###############################################################################

    # Concentration vector after averaging duplicates
    concunique = averages(concdata[:, 0], concdata[:, 1])
    if concunique[0,0] > 0.05:
        concunique = np.concatenate((np.array(([0],ct0)).T, concunique), axis=0)  # Add in seawater value (upper limit)
    return concunique, temp_gradient, bottom_conc, bottom_temp, bottom_temp_est, pordata, sedtimes, seddepths, sedrate, picks, age_depth_boundaries, advection, lithology, sed_thickness

# Temperature profile (degrees C)
def sedtemp(z, bottom_temp, temp_gradient):
    if z.all() == 0:
        return bottom_temp*np.ones(len(z))
    else:
        return bottom_temp + np.multiply(z, temp_gradient)


# Calculates viscosity from Mostafa H. Sharqawy 12-18-2009, MIT (mhamed@mit.edu) Sharqawy M. H., Lienhard J. H., and Zubair, S. M., Desalination and Water Treatment, 2009
# Viscosity used as input into Stokes-Einstein equation
# Td is the reference temperature (TempD), T is the in situ temperature
def d_stp(Td, T, Ds):
    # Viscosity at reference temperature
    muwd = 4.2844324477E-05 + 1/(1.5700386464E-01*(Td+6.4992620050E+01)**2+-9.1296496657E+01)
    A = 1.5409136040E+00 + 1.9981117208E-02 * Td + -9.5203865864E-05 * Td**2
    B = 7.9739318223E+00 + -7.5614568881E-02 * Td + 4.7237011074E-04 * Td**2
    visd = muwd*(1 + A*0.035 + B*0.035**2)

    # Viscosity vector
    muw = 4.2844324477E-05 + 1/(1.5700386464E-01*(T+6.4992620050E+01)**2+-9.1296496657E+01)
    C = 1.5409136040E+00 + 1.9981117208E-02 * T + -9.5203865864E-05 * T**2
    D = 7.9739318223E+00 + -7.5614568881E-02 * T + 4.7237011074E-04 * T**2
    vis = muw*(1 + C*0.035 + D*0.035**2)
    T = T+273.15
    Td = Td+273.15
    return T/vis*visd*Ds/Td  # Stokes-Einstein equation

# Calculate r-squared value of fit
def rsq(modeled, measured):
    yresid = measured - modeled
    sse = sum(yresid**2)
    sstotal = (len(measured)-1)*np.var(measured)
    return 1-sse/sstotal

# Fit exponential curve to concentration datapoints (specified as "dp")
def conc_curve(z,a,b,c):
    return b * np.exp(np.multiply(a,z)) + c  # a = (v - sqrt(v**2 * 4Dk))/2D
'''def conc_curve(z, a, b, k):
    return a*np.sin(np.sqrt(k)*z)+b*np.cos(np.sqrt(k)*z)
'''
'''def conc_curve(z, a):
    return (concunique[0,1]-concunique[dp-1,1]) * np.exp(np.multiply(np.multiply(-1, a), z)) + concunique[dp-1,1]
'''

def concentration_fit(concunique, dp):
    if np.mean(concunique[1:dp,1]) < concunique[0,1]:
        conc_fit, conc_cov = optimize.curve_fit(conc_curve, concunique[:dp,0], concunique[:dp,1], p0=[-0.1,10,40], bounds=([-1,-100,-1000],[1,1000,1000]), method='trf')
    else:
        conc_fit, conc_cov = optimize.curve_fit(conc_curve, concunique[:dp,0], concunique[:dp,1], p0=[-0.1,-10,60], bounds=([-1,-1000,-1000],[1,1000,1000]), method='trf')
    # conc_interp_depths = np.arange(0,3,intervalthickness)  # Three equally-spaced points
    # conc_interp_fit = conc_curve(conc_interp_depths, conc_fit)  # To be used if Boudreau method for conc gradient is used
    return conc_fit

# Porosity and solids fraction functions and data preparation

# Porosity curve fit (Modified Athy's Law, ) (Makes porosity at sed surface equal to greatest of first 3 measurements)
def por_curve(z, por, a):
    portop = np.max(por[:3,1])  # Greatest of top 3 porosity measurements for upper porosity boundary
    porbottom = np.mean(por[-3:,1])  # Takes lowest porosity measurement as the lower boundary
    return (portop-porbottom) * np.exp(np.multiply(a, z)) + porbottom


# Porosity vectors
def porosity_fit(por):
    porvalues = por[:, 1]
    pordepth = por[:, 0]

    por_fit, porcov = optimize.curve_fit(lambda z, a: por_curve(z,por,a), pordepth, porvalues, p0=-0.01)
    return por_fit

# Pore water burial mass flux
# Solids curve fit (based on porosity curve fit function)
def solid_curve(z, por, a):
    portop = np.max(por[:3,1])  # Greatest of top 3 porosity measurements for upper porosity boundary
    porbottom = np.mean(por[-3:,1])  # Takes lowest porosity measurement as the lower boundary
    return 1-((portop-porbottom) * np.exp(np.multiply(a, z)) + porbottom)

def pw_burial(seddepths, sedtimes, por_fit, por):
    # Sediment mass (1-dimensional volume of solids) accumulation rates for each age-depth section
    # Assumes constant sediment mass (really volume of solids) accumulation rates between age-depth measurements

    sectionmass = (integrate.quad(solid_curve, seddepths[0], seddepths[1], args=(por, por_fit)))[0]
    sedmassrate = (sectionmass/np.diff(sedtimes)[0])

    # Pore water burial flux calculation (ref?)
    deeppor = np.mean(por[-3:,1])  # Takes lowest porosity measurement as the lower boundary
    deepsolid = 1 - deeppor
    pwburialflux = deeppor*sedmassrate/deepsolid
    return pwburialflux

# Flux model
def flux_model(conc_fit, concunique, z, pwburialflux, porosity, Dsed, advection, dp, Site):
    # gradient = (-3*conc_interp_fit[0] + 4*conc_interp_fit[1] - conc_interp_fit[2])/(2*intervalthickness)  # Approximation according to Boudreau 1997 Diagenetic Models and Their Implementation. Matches well with derivative method
    gradient = conc_fit[1] * conc_fit[0] * np.exp(np.multiply(conc_fit[0], z))
    # gradient = (concunique[0, 1] - concunique[dp-1, 1]) * -a * np.exp(-a * z)  # Derivative of conc_curve @ z
    burial_flux = pwburialflux * conc_curve(z, *conc_fit)
    flux = porosity * Dsed * -gradient + (porosity * advection + pwburialflux) * conc_curve(z, *conc_fit)
    print('Site:', Site, 'Flux (mol/m^2 y^-1):', flux)
    return flux, burial_flux, gradient

def monte_carlo(cycles, Precision, concunique, bottom_temp_est, dp, por, por_fit, seddepths, sedtimes, TempD, bottom_temp, z, advection, Leg, Site, Solute_db, Ds, por_error, conc_fit, runtime_errors):
    # Porosity offsets - using full gaussian probability
    por_offsets = np.random.normal(scale=por_error, size=(cycles, len(por[:,1])))

    # Get randomized porosity matrix (within realistic ranges between 30% and 90%)
    por_rand = np.add(por_curve(por[:,0], por, *por_fit), por_offsets)
    por_rand[por_rand > 0.90] = 0.90
    por_rand[por_rand < 0.30] = 0.30

    portop = np.max(por[:3,1])
    portop_rand = np.add(portop, por_offsets[:,0])
    portop_rand = portop_rand[portop_rand > por_curve(por[-1,0], por, *por_fit)]
    portop_rand = portop_rand[portop_rand < 0.90]

    cycles = len(portop_rand)
    por_rand = por_rand[:cycles,:]

    # Concentration offsets - using full gaussian probability
    relativeerror = Precision
    conc_offsets = np.random.normal(scale=relativeerror, size=(cycles, len(concunique[:dp,1])))

    # Bottom water temperature offsets errors calculated in bottom_temp_error_estimate.py
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
    # Concentration offsets - truncate error at 2-sigma of gaussian distribution
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
    ###############################################################################
    # Calculate fluxes for random profiles

    # Get randomized concentration matrix (within realistic ranges)
    conc_rand = np.add(concunique[:dp,1], np.multiply(conc_offsets, concunique[:dp,1]))
    conc_rand[conc_rand < 0] = 0

    # Calculate flux
    conc_fits = np.empty((cycles, len(conc_fit)))
    por_fits = np.empty(cycles)
    pwburialfluxes = np.empty(cycles)
    conc_rand_n = np.stack((concunique[:dp,0], np.empty(dp)), axis=1)
    por_rand_n = np.stack((por[:,0], np.empty(len(por))), axis=1)
    for n in range(cycles):
        # Fit exponential curve to each randomized concentration profile
        conc_rand_n[:,1] = conc_rand[n,:]
        try:
            conc_fit = concentration_fit(conc_rand_n, dp)
        except RuntimeError:
            runtime_errors += 1
        conc_fits[n,:] = conc_fit

        # Fit exponential curve to each randomized porosity profile
        por_rand_n[:,1] =  por_rand[n,:]
        por_fit = porosity_fit(por_rand_n)
        por_fits[n] = por_fit

        # Pore water burial mass flux
        # Sediment mass (1-dimensional volume of solids) accumulation rates for each age-depth section
        # Assumes constant sediment mass (really volume of solids) accumulation rates between age-depth measurements
        pwburialflux = pw_burial(seddepths, sedtimes, por_fit, por)
        pwburialfluxes[n] = pwburialflux

    # Dsed vector
    tortuosity_rand = 1-np.log(portop_rand**2)
    bottom_temp_rand = bottom_temp+temp_offsets
    bottom_temp_rand[bottom_temp_rand < -2] = -2

    Dsed_rand = d_stp(TempD, bottom_temp_rand, Ds)/tortuosity_rand


    # Plot all the monte carlo runs
    # conc_interp_fit_plot = conc_curve(np.linspace(concunique[0,0], concunique[dp-1,0], num=50), conc_fits)
    # por_interp_fit_plot = conc_curve(np.linspace(concunique[0,0], concunique[dp-1,0], num=50), conc_fits)

    # Calculate fluxes
    gradient = conc_fits[:,1] * conc_fits[:,0] * np.exp(np.multiply(conc_fits[:,0], z))
    interface_fluxes = portop_rand * Dsed_rand * -gradient + (portop_rand * advection + pwburialfluxes) * conc_curve(z, conc_fits[:,0], conc_fits[:,1], conc_fits[:,2])

    ###############################################################################
    # Distribution statistics

    # Stats on normal distribution
    mean_flux = np.mean(interface_fluxes)
    median_flux = np.median(interface_fluxes)
    stdev_flux = np.std(interface_fluxes)
    skewness = stats.skew(abs(interface_fluxes))
    z_score, p_value = stats.skewtest(abs(interface_fluxes))

    # Stats on lognormal distribution
    interface_fluxes_log = -np.log(abs(interface_fluxes))
    mean_flux_log = np.mean(interface_fluxes_log)
    median_flux_log = np.median(interface_fluxes_log)
    stdev_flux_log = np.std(interface_fluxes_log)
    skewness_log = stats.skew(interface_fluxes_log)
    z_score_log, p_value_log = stats.skewtest(interface_fluxes_log)
    stdev_flux_lower = np.exp(-(median_flux_log-stdev_flux_log))
    stdev_flux_upper = np.exp(-(median_flux_log+stdev_flux_log))
    return interface_fluxes, interface_fluxes_log, cycles, por_error, mean_flux, median_flux, stdev_flux, skewness, z_score, mean_flux_log, median_flux_log, stdev_flux_log, stdev_flux_lower, stdev_flux_upper, skewness_log, z_score_log, runtime_errors

# Plotting
def flux_plots(concunique, conc_interp_fit_plot, por, por_all, porfit, bottom_temp, picks, sedtimes, seddepths, Leg, Site, Solute_db, flux, dp, temp_gradient):
    # Set up axes and subplot grid
    figure_1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6), facecolor='none')
    grid = gridspec.GridSpec(3, 6, wspace=0.7)
    ax1 = plt.subplot(grid[0:3, :2])
    ax1.grid()
    ax2 = plt.subplot(grid[0:3, 2:4], sharey=ax1)
    ax2.grid()
    ax3 = plt.subplot(grid[0:3, 4:6], sharey=ax1)
    ax3.grid()
    # ax4 = plt.subplot(grid[0:3, 6:8], sharey=ax1)
    # ax4.grid()

    # Figure title
    # figure_1.suptitle(r"$Expedition\ {},\ Site\ {}\ \ \ \ \ \ \ \ \ \ \ \ \ \ {}\ flux={}\ mol/m^2y$".format(Leg, Site, Solute_db, round(flux,4)), fontsize=20)

    # Plot input data
    ax1.plot(concunique[:,1], concunique[:,0], 'go')
    ax1.plot(concunique[0:dp,1],
             concunique[0:dp,0],
             'bo', label="Used for curve fit")
    ax1.plot(conc_interp_fit_plot,
             np.linspace(concunique[0,0],
                         concunique[dp-1,0],num=50),
                         'k-', label="Curve fit", linewidth=2)
    ax2.plot(por_all[:, 1], por_all[:, 0], 'mo', label='Measured')
    ax2.plot(por_curve(por[:,0],
                       por, *porfit),
                       por[:,0],
                       'k-', label='Curve fit', linewidth=3)
    ax3.plot(picks[:,1]/1000000, picks[:,0], 'ro', label='Biostrat picks')
    ax3.plot(sedtimes/1000000, seddepths, 'k-', label='Curve fit', linewidth=2)
    # ax4.plot(bottom_temp,0, 'ko', markersize=15)
    # ax4.plot(sedtemp(np.arange(concunique[-1,0]), bottom_temp, temp_gradient), np.arange(concunique[-1,0]), 'k-', linewidth=3)

    # Inset in concentration plot
    y2 = np.ceil(concunique[dp-1,0])
    x2 = max(concunique[:dp,1])+2
    x1 = min(concunique[:dp,1])-2
    axins1 = inset_axes(ax1, width="50%", height="30%", loc=5)
    axins1.plot(concunique[:,1], concunique[:,0], 'go')
    axins1.plot(concunique[0:dp,1], concunique[0:dp,0], 'bo', label="Used for curve fit")
    axins1.plot(conc_interp_fit_plot, np.linspace(concunique[0,0], concunique[dp-1,0], num=50), 'k-')
    axins1.set_xlim(x1-1, x2+1)
    axins1.set_ylim(0, y2)
    mark_inset(ax1, axins1, loc1=1, loc2=2, fc="none", ec="0.5")

    # Additional formatting
    ax1.legend(loc='best', fontsize='small')
    ax2.legend(loc='best', fontsize='small')
    ax3.legend(loc='best', fontsize='small')
    ax1.set_ylabel('Depth (mbsf)', fontsize=18)
    ax1.set_xlabel('Concentration (mM)', fontsize=18)
    ax2.set_xlabel('Porosity', fontsize=18)
    ax3.set_xlabel('Age (Ma)', fontsize=18)
    # ax4.set_xlabel('Temperature (\u00b0C)')
    ax1.locator_params(axis='x', nbins=4)
    ax2.locator_params(axis='x', nbins=4)
    ax3.locator_params(axis='x', nbins=4)
    # ax4.locator_params(axis='x', nbins=4)
    axins1.locator_params(axis='x', nbins=3)
    ax1.invert_yaxis()
    axins1.invert_yaxis()
    return figure_1


def monte_carlo_plot(interface_fluxes, median_flux, stdev_flux, skewness, z_score, interface_fluxes_log, median_flux_log, stdev_flux_log, skewness_log, z_score_log):
    mc_figure, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5),gridspec_kw={'wspace':0.2,
                                                        'top':0.92,
                                                        'bottom':0.13,
                                                        'left':0.1,
                                                        'right':0.90})

    # Plot histogram of results
    n_1, bins_1, patches_1 = ax5.hist(interface_fluxes*1000, normed=1, bins=30, facecolor='orange')

    # Best fit normal distribution line to results
    bf_line_1 = mlab.normpdf(bins_1, median_flux*1000, stdev_flux*1000)
    ax5.plot(bins_1, bf_line_1, 'k--', linewidth=2)
    ax5.set_xlabel("$Interface\ flux\ (mmol\ m^{-2}\ y^{-1})$", fontsize=20)

    [left_raw, right_raw] = ax5.get_xlim()
    [bottom_raw, top_raw] = ax5.get_ylim()
    ax5.text((left_raw+(right_raw-left_raw)/20), (top_raw-(top_raw-bottom_raw)/20), '$sk\ =\ {}$'.format(np.round(skewness, 2)))
    ax5.text((left_raw+(right_raw-left_raw)/20), (top_raw-(top_raw-bottom_raw)/10), "$z'\ =\ {}$".format(np.round(z_score, 2)))

    # Plot histogram of ln(results)
    n_2, bins_2, patches_2 = ax6.hist(interface_fluxes_log, normed=1, bins=30, facecolor='g')

    # Best fit normal distribution line to ln(results)
    bf_line_2 = mlab.normpdf(bins_2, median_flux_log, stdev_flux_log)
    ax6.plot(bins_2, bf_line_2, 'k--', linewidth=2)
    ax6.set_xlabel("$ln(abs(Interface flux)$", fontsize=20)

    [left_log, right_log] = ax6.get_xlim()
    [bottom_log, top_log] = ax6.get_ylim()
    ax6.text((left_log+(right_log-left_log)/20), (top_log-(top_log-bottom_log)/20), '$sk\ =\ {}$'.format(np.round(skewness_log, 2)))
    ax6.text((left_log+(right_log-left_log)/20), (top_log-(top_log-bottom_log)/10), "$z'\ =\ {}$".format(np.round(z_score_log, 2)))
    return mc_figure

def flux_to_sql(con, metadata_table_write, Solute_db, site_key,Leg,Site,Hole,Solute,flux,
                burial_flux,gradient,porosity,z,dp,bottom_conc,conc_fit,r_squared,
                           age_depth_boundaries,sedrate,advection,Precision,Ds,
                           TempD,bottom_temp,bottom_temp_est,cycles,
                           por_error,mean_flux,median_flux,stdev_flux,
                           skewness,z_score,mean_flux_log,median_flux_log,
                           stdev_flux_log,stdev_flux_lower,stdev_flux_upper,
                           skewness_log,z_score_log,runtime_errors,Date,Comments,Complete):
    # Send metadata to database
    sql= """insert into {} (site_key,leg,site,hole,solute,
                   interface_flux,burial_flux,gradient,top_por,flux_depth,datapoints,
                   bottom_conc,conc_fit,r_squared,age_depth_boundaries,sed_rate,advection,
                   measurement_precision,ds,ds_reference_temp,bottom_temp,
                   bottom_temp_est,mc_cycles,porosity_error,mean_flux,median_flux,
                   stdev_flux,skewness,z_score,mean_flux_log,median_flux_log,
                   stdev_flux_log,stdev_flux_lower,stdev_flux_upper,skewness_log,
                   z_score_log,runtime_errors,run_date,comments,complete)
                   VALUES ({}, '{}', '{}', '{}', '{}', {}, {}, {}, {}, {}, {},
                   {}, '{}', {},'{}', {}, {}, {}, {}, {}, {}, '{}', {}, {}, {}, {},
                   {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},'{}','{}', '{}')
                   ON DUPLICATE KEY UPDATE hole='{}', solute='{}', interface_flux={},
                   burial_flux={}, gradient={}, top_por={}, flux_depth={},
                   datapoints={}, bottom_conc={}, conc_fit='{}', r_squared={},
                   age_depth_boundaries='{}', sed_rate={}, advection={},
                   measurement_precision={}, ds={}, ds_reference_temp={},
                   bottom_temp={}, bottom_temp_est='{}', mc_cycles={},
                   porosity_error={}, mean_flux={}, median_flux={},
                   stdev_flux={}, skewness={}, z_score={}, mean_flux_log={},
                   median_flux_log={}, stdev_flux_log={}, stdev_flux_lower={},
                   stdev_flux_upper={}, skewness_log={}, z_score_log={},runtime_errors={},
                   run_date='{}', comments='{}', complete='{}'
                   ;""".format(metadata_table_write, site_key,Leg,Site,Hole,Solute,
                               flux,burial_flux,gradient,porosity,
                               z,dp,bottom_conc,conc_fit,r_squared,
                               age_depth_boundaries,sedrate,advection,
                               Precision,Ds,TempD,
                               bottom_temp,bottom_temp_est,cycles,
                               por_error,mean_flux,median_flux,stdev_flux,
                               skewness,z_score,mean_flux_log,median_flux_log,
                               stdev_flux_log,stdev_flux_lower,stdev_flux_upper,
                               skewness_log,z_score_log,runtime_errors,Date,Comments,Complete,
                               Hole,Solute,
                               flux,burial_flux,gradient,porosity,
                               z,dp,bottom_conc,conc_fit,r_squared,
                               age_depth_boundaries,sedrate,advection,
                               Precision,Ds,TempD,
                               bottom_temp,bottom_temp_est,cycles,
                               por_error,mean_flux,median_flux,stdev_flux,
                               skewness,z_score,mean_flux_log,median_flux_log,
                               stdev_flux_log,stdev_flux_lower,stdev_flux_upper,
                               skewness_log,z_score_log,runtime_errors,Date,Comments,Complete)
    con.execute(sql)


def flux_only_to_sql(con, metadata_table_write, Solute_db, site_key,Leg,Site,Hole,Solute,flux,
                burial_flux,gradient,porosity,z,dp,bottom_conc,conc_fit,r_squared,
                           age_depth_boundaries,sedrate,advection,Precision,Ds,
                           TempD,bottom_temp,bottom_temp_est,Date,Comments,Complete):
    # Send metadata to database
    sql= """insert into {} (site_key,leg,site,hole,solute,
                   interface_flux,burial_flux,gradient,top_por,flux_depth,datapoints,
                   bottom_conc,conc_fit,r_squared,age_depth_boundaries,sed_rate,advection,
                   measurement_precision,ds,ds_reference_temp,bottom_temp,
                   bottom_temp_est,run_date,comments,complete)
                   VALUES ({}, '{}', '{}', '{}', '{}', {}, {}, {}, {}, {}, {},
                   {}, '{}', {},'{}', {}, {}, {}, {}, {}, {}, '{}','{}','{}','{}')
                   ON DUPLICATE KEY UPDATE hole='{}', solute='{}', interface_flux={},
                   burial_flux={}, gradient={}, top_por={}, flux_depth={},
                   datapoints={}, bottom_conc={}, conc_fit='{}', r_squared={},
                   age_depth_boundaries='{}', sed_rate={}, advection={},
                   measurement_precision={}, ds={}, ds_reference_temp={},
                   bottom_temp={}, bottom_temp_est='{}',
                   run_date='{}', comments='{}', complete='{}'
                   ;""".format(metadata_table_write, site_key,Leg,Site,Hole,Solute,
                               flux,burial_flux,gradient,porosity,
                               z,dp,bottom_conc,conc_fit,r_squared,
                               age_depth_boundaries,sedrate,advection,
                               Precision,Ds,TempD,bottom_temp,bottom_temp_est,
                               Date,Comments,Complete,
                               Hole,Solute,flux,burial_flux,gradient,porosity,
                               z,dp,bottom_conc,conc_fit,r_squared,
                               age_depth_boundaries,sedrate,advection,
                               Precision,Ds,TempD,bottom_temp,bottom_temp_est,
                               Date,Comments,Complete)
    con.execute(sql)


# eof