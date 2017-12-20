# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:42:04 2017

@author: rickdberg

Data explorer

['etopo1_depth', 'surface_porosity', 'sed_thickness_combined',
  'crustal_age','coast_distance', 'ridge_distance', 'seamount',
  'surface_productivity','toc_seiter', 'opal', 'caco3',
  'sed_rate_burwicz', 'woa_temp', 'woa_salinity', 'woa_o2',
  'caco3_archer','acc_rate_archer','toc_combined','toc_wood'
  'sed_rate_combined','lithology','lith1','lith2','lith3','lith4','lith5',
  'lith6','lith7','lith8','lith9','lith10','lith11','lith12',
  'lith13']

"""
import numpy as np
import matplotlib.pyplot as plt
from site_metadata_compiler_completed import comp

database = "mysql://root:neogene227@localhost/iodp_compiled"
metadata = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"

# Load site data
data = comp(database, metadata, site_info, hole_info)
data = data[data['alpha'].notnull()]
epsilon = (data['alpha'].astype(float) - 1) * 1000

# Geographic patterns
size = abs(data['alpha'].astype(float)*100)
color = data['alpha'].astype(float)

plt.scatter(data['lon'], data['lat'], cmap='seismic',
            c=color, s=size)
plt.clim(vmin=min(color), vmax = -min(color))
plt.xlim((-180,180))
plt.ylim((-90,90))
plt.colorbar(shrink=0.5)
plt.show()

# Direct comparison
x = epsilon
y = data['burial_flux'].astype(float)/data['interface_flux'].astype(float)
size = data['toc_wood'].astype(float) * 100 + 100
color = data['lithology'].astype(float)
plt.scatter(x, y, s= size, c= color)
plt.show()

# Histogram
d = data['alpha'].astype(float)*1
h = d.hist(bins=100)
h.set_ylabel('$Count$', fontsize=20)
h.set_xlabel('$Mg\ Flux\ (mmol\ m^{-2}\ y^{-1})$', fontsize=20)
plt.show()

#data = data.loc[data['datapoints'].astype(float) > 2]

median = np.nanmedian(data['interface_flux'].astype(float))
median_pos = np.nanmedian(data.loc[data['interface_flux'].astype(float) > 0,'interface_flux'].astype(float))
median_neg = np.nanmedian(data.loc[data['interface_flux'].astype(float) < 0,'interface_flux'].astype(float))
print('Median:', median)
print('Median positive:', median_pos)
print('Median negative:', median_neg)
num_pos = np.count_nonzero(data.loc[data['interface_flux'].astype(float) > 0,'interface_flux'])
num_neg = np.count_nonzero(data.loc[data['interface_flux'].astype(float) < 0,'interface_flux'])
print('Count positive:', num_pos)
print('Count negative:', num_neg)


median_stdev = np.nanmedian(data['stdev_flux'].astype(float)/abs(data['interface_flux'].astype(float))*100)
print('Median stdev (%):', median_stdev)

# eof
