# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:05:55 2017

@author: rickdberg
"""






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


Solute = 'B'
con = "mysql://root:neogene227@localhost/iodp_compiled"
sql = "SELECT sample_depth, {} FROM iw_all;".format(Solute)

concentrations = pd.read_sql(sql, con)
concentrations = concentrations.loc[pd.notnull(concentrations).all(axis=1),:]
#concentrations = concentrations.iloc[0:100,:]

plt.plot(concentrations['B'], concentrations['sample_depth'],'o')

plt.legend(loc='lower right', fontsize='small')
plt.ylabel('Depth (mbsf)')
plt.xlabel('Concentration (uM)')
plt.locator_params(axis='x', nbins=3)
plt.gca().invert_yaxis()


