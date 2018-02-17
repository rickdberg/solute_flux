# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:19:14 2018

Author: Rick Berg, University of Washington School of Oceanography, Seattle WA

Initiation script for to load user-specific settings for the solute_flux
package.

Variables:
engine:         sqlalchemy formatted connection to database
conctable:      name of MySQL solute concentration table
portable:       name of MySQL porosity (MAD) table
metadata_table: name of MySQL metadata table for saving results
site_info:      name of MySQL site information table
hole_info:      name of MySQL hole information table
age_table:      name of MySQL age-depth table
por_cut_table:  name of porosity cutoff depth table
flux_fig_path:  Optional folder path to save flux input data figures
mc_fig_path:    Optional folder path to save monte carlo distribution figure
mc_text_path:   Optional folder path to save monte carlo run results matrix


"""
from sqlalchemy import create_engine

# Connect to database
engine = create_engine("mysql://root:neogene227@localhost/iodp_compiled")
conctable = "iw_all"
portable = "mad_all"
metadata_table = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"
age_table = "age_depth"
por_cut_table = "porosity_cutoff"

# Optional: Folder paths for saving flux input data figure, and monte carlo
# distribution figure and matrix.
# Example for Windows OS: r"C:\Users\root\flux_fig_output\\"
# If not saving to any folder, leave as ""
flux_fig_path = r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output flux figures\\"
mc_fig_path = r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output monte carlo distributions\\"
mc_text_path = r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\Output monte carlo distributions\\"




