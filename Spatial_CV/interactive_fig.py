import numpy as np
import pandas as pd
from scipy import stats, odr
import geoviews as gv
import geoviews.feature as gf
from cartopy import crs as ccrs
from geoviews import dim
# gv.extension('bokeh')
# gv.output(backend='bokeh')
import os
import holoviews as hv
from holoviews import opts
from holoviews.plotting.links import DataLink


def get_output():
    output_dir = '/my-projects/Projects/MLCNN_PM25_2021/code/Cross_Validation/MultipleModel_Spatial_withAreas_Cross_Validation_BenchMark/figures/'
    return

def plot_interactive_fig():
    # Plot
    TOOLS = ["pan,wheel_zoom,box_zoom,box_select,lasso_select,hover,reset,help"]
    height = 350
    extents=(-180,-90,180,90)
    aspect = (180*2) / (90+90)
    width = int(height*aspect)

    # map
    left_background = gf.coastline.options(
    'Feature', projection=ccrs.PlateCarree(), global_extent=True, height=height, width=width)
    scatter = hv.Scatter(ds, 'lon', vdims=['lat','sim', 'obs', 'num_obs'])
    left = scatter.opts(height=height, width=width, \
    # tools=TOOLS, \
    size=np.sqrt(dim('obs'))*1, color='obs', cmap='RdYlBu_r', logz=True,\
    xlabel=r'$$\rm Longitude$$',
    ylabel=r'$$\rm Latitude$$',
#   clabel=r'$$\rm Observation\ (\mu g/m^3)$$',
    ylim=(-90,90),
    colorbar=True, clim=(5,100))

    points = [(i, slope*i + offset) for i in range(0, 280)]
    fitline = hv.Curve(points).options(color=colors[1], line_width=1)
    points = [(i, i) for i in range(0, 280)]
    line11 = hv.Curve(points).options(color='black', line_dash='dashed', line_width=1)
    if offset>0:
        stat = hv.Text(10,270,r'y = {:.2f} x + {:.2f}''\n''R = {:.2f}''\n''NMB = {:2.1%}''\n''NRMSD = {:2.1%}'.\
            format(slope, offset, r, NMB, NRMSD) \
            + '\n' + r'N' + f' = {len(x)}', halign='left', valign='top', fontsize=11)
    else:
        stat = hv.Text(10,270,r'y = {:.2f} x - {:.2f}''\n''R = {:.2f}''\n''NMB = {:2.1%}''\n''NRMSD = {:2.1%}'.\
            format(slope, -offset, r, NMB, NRMSD) \
            + '\n' + r'N' + f' = {len(x)}', halign='left', valign='top', fontsize=11)
        
    scatter = hv.Scatter(ds, 'obs', vdims=['sim','lat', 'lon', 'num_obs'])
    right = scatter.opts(height=height, width=height,
    # tools=TOOLS, \
    size=2, color=colors[0],
    xlim=(0,280), ylim=(0,280),
    xlabel=r'$$\rm Observation\ (\mu g/m^3)$$', 
    ylabel=r'$$\rm Simulation\ (\mu g/m^3)$$')
    DataLink(left, right)
    fig = (left_background*left + right*fitline*line11*stat).opts(opts.Scatter(tools=['box_select', 'lasso_select']))

    # Save as HTML
    outfpath = figDir + '{}_LUO_Sim_vs_CompiledGM_noFILL_Obs_{}_AnnMean_interactive.html'.format(cres, year)
    hv.save(fig, outfpath)