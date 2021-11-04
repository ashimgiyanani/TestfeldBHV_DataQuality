# Script to perform IEC based site assessment


# %% user modules
# %matplotlib qt
import pandas as pd
import sys, os, glob
sys.path.append(r"../../userModules")
sys.path.append(r"../fun")

import numpy as np
import runpy as rp
import matplotlib.pyplot as plt
import tikzplotlib as tz
import re
import seaborn as sns
import datetime as DT
from datetime import datetime

sys.path.append(r"../../OneDasExplorer/Python Connector")
sys.path.append(r"c:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\OneDasExplorer\Python Connector")
sys.path.append(r"c:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\userModules")

from odc_exportData import odc_exportData
from FnImportOneDas import FnImportOneDas
from FnImportFiles import FnImportFiles
import matlab2py as m2p
from pythonAssist import *
from FnWsRange import *
from FnTurbLengthScale import FnTurbLengthScale

#%% user definitions
inp=struct()
inp.h115m = struct()

out = struct()
out.h115m = struct()

inp.tiny = 12
inp.Small = 14
inp.Large = 18
inp.Huge = 22
plt.rc('font', size=inp.Small)          # controls default text sizes
plt.rc('axes', titlesize=inp.Small)     # fontsize of the axes title
plt.rc('axes', labelsize=inp.Large)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=inp.Small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=inp.Small)    # fontsize of the tick labels
plt.rc('legend', fontsize=inp.Small)    # legend fontsize
plt.rc('figure', titlesize=inp.Huge)  # fontsize of the figure title

# %%  import channels and projects from OneDAS
from FnGetChannelNames import FnGetChannelNames

inp.prj_url = f"https://onedas.iwes.fraunhofer.de/api/v1/projects/%2FAIRPORT%2FAD8_PROTOTYPE%2FMETMAST_EXTENSION/channels"
inp.channelNames, inp.prj_paths, inp.channel_paths, inp.units = FnGetChannelNames(inp.prj_url)

## Initializations
inp.saveFig=1
inp.h115m.tstart = DT.datetime.strptime('2020-01-01_00-00-00', '%Y-%m-%d_%H-%M-%S') # Select start date in the form yyyy-mm-dd_HH-MM-SS
inp.h115m.tend = DT.datetime.strptime('2020-12-31_00-00-00', '%Y-%m-%d_%H-%M-%S') # Select start date in the form yyyy-mm-dd_HH-MM-SS
inp.h115m.sampleRate = [1/600]
inp.device = ['Ammonit']
inp.target_folder = r"../data"
inp.target_folder = ""
# comment the following line to read the data VerificationData.dat from ../data/
# inp.searchStr = ".*M0.*(V|D)\d" 
inp.statStr = 's_mean' 

# %% import Lidar data
try:
    inp.searchStr

    import pickle
    from more_itertools import chunked
    from FnFilterChannels import FnFilterChannels
    try:
        with open(r'../data/runVerification_MM.pickle', 'rb') as f:
            DF = pickle.load(f)
            print('[{0}] File (*.pickle) loaded into DF'.format(now()))
    except FileNotFoundError:
        print('[{0}] File (*.pickle) not found'.format(now()))
        DF = pd.DataFrame()
        for i in range(len(inp.device)):
            inp.h115m.ch_names, inp.h115m.paths = FnFilterChannels(inp.prj_url, inp.h115m.sampleRate[i], inp.device[i], inp.searchStr)
            
            for chunks in list(chunked(inp.h115m.ch_names,len(inp.ch_names))):
                _, df, t = FnImportOneDas(inp.h115m.tstart, inp.h115m.tend, inp.paths, chunks, inp.h115m.sampleRate[i], inp.target_folder)
                DF=pd.concat([DF,df],axis=1)
            
            print('[{0}] device {1} complete'.format(now(), inp.device[i]))
        # export dump file for variables as *.pickle
        import pickle
        with open(r'../data/runVerification_MM.pickle', 'wb') as f:
            pickle.dump(DF, f)
except NameError:
    load_workspace('../data/VerificationData', globals())
    print('Loading database from data folder')
except:
    inp.h115m.channel_paths = [
        '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0000_V1/600 s_mean',
        '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0000_V1/600 s_std',
        '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0010_V2/600 s_mean',
        '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0070_D1/600 s_mean_polar',
        '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0020_V3/600 s_mean',
        '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0021_V3_2/600 s_mean',
        '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0030_V4/600 s_mean',
        '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0040_V5/600 s_mean',
        '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0050_V6/600 s_mean',
        '/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_115m_Wind_Direction/600 s',
        '/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_115m_Wind_Speed/600 s',
        '/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_115m_CNR/600 s',
        '/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_115m_Data_Availability/600 s',
    	]

    inp.h115m.ch_names = [
        'mm_V1',
        'mm_V1_std',
        'mm_V2',
        'mm_D1',
        'mm_V3',
        'mm_V3_2',
        'mm_V4',
        'mm_V5',
        'mm_V6',
        'wc_d115m',
        'wc_V115m',
        'wc_cnr115m',
        'wc_Av115m',
    	]
    _, df, t = FnImportOneDas(inp.h115m.tstart, inp.h115m.tend, inp.h115m.channel_paths, inp.h115m.ch_names, inp.h115m.sampleRate[0], inp.target_folder)
    save_workspace('../data/VerificationData', dir(), globals())
    print('[{}]: Variables loaded from OneDAS'.format(now()))

# %% Apply Weibull to the wind time series
from FnWeibull import FnWeibull
from FnWsRange import FnWsRange

inp.Weibull = 1
inp.windRose = 1
inp.TurbDist = 1
inp.flag = np.nan
inp.verbose=0

u1 = df.mm_V1
u2 = df.mm_V2
d1 = df.mm_D1

# filter windspeed u1, u2 for range
inp.ws_range = [1, 25]
u1 = FnWsRange(u1, inp.ws_range[0], inp.ws_range[1], inp.flag, inp.verbose)
u2 = FnWsRange(u2, inp.ws_range[0], inp.ws_range[1], inp.flag, inp.verbose)

# filter wind direction d1 for range
inp.wdir_range = [0, 360]
d1 = FnWsRange(d1, inp.wdir_range[0], inp.wdir_range[1], inp.flag, inp.verbose)

#%% lay windrose over a map
if inp.windRose == 1:
    from windrose import WindroseAxes, WindAxes
    import matplotlib.cm as cm
    uu = u1[(np.isfinite(d1) & np.isfinite(u1))]
    wdir = d1[np.isfinite(d1) & np.isfinite(u1)]
    bins = np.linspace(0,28,8)
    deg = np.linspace(0,330, 12)
    deg_str = [str(int(x))+'°' for x in deg]
    deg_rolled = np.roll(np.linspace(330, 0, 12, dtype=int),-8)
    fig = plt.figure(figsize=(8,6))
    ax=WindroseAxes.from_ax(fig=fig)
    ax.bar(wdir, uu,normed=True, opening=1, edgecolor='white', nsector=24, bins=bins)
    ax.set_thetagrids(angles=deg, labels=deg_rolled)
    ax.set_legend(loc='best')
    ax.set(title='$u_{115m,1}$ windrose over $wind direction_{110m}$')
    # tikzplotlib not working for windroses, polar plots
    # tz.clean_figure()
    # tz.save("../results/windrose_u1d1.tikz", float_format=".3f")
    plt.savefig('../results/windrose_u1d1.png', bbox_inches='tight')
    plt.savefig('../results/windrose_u1d1.pdf', bbox_inches='tight')
    plt.close()
    del uu, wdir, bins, deg, deg_str, deg_rolled, fig, ax

if inp.Weibull== 1:
    out.h115m.weib = struct()

    fig,ax=plt.subplots(figsize=(10,4))
    ax.scatter(t, u1, marker='.', s=8, color='lightcoral',  label='u1')
    # ax.scatter(t, u2, marker='.', s=8, color='seagreen',  label='V2')
    plt.axis((np.min(t), np.max(t), 0, 25))
    plt.legend()
    plt.xlabel('Time [10-min]')
    plt.ylabel('Wind speed [m/s]')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
    tz.clean_figure()
    tz.save("../results/timeseries_u1.tikz", float_format=".3f")
    plt.savefig('../results/timeseries_u1.png', bbox_inches='tight')
    plt.savefig('../results/timeseries_u1.pdf', bbox_inches='tight')
    plt.close()
    # import numpy as np

    inp.nbins = 25
    inp.x_range = (0,25)
    x = u1[np.isfinite(u1)]
    out.h115m.weib.u1_k, out.h115m.weib.u1_A, out.h115m.weib.u1_binc, out.h115m.weib.u1_bins, out.h115m.weib.u1_vals,_,_= FnWeibull(x, inp.nbins, inp.x_range)
    tz.clean_figure()
    tz.save("../results/weibull_u1.tikz", float_format=".3f")
    plt.savefig('../results/weibull_u1.png', bbox_inches='tight')
    plt.savefig('../results/weibull_u1.pdf', bbox_inches='tight')
    plt.close()
    
    x = u2[np.isfinite(u2)]
    out.h115m.weib.u2_k, out.h115m.weib.u2_A, out.h115m.weib.u2_binc, out.h115m.weib.u2_bins,out.h115m.weib.u2_vals,_,_= FnWeibull(x, inp.nbins, inp.x_range)
    tz.clean_figure()
    tz.save("../results/weibull_u2.tikz", float_format=".3f")
    plt.savefig('../results/weibull_u2.png', bbox_inches='tight')
    plt.savefig('../results/weibull_u2.pdf', bbox_inches='tight')
    plt.close()

    import matlab2py as m2p
    m2p.printMatrixE(np.array([out.h115m.weib.u1_binc, out.h115m.weib.u1_vals]).transpose())

    inp.WriteExcel = 1
    if inp.WriteExcel == 1:
        import openpyxl
        out.h115m.weib.df = pd.DataFrame(np.transpose([out.h115m.weib.u1_binc, out.h115m.weib.u1_vals, out.h115m.weib.u2_binc, out.h115m.weib.u2_vals]), columns = ['V1_bin_center', 'V1_pdf','V2_bin_center', 'V2_pdf'] )
        out.h115m.weib.df.to_excel('../data/Weibull_statistics.xlsx', sheet_name='Weibull_2019')
    del  fig, ax

if inp.TurbDist == 1:

    # calculate 10-min avg. turbulence intensities
    Iu = df.mm_V1_std / df.mm_V1
    # correct for range
    inp.Iu_range = [0, 0.4]
    Iu = FnWsRange(Iu, inp.Iu_range[0], inp.Iu_range[1], inp.flag, inp.verbose)

    def FnPdf(x, y, xrange, yrange, xlab, ylab):
        import sys
        import numpy as np
        import pandas as pd
        sys.path.append(r"../../userModules")
        from FnWsRange import FnWsRange
        import matplotlib.pyplot as plt
        import seaborn as sns

        #%% user definitions
        tiny = 12
        Small = 14
        Large = 18
        Huge = 22
        plt.rc('font', size=Small)          # controls default text sizes
        plt.rc('axes', titlesize=Small)     # fontsize of the axes title
        plt.rc('axes', labelsize=Large)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=Small)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=Small)    # fontsize of the tick labels
        plt.rc('legend', fontsize=Small)    # legend fontsize
        plt.rc('figure', titlesize=Huge)  # fontsize of the figure title

        x[~np.isfinite(x) | ~np.isfinite(y)] = np.nan
        mask = pd.array(np.isnan(x) | np.isnan(y))
        y[mask] = np.nan
        x[mask] = np.nan

        flag = np.nan
        verbose = 0
        x = FnWsRange(x, xrange[0], xrange[1], flag, verbose)
        y = FnWsRange(y, yrange[0], yrange[1], flag, verbose)

        # compute binned mean and std dev 
        from scipy import stats
        bin_means, bin_edges, binnumber = stats.binned_statistic(y[np.isfinite(y)], x[np.isfinite(y)], statistic='mean', bins=26, range=(0,30))
        bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2
        ci = np.nanstd(x)

        # plot the y signal against x with confidence intervals
        fig, ax = plt.subplots()
        sns.scatterplot(x=x, y=y, color='seagreen', marker='.', ci=25, label='Iu')
        sns.lineplot(x=bin_centers, y=bin_means, color='darkgreen', lw=3, label='mean')
        ax = plt.gca()
        # ax.fill_between(bin_centers, bin_means+ci, bin_means-ci, interpolate=True, color='mediumseagreen', alpha =0.6, label='$ci=\pm\sigma$')
        plt.xlabel(xlab, fontsize=18)
        plt.ylabel(ylab, fontsize=18)
        plt.axis(ymin=0.0,ymax=0.5,xmin=0, xmax=25)
        plt.xticks(np.linspace(xrange[0],xrange[1], 26))
        plt.legend(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        ax.xaxis.set_tick_params(rotation=25)
        plt.tight_layout()
        plt.show()
        return bin_means, bin_centers, ci, x, y, fig, ax

    bin_means, bin_centers, ci, x, y, fig, ax = FnPdf(u1[np.isfinite(u1)], Iu[np.isfinite(u1)], xrange=[0,25], yrange=[0, 0.4], xlab='u1', ylab='pdf Iu')
    tz.clean_figure()
    tz.save("../results/pdf_Iu_115m.tikz", float_format=".3f")
    plt.savefig('../results/pdf_Iu_115m.png', bbox_inches='tight')
    plt.savefig('../results/pdf_Iu_115m.pdf', bbox_inches='tight')

    fig,ax=plt.subplots(figsize=(10,4))
    ax.scatter(t, Iu, marker='.', s=8, color='lightcoral',  label='Iu')
    plt.axis((np.min(t), np.max(t), 0, 0.4))
    plt.legend()
    plt.xlabel('Time [10-min]')
    plt.ylabel('Turbulent intensity [-]')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
    tz.clean_figure()
    tz.save("../results/ts_Iu_115m.tikz", float_format=".3f")
    plt.savefig('../results/ts_Iu_115m.png', bbox_inches='tight')
    plt.savefig('../results/ts_Iu_115m.pdf', bbox_inches='tight')
    plt.close()
    print('stop')

# %%
