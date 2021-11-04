#%% Script to correct the metmast signals for metmast shadow effects

# Steps to be followed in detail:
# Import Lidar data and metmast data for one height into Pandas DataFrame [#### 80%] -> 115m, 85m, 55m*, 25m*
# plot weibull and windrose 
# plot mast effects at the height [#### 80%]
# combine and correct the mast effects [#### 80%]
# verify the correction with metmast [#### 70%]
# verify the correction with windcube [#### 70%]


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
inp.flag = np.nan
inp.verbose=0

u1 = df.mm_V1
u2 = df.mm_V2
d1 = df.mm_D1
t = t

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

#%% scatter plot for Linear regression
inp.h115m = struct()
# out.h115m = struct()

out.h115m.Vratio =  u1/u2
inp.h115m.sectors= [[112,142], [287,317]]

from FnMastEffects_linreg import FnMastEffects_linreg
FnMastEffects_linreg(d1, u1, u2, sectors=inp.h115m.sectors, xstr='$u1$', ystr='$u2$')

#%% effect of mast
from FnMastEffects import  FnMastEffects
xlab= 'wind direction$_{110 m}$ [°]'
ylab = '$u_{115 m,1} / u_{115 m,2}$ [-]'
bin_means, bin_centers, ci, V_ratio, fig, ax = FnMastEffects(u1, u2, d1, xlab, ylab)

# %% Combine metmast sensors into one channel to correct for metmast blockage effects
out.h115m.u = (u1 + u2)/2
out.h115m.u[(d1>=60) & (d1<=180)] = u2
out.h115m.u[(d1>=240) & (d1<=360)] = u1

fig, ax = plt.subplots(3, 1, figsize=(10, 7))
sns.scatterplot(ax = ax[0], x=df.index, y=out.h115m.u,  color='seagreen', marker='.',ci=None, label='corrected')
sns.scatterplot(ax = ax[1], x=df.index, y=u1,  color='blue', marker='.', label='V1')
sns.scatterplot(ax = ax[2], x=df.index, y=u2,  color='gray', marker='.', label='V2')
ax = fig.axes
plt.xlabel('Timestamp [10-min]',fontsize=inp.Large)
ax[0].set(ylabel='$u_{115m}$ [m/s]')
ax[1].set(ylabel='$u_{115m,1}$ [m/s]')
ax[2].set(ylabel='$u_{115m,2}$ [m/s]')
# plt.axis(ymin=0.5,ymax=1.5,xmin=0, xmax=360)
# plt.legend(fontsize=Small)
# plt.yticks(fontsize=Small)
# plt.xticks(fontsize=Small)
# ax.xaxis.set_tick_params(rotation=25)
plt.tight_layout()
plt.show()

# verify the correction
xlab= 'wind direction$_{110 m}$ [°]'
ylab = '$u_{115 m} / u_{115 m,2}$ [-]'
bin_means, bin_centers, ci, V_ratio, fig, ax = FnMastEffects(out.h115m.u, u2, d1, xlab, ylab)

# %% Comparison of 115m WindCube Data with corrected metmast data at 115m
# filtering sectors
inp.h115m.sectors_basic = ((330,40),)
inp.h115m.sectors=((112,142), (287,317))
inp.sectors_range =((360,0),)
inp.cnr_range = [-24, 0]
inp.Av_threshold = 50
inp.h115m.Vratio_range = [0.8, 1.2]

# filtering conditions
cond2 = (df.wc_cnr115m > inp.cnr_range[0]) & (df.wc_cnr115m<inp.cnr_range[1]) & (df.wc_cnr115m!=0) # cnr
cond3 = (df.wc_Av115m > inp.Av_threshold) # availability
cond4 = ((out.h115m.Vratio > inp.h115m.Vratio_range[0]) & (out.h115m.Vratio < inp.h115m.Vratio_range[1]))  # met mast blockage effects
cond5 = (df.mm_D1 > inp.h115m.sectors_basic[0][1]) & (df.mm_D1 < inp.h115m.sectors_basic[0][0])   # sector filtering
cond55 = (df.wc_d115m > inp.h115m.sectors_basic[0][1]) & (df.wc_d115m < inp.h115m.sectors_basic[0][0])

# inputs
xx = df.wc_V115m[(cond2 & cond3 & cond4 & cond5)]
wcd1 = df.wc_d115m[cond2 & cond3 & cond4 & cond5]
d1 = df.mm_D1[cond2 & cond3 & cond4 & cond5]

# linear regression between cup V1 and WindCube at 115m
yy = u1[cond2 & cond3 & cond4]
out.h115m.R_sq_wc_u1, out.h115m.m_wc_u1, out.h115m.c_wc_u1, _, _, _ = FnMastEffects_linreg(d1, xx, yy, sectors=inp.h115m.sectors, xstr='$u_{115m,wc}$', ystr='$u_{115m,1}$')

# linear regression between cup V2 and WindCube at 115m
yy = u2[cond2 & cond3 & cond4]
out.h115m.R_sq_wc_u2, out.h115m.m_wc_u2, out.h115m.c_wc_u2, _, _, _ = FnMastEffects_linreg(d1, xx, yy, sectors=inp.h115m.sectors, xstr='$u_{115m,wc}$', ystr='$u_{115m,2}$')

# linear regression between averaged cup and WindCube at 115m
yy = out.h115m.u[cond2 & cond3 & cond4 & cond5]
Vratio = yy/xx
out.h115m.R_sq_wc_u, out.h115m.m_wc_u, out.h115m.c_wc_u, _, _, _ = FnMastEffects_linreg(d1, xx, yy, sectors=inp.h115m.sectors_basic, xstr='$u_{115m,wc}$', ystr='$u_{115m,avg}$')

# linear regression between the wind vane and WindCube at 115m
from FnLinReg_wnddir import FnLinReg_wnddir
xx = df.mm_D1
yy = df.wc_d115m
R_sq, m, c = FnLinReg_wnddir(xx, yy, xstr='$Dir_{115m,mm}$', ystr='$Dir_{115m,wc}$')
R_sq_wc_u, m_wc_u, c_wc_u, _, _, _ = FnMastEffects_linreg(d1, xx, yy, xx_range=inp.wdir_range, yy_range=inp.wdir_range, 
                                sectors = inp.sectors_range, xstr='$Dir_{115m,mm}$', ystr='$Dir_{115m,wc}$')

#%% comparison of wind speeds compared to the wind direction
d1 = df.wc_d115m
xx = df.wc_V115m
yy = out.h115m.u
xlab= 'WindCube wind direction$_{115 m}$ [°]'
ylab = '$u_{115 m,avg} / u_{115 m,wc}$ [-]'
bin_means, bin_centers, ci, V_ratio, fig, ax = FnMastEffects(yy, xx, d1, xlab, ylab)

#% comparison of WindCube and MM wind speeds wrt wind direction at 115m after filtering
Vratio = xx/yy
cond4 = ((Vratio > inp.h115m.Vratio_range[0]) & (Vratio < inp.h115m.Vratio_range[1])) # met mast blockage effects
d1 = df.mm_D1[cond2 & cond3 & cond4 & cond5 & cond55]
xx = df.wc_V115m[cond2 & cond3 & cond4 & cond5 & cond55]
yy = out.h115m.u[cond2 & cond3 & cond4 & cond5 & cond55]
xlab= 'Vane wind direction$_{115 m}$ [°]'
ylab = '$u_{115 m,avg} / u_{115 m,wc}$ [-]'
bin_means, bin_centers, ci, V_ratio, fig, ax = FnMastEffects(yy, xx, d1, xlab, ylab)

# linear regression after filtering the data
R_sq, m, c, _, _, _ = FnMastEffects_linreg(d1,xx,yy,sectors=inp.sectors_range, xstr='$u_{115m,wc}$', ystr='$u_{115m,mm}$')


#%% Comparison of wind speeds at 85m
inp.h85m = struct()
out.h85m = struct()

inp.h85m.channel_paths = [
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0070_D1/600 s_mean_polar',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0070_D4/600 s_mean_polar',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0070_D5/600 s_mean_polar',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0020_V3/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0021_V3_2/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_84m_Wind_Direction/600 s',
    '/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_84m_Wind_Speed/600 s',
    '/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_84m_CNR/600 s',
    '/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_84m_Data_Availability/600 s',
    ]

inp.h85m.ch_names = [
    'mm_D1',
    'mm_V3',
    'mm_V3_2',
    'wc_d84m',
    'wc_V84m',
    'wc_cnr84m',
    'wc_Av84m',
    ]
inp.h85m.tstart = DT.datetime.strptime('2020-11-01_00-00-00', '%Y-%m-%d_%H-%M-%S') # Select start date in the form yyyy-mm-dd_HH-MM-SS
inp.h85m.tend = DT.datetime.strptime('2021-07-17_00-00-00', '%Y-%m-%d_%H-%M-%S') # Select start date in the form yyyy-mm-dd_HH-MM-SS
inp.h85m.sampleRate = [1/600]
_, df2, t2 = FnImportOneDas(inp.h85m.tstart, inp.h85m.tend, inp.h85m.channel_paths, inp.h85m.ch_names, inp.h85m.sampleRate[0], inp.target_folder)
df = pd.concat([df, df2])

u3 = df.mm_V3
u3b = df.mm_V3_2
d1 = df.mm_D1

# comparison of wind direction between 115m and 85m
R_sq, m, c = FnLinReg_wnddir(d1, df.wc_d84m, xstr='$Dir_{115m,mm}$', ystr='$Dir_{84m,wc}$')

# initial linear regression 
R_sq, m, c, x, y, d = FnMastEffects_linreg(d1, u3, u3b, xx_range=(2,25), yy_range=(2, 25), 
                                sectors = inp.sectors_range, xstr='$u_{85m,mm_{V3}, 300^\circ}$', ystr='$u_{85m,mm_{V3_2}, 118^\circ}$')

# metmast effects
xlab= 'wind direction$_{115 m, mm}$ [°]'
ylab = '$u_{85 m,V3} / u_{85 m,V3_2}$ [-]'
bin_means, bin_centers, ci, V_ratio, fig, ax = FnMastEffects(u3, u3b, d1, xlab, ylab)

# second linear regression after filtering the metmast blockage effects
# linear regression between cup V1 and WindCube at 115m
from FnMastEffects_linreg import FnMastEffects_linreg
inp.h85m.sectors = ((110, 160), (275, 325), (350,360))
u3b = u3b
R_sq_wc_u1, m_wc_u1, c_wc_u1, x, y, d = FnMastEffects_linreg(d1, u3, u3b, sectors=inp.h85m.sectors, xstr='$u_{85m,mm_{V3}, 300^\circ}$', ystr='$u_{85m,mm_{V3_2}, 118^\circ}$')
bin_means, bin_centers, ci, V_ratio, fig, ax = FnMastEffects(x, y, d, xlab, ylab)

#%% averaged wind speed signal at 85m
out.h85m.u = (u3 + u3b)/2
out.h85m.u[(d1>=60) & (d1<=180)] = u3b
out.h85m.u[(d1>=240) & (d1<=360)] = u3

fig, ax = plt.subplots(3, 1, figsize=(10, 7))
sns.scatterplot(ax = ax[0], x=df.index, y=out.h85m.u,  color='seagreen', marker='.',ci=None, label='corrected')
sns.scatterplot(ax = ax[1], x=df.index, y=u3,  color='blue', marker='.', label='$V3$')
sns.scatterplot(ax = ax[2], x=df.index, y=u3b,  color='gray', marker='.', label='$V3_2$')
ax = fig.axes
plt.xlabel('Timestamp [10-min]',fontsize=inp.Large)
ax[0].set(ylabel='$u_{85m}$ [m/s]')
ax[1].set(ylabel='$u_{85m,V3}$ [m/s]')
ax[2].set(ylabel='$u_{85m,V3_2}$ [m/s]')
# plt.axis(ymin=0.5,ymax=1.5,xmin=0, xmax=360)
# plt.legend(fontsize=Small)
# plt.yticks(fontsize=Small)
# plt.xticks(fontsize=Small)
# ax.xaxis.set_tick_params(rotation=25)
plt.tight_layout()
plt.show()

# verify the correction
xlab= 'wind direction$_{115 m, mm}$ [°]'
ylab = '$u_{85 m, V3} / u_{84 m,wc}$ [-]'
bin_means, bin_centers, ci, V_ratio, fig, ax = FnMastEffects(out.h85m.u, df.wc_V84m, d1, xlab, ylab)

#% comparison of WindCube and MM wind speeds wrt wind direction at 85m after filtering
# filtering conditions
Vratio =  out.h85m.u /df.wc_V84m
cond2 = (df.wc_cnr84m >-24) & (df.wc_cnr84m<10) & (df.wc_cnr84m!=0) # cnr
cond3 = (df.wc_Av84m > 95) # availability
cond4 = ((Vratio < 1.2) & (Vratio > 0.8)) # met mast blockage effects
# cond5 = (df.mm_D1 > basic[0][1]) & (df.mm_D1 < basic[0][0])   # sector filtering
# cond55 = (df.wc_d84m > basic[0][1]) & (df.wc_d84m < basic[0][0])

d1 = df.mm_D1[cond2 & cond3 & cond4]
xx = df.wc_V84m[cond2 & cond3 & cond4]
yy = out.h85m.u[cond2 & cond3 & cond4]
xlab= 'wind vane wind direction$_{115 m}$ [°]'
ylab = '$u_{85 m,avg} / u_{84 m,wc}$ [-]'
bin_means, bin_centers, ci, V_ratio, fig, ax = FnMastEffects(yy, xx, d1, xlab, ylab)

R_sq_wc_u, m_wc_u, c_wc_u, _, _, _ = FnMastEffects_linreg(d1, xx, yy, sectors=inp.sectors_range, 
                                                    xstr='$u_{84m,wc}$', ystr='$u_{85m,avg}$')

#%% plot v3/v1, v4/v1, v5/v1, v6/v1 and compare the wind speeds ratios

# read in the ultrasonics data
file_55m = r"z:\Projekte\109797-TestfeldBHV\30_Technical_execution_Confidential\TP3\AP2_Aufbau_Infrastruktur\Infrastruktur_Windmessung\02_Equipment\01_Wartung_Messmast_GE-NET_DWG_20190226\Data\post_processed\gill_55m_sample_20210115_20210926.txt"
file_110m = r"z:\Projekte\109797-TestfeldBHV\30_Technical_execution_Confidential\TP3\AP2_Aufbau_Infrastruktur\Infrastruktur_Windmessung\02_Equipment\01_Wartung_Messmast_GE-NET_DWG_20190226\Data\post_processed\gill_110m_sample_20210115_20210926.txt"
df3 = pd.read_csv(file_55m)
df4 = pd.read_csv(file_110m)

# rename columns to include the height
df3.rename(columns = lambda x: x.split(' ')[0]+'_55m', inplace=True)
df4.rename(columns = lambda x: x.split(' ')[0]+'_110m', inplace= True)


tstart = DT.datetime.strptime(df3.Time_55m[0], '%d-%b-%Y %H:%M:%S')
tend = DT.datetime.strptime(df3.Time_55m.iloc[-1], '%d-%b-%Y %H:%M:%S')

channel_paths = [
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0000_V1/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0010_V2/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0020_V3/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0021_V3_2/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0030_V4/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0040_V5/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0050_V6/600 s_mean',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0070_D1/600 s_mean_polar',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0100_D4/600 s_mean_polar',
    '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0110_D5/600 s_mean_polar',
    ]

ch_names = [
    'mm_V1',
    'mm_V2',
    'mm_V3',
    'mm_V3_2',
    'mm_V4',
    'mm_V5',
    'mm_V6',
    'mm_D1',
    'mm_D4',
    'mm_D5',
    ]
sampleRate = 1/600
target_folder = r"../data/"
_, df5, t5 = FnImportOneDas(tstart, tend, channel_paths, ch_names, sampleRate, target_folder)
# concatenate the pandas dataframe
df_2021 = pd.concat([df3, df4, df5],axis=1)

frac = struct()

frac.v3v1 = df.mm_V2 / df.mm_V1
frac.v4v1 = df.mm_V4 / df.mm_V1
frac.v5v1 = df.mm_V5 / df.mm_V1
frac.v6v1 = df.mm_V6 / df.mm_V1

d1 = df.mm_D1
xx = df.mm_V3
yy = df.mm_V1
xlab= 'wind vane wind direction$_{115 m}$ [°]'
ylab = '$u_{85 m} / u_{115 m}$ [-]'
bin_means, bin_centers, ci, V_ratio, fig, ax = FnMastEffects(yy, xx, d1, xlab, ylab)

d1 = df.mm_D1
xx = df.mm_V4
yy = df.mm_V1
xlab= 'wind vane wind direction$_{115 m}$ [°]'
ylab = '$u_{55 m} / u_{115 m}$ [-]'
bin_means, bin_centers, ci, V_ratio, fig, ax = FnMastEffects(yy, xx, d1, xlab, ylab)

d1 = df.mm_D4
xx = df.mm_V5
yy = df.mm_V1
xlab= 'wind vane wind direction$_{115 m}$ [°]'
ylab = '$u_{25 m} / u_{115 m}$ [-]'
bin_means, bin_centers, ci, V_ratio, fig, ax = FnMastEffects(yy, xx, d1, xlab, ylab)

d1 = df.mm_D5
xx = df.mm_V6
yy = df.mm_V1
xlab= 'wind vane wind direction$_{115 m}$ [°]'
ylab = '$u_{10 m} / u_{115 m}$ [-]'
bin_means, bin_centers, ci, V_ratio, fig, ax = FnMastEffects(yy, xx, d1, xlab, ylab)

#%% corrections using sonic anemometers at 55m
d1 = df_2021.wind_55m
xx = df_2021.mm_V4
yy = df_2021.U_vec_55m
xlab= 'sonics wind direction$_{55 m}$ [°]'
ylab = '$u_{55 m, 4} / u_{55 m, gill}$ [-]'
bin_means, bin_centers, ci, V_ratio, fig, ax = FnMastEffects(yy, xx, d1, xlab, ylab)

sys.exit('manual stop')


#%% Writing a report template
# tab1 = pd.DataFrame({"Configuration": ['Begin Date', 'End Date', 'x', 'y', 'N_total'],
#                 "Details": [tstart, tend, xlab, ylab, len(xx)]})
# pivot = pd.pivot_table(tab1, index='Configuration', values='Details', aggfunc=np.sum, fill_value=0)

# tab2 = pd.DataFrame({
#     "Filters used": ['wind speed', 'wind direction', 'wind direction'],
#     "Range": ['{0} - {1} [m/s]'.format(2, 25), 
#                 '{0} - {1} [°]'.format(wdir_filtsectors[0][0],wdir_filtsectors[0][1]),
#                 '{} - {} [°]'.format(wdir_filtsectors[1][0],wdir_filtsectors[1][1])],
#     "N(#)": [len(xx), len(xx), len(xx)],
#     "x": ['Cup u1 (115.9 m)'],
#     "y": ['Cup u2 (115.9 m)'],
#     "m": [m_wc_u1, m_wc_u2, m_wc_u],
#     "c": [c_wc_u1, c_wc_u2, c_wc_u],
#     "R^2 Pearson": [R_sq_wc_u1, R_sq_wc_u2, R_sq_wc_u]
# })
# %%
