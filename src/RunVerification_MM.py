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
from odc_exportData import odc_exportData
from FnImportOneDas import FnImportOneDas
from FnImportFiles import FnImportFiles
import matlab2py as m2p
from pythonAssist import *
from FnWsRange import *
from FnTurbLengthScale import FnTurbLengthScale

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

# %%  import channels and projects from OneDAS
from FnGetChannelNames import FnGetChannelNames
prj_url = f"https://onedas.iwes.fraunhofer.de/api/v1/projects/%2FAIRPORT%2FAD8_PROTOTYPE%2FMETMAST_EXTENSION/channels"
channelNames, prj_paths, channel_paths, units = FnGetChannelNames(prj_url)

## Initializations
saveFig=0
tstart = DT.datetime.strptime('2020-01-01_00-00-00', '%Y-%m-%d_%H-%M-%S') # Select start date in the form yyyy-mm-dd_HH-MM-SS
tend = DT.datetime.strptime('2020-12-31_00-00-00', '%Y-%m-%d_%H-%M-%S') # Select start date in the form yyyy-mm-dd_HH-MM-SS
sampleRate = [600]
device = ['Ammonit']
target_folder = r"../data"
target_folder = ""
# searchStr = ".*M0.*(V|D)\d" 
statStr = 's_mean' 

# %% import Lidar data
try:
    searchStr

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
        for i in range(len(device)):
            ch_names, paths = FnFilterChannels(prj_url, sampleRate[i], device[i], searchStr)
            
            for chunks in list(chunked(ch_names,len(ch_names))):
                odcData, pdData, t = FnImportOneDas(tstart, tend, paths, chunks, sampleRate[i], target_folder)
                DF=pd.concat([DF,pdData],axis=1)
            
            print('[{0}] device {1} complete'.format(now(), device[i]))
        # export dump file for variables as *.pickle
        import pickle
        with open(r'../data/runVerification_MM.pickle', 'wb') as f:
            pickle.dump(DF, f)
except NameError:
    load_workspace('../data/VerificationData', globals())
    print('Loading database from data folder')
except:
    channel_paths = [
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
        '/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_115m_Wind_Direction/600 s',
        '/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_115m_Wind_Speed/600 s',
        '/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_115m_CNR/600 s',
        '/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_115m_Data_Availability/600 s',
        '/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_115m_Wind_Direction/600 s',
        '/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_115m_Wind_Speed/600 s',
        '/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_115m_CNR/600 s',
        '/AIRPORT/AD8_PROTOTYPE/WIND_CUBE/WC_115m_Data_Availability/600 s',
    	]

    ch_names = [
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
        'wc_d115m',
        'wc_V115m',
        'wc_cnr115m',
        'wc_Av115m',
        'wc_d115m',
        'wc_V115m',
        'wc_cnr115m',
        'wc_Av115m',
    	]
    odcData, pdData, t = FnImportOneDas(tstart, tend, channel_paths, ch_names, sampleRate[0], target_folder)
    save_workspace('../data/VerificationData', dir(), globals())
    print('[{}]: Variables loaded from OneDAS'.format(now()))

# %% Apply Weibull to the wind time series
from FnWeibull import FnWeibull
from FnWsRange import FnWsRange

u1 = pdData.mm_V1
u2 = pdData.mm_V2
d1 = pdData.mm_D1
Weibull = 0
flag = np.nan
verbose=0

if Weibull== 1:

    WsMin = 1
    WsMax = 25
    flag = np.nan
    verbose=0
    u1 = FnWsRange(u1.values, WsMin, WsMax, flag, verbose)
    u2 = FnWsRange(u2.values, WsMin, WsMax, flag, verbose)

    fig,ax=plt.subplots(figsize=(10,4))
    ax.scatter(t, u1, marker='.', s=8, color='lightcoral',  label='V1')
    # ax.scatter(t, u2, marker='.', s=8, color='seagreen',  label='V2')
    plt.axis((np.min(t), np.max(t), 0, 25))
    plt.legend()
    plt.xlabel('Time [10-min]')
    plt.ylabel('Wind speed [m/s]')
    plt.show()

    # import numpy as np

    nbins = 25
    x_range = (0,25)
    u1 = u1[np.isfinite(u1)]
    k1, A1, bin_center1, bins1, vals1 = FnWeibull(u1, nbins, x_range)

    nbins = 25
    x_range = (0,25)
    u2 = u2[np.isfinite(u2)]
    k2, A2, bin_center2, bins2, vals2 = FnWeibull(u2, nbins, x_range)

    import matlab2py as m2p
    m2p.printMatrixE(np.array([bin_center1, vals1]).transpose())

    WriteExcel = 1
    if WriteExcel == 1:
        import openpyxl
        df = pd.DataFrame(np.transpose([bin_center1, vals1, bin_center2, vals2]), columns = ['V1_bin_center', 'V1_pdf','V2_bin_center', 'V2_pdf'] )
        df.to_excel('Weibull_statistics.xlsx', sheet_name='Weibull_2019')

#%% scatter plot for Linear regression
xx = u1
yy = u2
Vratio = xx/yy
sectors=[[112,142], [287,317]]

from FnMastEffects_linreg import FnMastEffects_linreg
FnMastEffects_linreg(d1, xx, yy, sectors=sectors, xstr='u1', ystr='u2')

#%% effect of mast
from FnMastEffects import  FnMastEffects
xlab= 'wind direction$_{110 m}$ [°]'
ylab = '$u_{116 m,1} / u_{116 m,2}$ [-]'
bin_means, bin_centers, ci, V_ratio, fig, ax = FnMastEffects(u1, u2, d1, xlab, ylab)

# %% Combine metmast sensors into one channel to correct for metmast blockage effects
u_116m = (u1 + u2)/2
u_116m[(d1>=60) & (d1<=180)] = u2
u_116m[(d1>=240) & (d1<=360)] = u1

fig, ax = plt.subplots(3, 1, figsize=(10, 7))
sns.scatterplot(ax = ax[0], x=pdData.index, y=u_116m,  color='seagreen', marker='.',ci=None, label='corrected')
sns.scatterplot(ax = ax[1], x=pdData.index, y=u1,  color='blue', marker='.', label='V1')
sns.scatterplot(ax = ax[2], x=pdData.index, y=u2,  color='gray', marker='.', label='V2')
ax = fig.axes
plt.xlabel('Timestamp [10-min]',fontsize=Large)
ax[0].set(ylabel='$u_{116m}$ [m/s]')
ax[1].set(ylabel='$u_{116m,1}$ [m/s]')
ax[2].set(ylabel='$u_{116m,2}$ [m/s]')
# plt.axis(ymin=0.5,ymax=1.5,xmin=0, xmax=360)
# plt.legend(fontsize=Small)
# plt.yticks(fontsize=Small)
# plt.xticks(fontsize=Small)
# ax.xaxis.set_tick_params(rotation=25)
plt.tight_layout()
plt.show()

# verify the correction
xlab= 'wind direction$_{110 m}$ [°]'
ylab = '$u_{116 m} / u_{116 m,2}$ [-]'
bin_means, bin_centers, ci, V_ratio, fig, ax = FnMastEffects(u_116m, u2, d1, xlab, ylab)

# %% Comparison of 115m WindCube Data with corrected metmast data at 116m
# filtering sectors
basic = ((330,40),)
sectors=((112,142), (287,317))
default_sectors =((360,0),)

# filtering conditions
cond2 = (pdData.wc_cnr115m >-17) & (pdData.wc_cnr115m<10) & (pdData.wc_cnr115m!=0) # cnr
cond3 = (pdData.wc_Av115m > 95) # availability
cond4 = ((Vratio < 1.3) & (Vratio > 0.7)) # met mast blockage effects
cond5 = (pdData.mm_D1 > basic[0][1]) & (pdData.mm_D1 < basic[0][0])   # sector filtering
cond55 = (pdData.wc_d115m > basic[0][1]) & (pdData.wc_d115m < basic[0][0])

# inputs
xx = pdData.wc_V115m[(cond2 & cond3 & cond4 & cond5)]
wcd1 = pdData.wc_d115m[cond2 & cond3 & cond4 & cond5]
d1 = pdData.mm_D1[cond2 & cond3 & cond4 & cond5]

# linear regression between cup V1 and WindCube at 115m
yy = u1[cond2 & cond3 & cond4]
R_sq_wc_u1, m_wc_u1, c_wc_u1 = FnMastEffects_linreg(d1, xx, yy, sectors=sectors, xstr='$u_{115m,wc}$', ystr='$u_{116m,1}$')

# linear regression between cup V2 and WindCube at 115m
yy = u2[cond2 & cond3 & cond4]
R_sq_wc_u2, m_wc_u2, c_wc_u2 = FnMastEffects_linreg(d1, xx, yy, sectors=sectors, xstr='$u_{115m,wc}$', ystr='$u_{116m,2}$')

# linear regression between averaged cup and WindCube at 115m
yy = u_116m[cond2 & cond3 & cond4 & cond5]
Vratio = yy/xx
R_sq_wc_u, m_wc_u, c_wc_u = FnMastEffects_linreg(d1, xx, yy, sectors=basic, xstr='$u_{115m,wc}$', ystr='$u_{116m,avg}$')

# linear regression between the wind vane and WindCube at 115m
from FnLinReg_wnddir import FnLinReg_wnddir
xx = pdData.mm_D1
yy = pdData.wc_d115m
R_sq, m, c = FnLinReg_wnddir(xx, yy, xstr='$Dir_{115m,mm}$', ystr='$Dir_{115m,wc}$')
R_sq_wc_u, m_wc_u, c_wc_u = FnMastEffects_linreg(d1, xx, yy, xx_range=(0,360), yy_range=(0, 360), 
                                sectors = default_sectors, xstr='$Dir_{115m,mm}$', ystr='$Dir_{115m,wc}$')

#%% comparison of wind speeds compared to the wind direction
d1 = pdData.wc_d115m
xx = pdData.wc_V115m
yy = u_116m
xlab= 'WindCube wind direction$_{116 m}$ [°]'
ylab = '$u_{116 m,avg} / u_{115 m,wc}$ [-]'
bin_means, bin_centers, ci, V_ratio, fig, ax = FnMastEffects(yy, xx, d1, xlab, ylab)

#% comparison of WindCube and MM wind speeds wrt wind direction at 115m after filtering
d1 = pdData.wc_d115m[cond2 & cond3 & cond5 & cond55]
xx = pdData.wc_V115m[cond2 & cond3 & cond5 & cond55]
yy = u_116m[cond2 & cond3 &cond5 & cond55]
xlab= 'WindCube wind direction$_{116 m}$ [°]'
ylab = '$u_{116 m,avg} / u_{115 m,wc}$ [-]'
bin_means, bin_centers, ci, V_ratio, fig, ax = FnMastEffects(yy, xx, d1, xlab, ylab)

#%% Comparison of wind speeds at 85m
xx = pdData.mm_V3
yy = pdData.mm_V3_2
d1 = pdData.mm_D1

# initial linear regression 
R_sq, m, c = FnMastEffects_linreg(d1, xx, yy, xx_range=(2,25), yy_range=(2, 25), 
                                sectors = default_sectors, xstr='$u_{85m,mm_1}$', ystr='$u_{85m,mm_2}$')

# metmast effects
xlab= 'wind direction$_{110 m}$ [°]'
ylab = '$u_{85 m,1} / u_{85 m,2}$ [-]'
bin_means, bin_centers, ci, V_ratio, fig, ax = FnMastEffects(xx, yy, d1, xlab, ylab)

# second linear regression after filtering the metmast blockage effects
# linear regression between cup V1 and WindCube at 115m
from FnMastEffects_linreg import FnMastEffects_linreg
yy = yy
R_sq_wc_u1, m_wc_u1, c_wc_u1, x, y, d = FnMastEffects_linreg(d1, xx, yy, sectors=sectors, xstr='$u_{85m,mm_1}$', ystr='$u_{85m,mm_2}$')


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
