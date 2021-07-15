# %% user modules
import pandas as pd
import sys,  os, glob
sys.path.append(r"c:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\testfeld-bhv\userModules")

import numpy as np
import runpy as rp
import matplotlib.pyplot as plt
import tikzplotlib as tz
import re
import seaborn as sns
import datetime as DT
from datetime import datetime

sys.path.append("c:/Users/giyash/OneDrive - Fraunhofer/Python/Scripts/testfeld-bhv/OneDasExplorer/Python Connector")
from odc_exportData import odc_exportData
from FnImportOneDas import FnImportOneDas
from FnImportFiles import FnImportFiles
import matlab2py as m2p
from pythonAssist import *
from FnWsRange import *
from FnTurbLengthScale import FnTurbLengthScale

#%% user definitions
Small = 14
Large = 18
Huge = 22


# %%  import channels and projects from OneDAS
from FnGetChannelNames import FnGetChannelNames
prj_url = f"https://onedas.iwes.fraunhofer.de/api/v1/projects/%2FAIRPORT%2FAD8_PROTOTYPE%2FMETMAST_EXTENSION/channels"
channelNames, prj_paths, channel_paths = FnGetChannelNames(prj_url)

## Initializations
saveFig=0
tstart = DT.datetime.strptime('2021-01-14_00-00-00', '%Y-%m-%d_%H-%M-%S') # Select start date in the form yyyy-mm-dd_HH-MM-SS
tend = DT.datetime.strptime('2021-01-31_00-00-00', '%Y-%m-%d_%H-%M-%S') # Select start date in the form yyyy-mm-dd_HH-MM-SS
sampleRate = [600]
device = ['BluePO']
dev = ['bpo']
target_folder = r"c:/Users/giyash/OneDrive - Fraunhofer/Python/Scripts/OneDasExplorer/Python Connector/data"
target_folder = ""
searchStr = ".*{0}.*450m_HWS.*".format(device)

# %% import Lidar data

import pickle
from more_itertools import chunked

def FnFilterChannels(channelNames, channel_paths, sampleRate, device, searchStr):
	import re

	if device == 'GreenPO':
		dev='gpo'
	elif device == 'BluePO':
		dev = 'bpo'
	elif device == 'BlackTC':
		dev = 'btc'
	elif device == 'iSpin':
		dev = 'ss'
	else:
		dev = 'mm'

	filtpaths = []
	newnames = []
	ch_names = []
	paths = []
	all_ch_names = []
	all_paths = []

	r = re.compile(searchStr, re.I)
	filtpaths = list(filter(r.match, channel_paths))
	newpaths = []
	[newpaths.append(x) for x in filtpaths if x not in newpaths]		
	filtnames = list(filter(r.match, channelNames))
	newnames = []
	[newnames.append(x) for x in filtnames if x not in newnames]		
	ch_names = [x.replace(device, dev) for x in newnames]
	paths =  [ x + '/{0} s'.format(sampleRate) for x in newpaths]
	all_ch_names += ch_names
	all_paths += paths
	return ch_names,paths

try:
    with open('runVerification_NL_MM.pickle', 'rb') as f:
        DF = pickle.load(f)
        print('[{0}] File (*.pickle) loaded into DF'.format(now()))
except FileNotFoundError:
	print('[{0}] File (*.pickle) not found'.format(now()))
	DF = pd.DataFrame()
	for i in range(len(device)):
		ch_names, paths = FnFilterChannels(channelNames, channel_paths, sampleRate[i], device[i], searchStr)
		
		for chunks in list(chunked(ch_names,len(ch_names))):
			odcData, pdData, t = FnImportOneDas(tstart, tend, paths, chunks, sampleRate[i], target_folder)
			DF=pd.concat([DF,pdData],axis=1)
		
		print('[{0}] device {1} complete'.format(now(), device[i]))
	# export dump file for variables as *.pickle
	import pickle
	with open('runVerification_NL_MM.pickle', 'wb') as f:
		pickle.dump(DF, f)

# %%