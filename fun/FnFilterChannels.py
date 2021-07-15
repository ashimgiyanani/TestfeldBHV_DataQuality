def FnFilterChannels(prj_url, sampleRate, device, searchStr):
    import re
    import sys
    sys.path.append(r"../../userModules")
    sys.path.append(r"../../OneDasExplorer/Python Connector")
    from FnGetChannelNames import FnGetChannelNames

    # prj_url = f"https://onedas.iwes.fraunhofer.de/api/v1/projects/%2FAIRPORT%2FAD8_PROTOTYPE%2FMETMAST_EXTENSION/channels"
    channelNames, prj_paths, channel_paths, units = FnGetChannelNames(prj_url)

    # assigning default variables
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

    # initialisations
    filtpaths = []
    newnames = []
    ch_names = []
    paths = []
    all_ch_names = []
    all_paths = []

    # create a regex compile pattern
    r = re.compile(searchStr, re.I)
    # matching channel paths
    filtpaths = list(filter(r.match, channel_paths))
    # index for matching pattern
    match = [re.search(r,x) for x in channel_paths]
    idx = np.where(np.array(match) != None)[0]
    
    newpaths = []
    [newpaths.append(x) for x in filtpaths if x not in newpaths]		
    filtnames = list(filter(r.match, channelNames))
    newnames = []
    [newnames.append(x) for x in filtnames if x not in newnames]		
    # short names for channels
    ch_names = [x.replace(device[0], dev) for x in newnames]

    ch_units =[]
    statStr = []
    for i in range(len(idx)):
        ch_units.append(units[idx[i]])
        statStr.append(' s_mean_polar' if ch_units[i] == 'deg' else ' s_mean')
        paths.append(newpaths[i] + '/{0}'.format(sampleRate) + statStr[i])

    all_ch_names += ch_names
    all_paths += paths
    return ch_names,paths
