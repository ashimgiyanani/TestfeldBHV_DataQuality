def FnMastEffects_linreg(d1, xx, yy, **kwargs):
# FnMastEffects_linreg - Functin defintiion to perform linear regression between two variables xx and yy with d1 as the 
#                           reference direction on a metmast

# Syntax:  R_sq, m, c = FnMastEffects_linreg(d1, v1, v2,sectors=((360,0),) xstr='str', ystr='str')
#
# Inputs:
#    d1 - Reference wind direction, relative to North [°]
#    xx - Variable on the x-axis for linear regression, mostly wind speed [m/s]
#    yy - Variable on the y-axis for linear regression, mostly wind speed [m/s]
#   xx_range - min. and max. range of xx values for filtering
#   yy_range - min. and max. range of yy values for filtering
#   flag - replace value for filtering purposes
#   verbose - verbose output after filtering, 0/1 boolean, provides no. of filtered values
#   xstr, ystr - xlabel and ylabel for the plots
#   sectors - array of form ((112,142), (287,317)) or ((112,142), ) used for filtering the sectors out of wind direction
# 
# Outputs:
#    R_sq - Pearson's R^2 coeffficient
#    m - slope of the linear regression
#    c -  bias in the relationship
#
# Example:
# import numpy as np
# import pandas as pd
# d1 = pd.Series(180*np.random.randn(1000), name='d1')
# v1 = pd.Series(8 * np.random.randn(1000), name='v1')
# v2 = pd.Series(8 * np.random.randn(1000), name='v2')
# R_sq, m, c = FnMastEffects_linreg(d1, v1, v2, xstr='$u_{115m,wc}$', ystr='$u_{116m,1}$')
# 
# modules required: none
# classes required: none
# Data-files required: none
#
# See also: OTHER_FUNCTION_NAME1,  OTHER_FUNCTION_NAME2
#
# References:
# Author name, year, Title, Link
# Website, Title, link, 
#
# Author: Ashim Giyanani, Research Associate 
# Fraunhofer Institute of Wind Energy 
# Windpark planning and operation department
# Am Seedeich 45, Bremerhaven 
# email: ashim.giyanani@iwes.fraunhofer.de
# Git site: https://gitlab.cc-asp.fraunhofer.de/giyash/testfeld-bhv.git  
# Created: 06-08-2020; Last revision: 12-May-200406-08-2020

#------------- BEGIN CODE --------------


    from seaborn.palettes import set_color_codes
    import numpy as np
    import sys
    sys.path.append(r"../../userModules")
    from FnWsRange import FnWsRange
    import matplotlib.pyplot as plt

    xx_range = kwargs.pop('xx_range', (4,25))
    yy_range = kwargs.pop('yy_range', (4,25))
    xstr = kwargs.pop('xstr', '${}$ [m/s]'.format(str(xx.name)))
    ystr = kwargs.pop('ystr', '${}$ [m/s]'.format(str(yy.name)))
    flag = kwargs.pop('flag', np.nan)
    verbose = kwargs.pop('verbose', 0)
    sectors = kwargs.pop('sectors', ((360,0),))

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

    # filtering wind speed range
    xx = FnWsRange(xx, xx_range[0], xx_range[1], flag, verbose)
    yy = FnWsRange(yy, yy_range[0], yy_range[1], flag, verbose)

    # filtering for finite values
    cond0 = (np.isfinite(xx) & np.isfinite(yy) & np.isfinite(d1))

    # filtering for metmast blockage, sectors from wind speed ratio plots
    cond_mmb = np.zeros(d1.shape, dtype=bool)
    for i in range(len(sectors)):
        print('{} sector filtering used'.format(sectors))
        cond_mmb = np.logical_or(cond_mmb, ((d1 > sectors[i][0]) & (d1 < sectors[i][1]))) 


    #  filtering the time series
    xx = xx[cond0 & np.logical_not(cond_mmb)]
    yy = yy[cond0 & np.logical_not(cond_mmb)]
    d1 = d1[cond0 & np.logical_not(cond_mmb)]
    
    #  getting least square regression coefficients
    from sklearn.linear_model import LinearRegression
    Yarr = np.asarray(yy).reshape(-1,1)
    Xarr = np.asarray(xx).reshape(-1,1)
    model= LinearRegression().fit(Xarr,Yarr)
    R_sq = model.score(Xarr,Yarr)
    m = model.coef_
    c = model.intercept_ 
    print(R_sq, m, c)

    from scipy import stats
    rho_P, p_P = stats.pearsonr(xx,yy)
    rho_S, p_S = stats.spearmanr(xx,yy)

## seaborn regression plots
    import seaborn as sns
    sns.regplot(x=xx,y=yy,color='darkgreen',marker='.',ci=75, scatter_kws={'s':8, 'color':'seagreen'})
    plt.text(0.6*np.max(xx), 0.4*np.max(yy),'R^2={:.4f} (N={})'.format(R_sq, len(xx)), fontsize=Small)
    plt.text(0.6*np.max(xx), 0.3*np.max(yy),'y={:.4f}x + ({:.4f})'.format(np.float64(m), np.float64(c)), fontsize=Small)
#plt.plot(np.sin(np.deg2rad(odcData.nwd1)), np.sin(np.deg2rad(odcData.nwd2)),'k.')
    plt.xlabel(xstr, fontsize=14)
    plt.ylabel(ystr, fontsize=14)
    plt.legend(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
# ax.xaxis.set_tick_params(rotation=25)
    plt.tight_layout()
    plt.show()
    return R_sq, m, c, xx, yy, d1

# Example:
# import numpy as np
# import pandas as pd
# d1 = pd.Series(180*np.random.randn(1000), name='d1')
# v1 = pd.Series(8 * np.random.randn(1000), name='v1')
# v2 = pd.Series(8 * np.random.randn(1000), name='v2')
# R_sq, m, c = FnMastEffects_linreg(d1, v1, v2, xstr='$u_{115m,wc}$', ystr='$u_{116m,1}$')