def FnMastEffects(u1, u2, d1, xlab, ylab):
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


    V_ratio = u1/u2
    V_ratio[(V_ratio > 1.5) | (V_ratio < 0)] = np.nan
    V_ratio[~np.isfinite(V_ratio)] = np.nan
    mask = pd.array(np.isnan(V_ratio))
    d1[mask] = np.nan

    flag = np.nan
    verbose = 0
    d1 = FnWsRange(d1, 0, 360, flag, verbose)

# compute binned mean and std dev 
    from scipy import stats
    bin_means, bin_edges, binnumber = stats.binned_statistic(d1[np.isfinite(d1)], V_ratio[np.isfinite(d1)],statistic='mean', bins=361, range=(0,360))
    bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2
    ci = np.nanstd(V_ratio)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.scatterplot(x=d1, y=V_ratio, color='seagreen', marker='.', ci=25, label='$u_1/u_2$')
    sns.lineplot(x=bin_centers, y=bin_means, color='darkgreen', lw=3, label='mean')
    ax = plt.gca()
    ax.fill_between(bin_centers, bin_means+ci, bin_means-ci, interpolate=True, color='mediumseagreen', alpha =0.6, label='$ci=\pm\sigma$')
    plt.xlabel(xlab, fontsize=18)
    plt.ylabel(ylab, fontsize=18)
    plt.axis(ymin=0.5,ymax=1.5,xmin=0, xmax=360)
    plt.xticks(np.linspace(0,360, 13))
    plt.legend(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
# ax.xaxis.set_tick_params(rotation=25)
    plt.tight_layout()
    plt.show()
    return bin_means, bin_centers, ci, V_ratio, fig, ax
