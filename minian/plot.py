import numpy as np
import pylab as pl
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import sem


def plot_overlaps(overlaps, subset, kind="box"):
    """[summary]

    Args:
        overlaps ([type]): [description]
        subset ([type]): [description]
        kind (str, optional): [description]. Defaults to 'box'.
    """
    # overlaps['group'].replace(
    #    {
    #        'shock': 'neutral',
    #        'non-shock': 'valence'
    #    }, inplace=True)
    overlap_sub = overlaps.loc[
        overlaps["mappings"].apply(lambda i: set(i) <= subset), :
    ]
    if kind == "box":
        overlap_sub.boxplot(column=["freq"], by=["mappings", "on", "group"], rot=50)
        gcount = 1
        for gname, group in overlap_sub.groupby(["mappings", "on", "group"]):
            y = np.array(group["freq"])
            x = np.random.normal(gcount, 0.04, len(y))
            n = np.array(group["animal"])
            color = (
                group["group"]
                .replace({"neutral": "orange", "negative": "blue"})
                .iloc[0]
            )
            plt.plot(x, y, marker="o", linestyle="None", mec="k", ms=9, mfc=color)
            for iname, name in enumerate(n):
                plt.annotate(
                    name, xy=(x[iname], y[iname]), xytext=(x[iname] + 0.1, y[iname])
                )
            gcount += 1
    elif kind == "bar":
        ovlp_grouped = overlap_sub.groupby(["mappings", "on", "group"])
        overlap_mean = ovlp_grouped.apply(np.mean).unstack("group").reset_index()
        overlap_std = ovlp_grouped.aggregate(sem).unstack("group").reset_index()
        overlap_mean.set_index(["mappings", "on"]).plot(
            kind="bar", yerr=overlap_std.set_index(["mappings", "on"])
        )


def plot_overlaps_temporal(
    overlaps, subset, suppress_same_session=True, ax=None, kind="box"
):
    """[summary]

    Args:
        overlaps ([type]): [description]
        subset ([type]): [description]
        suppress_same_session (bool, optional): [description]. Defaults to True.
        ax ([type], optional): [description]. Defaults to None.
        kind (str, optional): [description]. Defaults to 'box'.
    """
    overlap_sub = overlaps.loc[
        overlaps["inter"].apply(lambda x: set([i[0] for i in x]) <= set(subset)), :
    ]
    if suppress_same_session:
        overlap_sub = overlap_sub.loc[
            overlap_sub["inter"].apply(lambda x: x[0][0] != x[1][0]), :
        ]
    if kind == "box":
        overlap_sub.boxplot(column=["freq"], by=["inter", "on", "group"], rot=90)
        gcount = 1
        for gname, group in overlap_sub.groupby(["inter", "on", "group"]):
            y = np.array(group["freq"])
            x = np.random.normal(gcount, 0.04, len(y))
            n = np.array(group["animal"])
            color = (
                group["group"]
                .replace({"neutral": "orange", "negative": "blue"})
                .iloc[0]
            )
            plt.plot(x, y, marker="o", linestyle="None", mec="k", ms=9, mfc=color)
            for iname, name in enumerate(n):
                plt.annotate(
                    name, xy=(x[iname], y[iname]), xytext=(x[iname] + 0.1, y[iname])
                )
            gcount += 1
    elif kind == "bar":
        ovlp_grouped = overlap_sub.groupby(["mappings", "on", "group"])
        overlap_mean = ovlp_grouped.apply(np.mean).unstack("group").reset_index()
        overlap_std = ovlp_grouped.aggregate(sem).unstack("group").reset_index()
        overlap_mean.set_index(["mappings", "on"]).plot(
            kind="bar", yerr=overlap_std.set_index(["mappings", "on"])
        )
    elif kind == "line":
        for cur_grp, grp in overlap_sub.groupby("group", sort=False):
            overlap_grp = grp.groupby("inter", sort=False)["freq"]
            mean = overlap_grp.apply(np.mean)
            std = overlap_grp.aggregate(sem)
            cur_ax = mean.plot(yerr=std, rot=90, ax=ax, label=cur_grp)
            cur_ax.set_xticks(np.arange(len(mean)))
            cur_ax.set_title("inter = " + str(subset))
            cur_ax.legend()


def plot_active_cells(active):
    """[summary]

    Args:
        active ([type]): [description]

    Returns:
        [type]: [description]
    """
    fg = sns.factorplot(
        data=active,
        x="segment_id",
        y="active",
        hue="group",
        col="session_id",
        sharey=True,
        order=["first", "second", "third", "fourth", "last"],
    )
    return fg


def plot_spatial(alist, idlist=None, dims=None, ax=None, cmaplist=None):
    """[summary]

    Args:
        alist ([type]): [description]
        idlist ([type], optional): [description]. Defaults to None.
        dims ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.
        cmaplist ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if not idlist:
        idlist = []
        for cur_a in alist:
            idlist.append(np.arange(cur_a.shape[-1]))
    if not cmaplist:
        cmaplist = []
        cmaplist = ["gray"] * len(alist)
    if not ax:
        ax = pl.gca()
        ax.set_facecolor("white")
    for ida, a in enumerate(alist):
        if np.ndim(a) < 3:
            a = a.reshape(np.append(dims, [-1]))
        for idx in idlist[ida]:
            ax.imshow(
                np.ma.masked_equal(a[:, :, idx], 0), alpha=0.5, cmap=cmaplist[ida]
            )
    return ax


def plot_dist_vs_corr(
    dist_centroids, dist_mask, correlations, savepath=None, suffix=""
):
    """[summary]

    Args:
        dist_centroids ([type]): [description]
        dist_mask ([type]): [description]
        correlations ([type]): [description]
        savepath ([type], optional): [description]. Defaults to None.
        suffix (str, optional): [description]. Defaults to ''.
    """
    fig = pl.figure()
    ax = fig.add_subplot(111)
    y = dist_centroids.flatten()
    x = correlations[:, :, 0].flatten()
    mask = dist_mask.flatten()
    outpoints = ax.scatter(x=x[~mask], y=y[~mask], c="b")
    inpoints = ax.scatter(x=x[mask], y=y[mask], c="r")
    ax.set_xlabel("correlation")
    ax.set_ylabel("distance of centroids")
    ax.legend((inpoints, outpoints), ("masked", "residule"))
    if savepath:
        fig.savefig(savepath + "dist_vs_corr" + suffix + ".svg")


def plot_dist_hist(dist_centroids, cut_range=(1, 20, 1), restrict_min=True):
    """[summary]

    Args:
        dist_centroids ([type]): [description]
        cut_range (tuple, optional): [description]. Defaults to (1, 20, 1).
        restrict_min (bool, optional): [description]. Defaults to True.
    """
    cuts = range(cut_range[0], cut_range[1], cut_range[2])
    ncut = len(cuts)
    fig = pl.figure(figsize=(6 * 5.5, ncut * 5.5))
    for id_cut, cut in enumerate(cuts):
        dist_filtered = np.ones_like(dist_centroids)
        if restrict_min:
            dist_min0 = np.tile(
                np.min(dist_centroids, axis=1), (dist_centroids.shape[1], 1)
            ).transpose()
            dist_min0 = dist_centroids == dist_min0
            dist_min1 = np.tile(
                np.min(dist_centroids, axis=0), (dist_centroids.shape[0], 1)
            )
            dist_min1 = dist_centroids == dist_min1
            dist_filtered = np.logical_and(dist_min0, dist_min1)
        dist_filtered = np.logical_and(dist_filtered, dist_centroids < cut)
        nmatch0 = np.sum(dist_filtered, axis=1)
        nmatch1 = np.sum(dist_filtered, axis=0)
        plt1 = fig.add_subplot(ncut, 2, id_cut * 2 + 1)
        plt2 = fig.add_subplot(ncut, 2, id_cut * 2 + 2)
        plt1.hist(nmatch0, bins=30, range=(0, 30))
        plt2.hist(nmatch1, bins=30, range=(0, 30))
        plt1.set_title("matches for first spatial matrix. dist_cutoff: " + str(cut))
        plt2.set_title("matches for second spatial matrix. dist_cutoff: " + str(cut))
        plt1.set_xlabel("number of matches")
        plt2.set_xlabel("number of matches")
    fig.savefig("/home/phild/dist_hist_min.svg", bboxinches="tight", dpi=300)


def plot_components(a, c, dims, savepath=""):
    """[summary]

    Args:
        a ([type]): [description]
        c ([type]): [description]
        dims ([type]): [description]
        savepath (str, optional): [description]. Defaults to ''.
    """
    try:
        a = a.reshape(np.append(dims, -1), order="F")
    except NotImplementedError:
        a = a.toarray().reshape(np.append(dims, -1), order="F")
    if savepath:
        pl.ioff()
    for cmp_id, temp_sig in enumerate(c):
        fig = pl.figure()
        ax_a = fig.add_subplot(211)
        ax_c = fig.add_subplot(212)
        ax_a.imshow(a[:, :, cmp_id])
        ax_c.plot(temp_sig)
        fig.suptitle("component " + str(cmp_id))
        if savepath:
            fig.savefig(savepath + "component_" + str(cmp_id) + ".svg")
            print("saving component " + str(cmp_id))
    pl.ion()


def plot_temporal_components(temp_comp, ax=None):
    """[summary]

    Args:
        temp_comp ([type]): [description]
        ax ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    temp_squeezed = temp_comp.dropna("mapping_id", how="all").dropna("frame", how="all")
    temp_squeezed.coords["mapping_id"] = np.arange(
        len(temp_squeezed.coords["mapping_id"])
    )
    if ax is not None:
        return temp_squeezed.plot(ax=ax)
    else:
        return temp_squeezed.plot()
