import cv2
import xarray as xr
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import uniform_filter
from skimage.morphology import disk


def remove_background(varr, method, wnd):
    selem = disk(wnd)
    res = xr.apply_ufunc(
        remove_background_perframe,
        varr.chunk(dict(height=-1, width=-1)),
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[varr.dtype],
        kwargs=dict(method=method, wnd=wnd, selem=selem),
    )
    res = res.astype(varr.dtype)
    return res.rename(varr.name + "_subtracted")


def remove_background_perframe(fm, method, wnd, selem):
    if method == "uniform":
        return fm - uniform_filter(fm, wnd)
    elif method == "tophat":
        return cv2.morphologyEx(fm, cv2.MORPH_TOPHAT, selem)


def stripe_correction(varr, reduce_dim="height", on="mean"):
    if on == "mean":
        temp = varr.mean(dim="frame")
    elif on == "max":
        temp = varr.max(dim="frame")
    elif on == "perframe":
        temp = varr
    else:
        raise NotImplementedError("on {} not understood".format(on))
    mean1d = temp.mean(dim=reduce_dim)
    varr_sc = varr - mean1d
    return varr_sc.rename(varr.name + "_Stripe_Corrected")


def denoise(varr, method, **kwargs):
    if method == "gaussian":
        func = cv2.GaussianBlur
    elif method == "anisotropic":
        func = anisotropic_diffusion
    elif method == "median":
        func = cv2.medianBlur
    elif method == "bilateral":
        func = cv2.bilateralFilter
    else:
        raise NotImplementedError("denoise method {} not understood".format(method))
    res = xr.apply_ufunc(
        func,
        varr,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[varr.dtype],
        kwargs=kwargs,
    )
    res = res.astype(varr.dtype)
    return res.rename(varr.name + "_denoised")
