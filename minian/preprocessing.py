import cv2
import xarray as xr
from scipy.ndimage import uniform_filter
from skimage.morphology import disk


def remove_background(varr, method, wnd):
    """
    Remove background from a video.

    Args:
        varr (xarray.DataArray): xarray.DataArray a labeled 3-d array representation of the videos with dimensions: frame, height and width.
        method (string): ‘uniform‘ or ‘tophat’
        wnd (int): size of the disk shaped kernel to use for filtering (in pixels)

    Returns:
        xarray.DataArray: xarray.DataArray a labeled 3-d array with name <name>_substracted
    """
    selem = disk(wnd)
    res = xr.apply_ufunc(
        remove_background_perframe,
        varr,
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
    """
    Remove background per frame by applying the filtering method

    Args:
        fm (uint8[]): frame, array of unsigned int 8 (bytes)
        method (string): ‘uniform‘ or ‘tophat’
        wnd (int): size of the disk shaped kernel to use for filtering (in pixels)
        selem (uint8[]): kernel (mask) for filtering, array of unsigned int 8 (bytes)

    Returns:
        uint8[]: frame, array of unsigned int 8 (bytes)
    """
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


def gaussian_blur(varray, ksize=(3, 3), sigmaX=0):
    return varray.groupby("frame").apply(
        lambda fm: cv2.GaussianBlur(fm.values, ksize, sigmaX)
    )


def denoise(varr, method, **kwargs):
    """
    Remove noise from a video

    Args:
        varr (xarray.DataArray): xarray.DataArray a labeled 3-d array representation of the videos with dimensions: frame, height and width.
        method (string): "gaussian", "anisotropic", "median" or "bilateral"

    Raises:
        NotImplementedError: raised when the method is not one of "gaussian", "anisotropic", "median" or "bilateral"

    Returns:
        xarray.DataArray: xarray.DataArray a labeled 3-d array with name <name>_denoised
    """
    if method == "gaussian":
        func = cv2.GaussianBlur
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
