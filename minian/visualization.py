import functools as fct
import itertools as itt
import numpy as np
import holoviews as hv
import xarray as xr
import ipywidgets as iwgt
import pandas as pd
import colorsys
import param
import dask.array as da
import panel as pn
import os
import dask
import ffmpeg
import cv2
import av
from .utilities import scale_varr
from .motion_correction import match_temp
from uuid import uuid4
from collections import OrderedDict
from holoviews.streams import (
    Stream,
    Pipe,
    RangeXY,
    DoubleTap,
    Tap,
    Selection1D,
    BoxEdit,
)
from holoviews.operation import contours, threshold
from holoviews.operation.datashader import datashade, regrid, dynspread
from holoviews.util import Dynamic
from datashader.colors import Sets1to3
from datashader import count_cat
from datashader import mean as dsmean
from datashader import count as dscount
from IPython.core.display import display, clear_output
from bokeh.io import push_notebook, show
from bokeh.layouts import layout
from bokeh.plotting import figure
from bokeh.models import Slider, Range1d, LinearAxis
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.palettes import Category10_10, Viridis256
from matplotlib.colors import rgb_to_hsv
from scipy.ndimage.measurements import center_of_mass
from IPython.core.debugger import set_trace
from dask.diagnostics import ProgressBar
from skvideo.io import FFmpegWriter, vwrite
from scipy import linalg
from bokeh import models
from bokeh.io import export_svgs
from panel.interact import interact as pninteract
from panel.layout import Row, Column
from panel import widgets as pnwgt
from scipy.spatial import cKDTree

# def update_override(self, **kwargs):
#     self._set_stream_parameters(**kwargs)
#     transformed = self.transform()
#     if transformed:
#         self._set_stream_parameters(**transformed)
#     self.trigger([self])


class VArrayViewer:
    def __init__(
        self,
        varr,
        framerate=30,
        rerange=None,
        summary=["mean"],
        meta_dims=None,
        datashading=True,
        layout=False,
    ):
        if isinstance(varr, list):
            for iv, v in enumerate(varr):
                varr[iv] = v.assign_coords(data_var=v.name)
            self.ds = xr.concat(varr, dim="data_var")
            meta_dims = ["data_var"]
        elif isinstance(varr, xr.DataArray):
            self.ds = varr.to_dataset()
        elif isinstance(varr, xr.Dataset):
            self.ds = varr
        else:
            raise NotImplementedError(
                "video array of type {} not supported".format(type(varr))
            )
        try:
            self.meta_dicts = OrderedDict(
                [(d, list(self.ds.coords[d].values)) for d in meta_dims]
            )
            self.cur_metas = OrderedDict(
                [(d, v[0]) for d, v in self.meta_dicts.items()]
            )
        except TypeError:
            self.meta_dicts = dict()
            self.cur_metas = dict()
        self._datashade = datashading
        self._layout = layout
        self.framerate = framerate
        self._f = self.ds.coords["frame"].values
        self._h = self.ds.sizes["height"]
        self._w = self.ds.sizes["width"]
        self.mask = dict()
        self.rerange = rerange
        CStream = Stream.define(
            "CStream",
            f=param.Integer(
                default=int(self._f.min()), bounds=(self._f.min(), self._f.max())
            ),
        )
        self.strm_f = CStream()
        self.str_box = BoxEdit()
        self.widgets = self._widgets()
        if type(summary) is list:
            summ_all = {
                "mean": self.ds.mean(["height", "width"]),
                "max": self.ds.max(["height", "width"]),
                "min": self.ds.min(["height", "width"]),
                "diff": self.ds.diff("frame").mean(["height", "width"]),
            }
            try:
                summ = {k: summ_all[k] for k in summary}
            except KeyError:
                print("{} Not understood for specifying summary".format(summary))
            if summ:
                print("computing summary")
                sum_list = []
                for k, v in summ.items():
                    sum_list.append(v.compute().assign_coords(sum_var=k))
                summary = xr.concat(sum_list, dim="sum_var")
        self.summary = summary
        if layout:
            self.ds_sub = self.ds
            self.sum_sub = self.summary
        else:
            self.ds_sub = self.ds.sel(**self.cur_metas)
            try:
                self.sum_sub = self.summary.sel(**self.cur_metas)
            except AttributeError:
                self.sum_sub = self.summary
        self.pnplot = pn.panel(self.get_hvobj())

    def get_hvobj(self):
        def get_im_ovly(meta):
            def img(f, ds):
                return hv.Image(ds.sel(frame=f).compute(), kdims=["width", "height"])

            try:
                curds = self.ds_sub.sel(**meta).rename("_".join(meta.values()))
            except ValueError:
                curds = self.ds_sub
            fim = fct.partial(img, ds=curds)
            im = regrid(hv.DynamicMap(fim, streams=[self.strm_f])).opts(
                frame_width=500, aspect=self._w / self._h, cmap="Viridis"
            )
            if self.rerange:
                im = im.redim.range(**{vname: self.rerange})
            self.xyrange = RangeXY(source=im).rename(x_range="w", y_range="h")
            if not self._layout:
                hv_box = hv.Polygons([]).opts(
                    style={"fill_alpha": 0.3, "line_color": "white"}
                )
                self.str_box = BoxEdit(source=hv_box)
                im_ovly = im * hv_box
            else:
                im_ovly = im

            def hist(f, w, h, ds):
                if w and h:
                    cur_im = hv.Image(
                        ds.sel(frame=f).compute(), kdims=["width", "height"]
                    ).select(height=h, width=w)
                else:
                    cur_im = hv.Image(
                        ds.sel(frame=f).compute(), kdims=["width", "height"]
                    )
                return hv.operation.histogram(cur_im, num_bins=50).opts(
                    xlabel="fluorescence", ylabel="freq"
                )

            fhist = fct.partial(hist, ds=curds)
            his = hv.DynamicMap(fhist, streams=[self.strm_f, self.xyrange]).opts(
                frame_height=int(500 * self._h / self._w), width=150, cmap="Viridis"
            )
            im_ovly = (im_ovly << his).map(lambda p: p.opts(style=dict(cmap="Viridis")))
            return im_ovly

        if self._layout and self.meta_dicts:
            im_dict = OrderedDict()
            for meta in itt.product(*list(self.meta_dicts.values())):
                mdict = {k: v for k, v in zip(list(self.meta_dicts.keys()), meta)}
                im_dict[meta] = get_im_ovly(mdict)
            ims = hv.NdLayout(im_dict, kdims=list(self.meta_dicts.keys()))
        else:
            ims = get_im_ovly(self.cur_metas)
        if self.summary is not None:
            hvsum = (
                hv.Dataset(self.sum_sub)
                .to(hv.Curve, kdims=["frame"])
                .overlay("sum_var")
            )
            if self._datashade:
                hvsum = datashade_ndcurve(hvsum, kdim="sum_var")
            try:
                hvsum = hvsum.layout(list(self.meta_dicts.keys()))
            except:
                pass
            vl = hv.DynamicMap(lambda f: hv.VLine(f), streams=[self.strm_f]).opts(
                style=dict(color="red")
            )
            summ = (hvsum * vl).map(
                lambda p: p.opts(frame_width=500, aspect=3), [hv.RGB, hv.Curve]
            )
        else:
            summ = hv.Div("")
        hvobj = (ims + summ).cols(1)
        return hvobj

    def show(self):
        return pn.layout.Column(self.widgets, self.pnplot)

    def _widgets(self):
        w_play = pnwgt.Player(
            length=len(self._f), interval=10, value=0, width=650, height=90
        )

        def play(f):
            if not f.old == f.new:
                self.strm_f.event(f=int(self._f[f.new]))

        w_play.param.watch(play, "value")
        w_box = pnwgt.Button(
            name="Update Mask", button_type="primary", width=100, height=30
        )
        w_box.param.watch(self._update_box, "clicks")
        if not self._layout:
            wgt_meta = {
                d: pnwgt.Select(name=d, options=v, height=45, width=120)
                for d, v in self.meta_dicts.items()
            }

            def make_update_func(meta_name):
                def _update(x):
                    self.cur_metas[meta_name] = x.new
                    self._update_subs()

                return _update

            for d, wgt in wgt_meta.items():
                cur_update = make_update_func(d)
                wgt.param.watch(cur_update, "value")
            wgts = pn.layout.WidgetBox(w_box, w_play, *list(wgt_meta.values()))
        else:
            wgts = pn.layout.WidgetBox(w_box, w_play)
        return wgts

    def _update_subs(self):
        self.ds_sub = self.ds.sel(**self.cur_metas)
        if self.sum_sub is not None:
            self.sum_sub = self.summary.sel(**self.cur_metas)
        self.pnplot.objects[0].object = self.get_hvobj()

    def _update_box(self, click):
        box = self.str_box.data
        self.mask.update(
            {
                tuple(self.cur_metas.values()): {
                    "height": slice(box["y0"][0], box["y1"][0]),
                    "width": slice(box["x0"][0], box["x1"][0]),
                }
            }
        )


class MCViewer:
    def __init__(self, varr, marr=None, framerate=30):
        if isinstance(varr, list):
            varr = xr.merge(varr)
        self.varr = hv.Dataset(varr)
        if marr is not None:
            self.marr = hv.Dataset(marr)
        else:
            self.marr = None
        self.framerate = framerate
        CStream = Stream.define(
            "CStream", f=param.Integer(default=0, bounds=self.varr.range("frame"))
        )
        self.stream = CStream()
        self.widgets = self._widgets()

    def show(self, use_datashade=False):
        vh = self.varr.range("height")
        vw = self.varr.range("width")
        him = int(vh[1] - vh[0] + 1)
        wim = int(vw[1] - vw[0] + 1)
        if self.marr is not None:
            mh = self.marr.range("height")
            mw = self.marr.range("width")
            hma = int(mh[1] - mh[0] + 1)
            wma = int(mw[1] - mw[0] + 1)
        varrdict = OrderedDict()
        for dim in self.varr.vdims:
            cur_d = self.varr.reindex(vdims=[dim.name])
            fim = fct.partial(self._img, dat=cur_d)
            im = hv.DynamicMap(fim, streams=[self.stream])
            im = regrid(im, height=him, width=wim)
            im = im(plot={"width": wim, "height": him})
            image = im.relabel(group=dim.name, label="Image")
            varrdict[dim.name] = image
        if self.marr is not None:
            fma = fct.partial(self._rgb, dat=self.marr)
            ma = hv.DynamicMap(fma, streams=[self.stream])
            ma = regrid(ma, height=hma, width=wma)
            ma = ma.relabel(label="Match Image")
            ma = ma(plot={"height": hma, "width": wma})
        else:
            ma = hv.Empty()
        return hv.Layout(ma + hv.NdLayout(varrdict, kdims=["name"])).cols(1)

    def _widgets(self):
        dfrange = self.varr.range("frame")
        w_frame = iwgt.IntSlider(
            value=0,
            min=dfrange[0],
            max=dfrange[1],
            continuous_update=False,
            description="Frame:",
        )
        w_paly = iwgt.Play(
            value=0, min=dfrange[0], max=dfrange[1], interval=1000 / self.framerate
        )
        iwgt.jslink((w_paly, "value"), (w_frame, "value"))
        iwgt.interactive(self.stream.event, f=w_frame)
        return iwgt.HBox([w_paly, w_frame])

    def _img(self, dat, f):
        return hv.Image(dat.select(frame=f), kdims=["width", "height"])

    def _rgb(self, dat, f):
        return hv.RGB(dat.select(frame=f), kdims=["width", "height"])


class CNMFViewer_old:
    def __init__(self, cnmf, Y, framerate=30):
        self.cnmf = cnmf
        # self.cnmf_vld = cnmf.sel(unit_id=cnmf.attrs['unit_mask'])
        self.cnmf_vld = cnmf
        # self.ds = hv.Dataset(cnmf)
        # self.ds_vld = hv.Dataset(self.cnmf_vld)
        self.framerate = framerate
        self.Y = Y
        # _rh = self.ds_vld.range('height')
        # _rw = self.ds_vld.range('width')
        # self._h = int(_rh[1] - _rh[0])
        # self._w = int(_rw[1] - _rw[0])
        self._h = self.cnmf_vld.sizes["height"]
        self._w = self.cnmf_vld.sizes["width"]
        self._f = self.cnmf_vld.sizes["frame"]
        self._u = self.cnmf.sizes["unit_id"]
        self._update_mov = False
        self.strm_f = DoubleTap(x=0).rename(x="f", y=None)
        self.strm_f.add_subscriber(self._update_f)
        self.strm_uid = Selection1D()
        self.pipY = Pipe(data=[])
        self.pipAdC = Pipe(data=[])
        self.pipbdf = Pipe(data=[])
        self.pipYr = Pipe(data=[])
        self._cur_sel = (0, 5)
        self._overlay = True
        self.widgets = self._widgets()
        self.hvobjs = self.get_plot()

    def get_plot(self):
        cur_sel = (self._cur_sel[0], self._cur_sel[1])
        cur_units = self.cnmf_vld.isel(unit_id=slice(*cur_sel)).coords["unit_id"].values
        cont_dict = OrderedDict()
        A = self.cnmf_vld["A"]
        for uid in self.cnmf_vld.coords["unit_id"]:
            cur_A = A.sel(unit_id=uid).load()
            cur_thres = float(cur_A.max() * 0.3)
            cur_cent = center_of_mass(cur_A.data)
            cur_im = hv.Image(cur_A, kdims=["width", "height"])
            cur_cont = contours(cur_im, levels=[cur_thres])
            cur_cont = cur_cont(
                plot={
                    "show_legend": False,
                    "tools": ["hover", "box_select", "lasso_select", "tap"],
                },
                style={"cmap": ["white"]},
            )
            cur_text = hv.Text(cur_cent[1], cur_cent[0], str(int(uid)))
            cur_text = cur_text(style={"color": "white", "text_font_size": "8pt"})
            cont_dict[int(uid)] = cur_cont * cur_text
        ovly = hv.NdOverlay(cont_dict, kdims=["unit_id"])
        self.strm_uid.source = ovly
        fim = fct.partial(hv.Image, kdims=["width", "height"])
        img_Y = hv.DynamicMap(fim, streams=[self.pipY])
        img_Y = regrid(img_Y, height=int(self._h / 10), width=int(self._w / 10))
        img_Y = img_Y(
            plot={"height": self._h, "width": self._w, "title_format": "Y (Original)"}
        )
        img_AdC = hv.DynamicMap(fim, streams=[self.pipAdC])
        img_AdC = regrid(img_AdC, height=int(self._h / 10), width=int(self._w / 10))
        img_AdC = img_AdC(
            plot={
                "height": self._h,
                "width": self._w,
                "title_format": "A dot C (Units)",
            }
        )
        img_bdf = hv.DynamicMap(fim, streams=[self.pipbdf])
        img_bdf = regrid(img_bdf, height=int(self._h / 10), width=int(self._w / 10))
        img_bdf = img_bdf(
            plot={
                "height": self._h,
                "width": self._w,
                "title_format": "b dot f (Background)",
            }
        )
        img_Yr = hv.DynamicMap(fim, streams=[self.pipYr])
        img_Yr = regrid(img_Yr, height=int(self._h / 10), width=int(self._w / 10))
        img_Yr = img_Yr(
            plot={"height": self._h, "width": self._w, "title_format": "Yr (Residule)"}
        )
        if self._overlay:
            nunits = len(cur_units)
            cur_A = scale_varr(self.cnmf_vld["A"].sel(unit_id=cur_units))
            cur_C = scale_varr(self.cnmf_vld["C"].sel(unit_id=cur_units))
            cur_A = cur_A.load()
            cur_C = cur_C.load()
            clr_rgb = sns.color_palette("hsv", nunits)
            clr_rgb_xr = xr.DataArray(
                clr_rgb,
                coords={"unit_id": cur_A.coords["unit_id"], "cspace": ["R", "G", "B"]},
                dims=["unit_id", "cspace"],
            )
            im_rgb = (cur_A.dot(clr_rgb_xr) / cur_A.sum("unit_id")).fillna(0)
            hsv_coords = im_rgb.coords
            hsv_coords["cspace"] = ["H", "S", "V"]
            im_hsv = xr.DataArray(im_rgb.values, coords=hsv_coords, dims=im_rgb.dims)
            im_hsv.values = rgb_to_hsv(im_hsv.values)
            fim = fct.partial(self._im_overlay, A=cur_A, C=cur_C, im_hsv=im_hsv)
            imgs_pu = hv.DynamicMap(fim, streams=[self.strm_f])
            imgs_pu = regrid(imgs_pu, height=int(self._h / 10), width=int(self._w / 10))
            # imgs_pu = regrid(imgs_pu, x_sampling=4, y_sampling=4)
            imgs_pu = (
                imgs_pu(
                    plot={
                        "height": self._h,
                        "width": self._w,
                        "title_format": "overlay - A",
                    }
                )
                * ovly
            )
            c_pu = OrderedDict()
            for uid in cur_units:
                cc = hv.Curve(self.cnmf_vld.sel(unit_id=uid)["C"], kdims=["frame"])
                cc = cc(
                    plot={
                        "height": self._h,
                        "width": self._w * 2,
                        "title_format": "overlay - C",
                        "tools": [],
                    },
                    style={"color": hv.Cycle(values=clr_rgb)},
                )
                c_pu[uid] = cc
            c_pu = hv.NdOverlay(c_pu, kdims=["unit_id"])
            self.strm_f.source = c_pu
            vl = hv.DynamicMap(lambda f: hv.VLine(f), streams=[self.strm_f])
            vl = vl(style={"color": "red"})
            c_pu = c_pu * vl
        else:
            imgs_pu = OrderedDict()
            c_pu = OrderedDict()
            for uid in cur_units:
                im = hv.Image(
                    self.cnmf_vld.sel(unit_id=uid)["A"], kdims=["width", "height"]
                )
                im = regrid(im)
                im = im(plot={"height": self._h, "width": self._w})
                imgs_pu[uid] = im
                cc = hv.Curve(self.cnmf_vld.sel(unit_id=uid)["C"], kdims=["frame"])
                cs = hv.Curve(self.cnmf_vld.sel(unit_id=uid)["S"], kdims=["frame"])
                cs = cs(
                    plot={"finalize_hooks": [self._twinx], "apply_ranges": False},
                    style={"color": "red"},
                )
                vl = hv.DynamicMap(lambda f: hv.VLine(f), streams=[self.strm_f])
                vl = vl(style={"color": "red"})
                c = cc * vl
                c = c(plot={"height": self._h, "width": self._w * 2})
                c_pu[uid] = c
            imgs_pu = hv.NdLayout(imgs_pu, kdims=["unit_id"]).cols(1)
            c_pu = hv.NdLayout(c_pu, kdims=["unit_id"]).cols(1)
        hvobjs = (img_Y + img_AdC + img_bdf + img_Yr + imgs_pu + c_pu).cols(2)
        self.hvobjs = hvobjs
        return hvobjs

    def show(self):
        display(self.widgets)
        display(self.hvobjs)

    def _set_sel(self, change):
        self._cur_sel = change["new"]

    def _set_update(self, change):
        self._update_mov = change["new"]
        if change["new"]:
            self._update_f(f=self.strm_f.contents["f"])

    def _set_overlay(self, change):
        self._overlay = change["new"]

    def _update_plot(self, b):
        clear_output()
        self.get_plot()
        self.show()

    def _update_f(self, f):
        if f is not None:
            f = int(f)
            if self._update_mov:
                self.pipY.send([])
                self.pipAdC.send([])
                self.pipbdf.send([])
                self.pipYr.send([])
                cur_Y = self.Y.sel(frame=f)
                cur_AdC = self.cnmf_vld["A"].dot(self.cnmf_vld["C"].sel(frame=f))
                cur_bdf = self.cnmf_vld["b"].dot(self.cnmf_vld["f"].sel(frame=f))
                cur_Yr = cur_Y - cur_AdC - cur_bdf
                self.pipY.send(cur_Y)
                self.pipAdC.send(cur_AdC)
                self.pipbdf.send(cur_bdf)
                self.pipYr.send(cur_Yr)

    def _f_vl(self, f):
        if f is not None:
            self._update_f(f)
            return hv.VLine(f)
        else:
            return hv.VLine(0)

    def _im_overlay(self, f, A, C, im_hsv, contour=None):
        f = int(f)
        AdC = A.dot(C.sel(frame=f))
        im_hue = im_hsv.sel(cspace="H").rename("H").drop("cspace")
        im_sat = (im_hsv.sel(cspace="S")).rename("S").drop("cspace")
        im_val = (
            (im_hsv.sel(cspace="V") * AdC * 4).clip(0, 1).rename("V").drop("cspace")
        )
        ds = xr.merge([im_hue, im_sat, im_val])
        im = hv.HSV(ds, kdims=["width", "height"])
        # if contour is None:
        #     contour = hv.operation.contours(im)
        return im

    def _twinx(self, plot, element):
        # Setting the second y axis range name and range
        start, end = element.range(1)
        label = element.dimensions()[1].pprint_label
        plot.state.extra_y_ranges = {"foo": Range1d(start=start, end=end)}
        # Adding the second axis to the plot.
        linaxis = LinearAxis(axis_label=label, y_range_name="foo")
        plot.state.add_layout(linaxis, "right")

    def _widgets(self):
        dfrange = [0, self._f]
        w_frame = iwgt.IntSlider(
            value=0,
            min=dfrange[0],
            max=dfrange[1],
            continuous_update=False,
            description="Frame:",
            layout=iwgt.Layout(width="50%"),
        )
        w_play = iwgt.Play(
            value=0, min=dfrange[0], max=dfrange[1], interval=1000 / self.framerate
        )
        iwgt.jslink((w_play, "value"), (w_frame, "value"))
        iwgt.interactive(self.strm_f.event, x=w_frame)
        uidrange = [0, self._u]
        w_select = iwgt.IntRangeSlider(
            value=self._cur_sel,
            min=uidrange[0],
            max=uidrange[1],
            continuous_update=False,
            description="Unit ID:",
            layout=iwgt.Layout(width="50%"),
        )
        w_select.observe(self._set_sel, names="value")
        w_update = iwgt.Button(description="Update")
        w_update.on_click(self._update_plot)
        w_update_mov = iwgt.Checkbox(
            value=self._update_mov, description="Update Movies"
        )
        w_update_mov.observe(self._set_update, names="value")
        w_overlay = iwgt.Checkbox(value=self._overlay, description="Overlay")
        w_overlay.observe(self._set_overlay, names="value")
        return iwgt.VBox(
            [
                iwgt.HBox([w_frame, w_play, w_update_mov]),
                iwgt.HBox([w_select, w_update, w_overlay]),
            ]
        )


class CNMFViewer:
    def __init__(self, minian=None, A=None, C=None, S=None, org=None, sortNN=True):
        self._A = A if A is not None else minian["A"]
        self._C = C if C is not None else minian["C"]
        self._S = S if S is not None else minian["S"]
        self._org = org if org is not None else minian["org"]
        try:
            self.unit_labels = minian["unit_labels"].compute()
        except:
            self.unit_labels = xr.DataArray(
                self._A["unit_id"].values.copy(),
                dims=self._A["unit_id"].dims,
                coords=self._A["unit_id"].coords,
            ).rename("unit_labels")
        self._C_norm = xr.apply_ufunc(
            normalize,
            self._C.chunk(dict(frame=-1, unit_id="auto")),
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[self._C.dtype],
        )
        self._S_norm = xr.apply_ufunc(
            normalize,
            self._S.chunk(dict(frame=-1, unit_id="auto")),
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[self._C.dtype],
        )
        self.cents = centroid(self._A, verbose=True)
        print("computing sum projection")
        with ProgressBar():
            self.Asum = self._A.sum("unit_id").compute()
        self._NNsort = sortNN
        self._normalize = False
        self._useAC = True
        self._showC = True
        self._showS = True
        meta_dims = list(set(self._org.dims) - {"frame", "height", "width"})
        self.meta_dicts = {d: list(self._org.coords[d].values) for d in meta_dims}
        self.metas = {d: v[0] for d, v in self.meta_dicts.items()}
        if self._NNsort:
            try:
                self.cents["NNord"] = self.cents.groupby(
                    meta_dims, group_keys=False
                ).apply(NNsort)
            except ValueError:
                self.cents["NNord"] = NNsort(self.cents)
            NNcoords = self.cents.set_index(meta_dims + ["unit_id"])[
                "NNord"
            ].to_xarray()
            self._A = self._A.assign_coords(NNord=NNcoords)
            self._C = self._C.assign_coords(NNord=NNcoords)
            self._S = self._S.assign_coords(NNord=NNcoords)
            self._C_norm = self._C_norm.assign_coords(NNord=NNcoords)
            self._S_norm = self._S_norm.assign_coords(NNord=NNcoords)
        self.update_subs()
        self.strm_f = DoubleTap(rename=dict(x="f"))
        self.strm_f.add_subscriber(self.callback_f)
        self.strm_uid = Selection1D()
        self.strm_uid.add_subscriber(self.callback_uid)
        Stream_usub = Stream.define("Stream_usub", usub=param.List())
        self.strm_usub = Stream_usub()
        self.strm_usub.add_subscriber(self.callback_usub)
        self.usub_sel = self.strm_usub.usub
        self._AC = self._org.sel(**self.metas)
        self._mov = self._org.sel(**self.metas)
        self.pipAC = Pipe([])
        self.pipmov = Pipe([])
        self.pipusub = Pipe([])
        self.wgt_meta = self._meta_wgt()
        self.wgt_spatial_all = self._spatial_all_wgt()
        self.spatial_all = self._spatial_all()
        self.temp_comp_sub = self._temp_comp_sub(self._u[:5])
        self.wgt_man = self._man_wgt()
        self.wgt_temp_comp = self._temp_comp_wgt()

    def update_subs(self):
        self.A_sub = self._A.sel(**self.metas)
        self.C_sub = self._C.sel(**self.metas)
        self.S_sub = self._S.sel(**self.metas)
        self.org_sub = self._org.sel(**self.metas)
        self.C_norm_sub = self._C_norm.sel(**self.metas)
        self.S_norm_sub = self._S_norm.sel(**self.metas)
        if self._NNsort:
            self.A_sub = self.A_sub.sortby("NNord")
            self.C_sub = self.C_sub.sortby("NNord")
            self.S_sub = self.S_sub.sortby("NNord")
            self.C_norm_sub = self.C_norm_sub.sortby("NNord")
            self.S_norm_sub = self.S_norm_sub.sortby("NNord")
        self._h = (
            self.A_sub.isel(unit_id=0)
            .dropna("height", how="all")
            .coords["height"]
            .values
        )
        self._w = (
            self.A_sub.isel(unit_id=0).dropna("width", how="all").coords["width"].values
        )
        self._f = self.C_sub.isel(unit_id=0).dropna("frame").coords["frame"].values
        self._u = self.C_sub.isel(frame=0).dropna("unit_id").coords["unit_id"].values
        if self.meta_dicts:
            sub = pd.concat(
                [self.cents[d] == v for d, v in self.metas.items()], axis="columns"
            ).all(axis="columns")
            self.cents_sub = self.cents[sub]
        else:
            self.cents_sub = self.cents

    def compute_subs(self, clicks=None):
        self.A_sub = self.A_sub.compute()
        self.C_sub = self.C_sub.compute()
        self.S_sub = self.S_sub.compute()
        self.org_sub = self.org_sub.compute()
        self.C_norm_sub = self.C_norm_sub.compute()
        self.S_norm_sub = self.S_norm_sub.compute()

    def update_all(self, clicks=None):
        self.update_subs()
        self.strm_uid.event(index=[])
        self.strm_f.event(x=0)
        self.update_spatial_all()

    def callback_uid(self, index=None):
        self.update_temp()
        self.update_AC()
        self.update_usub_lab()

    def callback_f(self, f, y):
        if len(self._AC) > 0 and len(self._mov) > 0:
            fidx = np.abs(self._f - f).argmin()
            f = self._f[fidx]
            if self._useAC:
                AC = self._AC.sel(frame=f)
            else:
                AC = self._AC
            mov = self._mov.sel(frame=f)
            self.pipAC.send(AC)
            self.pipmov.send(mov)
            try:
                self.wgt_temp_comp[1].value = int(fidx)
            except AttributeError:
                pass
        else:
            self.pipAC.send([])
            self.pipmov.send([])

    def callback_usub(self, usub=None):
        self.update_temp_comp_sub(usub)
        self.update_AC(usub)
        self.update_usub_lab(usub)

    def _meta_wgt(self):
        wgt_meta = {
            d: pnwgt.Select(name=d, options=v, height=45, width=120)
            for d, v in self.meta_dicts.items()
        }

        def make_update_func(meta_name):
            def _update(x):
                self.metas[meta_name] = x.new
                self.update_subs()

            return _update

        for d, wgt in wgt_meta.items():
            cur_update = make_update_func(d)
            wgt.param.watch(cur_update, "value")
        wgt_update = pnwgt.Button(
            name="Refresh", button_type="primary", height=30, width=120
        )
        wgt_update.param.watch(self.update_all, "clicks")
        wgt_load = pnwgt.Button(
            name="Load Data", button_type="danger", height=30, width=120
        )
        wgt_load.param.watch(self.compute_subs, "clicks")
        return pn.layout.WidgetBox(
            *(list(wgt_meta.values()) + [wgt_update, wgt_load]), width=150
        )

    def show(self):
        return pn.layout.Column(
            self.spatial_all,
            pn.layout.Row(
                pn.layout.Column(
                    pn.layout.Row(self.wgt_meta, self.wgt_spatial_all),
                    self.wgt_temp_comp,
                ),
                self.temp_comp_sub,
                self.wgt_man,
            ),
        )

    def _temp_comp_sub(self, usub=None):
        if usub is None:
            usub = self.strm_usub.usub
        if self._normalize:
            C, S = self.C_norm_sub, self.S_norm_sub
        else:
            C, S = self.C_sub, self.S_sub
        cur_temp = dict()
        if self._showC:
            cur_temp["C"] = hv.Dataset(
                C.sel(unit_id=usub)
                .compute()
                .rename("Intensity (A. U.)")
                .dropna("frame", how="all")
            ).to(hv.Curve, "frame")
        if self._showS:
            cur_temp["S"] = hv.Dataset(
                S.sel(unit_id=usub)
                .compute()
                .rename("Intensity (A. U.)")
                .dropna("frame", how="all")
            ).to(hv.Curve, "frame")
        cur_vl = hv.DynamicMap(
            lambda f, y: hv.VLine(f) if f else hv.VLine(0), streams=[self.strm_f]
        ).opts(style=dict(color="red"))
        cur_cv = hv.Curve([], kdims=["frame"], vdims=["Internsity (A.U.)"])
        self.strm_f.source = cur_cv
        h_cv = len(self._w) // 8
        w_cv = len(self._w) * 2
        temp_comp = (
            cur_cv
            * datashade_ndcurve(
                hv.HoloMap(cur_temp, "trace")
                .collate()
                .overlay("trace")
                .grid("unit_id")
                .add_dimension("time", 0, 0),
                "trace",
            )
            .opts(plot=dict(shared_xaxis=True))
            .map(
                lambda p: p.opts(plot=dict(frame_height=h_cv, frame_width=w_cv)), hv.RGB
            )
            * cur_vl
        )
        temp_comp[temp_comp.keys()[0]] = temp_comp[temp_comp.keys()[0]].opts(
            plot=dict(height=h_cv + 75)
        )
        return pn.panel(temp_comp)

    def update_temp_comp_sub(self, usub=None):
        self.temp_comp_sub.object = self._temp_comp_sub(usub).object
        self.wgt_man.objects = self._man_wgt().objects

    def update_norm(self, norm):
        self._normalize = norm.new
        self.update_temp_comp_sub()

    def _temp_comp_wgt(self):
        if self.strm_uid.index:
            cur_idxs = self.strm_uid.index
        else:
            cur_idxs = self._u
        ntabs = np.ceil(len(cur_idxs) / 5)
        sub_idxs = np.array_split(cur_idxs, ntabs)
        idxs_dict = OrderedDict(
            [("group{}".format(i), g.tolist()) for i, g in enumerate(sub_idxs)]
        )
        def_idxs = list(idxs_dict.values())[0]
        wgt_grp = pnwgt.Select(
            name="", options=idxs_dict, width=120, height=30, value=def_idxs
        )

        def update_usub(usub):
            self.usub_sel = []
            self.strm_usub.event(usub=usub.new)

        wgt_grp.param.watch(update_usub, "value")
        wgt_grp.value = def_idxs
        self.strm_usub.event(usub=def_idxs)
        wgt_grp_prv = pnwgt.Button(
            name="Previous Group", width=120, height=30, button_type="primary"
        )

        def prv(clicks):
            cur_val = wgt_grp.value
            ig = list(idxs_dict.values()).index(cur_val)
            try:
                prv_val = idxs_dict[list(idxs_dict.keys())[ig - 1]]
                wgt_grp.value = prv_val
            except:
                pass

        wgt_grp_prv.param.watch(prv, "clicks")
        wgt_grp_nxt = pnwgt.Button(
            name="Next Group", width=120, height=30, button_type="primary"
        )

        def nxt(clicks):
            cur_val = wgt_grp.value
            ig = list(idxs_dict.values()).index(cur_val)
            try:
                nxt_val = idxs_dict[list(idxs_dict.keys())[ig + 1]]
                wgt_grp.value = nxt_val
            except:
                pass

        wgt_grp_nxt.param.watch(nxt, "clicks")
        wgt_norm = pnwgt.Checkbox(
            name="Normalize", value=self._normalize, width=120, height=10
        )
        wgt_norm.param.watch(self.update_norm, "value")
        wgt_showC = pnwgt.Checkbox(
            name="ShowC", value=self._showC, width=120, height=10
        )

        def callback_showC(val):
            self._showC = val.new
            self.update_temp_comp_sub()

        wgt_showC.param.watch(callback_showC, "value")
        wgt_showS = pnwgt.Checkbox(
            name="ShowS", value=self._showS, width=120, height=10
        )

        def callback_showS(val):
            self._showS = val.new
            self.update_temp_comp_sub()

        wgt_showS.param.watch(callback_showS, "value")
        wgt_play = pnwgt.Player(length=len(self._f), interval=10, value=0, width=280)

        def play(f):
            if not f.old == f.new:
                self.strm_f.event(x=self._f[f.new])

        wgt_play.param.watch(play, "value")
        wgt_groups = pn.layout.Row(
            pn.layout.WidgetBox(wgt_norm, wgt_showC, wgt_showS, wgt_grp, width=150),
            pn.layout.WidgetBox(wgt_grp_prv, wgt_grp_nxt, width=150),
        )
        return pn.layout.Column(wgt_groups, wgt_play)

    def _man_wgt(self):
        usub = self.strm_usub.usub
        usub.sort()
        usub.reverse()
        ulabs = self.unit_labels.sel(unit_id=usub).values
        wgt_sel = {
            uid: pnwgt.Select(
                name="Unit Label",
                options=usub + [-1] + ulabs.tolist(),
                value=ulb,
                height=50,
                width=100,
            )
            for uid, ulb in zip(usub, ulabs)
        }

        def callback_ulab(value, uid):
            self.unit_labels.loc[uid] = value.new

        for uid, sel in wgt_sel.items():
            cb = fct.partial(callback_ulab, uid=uid)
            sel.param.watch(cb, "value")
        wgt_check = {
            uid: pnwgt.Checkbox(
                name="Unit ID: {}".format(uid), value=False, height=50, width=100
            )
            for uid in usub
        }

        def callback_chk(val, uid):
            if not val.old == val.new:
                if val.new:
                    self.usub_sel.append(uid)
                else:
                    self.usub_sel.remove(uid)

        for uid, chk in wgt_check.items():
            cb = fct.partial(callback_chk, uid=uid)
            chk.param.watch(cb, "value")
        wgt_discard = pnwgt.Button(
            name="Discard Selected", button_type="primary", width=200
        )

        def callback_discard(clicks):
            for uid in self.usub_sel:
                wgt_sel[uid].value = -1

        wgt_discard.param.watch(callback_discard, "clicks")
        wgt_merge = pnwgt.Button(
            name="Merge Selected", button_type="primary", width=200
        )

        def callback_merge(clicks):
            for uid in self.usub_sel:
                wgt_sel[uid].value = self.usub_sel[0]

        wgt_merge.param.watch(callback_merge, "clicks")
        return pn.layout.Column(
            pn.layout.WidgetBox(wgt_discard, wgt_merge, width=200),
            pn.layout.Row(
                pn.layout.WidgetBox(*wgt_check.values(), width=100),
                pn.layout.WidgetBox(*wgt_sel.values(), width=100),
            ),
        )

    def update_temp_comp_wgt(self):
        self.wgt_temp_comp.objects = self._temp_comp_wgt().objects

    def update_temp(self):
        self.update_temp_comp_wgt()

    def update_AC(self, usub=None):
        if usub is None:
            usub = self.strm_usub.usub
        if usub:
            if self._useAC:
                umask = (self.A_sub.sel(unit_id=usub) > 0).any("unit_id")
                A_sub = self.A_sub.sel(unit_id=usub).where(umask, drop=True).fillna(0)
                C_sub = self.C_sub.sel(unit_id=usub)
                AC = xr.apply_ufunc(
                    da.dot,
                    A_sub,
                    C_sub,
                    input_core_dims=[
                        ["height", "width", "unit_id"],
                        ["unit_id", "frame"],
                    ],
                    output_core_dims=[["height", "width", "frame"]],
                    dask="allowed",
                )
                self._AC = AC.compute()
                wndh, wndw = AC.coords["height"].values, AC.coords["width"].values
                window = self.A_sub.sel(
                    height=slice(wndh.min(), wndh.max()),
                    width=slice(wndw.min(), wndw.max()),
                )
                self._AC = self._AC.reindex_like(window).fillna(0)
                self._mov = (self.org_sub.reindex_like(window)).compute()
            else:
                self._AC = self.A_sub.sel(unit_id=usub).sum("unit_id")
                self._mov = self.org_sub
            self.strm_f.event(x=0)
        else:
            self._AC = xr.DataArray([])
            self._mov = xr.DataArray([])
            self.strm_f.event(x=0)

    def update_usub_lab(self, usub=None):
        if usub is None:
            usub = self.strm_usub.usub
        if usub:
            self.pipusub.send(self.cents_sub[self.cents_sub["unit_id"].isin(usub)])
        else:
            self.pipusub.send([])

    def _spatial_all_wgt(self):
        wgt_useAC = pnwgt.Checkbox(
            name="UseAC", value=self._useAC, width=120, height=15
        )

        def callback_useAC(val):
            self._useAC = val.new
            self.update_AC()

        wgt_useAC.param.watch(callback_useAC, "value")
        return pn.layout.WidgetBox(wgt_useAC, width=150)

    def _spatial_all(self):
        metas = self.metas
        Asum = regrid(
            hv.Image(self.Asum.sel(**metas), ["width", "height"]), precompute=True
        ).opts(
            plot=dict(frame_height=len(self._h), frame_width=len(self._w)),
            style=dict(cmap="Viridis"),
        )
        cents = (
            hv.Dataset(
                self.cents_sub.drop(list(self.meta_dicts.keys()), axis="columns"),
                kdims=["width", "height", "unit_id"],
            )
            .to(hv.Points, ["width", "height"])
            .opts(
                style=dict(
                    alpha=0.1,
                    line_alpha=0,
                    size=5,
                    nonselection_alpha=0.1,
                    selection_alpha=0.9,
                )
            )
            .collate()
            .overlay("unit_id")
            .opts(plot=dict(tools=["hover", "box_select"]))
        )
        self.strm_uid.source = cents
        fim = fct.partial(hv.Image, kdims=["width", "height"])
        AC = regrid(hv.DynamicMap(fim, streams=[self.pipAC]), precompute=True).opts(
            plot=dict(frame_height=len(self._h), frame_width=len(self._w)),
            style=dict(cmap="Viridis"),
        )
        mov = regrid(hv.DynamicMap(fim, streams=[self.pipmov]), precompute=True).opts(
            plot=dict(frame_height=len(self._h), frame_width=len(self._w)),
            style=dict(cmap="Viridis"),
        )
        lab = fct.partial(hv.Labels, kdims=["width", "height"], vdims=["unit_id"])
        ulab = hv.DynamicMap(lab, streams=[self.pipusub]).opts(
            style=dict(text_color="red")
        )
        return pn.panel(Asum * cents + AC * ulab + mov)

    def update_spatial_all(self):
        self.spatial_all.objects = self._spatial_all().objects


class AlignViewer:
    def __init__(self, shiftds, sampling=2, pct_thres=99.9, on=0):
        self.shiftds = shiftds
        self.temps = shiftds["temps"]
        self.on = on
        self.pct_thres = pct_thres
        self.hh = self.temps.sizes["height"]
        self.ww = self.temps.sizes["width"]
        self.mask = xr.zeros_like(self.temps, dtype=bool)
        self.ls_anm = np.unique(shiftds.coords["animal"].values)
        self.ls_ss = np.unique(shiftds.coords["session"].values)
        Selection = Stream.define(
            "selection", anm=param.Selector(self.ls_anm), ss=param.Selector(self.ls_ss)
        )
        self.str_sel = Selection(anm=self.ls_anm[0], ss=self.ls_ss[0])
        # self.sampling = sampling
        self.str_box = BoxEdit()
        self.box = hv.DynamicMap(self._box, streams=[self.str_box])
        self.box = self.box.opts(style=dict(fill_alpha=0.3, line_color="white"))
        self.wgts = self._widgets()
        self.hvobjs = self._get_objs()

    def show(self):
        display(self.wgts)
        display(self.hvobjs)

    def _box(self, data):
        if data is None:
            return hv.Polygons([])
        else:
            diag = pd.DataFrame(data)
            corners = []
            for ir, r in diag.iterrows():
                corners.append(
                    np.array(
                        [
                            (r["x0"], r["y0"]),
                            (r["x0"], r["y1"]),
                            (r["x1"], r["y1"]),
                            (r["x1"], r["y0"]),
                        ]
                    )
                )
            return hv.Polygons([{("x", "y"): cnr.squeeze()} for cnr in corners])

    def _temps(self, anm, ss):
        ims = hv.Image(self.temps.sel(animal=anm, session=ss), ["width", "height"])
        return ims

    def _temp0(self, anm, ss):
        im0 = hv.Image(
            self.temps.sel(animal=anm).isel(session=self.on), ["width", "height"]
        )
        return im0

    def _re(self, anm, ss):
        re = hv.Image(
            self.shiftds["temps_shifted"].sel(animal=anm, session=ss),
            ["width", "height"],
        )
        return re

    def _widgets(self):
        sel_anm = iwgt.Dropdown(options=self.ls_anm, description="Animal:")
        sel_ss = iwgt.Dropdown(options=self.ls_ss, description="Session:")
        bt_mask0 = iwgt.Button(description="Update Template")
        bt_mask = iwgt.Button(description="Update Target")
        bt_mask0.on_click(self._save_mask0)
        bt_mask.on_click(self._save_mask)
        iwgt.interactive(self.str_sel.event, anm=sel_anm, ss=sel_ss)
        return iwgt.HBox([sel_anm, sel_ss, bt_mask0, bt_mask])

    def _get_objs(self):
        opts = {
            "plot": {"height": self.hh, "width": self.ww, "colorbar": True},
            "style": {"cmap": "Viridis"},
        }
        im0 = regrid(hv.DynamicMap(self._temp0, streams=[self.str_sel])).opts(**opts)
        ims = regrid(hv.DynamicMap(self._temps, streams=[self.str_sel])).opts(**opts)
        re = regrid(hv.DynamicMap(self._re, streams=[self.str_sel])).opts(**opts)
        return (
            (im0 * self.box).relabel("Template")
            + (ims * self.box.clone(link_inputs=False)).relabel("Target")
            + re.relabel("Shifted Target")
        )

    def _save_mask0(self, _):
        print("save mask 0")
        cur_anm = self.str_sel.anm
        cur_ss = self.str_sel.ss
        h0, hd = (
            self.str_box.data["y0"][-1],
            np.round(self.str_box.data["y1"][-1] - self.str_box.data["y0"][-1]),
        )
        w0, wd = (
            self.str_box.data["x0"][-1],
            np.round(self.str_box.data["x1"][-1] - self.str_box.data["x0"][-1]),
        )
        self.mask[dict(session=0)].loc[dict(animal=self.str_sel.anm)] = False
        self.mask[dict(session=0)].loc[
            dict(
                animal=self.str_sel.anm,
                height=slice(h0, h0 + hd),
                width=slice(w0, w0 + wd),
            )
        ] = True

    def _save_mask(self, _):
        print("entering")
        cur_anm = self.str_sel.anm
        cur_ss = self.str_sel.ss
        h1, hd = (
            self.str_box.data["y0"][-1],
            np.round(self.str_box.data["y1"][-1] - self.str_box.data["y0"][-1]),
        )
        w1, wd = (
            self.str_box.data["x0"][-1],
            np.round(self.str_box.data["x1"][-1] - self.str_box.data["x0"][-1]),
        )
        self.mask.loc[dict(animal=self.str_sel.anm, session=self.str_sel.ss)] = False
        self.mask.loc[
            dict(
                animal=self.str_sel.anm,
                session=self.str_sel.ss,
                height=slice(h1, h1 + hd),
                width=slice(w1, w1 + wd),
            )
        ] = True
        temp_src = self.temps.sel(animal=cur_anm).isel(session=0)
        mask_src = self.mask.sel(animal=cur_anm).isel(session=0)
        temp_dst = self.temps.sel(animal=cur_anm, session=cur_ss)
        mask_dst = self.mask.sel(animal=cur_anm, session=cur_ss)
        temp_src = temp_src.where(mask_src, drop=True)
        temp_dst = temp_dst.where(mask_dst, drop=True)
        man_sh_w = (
            temp_src.coords["width"][0].values - temp_dst.coords["width"][0].values
        )
        man_sh_h = (
            temp_src.coords["height"][0].values - temp_dst.coords["height"][0].values
        )
        man_sh = xr.DataArray(
            [man_sh_w, man_sh_h],
            coords=dict(shift_dim=["width", "height"]),
            dims=["shift_dim"],
        )
        print("done selection")
        temp_src_fft = np.fft.fft2(temp_src)
        temp_dst_fft = np.fft.fft2(temp_dst)
        cur_sh = match_temp(temp_src_fft, temp_dst_fft, pct_thres=self.pct_thres)
        cur_sh = xr.DataArray(
            np.round(cur_sh),
            coords=dict(shift_dim=list(temp_dst.dims)),
            dims=["shift_dim"],
        )
        cur_sh = cur_sh + man_sh
        sh_dict = cur_sh.astype(int).to_series().to_dict()
        print(sh_dict)
        self.shiftds["temps_shifted"].loc[
            dict(animal=cur_anm, session=cur_ss)
        ] = self.temps.sel(animal=cur_anm, session=cur_ss).shift(**sh_dict)
        self.shiftds["shifts"].loc[dict(animal=cur_anm, session=cur_ss)] = cur_sh


def write_vid_blk(arr, vpath, options):
    uid = uuid4()
    vname = "{}.mp4".format(uid)
    fpath = os.path.join(vpath, vname)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    container = av.open(fpath, mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width = arr.shape[2]
    stream.height = arr.shape[1]
    stream.pix_fmt = "yuv420p"
    stream.options = options
    for fm in arr:
        fm = cv2.cvtColor(fm, cv2.COLOR_GRAY2RGB)
        fmav = av.VideoFrame.from_ndarray(fm, format="rgb24")
        for p in stream.encode(fmav):
            container.mux(p)
    for p in stream.encode():
        container.mux(p)
    container.close()
    return fpath


def write_video(arr, vname=None, vpath=".", options={'crf': '18', 'preset': 'ultrafast'}):
    if not vname:
        vname = "{}.mp4".format(uuid4())
    fname = os.path.join(vpath, vname)
    paths = [
        dask.delayed(write_vid_blk)(np.asscalar(a), vpath, options)
        for a in arr.data.to_delayed()
    ]
    with dask.config.set(scheduler='processes'):
        paths = dask.compute(paths)[0]
    streams = [ffmpeg.input(p) for p in paths]
    (ffmpeg.concat(*streams).output(fname).run(overwrite_output=True))
    for vp in paths:
        os.remove(vp)
    return fname


def generate_videos(minian, varr, vpath=".", vname="minian.mp4", scale="auto", options={'crf': '18', 'preset': 'ultrafast'}):
    print("generating traces")
    A = minian["A"].compute().transpose("unit_id", "height", "width")
    C = minian["C"].chunk(dict(unit_id=-1)).transpose("frame", "unit_id")
    Y = (
        minian["Y"]
        .chunk(dict(height=-1, width=-1))
        .transpose("frame", "height", "width")
    )
    org = varr
    try:
        bl = minian["bl"].chunk(dict(unit_id=-1))
    except KeyError:
        print("cannot find background term")
        bl = 0
    C = C + bl
    AC = xr.apply_ufunc(
        da.tensordot,
        C,
        A,
        input_core_dims=[["frame", "unit_id"], ["unit_id", "height", "width"]],
        output_core_dims=[["frame", "height", "width"]],
        dask="allowed",
        kwargs=dict(axes=(1, 0)),
        output_dtypes=[A.dtype],
    )
    org_norm = org
    if scale == "auto":
        Y_max = Y.max().compute()
        Y_norm = Y * (255 / Y_max)
        AC_norm = AC * (255 / Y_max)
    else:
        Y_norm = Y * scale
        AC_norm = AC * scale
    res_norm = Y_norm - AC_norm
    print("writing videos")
    path_org = write_video(org_norm, vpath=vpath, options=options)
    path_Y = write_video(Y_norm, vpath=vpath, options=options)
    path_AC = write_video(AC_norm, vpath=vpath, options=options)
    path_res = write_video(res_norm, vpath=vpath, options=options)
    print("concatenating results")
    str_org = ffmpeg.input(path_org)
    str_Y = ffmpeg.input(path_Y)
    str_AC = ffmpeg.input(path_AC)
    str_res = ffmpeg.input(path_res)
    vtop = ffmpeg.filter([str_org, str_Y], "hstack")
    vbot = ffmpeg.filter([str_res, str_AC], "hstack")
    vid = ffmpeg.filter([vtop, vbot], "vstack")
    fname = os.path.join(vpath, vname)
    vid.output(fname).overwrite_output().run()
    for p in [path_res, path_AC, path_Y, path_org]:
        os.remove(p)
    return fname


def datashade_ndcurve(ovly, kdim=None, spread=False):
    if not kdim:
        kdim = ovly.kdims[0].name
    var = np.unique(ovly.dimension_values(kdim)).tolist()
    color_key = [(v, Category10_10[iv]) for iv, v in enumerate(var)]
    color_pts = hv.NdOverlay(
        {
            k: hv.Points([0, 0], label=str(k)).opts(style=dict(color=v))
            for k, v in color_key
        }
    )
    ds_ovly = datashade(
        ovly,
        aggregator=count_cat(kdim),
        color_key=dict(color_key),
        min_alpha=200,
        normalization="linear",
    )
    if spread:
        ds_ovly = dynspread(ds_ovly)
    return ds_ovly * color_pts


def construct_G(g, T):
    cur_c, cur_r = np.zeros(T), np.zeros(T)
    cur_c[0] = 1
    cur_r[0] = 1
    cur_c[1 : len(g) + 1] = -g
    return linalg.toeplitz(cur_c, cur_r)


def normalize(a):
    return np.interp(a, (np.nanmin(a), np.nanmax(a)), (0, +1))


def convolve_G(s, g):
    G = construct_G(g, len(s))
    try:
        c = np.linalg.inv(G).dot(s)
    except np.linalg.LinAlgError:
        c = s.copy()
    return c


def construct_pulse_response(g, length=500):
    s = np.zeros(length)
    s[np.arange(0, length, 500)] = 1
    c = convolve_G(s, g)
    return s, c


def centroid(A, verbose=False):
    def rel_cent(im):
        im_nan = np.isnan(im)
        if im_nan.all():
            return np.array([np.nan, np.nan])
        if im_nan.any():
            im = np.nan_to_num(im)
        cent = np.array(center_of_mass(im))
        return cent / im.shape

    gu_rel_cent = da.gufunc(
        rel_cent,
        signature="(h,w)->(d)",
        output_dtypes=float,
        output_sizes=dict(d=2),
        vectorize=True,
    )
    cents = xr.apply_ufunc(
        gu_rel_cent,
        A.chunk(dict(height=-1, width=-1)),
        input_core_dims=[["height", "width"]],
        output_core_dims=[["dim"]],
        dask="allowed",
    ).assign_coords(dim=["height", "width"])
    if verbose:
        print("computing centroids")
        with ProgressBar():
            cents = cents.compute()
    cents_df = (
        cents.rename("cents")
        .to_series()
        .dropna()
        .unstack("dim")
        .rename_axis(None, axis="columns")
        .reset_index()
    )
    h_rg = (A.coords["height"].min().values, A.coords["height"].max().values)
    w_rg = (A.coords["width"].min().values, A.coords["width"].max().values)
    cents_df["height"] = cents_df["height"] * (h_rg[1] - h_rg[0]) + h_rg[0]
    cents_df["width"] = cents_df["width"] * (w_rg[1] - w_rg[0]) + w_rg[0]
    return cents_df


def visualize_preprocess(fm, fn=None, include_org=True, **kwargs):
    fh, fw = fm.sizes["height"], fm.sizes["width"]
    asp = fw / fh
    opts_im = {
        "plot": {
            "frame_width": 500,
            "aspect": asp,
            "title": "Image {label} {group} {dimensions}",
        },
        "style": {"cmap": "viridis"},
    }
    opts_cnt = {
        "plot": {
            "frame_width": 500,
            "aspect": asp,
            "title": "Contours {label} {group} {dimensions}",
        },
        "style": {"cmap": "viridis"},
    }

    def _vis(f):
        im = hv.Image(f, kdims=["width", "height"]).opts(**opts_im)
        cnt = hv.operation.contours(im).opts(**opts_cnt)
        return im, cnt

    if fn is not None:
        pkey = kwargs.keys()
        pval = kwargs.values()
        im_dict = dict()
        cnt_dict = dict()
        for params in itt.product(*pval):
            fm_res = fn(fm, **dict(zip(pkey, params)))
            cur_im, cur_cnt = _vis(fm_res)
            cur_im = cur_im.relabel("After")
            cur_cnt = cur_cnt.relabel("After")
            p_str = tuple(
                [str(p) if not isinstance(p, (int, float)) else p for p in params]
            )
            im_dict[p_str] = cur_im
            cnt_dict[p_str] = cur_cnt
        hv_im = regrid(hv.HoloMap(im_dict, kdims=list(pkey)), precompute=True).opts(
            **opts_im
        )
        hv_cnt = datashade(
            hv.HoloMap(cnt_dict, kdims=list(pkey)), precompute=True, cmap=Viridis256
        ).opts(**opts_cnt)
        if include_org:
            im, cnt = _vis(fm)
            im = regrid(im, precompute=True).relabel("Before").opts(**opts_im)
            cnt = (
                datashade(cnt, precompute=True, cmap=Viridis256)
                .relabel("Before")
                .opts(**opts_cnt)
            )
        return (im + cnt + hv_im + hv_cnt).cols(2)
    else:
        im, cnt = _vis(fm)
        im = im.relabel("Before")
        cnt = cnt.relabel("Before")
        return im + cnt


def visualize_seeds(max_proj, seeds, mask=None, datashade=False):
    h, w = max_proj.sizes["height"], max_proj.sizes["width"]
    asp = w / h
    pt_cmap = {True: "white", False: "red"}
    opts_im = dict(plot=dict(frame_width=600, aspect=asp), style=dict(cmap="Viridis"))
    opts_pts = dict(
        plot=dict(
            frame_width=600,
            aspect=asp,
            size_index="seeds",
            color_index=mask,
            tools=["hover"],
        ),
        style=dict(fill_alpha=0.8, line_alpha=0, cmap=pt_cmap),
    )
    if mask:
        vdims = ["index", "seeds", mask]
    else:
        vdims = ["index", "seeds"]
        opts_pts["style"]["color"] = "white"
    im = hv.Image(max_proj, kdims=["width", "height"])
    pts = hv.Points(seeds, kdims=["width", "height"], vdims=vdims)
    if datashade:
        im = regrid(im)
    return im.opts(**opts_im) * pts.opts(**opts_pts)


def visualize_gmm_fit(values, gmm, bins):
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

    hist = np.histogram(values, bins=bins, density=True)
    gss_dict = dict()
    for igss, (mu, sig) in enumerate(zip(gmm.means_, gmm.covariances_)):
        gss = gaussian(hist[1], np.asscalar(mu), np.asscalar(np.sqrt(sig)))
        gss_dict[igss] = hv.Curve((hist[1], gss))
    return (
        hv.Histogram(((hist[0] - hist[0].min()) / np.ptp(hist[0]), hist[1])).opts(
            style=dict(alpha=0.6, fill_color="gray")
        )
        * hv.NdOverlay(gss_dict)
    ).opts(plot=dict(height=350, width=500))


def visualize_spatial_update(A_dict, C_dict, kdims=None, norm=True, datashading=True):
    if not kdims:
        A_dict = dict(dummy=A_dict)
        C_dict = dict(dummy=C_dict)
    hv_pts_dict, hv_A_dict, hv_Ab_dict, hv_C_dict = (dict(), dict(), dict(), dict())
    for key, A in A_dict.items():
        A = A.compute()
        C = C_dict[key]
        if norm:
            C = xr.apply_ufunc(
                normalize,
                C.chunk(dict(frame=-1)),
                input_core_dims=[["frame"]],
                output_core_dims=[["frame"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[C.dtype],
            )
        C = C.compute()
        h, w = A.sizes["height"], A.sizes["width"]
        cents_df = centroid(A)
        hv_pts_dict[key] = hv.Points(
            cents_df, kdims=["width", "height"], vdims=["unit_id"]
        ).opts(
            plot=dict(tools=["hover"]), style=dict(fill_alpha=0.2, line_alpha=0, size=8)
        )
        hv_A_dict[key] = hv.Image(
            A.sum("unit_id").rename("A"), kdims=["width", "height"]
        )
        hv_Ab_dict[key] = hv.Image(
            (A > 0).sum("unit_id").rename("A_bin"), kdims=["width", "height"]
        )
        hv_C_dict[key] = hv.Dataset(C.rename("C")).to(hv.Curve, kdims="frame")
    hv_pts = hv.HoloMap(hv_pts_dict, kdims=kdims)
    hv_A = hv.HoloMap(hv_A_dict, kdims=kdims)
    hv_Ab = hv.HoloMap(hv_Ab_dict, kdims=kdims)
    hv_C = (
        hv.HoloMap(hv_C_dict, kdims=kdims)
        .collate()
        .grid("unit_id")
        .add_dimension("time", 0, 0)
    )
    if datashading:
        hv_A = regrid(hv_A)
        hv_Ab = regrid(hv_Ab)
        hv_C = datashade(hv_C)
    hv_A = hv_A.opts(frame_width=400, aspect=w / h, colorbar=True, cmap="viridis")
    hv_Ab = hv_Ab.opts(frame_width=400, aspect=w / h, colorbar=True, cmap="viridis")
    hv_C = hv_C.map(
        lambda cr: cr.opts(frame_width=500, frame_height=50),
        hv.RGB if datashading else hv.Curve,
    )
    return hv.NdLayout(
        {"pseudo-color": (hv_pts * hv_A), "binary": (hv_pts * hv_Ab)},
        kdims="Spatial Matrix",
    ).cols(1) + hv_C.relabel("Temporal Components")


def visualize_temporal_update(
    YA_dict,
    C_dict,
    S_dict,
    g_dict,
    sig_dict,
    A_dict,
    kdims=None,
    norm=True,
    datashading=True,
):
    inputs = [YA_dict, C_dict, S_dict, sig_dict, g_dict]
    if not kdims:
        inputs = [dict(dummy=i) for i in inputs]
        A_dict = dict(dummy=A_dict)
    input_dict = {k: [i[k] for i in inputs] for k in inputs[0].keys()}
    hv_YA, hv_C, hv_S, hv_sig, hv_C_pul, hv_S_pul, hv_A = [dict() for _ in range(7)]
    for k, ins in input_dict.items():
        if norm:
            ins[:-1] = [
                xr.apply_ufunc(
                    normalize,
                    i.chunk(dict(frame=-1)),
                    input_core_dims=[["frame"]],
                    output_core_dims=[["frame"]],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[i.dtype],
                )
                for i in ins[:-1]
            ]
        ins[:] = [i.compute() for i in ins]
        ya, c, s, sig, g = ins
        f_crd = ya.coords["frame"]
        pul_crd = f_crd.values[:500]
        s_pul, c_pul = xr.apply_ufunc(
            construct_pulse_response,
            g,
            input_core_dims=[["lag"]],
            output_core_dims=[["t"], ["t"]],
            vectorize=True,
            kwargs=dict(length=len(pul_crd)),
            output_sizes=dict(t=len(pul_crd)),
        )
        s_pul, c_pul = (s_pul.assign_coords(t=pul_crd), c_pul.assign_coords(t=pul_crd))
        if norm:
            c_pul = xr.apply_ufunc(
                normalize,
                c_pul.chunk(dict(t=-1)),
                input_core_dims=[["t"]],
                output_core_dims=[["t"]],
                dask="parallelized",
                output_dtypes=[c_pul.dtype],
            ).compute()
        pul_range = (
            f_crd.min(),
            int(np.around(f_crd.min() + (f_crd.max() - f_crd.min()) / 2)),
        )
        hv_S_pul[k], hv_C_pul[k] = [
            (hv.Dataset(tr.rename("Response (A.U.)")).to(hv.Curve, kdims=["t"]))
            for tr in [s_pul, c_pul]
        ]
        hv_YA[k] = hv.Dataset(ya.rename("Intensity (A.U.)")).to(
            hv.Curve, kdims=["frame"]
        )
        if c.sizes["unit_id"] > 0:
            hv_C[k], hv_S[k], hv_sig[k] = [
                (
                    hv.Dataset(tr.rename("Intensity (A.U.)")).to(
                        hv.Curve, kdims=["frame"]
                    )
                )
                for tr in [c, s, sig]
            ]
        hv_A[k] = hv.Dataset(A_dict[k].rename("A")).to(
            hv.Image, kdims=["width", "height"]
        )
        h, w = A_dict[k].sizes["height"], A_dict[k].sizes["width"]
    hvobjs = [hv_YA, hv_C, hv_S, hv_sig, hv_C_pul, hv_S_pul, hv_A]
    hvobjs[:] = [hv.HoloMap(hvobj, kdims=kdims).collate() for hvobj in hvobjs]
    hv_unit = {
        "Raw Signal": hvobjs[0],
        "Fitted Calcium Trace": hvobjs[1],
        "Fitted Spikes": hvobjs[2],
        "Fitted Signal": hvobjs[3],
    }
    hv_pul = {"Simulated Calcium": hvobjs[4], "Simulated Spike": hvobjs[5]}
    hv_unit = hv.HoloMap(hv_unit, kdims="traces").collate().overlay("traces")
    hv_pul = hv.HoloMap(hv_pul, kdims="traces").collate().overlay("traces")
    hv_A = hvobjs[6]
    if datashading:
        hv_unit = datashade_ndcurve(hv_unit, "traces")
        hv_A = regrid(hv_A)
    else:
        hv_unit = Dynamic(hv_unit)
        hv_A = Dynamic(hv_A)
    hv_pul = Dynamic(hv_pul)
    hv_unit = hv_unit.map(
        lambda p: p.opts(plot=dict(frame_height=400, frame_width=1000))
    )
    hv_pul = hv_pul.opts(plot=dict(frame_width=500, aspect=w / h)).redim(
        t=hv.Dimension("t", soft_range=pul_range)
    )
    hv_A = hv_A.opts(
        plot=dict(frame_width=500, aspect=w / h), style=dict(cmap="Viridis")
    )
    return (
        hv_unit.relabel("Current Unit: Temporal Traces")
        + hv.NdLayout(
            {"Simulated Pulse Response": hv_pul, "Spatial Footprint": hv_A},
            kdims="Current Unit",
        )
    ).cols(1)


def roi_draw(im):
    h, w = im.sizes["height"], im.sizes["width"]
    opts_im = {"plot": {"height": h, "width": w}, "style": {"cmap": "Viridis"}}
    opts_box = {"style": {"fill_alpha": 0.3, "line_color": "white"}}
    hv_im = regrid(hv.Image(im, kdims=["width", "height"])).opts(**opts_im)
    hv_box = hv.Polygons([]).opts(**opts_box)
    str_box = BoxEdit(source=hv_box)
    return hv_im * hv_box, str_box


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el


def _get_figures_core(objs):
    if isinstance(objs, list):
        objs = [_get_figures_core(plot) for plot in objs]
    elif isinstance(objs, (models.Column, models.Row)):
        objs = [
            _get_figures_core(child)
            for child in objs.children
            if not isinstance(child, (models.ToolbarBox, models.WidgetBox))
        ]
    return objs


def _get_figures(objs):
    try:
        return list(flatten(_get_figures_core(objs)))
    except TypeError:
        return [_get_figures_core(objs)]


def _save_to_svg(hv_obj, save):
    bokeh_obj = hv.renderer("bokeh").get_plot(hv_obj).state
    figures = _get_figures(bokeh_obj)
    for i, figure in enumerate(figures):
        figure.output_backend = "svg"

        if len(figures) != 1:
            if not os.path.exists(save):
                os.makedirs(save)
            tidied_title = figure.title.text
            save_fp = os.path.join(save, "{0}_{1}".format(tidied_title, i))
        else:
            save_fp = save

        if not save_fp.endswith("svg"):
            save_fp = "{0}.{1}".format(save_fp, "svg")

        export_svgs(figure, save_fp)


def NNsort(cents):
    cents_hw = cents[["height", "width"]]
    kdtree = cKDTree(cents_hw)
    idu_start = cents_hw.sum(axis="columns").idxmin()
    result = pd.Series(0, index=cents.index)
    remain_list = cents.index.tolist()
    idu_next = idu_start
    NNord = 0
    while remain_list:
        result.loc[idu_next] = NNord
        remain_list.remove(idu_next)
        for k in range(1, int(np.ceil(np.log2(len(result)))) + 1):
            qry = kdtree.query(cents_hw.loc[idu_next], 2 ** k)
            NNs = qry[1][np.isfinite(qry[0])].squeeze()
            NNs = NNs[np.sort(np.unique(NNs, return_index=True)[1])]
            NNs = np.array(result.iloc[NNs].index)
            NN_idxs = np.argwhere(np.isin(NNs, remain_list, assume_unique=True))
            if len(NN_idxs) > 0:
                NN = NNs[NN_idxs[0]][0]
                idu_next = NN
                NNord = NNord + 1
                break
    return result
