import functools as fct
import numpy as np
import holoviews as hv
import xarray as xr
import ipywidgets as iwgt
import seaborn as sns
import colorsys
import param
import dash
import dash_core_components as dcc
import dash_html_components as dhc
import plotly
import plotly.graph_objs as go
import pandas as pd
import socket
import dask
import json
import cv2
from time import time
from dash.dependencies import Input, Output, State
from .utilities import scale_varr
from collections import OrderedDict
from holoviews.streams import Stream, Pipe, RangeXY, DoubleTap, Tap, Selection1D
from holoviews.operation import contours, threshold
from holoviews.operation.datashader import datashade, regrid
from datashader.colors import Sets1to3
from datashader import count_cat
from IPython.core.display import display, clear_output, display_html
from bokeh.io import push_notebook, show
from bokeh.plotting import figure
from bokeh.models import Slider, Range1d, LinearAxis
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from matplotlib.colors import rgb_to_hsv
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import find_contours
from IPython.core.debugger import set_trace

def update_override(self, **kwargs):
    self._set_stream_parameters(**kwargs)
    transformed = self.transform()
    if transformed:
        self._set_stream_parameters(**transformed)
    self.trigger([self])


class VArrayViewer():
    def __init__(self, varr, framerate=30):
        if isinstance(varr, list):
            varr = xr.merge(varr)
        self.ds = hv.Dataset(varr)
        self.framerate = framerate
        self._f = self.ds.range('frame')
        CStream = Stream.define(
            'CStream',
            f=param.Integer(default=int(self._f[0]), bounds=self._f))
        self.stream = CStream()
        self.widgets = self._widgets()

    def show(self, use_datashade=False):
        imdict = OrderedDict()
        meandict = OrderedDict()
        ds_mean = self.ds.aggregate(dimensions='frame', function=np.mean)
        ds_max = self.ds.aggregate(dimensions='frame', function=np.max)
        ds_min = self.ds.aggregate(dimensions='frame', function=np.min)
        for dim in self.ds.vdims:
            cur_d = self.ds.reindex(vdims=[dim.name])
            fim = fct.partial(self._img, dat=cur_d)
            im = hv.DynamicMap(fim, streams=[self.stream])
            im = regrid(im, height=480, width=752)
            im = im(plot={'width': 752, 'height': 480})
            xyrange = RangeXY(source=im)
            xyrange = xyrange.rename(x_range='w', y_range='h')
            fhist = fct.partial(self._hist, dat=cur_d)
            hist = hv.DynamicMap(fhist, streams=[self.stream, xyrange])
            hist = hist(plot={'width': 200})
            cur_mdict = OrderedDict()
            dmean = hv.Curve(ds_mean, kdims='frame', vdims=dim.name)
            cur_mdict['mean'] = dmean(plot={'tools': ['hover']})
            dmax = hv.Curve(ds_max, kdims='frame', vdims=dim.name)
            cur_mdict['max'] = dmax(plot={'tools': ['hover']})
            dmin = hv.Curve(ds_min, kdims='frame', vdims=dim.name)
            cur_mdict['min'] = dmin(plot={'tools': ['hover']})
            mean = hv.NdOverlay(cur_mdict, kdims=['variable'])
            if use_datashade:
                c_keys = OrderedDict()
                legends = OrderedDict()
                c_keys['mean'] = Sets1to3[0]
                legends['mean'] = hv.Points([0, 0]).opts(
                    style=dict(color=Sets1to3[0]))
                c_keys['max'] = Sets1to3[1]
                legends['max'] = hv.Points([0, 0]).opts(
                    style=dict(color=Sets1to3[1]))
                c_keys['min'] = Sets1to3[2]
                legends['min'] = hv.Points([0, 0]).opts(
                    style=dict(color=Sets1to3[2]))
                mean = datashade(
                    mean,
                    aggregator=count_cat('variable'),
                    height=300,
                    width=752,
                    color_key=c_keys)
                leg = hv.NdOverlay(legends)
                leg = leg(plot={'width': 752, 'height': 300})
                mean = mean * leg
            mean = mean(plot={'width': 752, 'height': 300})
            vl = hv.DynamicMap(lambda f: hv.VLine(f), streams=[self.stream])
            vl = vl(style={'color': 'red'})
            image = im.relabel(group=dim.name, label='Image')
            histtogram = hist.relabel(group=dim.name, label='Histogram')
            mean = (mean * vl).relabel(group=dim.name, label='Mean')
            imdict[dim.name] = image << histtogram
            meandict[dim.name] = mean
        return hv.Layout(list(imdict.values()) + list(meandict.values())).cols(
            len(self.ds.vdims))

    def _widgets(self):
        dfrange = self.ds.range('frame')
        w_frame = iwgt.IntSlider(
            value=0,
            min=dfrange[0],
            max=dfrange[1],
            continuous_update=False,
            description="Frame:")
        w_paly = iwgt.Play(
            value=0,
            min=dfrange[0],
            max=dfrange[1],
            interval=1000 / self.framerate)
        iwgt.jslink((w_paly, 'value'), (w_frame, 'value'))
        iwgt.interactive(self.stream.event, f=w_frame)
        return iwgt.HBox([w_paly, w_frame])

    def _img(self, dat, f):
        return hv.Image(dat.select(frame=f), kdims=['width', 'height'])

    def _hist(self, dat, f, w, h):
        if w and h:
            im = self._img(dat=dat, f=f).select(frame=f, height=h, width=w)
        else:
            im = self._img(dat=dat, f=f)
        return hv.operation.histogram(im, frequency_label='freq', num_bins=50)


class MCViewer():
    def __init__(self, varr, match_dict, framerate=30):
        if isinstance(varr, list):
            varr = xr.merge(varr)
        self.varr = varr
        self.varr_hv = hv.Dataset(varr)
        vh = self.varr_hv.range('height')
        vw = self.varr_hv.range('width')
        frange = self.varr_hv.range('frame')
        self._h = int(vh[1] - vh[0] + 1)
        self._w = int(vw[1] - vw[0] + 1)
        self.match = match_dict
        self.framerate = framerate
        CStream = Stream.define(
            'CStream', f=param.Integer(default=int(frange[0]), bounds=frange))
        self.stream = CStream()
        self.widgets = self._widgets()

    def show(self, use_datashade=False):
        varrdict = OrderedDict()
        for dim in self.varr_hv.vdims:
            cur_d = self.varr_hv.reindex(vdims=[dim.name])
            fim = fct.partial(self._img, dat=cur_d)
            im = hv.DynamicMap(fim, streams=[self.stream])
            im = regrid(im, height=self._h, width=self._w)
            im = im(plot={'width': self._w, 'height': self._h})
            image = im.relabel(group=dim.name, label='Image')
            varrdict[dim.name] = image
        ma = hv.DynamicMap(self._rgb, streams=[self.stream])
        ma = regrid(ma, height=self._h, width=self._w * 2)
        ma = ma.relabel(label="Match Image")
        ma = ma(plot={'height': self._h, 'width': self._w * 2})
        return hv.Layout(ma + hv.NdLayout(varrdict, kdims=['name'])).cols(1)

    def _widgets(self):
        dfrange = self.varr_hv.range('frame')
        w_frame = iwgt.IntSlider(
            value=0,
            min=dfrange[0],
            max=dfrange[1],
            continuous_update=False,
            description="Frame:")
        w_paly = iwgt.Play(
            value=0,
            min=dfrange[0],
            max=dfrange[1],
            interval=1000 / self.framerate)
        iwgt.jslink((w_paly, 'value'), (w_frame, 'value'))
        iwgt.interactive(self.stream.event, f=w_frame)
        return iwgt.HBox([w_paly, w_frame])

    def _img(self, dat, f):
        return hv.Image(dat.select(frame=f), kdims=['width', 'height'])

    def _rgb(self, f):
        dat_orig = list(self.varr.data_vars.values())[0]
        dat_mocc = list(self.varr.data_vars.values())[1]
        w = dat_orig.coords['width']
        h = dat_orig.coords['height']
        try:
            ma = self.match[f]
        except KeyError:
            ma = None
        if ma:
            up = ma['upsample']
            im_src = dat_orig.sel(frame=ma['src_fid']).reindex(
                method='nearest',
                width=np.linspace(w[0], w[-1], len(w) * up),
                height=np.linspace(h[0], h[-1], len(h) * up))
            im_dst = dat_orig.sel(frame=f).reindex(
                method='nearest',
                width=np.linspace(w[0], w[-1], len(w) * up),
                height=np.linspace(h[0], h[-1], len(h) * up))
            img = cv2.drawMatches(im_src.values, ma['src'][0], im_dst.values,
                                  ma['dst'][0], ma['match'], None)
        else:
            try:
                im_src = dat_orig.sel(frame=f - 1)
            except KeyError:
                im_src = dat_orig.sel(frame=f)
            im_dst = dat_orig.sel(frame=f)
            img = xr.concat([im_src, im_dst], dim='width')
            img = xr.concat([img] * 3, dim='color')
            img = img.transpose('height', 'width', 'color').values
        return hv.RGB(img, kdims=['width', 'height'])


class CNMFViewer():
    def __init__(self, cnmf, Y, framerate=30, load=True):
        self.cnmf = cnmf
        if load:
            self.Y = Y.load()
            self.A = self.cnmf['A'].load()
            self.C = self.cnmf['C'].load()
            self.S = self.cnmf['S'].load()
            self.b = self.cnmf['b'].load()
            self.f = self.cnmf['f'].load()
        else:
            self.Y = Y
            self.A = self.cnmf['A']
            self.C = self.cnmf['C']
            self.S = self.cnmf['S']
            self.b = self.cnmf['b']
            self.f = self.cnmf['f']
        self.framerate = framerate
        self._ht = (self.cnmf.coords['height'].min(),
                    self.cnmf.coords['height'].max())
        self._wt = (self.cnmf.coords['width'].min(),
                    self.cnmf.coords['width'].max())
        self._ft = (self.cnmf.coords['frame'].min(),
                    self.cnmf.coords['frame'].max())
        self._h = int(self.cnmf.coords['height'].max() -
                      self.cnmf.coords['height'].min() + 1)
        self._w = int(self.cnmf.coords['width'].max() -
                      self.cnmf.coords['width'].min() + 1)
        self._f = int(self.cnmf.coords['frame'].max() -
                      self.cnmf.coords['frame'].min() + 1)
        self._u = self.cnmf.sizes['unit_id']
        self._dims = dict(
            height=self._h, width=self._w, frame=self._f, unit_id=self._u)
        self.mean_fm = self._calculate_mean_frame()
        self.contours, self.centroids = self._calculate_contours_centroids()
        # self.sel_table = pd.DataFrame(
        #     dict(
        #         unit_id=self.cnmf.coords['unit_id'],
        #         active=[
        #             False,
        #         ] * self._u)).set_index('unit_id')
        # self.sel_table.at[self.cnmf.attrs['unit_mask'], 'active'] = True
        # self.sel_table = self.sel_table.reset_index()
        self.app = self.get_app()
        self._update_temporal_overlay = self._decorate(
            self._update_temporal_overlay,
            output=Output('temporal_overlay', 'figure'),
            inputs=[
                Input('spatial_overlay', 'selectedData'),
                Input('temporal_overlay', 'clickData'),
                Input('temporal_overlay_dropdown', 'value'),
                Input('temporal_unit', 'relayoutData')
            ],
            state=[State('temporal_overlay', 'figure')])
        self._update_temporal_unit = self._decorate(
            self._update_temporal_unit,
            output=Output('temporal_unit', 'figure'),
            inputs=[
                Input('unit_sel', 'value'),
                Input('temporal_overlay', 'clickData'),
                Input('temporal_overlay', 'relayoutData')
            ],
            state=[State('temporal_unit', 'figure')])
        self._update_spatial_overlay = self._decorate(
            self._update_spatial_overlay,
            output=Output('spatial_overlay', 'figure'),
            inputs=[
                Input('signal_chk', 'children'),
                Input('mov_Y', 'relayoutData'),
                Input('mov_AdC', 'relayoutData'),
                Input('mov_bdf', 'relayoutData'),
                Input('mov_res', 'relayoutData'),
                Input('spatial_unit', 'relayoutData')
            ],
            state=[State('spatial_overlay', 'figure')])
        self._update_spatial_unit = self._decorate(
            self._update_spatial_unit,
            output=Output('spatial_unit', 'figure'),
            inputs=[
                Input('unit_sel', 'value'),
                Input('mov_Y', 'relayoutData'),
                Input('mov_AdC', 'relayoutData'),
                Input('mov_bdf', 'relayoutData'),
                Input('mov_res', 'relayoutData'),
                Input('spatial_overlay', 'relayoutData')
            ],
            state=[State('spatial_unit', 'figure')])
        self._update_sel_list = self._decorate(
            self._update_sel_list,
            output=Output('unit_sel', 'options'),
            inputs=[Input('spatial_overlay', 'selectedData')])
        self._update_components = self._decorate(
            self._update_components,
            output=Output('signal_chk', 'children'),
            inputs=[Input('active', 'value')],
            state=[State('unit_sel', 'value')])
        self._save_result = self._decorate(
            self._save_result,
            output=Output('status', 'children'),
            inputs=[Input('save', 'n_clicks')])
        self._calculate_movies = self._decorate(
            self._calculate_movies,
            output=Output('signal_mov', 'children'),
            inputs=[
                Input('temporal_overlay', 'clickData'),
                Input('signal_chk', 'children')
            ])
        self._update_active = self._decorate(
            self._update_active,
            output=Output('active', 'value'),
            inputs=[Input('unit_sel', 'value')])
        self._update_movies_Y = self._decorate(
            self._update_movies_Y,
            output=Output('mov_Y', 'figure'),
            inputs=[
                Input('signal_mov', 'children'),
                Input('spatial_overlay', 'relayoutData'),
                Input('mov_AdC', 'relayoutData'),
                Input('mov_bdf', 'relayoutData'),
                Input('mov_res', 'relayoutData'),
                Input('spatial_unit', 'relayoutData')
            ],
            state=[State('mov_Y', 'figure')])
        self._update_moviesn_AdC = self._decorate(
            self._update_movies_AdC,
            output=Output('mov_AdC', 'figure'),
            inputs=[
                Input('signal_mov', 'children'),
                Input('spatial_overlay', 'relayoutData'),
                Input('mov_Y', 'relayoutData'),
                Input('mov_bdf', 'relayoutData'),
                Input('mov_res', 'relayoutData'),
                Input('spatial_unit', 'relayoutData')
            ],
            state=[State('mov_AdC', 'figure')])
        self._update_movies_bdf = self._decorate(
            self._update_movies_bdf,
            output=Output('mov_bdf', 'figure'),
            inputs=[
                Input('signal_mov', 'children'),
                Input('spatial_overlay', 'relayoutData'),
                Input('mov_Y', 'relayoutData'),
                Input('mov_AdC', 'relayoutData'),
                Input('mov_res', 'relayoutData'),
                Input('spatial_unit', 'relayoutData')
            ],
            state=[State('mov_bdf', 'figure')])
        self._update_movies_res = self._decorate(
            self._update_movies_res,
            output=Output('mov_res', 'figure'),
            inputs=[
                Input('signal_mov', 'children'),
                Input('spatial_overlay', 'relayoutData'),
                Input('mov_Y', 'relayoutData'),
                Input('mov_AdC', 'relayoutData'),
                Input('mov_bdf', 'relayoutData'),
                Input('spatial_unit', 'relayoutData')
            ],
            state=[State('mov_res', 'figure')])

    def get_app(self):
        app = dash.Dash()
        app.layout = dhc.Div(
            className='container-fluid',
            children=[
                dhc.H1(children='CNMF Viewer'),
                dhc.Div(
                    className='row justify-content-end',
                    children=[
                        dhc.Div(
                            className='col-2',
                            children=[
                                dcc.Dropdown(
                                    id='temporal_overlay_dropdown',
                                    options=[
                                        dict(
                                            label='Line Plot - C + S',
                                            value='line'),
                                        dict(
                                            label='Heatmap - C',
                                            value='heat_c'),
                                        dict(
                                            label='Heatmap - S',
                                            value='heat_s')
                                    ],
                                    value='line',
                                    searchable=False,
                                    clearable=False)
                            ])
                    ]),
                dhc.Div(
                    className='row',
                    children=[
                        dhc.Div(
                            className='col-auto',
                            children=[dcc.Graph(id='spatial_overlay')]),
                        dhc.Div(
                            className='col',
                            children=[dcc.Graph(id='temporal_overlay')])
                    ]),
                dhc.Div(
                    className='row justify-content-end',
                    children=[
                        dhc.Div(
                            className='col-2',
                            children=[
                                dcc.Dropdown(
                                    id='unit_sel',
                                    placeholder="Select a single unit to plot")
                            ]),
                        dhc.Div(
                            className='col-auto',
                            children=[
                                dcc.RadioItems(
                                    id='active',
                                    options=[
                                        dict(label='Accepted', value=True),
                                        dict(label='Rejected', value=False)
                                    ],
                                    value=False,
                                    labelStyle=dict(display='inline-block'))
                            ]),
                        dhc.Div(
                            className='col-auto',
                            children=[dhc.Button('Save Result', id='save')]),
                        dhc.Div(
                            className='col-auto',
                            children=[dhc.Div(id='status', children='Ready')],
                            style=dict(display='none'))
                    ]),
                dhc.Div(
                    className='row',
                    children=[
                        dhc.Div(
                            className='col-auto',
                            children=[dcc.Graph(id='spatial_unit')]),
                        dhc.Div(
                            className='col',
                            children=[dcc.Graph(id='temporal_unit')])
                    ]),
                dhc.Div(
                    className='row',
                    children=[
                        dhc.Div(
                            className='col-auto',
                            children=[dcc.Graph(id='mov_Y')]),
                        dhc.Div(
                            className='col-auto',
                            children=[dcc.Graph(id='mov_AdC')])
                    ]),
                dhc.Div(
                    className='row',
                    children=[
                        dhc.Div(
                            className='col-auto',
                            children=[dcc.Graph(id='mov_bdf')]),
                        dhc.Div(
                            className='col-auto',
                            children=[dcc.Graph(id='mov_res')])
                    ]),
                dhc.Div(id='signal_chk', style=dict(display='none')),
                dhc.Div(id='signal_mov', style=dict(display='none'))
            ])
        return app

    def show(self,
             port=9999,
             width=1740,
             height=2000,
             offline=False,
             style=True,
             **dash_flask_kwargs):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        localip = s.getsockname()[0]
        s.close()
        url = 'http://{ip}:{port}'.format(ip=localip, port=port)
        iframe = '<iframe src="{url}" width={width} height={height}></iframe>'.format(
            url=url, width=width, height=height)
        display_html(iframe, raw=True)
        if offline:
            self.app.css.config.serve_locally = True
            self.app.scripts.config.serve_locally = True
        if style:
            external_css = [
                "https://fonts.googleapis.com/css?family=Raleway:400,300,600",
                "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
                "http://getbootstrap.com/dist/css/bootstrap.min.css",
            ]
            for css in external_css:
                self.app.css.append_css({"external_url": css})
            external_js = [
                "https://code.jquery.com/jquery-3.2.1.min.js",
                "https://cdn.rawgit.com/plotly/dash-app-stylesheets/a3401de132a6d0b652ba11548736b1d1e80aa10d/dash-goldman-sachs-report-js.js",
                "http://getbootstrap.com/dist/js/bootstrap.min.js"
            ]
            for js in external_js:
                self.app.scripts.append_script({"external_url": js})
        self.app.run_server(
            debug=False, port=port, host='0.0.0.0', **dash_flask_kwargs)

    def _decorate(self, func, *args, **kwargs):
        return self.app.callback(*args, **kwargs)(func)

    def _calculate_contours_centroids(self):
        cnts_df_list = []
        cts_df_list = []
        A = self.cnmf['A'].load()
        for uid in range(self._u):
            cur_A = A.sel(unit_id=uid)
            cur_idxs = cur_A.squeeze().dims
            cur_thres = dask.delayed(cur_A.max)()
            cur_thres = dask.delayed(float)(cur_thres * .3)
            cur_cnts = dask.delayed(find_contours)(cur_A, cur_thres)
            cur_cnts = dask.delayed(np.concatenate)(cur_cnts)
            cur_cnts = dask.delayed(pd.DataFrame)(cur_cnts, columns=cur_idxs)
            cur_cnts = cur_cnts.assign(unit_id=uid)
            cur_cts = dask.delayed(center_of_mass)(cur_A.values)
            cur_cts = dask.delayed(pd.Series)(cur_cts, index=cur_idxs)
            cur_cts = cur_cts.append(pd.Series(dict(unit_id=uid)))
            cnts_df_list.append(cur_cnts)
            cts_df_list.append(cur_cts)
        cnts_df_list = dask.compute(*cnts_df_list)
        cts_df_list = dask.compute(*cts_df_list)
        cnts_df = pd.concat(cnts_df_list)
        cts_df = pd.concat(cts_df_list, axis=1).T
        for dim in cur_idxs:
            cnts_df[dim].update(cnts_df[dim] / A.sizes[dim] * self._dims[dim])
            cts_df[dim].update(cts_df[dim] / A.sizes[dim] * self._dims[dim])
        return cnts_df, cts_df

    def _calculate_mean_frame(self):
        cur_Y = self.Y.chunk(dict(width=100, height=100))
        return cur_Y.max('frame').compute()

    def _calculate_movies(self, frm, signal_chk):
        print("calculating movies")
        ts = time()
        signal_chk = json.loads(signal_chk)
        cur_fm = frm['points'][0]['x'] if frm else int(self._ft[0])
        cur_Y = self.Y.sel(frame=cur_fm).load()
        cur_A = self.A.sel(unit_id=self.cnmf.attrs['unit_mask'])
        cur_C = self.C.sel(unit_id=self.cnmf.attrs['unit_mask'], frame=cur_fm)
        cur_b = self.b
        cur_f = self.f.sel(frame=cur_fm)
        print("time spent in subsetting: {}".format(time() - ts))
        ts = time()
        cur_AdC = cur_A.dot(cur_C)
        print("AdC time: {}".format(time() - ts))
        ts = time()
        cur_bdf = cur_b.dot(cur_f)
        print("bdf time: {}".format(time() - ts))
        ts = time()
        # chk = dict(width=50)
        # cur_Y_chk = cur_Y.chunk(chk)
        # cur_AdC_chk = cur_AdC.chunk(chk)
        # cur_bdf_chk = cur_bdf.chunk(chk)
        # cur_res = cur_Y_chk - cur_AdC_chk - cur_bdf_chk
        # cur_res = cur_res.compute()
        cur_res = cur_Y.reindex_like(
            cur_AdC, method='nearest') - cur_AdC - cur_bdf
        print("Y time: {}".format(time() - ts))
        ts = time()
        self.cur_Y, self.cur_AdC, self.cur_bdf, self.cur_res = cur_Y, cur_AdC, cur_bdf, cur_res
        # cur_Y_df = dask.delayed(cur_Y).transpose('height', 'width').to_pandas()
        # cur_AdC_df = dask.delayed(cur_AdC).transpose('height', 'width').to_pandas()
        # cur_bdf_df = dask.delayed(cur_bdf).transpose('height', 'width').to_pandas()
        # cur_res_df = dask.delayed(cur_res).transpose('height', 'width').to_pandas()
        # self.cur_Y_df, self.cur_AdC_df, self.cur_bdf_df, self.cur_res_df = dask.compute(
        #     cur_Y_df, cur_AdC_df, cur_bdf_df, cur_res_df)
        # print("time spent in to pandas: {}".format(time() - ts))
        signal_chk['f'] = cur_fm
        signal_chk = json.dumps(signal_chk)
        return signal_chk

    def _update_components(self, active, uid):
        print("update_components")
        print(active)
        print(uid)
        if uid is not None:
            cur_mask = self.cnmf.attrs['unit_mask']
            if active and uid not in cur_mask:
                self.cnmf.attrs['unit_mask'] = np.append(cur_mask, uid)
            if not active and uid in cur_mask:
                idx = np.where(cur_mask == uid)[0]
                self.cnmf.attrs['unit_mask'] = np.delete(cur_mask, idx)
        return json.dumps(dict(active=active, uid=uid))

    def _update_active(self, uid):
        if uid is None:
            return True
        cur_mask = self.cnmf.attrs['unit_mask']
        if uid in cur_mask:
            return True
        else:
            return False

    def _save_result(self, nclicks):
        fpath = self.cnmf.attrs['file_path']
        newmask = self.cnmf.attrs['unit_mask']
        newds = xr.Dataset()
        newds = newds.assign_attrs(dict(unit_mask=newmask))
        self.cnmf.close()
        newds.to_netcdf(fpath, mode='a')
        return 'Ready'

    def _plot_spatial_overlay(self):
        pass

    def _update_spatial_overlay(self, sig_chk, rely1, rely2, rely3, rely4,
                                rely5, state):
        # cur_sel = [pt['text'] for pt in sel_uid['points']] if sel_uid else []
        # cur_sel = np.array(cur_sel)
        # cur_f = frm['points'][0]['x'] if frm else 0
        # AdC_list = []
        # for uid in cur_sel:
        #     cur_A = dask.delayed(self.cnmf['A'].sel(unit_id=uid))
        #     cur_C = dask.delayed(self.cnmf['C'].sel(unit_id=uid, frame=cur_f))
        #     cur_A = cur_A.where(cur_A > 0, drop=True)
        #     cur_AdC = cur_A * cur_C
        #     cur_AdC_df = cur_AdC.to_series().rename(
        #         'AdC').dropna().reset_index()
        #     cur_AdC_df = cur_AdC_df.assign(unit_id=uid, frame=cur_f)
        #     AdC_list.append(cur_AdC_df)
        # AdC_list = dask.compute(*AdC_list)
        # trace = state['data']
        # layout = state['layout']
        # for AdC in AdC_list:
        #     trace.append(
        #         go.Heatmap(
        #             x=AdC['width'],
        #             y=AdC['height'],
        #             z=AdC['AdC'],
        #             name=AdC['unit_id'],
        #             colorbar=False
        #         ))
        # state.update(data=trace, layout=layout)
        print('update_spatial_overlay')
        sig_chk = json.loads(sig_chk)
        _initial = False
        _update_cont = False
        if not state:
            _initial = True
        else:
            _update_cont = True
            last = state['data'][0]['customdata']
            if sig_chk['uid'] == last['uid'] and not sig_chk['active'] == last['active']:
                _update_cont = True
        if _initial or _update_cont:
            trace = []
            trace.append(
                go.Heatmap(
                    x=self.mean_fm.coords['width'].values,
                    y=self.mean_fm.coords['height'].values,
                    z=self.mean_fm.values,
                    name="mean_frame",
                    hoverinfo='none',
                    colorscale='Viridis',
                    colorbar=dict(x=1),
                    customdata=sig_chk))
            leg_showed = dict(accepted=False, rejected=False)
            for uid, cont in self.contours.groupby('unit_id'):
                if uid in self.cnmf.attrs['unit_mask']:
                    color = 'white'
                    group = 'accepted'
                else:
                    color = 'red'
                    group = 'rejected'
                cur_cont = go.Scatter(
                    x=cont['width'],
                    y=cont['height'],
                    text=cont['unit_id'],
                    mode='lines',
                    name='contours - ' + group,
                    hoverinfo='none',
                    legendgroup=group,
                    showlegend=False,
                    line=dict(color=color, width=1))
                if not leg_showed[group]:
                    cur_cont['showlegend'] = True
                    leg_showed[group] = True
                trace.append(cur_cont)
            trace.append(
                go.Scatter(
                    x=self.centroids['width'],
                    y=self.centroids['height'],
                    text=self.centroids['unit_id'],
                    mode='text',
                    name='unit id',
                    textfont=dict(color='white')))
            layout = go.Layout(
                title="Spatial Overlay",
                hovermode='closest',
                legend=dict(
                    x=1,
                    xanchor='right',
                    bgcolor='black',
                    font=dict(color='white'),
                    tracegroupgap=0),
                xaxis=dict(
                    title='width',
                    range=[0, self._w],
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                    constrain='domain',
                    constraintoward='right'),
                yaxis=dict(
                    title='height',
                    range=[0, self._h],
                    scaleanchor='x',
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                    constrain='domain',
                    constraintoward='top'))
            if _update_cont:
                state.update(data=trace)
            elif _initial:
                state = go.Figure(data=trace, layout=layout)
        state = self._sync_zoom([rely1, rely2, rely3, rely4, rely5], state)
        return state

    def _update_spatial_unit(self, uid, rely1, rely2, rely3, rely4, rely5,
                             state):
        print('update_spatial_unit')
        _initial = False
        _update = False
        if not state:
            _initial = True
        elif not state['data'][0]['customdata'] == uid:
            _update = True
        if _initial or _update:
            cur_A = self.cnmf['A'].sel(unit_id=uid if not uid is None else [])
            trace = [
                go.Heatmap(
                    x=cur_A.coords['width'].values,
                    y=cur_A.coords['height'].values,
                    z=cur_A.values,
                    colorscale='Viridis',
                    colorbar=dict(x=1),
                    hoverinfo='none',
                    customdata=uid)
            ]
            if _update:
                state.update(data=trace)
        if _initial:
            layout = go.Layout(
                title="Spatial Component of unit: {}".format(uid),
                xaxis=dict(
                    title='width',
                    range=[0, self._w],
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                    constrain='domain',
                    constraintoward='right'),
                yaxis=dict(
                    title='height',
                    range=[0, self._h],
                    scaleanchor='x',
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                    constrain='domain',
                    constraintoward='top'))
            state = go.Figure(data=trace, layout=layout)
        elif _update:
            state['layout']['title'] = "Spatial Component of unit: {}".format(uid)
        state = self._sync_zoom([rely1, rely2, rely3, rely4, rely5], state)
        return state

    def _update_temporal_overlay(self, sel_uid, frm, trace_type, rely1, state):
        print('update_temporal_overlay')
        ts = time()
        _initial = False
        _update_sel = False
        _update_f = False
        _update_type = False
        cur_sel = [pt['text'] for pt in sel_uid['points']] if sel_uid else []
        cur_f = frm['points'][0]['x'] if frm else 0
        custom_dict = dict(sel=cur_sel, f=cur_f, type=trace_type)
        if not state:
            _initial = True
        else:
            if not set(state['data'][0]['customdata']['sel']) == set(cur_sel):
                _update_sel = True
            if not state['data'][0]['customdata']['f'] == cur_f:
                _update_f = True
            if not state['data'][0]['customdata']['type'] == trace_type:
                _update_type = True
        if _initial or _update_type or _update_sel:
            ts = time()
            # cur_C = dask.delayed(self.cnmf['C'])
            # cur_S = dask.delayed(self.cnmf['S'])
            # cur_C = cur_C.load().sel(unit_id=cur_sel)
            # cur_S = cur_S.load().sel(unit_id=cur_sel)
            # cur_C_df = cur_C.transpose('unit_id', 'frame').to_pandas()
            # cur_S_df = cur_S.transpose('unit_id', 'frame').to_pandas()
            # cur_C_df, cur_S_df = dask.compute(cur_C_df, cur_S_df)
            cur_C = self.C.sel(unit_id=cur_sel)
            cur_S = self.S.sel(unit_id=cur_sel)
            print("time spent in generating data: {}".format(time() - ts))
            ts = time()
            trace = []
            if trace_type == 'line':
                trace.append(go.Scatter(visible=False, customdata=custom_dict))
                for uid in cur_sel:
                    trace.append(
                        go.Scatter(
                            x=cur_S.coords['frame'].values,
                            y=cur_S.sel(unit_id=uid).values,
                            mode='lines',
                            legendgroup='S',
                            name='S - {}'.format(uid),
                            yaxis='y2',
                            opacity=0.6,
                            line=dict(width=1)))
                    trace.append(
                        go.Scatter(
                            x=cur_C.coords['frame'].values,
                            y=cur_C.sel(unit_id=uid).values,
                            mode='lines',
                            legendgroup='C',
                            name='C - {}'.format(uid),
                            line=dict(width=1.5)))
                if _initial or _update_type:
                    layout = go.Layout(
                        title="Temporal Overlay",
                        hovermode='x',
                        xaxis=dict(title='frame', range=[0, self._f]),
                        yaxis=dict(title='Temporal Component', nticks=5),
                        yaxis2=dict(
                            title='Spike Component',
                            nticks=5,
                            overlaying='y',
                            side='right'),
                        legend=dict(x=1, xanchor='right'),
                        shapes=[
                            dict(
                                type='line',
                                xref='x',
                                yref='paper',
                                x0=cur_f,
                                y0=0,
                                x1=cur_f,
                                y1=1)
                        ])
            elif trace_type == 'heat_c':
                trace.append(
                    go.Heatmap(
                        x=cur_C.coords['frame'].values,
                        y=cur_C.coords['unit_id'].values,
                        z=cur_C.values,
                        colorscale='YlOrRd',
                        colorbar=dict(title="C Component"),
                        customdata=custom_dict))
                if _initial or _update_type:
                    layout = go.Layout(
                        title="Temporal Overlay",
                        hovermode='x',
                        xaxis=dict(title='frame', range=[0, self._f]),
                        yaxis=dict(title='Unit ID', type='category'),
                        shapes=[
                            dict(
                                type='line',
                                xref='x',
                                yref='paper',
                                x0=cur_f,
                                y0=0,
                                x1=cur_f,
                                y1=1,
                                line=dict(color='white'))
                        ])
            elif trace_type == 'heat_s':
                trace.append(
                    go.Heatmap(
                        x=cur_S.coords['frame'].values,
                        y=cur_S.coords['unit_id'].values,
                        z=cur_S.values,
                        colorscale='YlOrRd',
                        colorbar=dict(title="S Component"),
                        customdata=custom_dict))
                if _initial or _update_type:
                    layout = go.Layout(
                        title="Temporal Overlay",
                        hovermode='x',
                        xaxis=dict(title='frame', range=[0, self._f]),
                        yaxis=dict(title='Unit ID', type='category'),
                        shapes=[
                            dict(
                                type='line',
                                xref='x',
                                yref='paper',
                                x0=cur_f,
                                y0=0,
                                x1=cur_f,
                                y1=1,
                                line=dict(color='white'))
                        ])
            print("time spent in generating plot:{}".format(time() - ts))
            ts = time()
            if _initial:
                state = go.Figure(data=trace, layout=layout)
            else:
                state.update(data=trace)
                if _update_type:
                    state.update(layout=layout)
        if _update_f:
            state['layout']['shapes'][0]['x0'] = cur_f
            state['layout']['shapes'][0]['x1'] = cur_f
        print("time spent in updating: {}".format(time() - ts))
        state = self._sync_zoom([rely1], state, sync_y=False)
        return state

    def _update_temporal_unit(self, uid, frm, rely1, state):
        print('update_temporal_unit')
        cur_f = frm['points'][0]['x'] if frm else 0
        cur_C = self.cnmf['C'].load().sel(unit_id=uid if not uid is None else [])
        cur_S = self.cnmf['S'].load().sel(unit_id=uid if not uid is None else [])
        trace = []
        trace.append(
            go.Scatter(
                x=cur_S.coords['frame'].values,
                y=cur_S.values,
                mode='lines',
                legendgroup='S',
                name='S - {}'.format(uid),
                yaxis='y2',
                opacity=0.6,
                line=dict(width=1)))
        trace.append(
            go.Scatter(
                x=cur_C.coords['frame'].values,
                y=cur_C.values,
                mode='lines',
                legendgroup='C',
                name='C - {}'.format(uid),
                line=dict(width=1.5)))
        layout = go.Layout(
            title="Temporal Components of unit: {}".format(uid),
            hovermode='x',
            xaxis=dict(title='frame', range=[0, self._f]),
            yaxis=dict(title='Temporal Component', nticks=5),
            yaxis2=dict(
                title='Spike Component',
                nticks=5,
                overlaying='y',
                side='right'),
            legend=dict(x=1, xanchor='right'),
            shapes=[
                dict(
                    type='line',
                    xref='x',
                    yref='paper',
                    x0=cur_f,
                    y0=0,
                    x1=cur_f,
                    y1=1)
            ])
        if not state:
            state = go.Figure(data=trace, layout=layout)
        else:
            state.update(data=trace)
            state['layout']['title'] = layout['title']
            state['layout']['shapes'] = layout['shapes']
            state = self._sync_zoom([rely1], state, sync_y=False)
        return state

    def _update_sel_list(self, sel_uid):
        print('update selection list')
        cur_sel = [pt['text'] for pt in sel_uid['points']] if sel_uid else []
        if cur_sel:
            sel_list = [dict(label=str(s), value=int(s)) for s in cur_sel]
        else:
            sel_list = [
                dict(label=str(s), value=int(s))
                for s in self.cnmf.coords['unit_id'].values
            ]
        return sel_list

    def _update_movies_Y(self, sig_mov, rely1, rely2, rely3, rely4, rely5,
                         state):
        print('update movie Y')
        sig_mov = json.loads(sig_mov)
        _initial = False
        _update = False
        if state:
            if not state['data'][0]['customdata'] == sig_mov:
                _update = True
        else:
            _initial = True
        if _initial or _update:
            print("updating Y trace")
            ts = time()
            trace = [
                go.Heatmap(
                    x=self.cur_Y.coords['width'].values,
                    y=self.cur_Y.coords['height'].values,
                    z=self.cur_Y.values,
                    colorscale='Viridis',
                    colorbar=dict(x=1),
                    customdata=sig_mov,
                    hoverinfo='none')
            ]
            print("time spent in generating Y trace: {}".format(time() - ts))
            if _update:
                state.update(data=trace)
        if _initial:
            layout = go.Layout(
                title="Y (Original) at frame: {}".format(sig_mov['f']),
                xaxis=dict(
                    title='width',
                    range=[0, self._w],
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                    constrain='domain',
                    constraintoward='right'),
                yaxis=dict(
                    title='height',
                    range=[0, self._h],
                    scaleanchor='x',
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                    constrain='domain',
                    constraintoward='top'))
            state = go.Figure(data=trace, layout=layout)
        elif _update:
            state['layout']['title'] = "Y (Original) at frame: {}".format(
                sig_mov['f'])
        state = self._sync_zoom([rely1, rely2, rely3, rely4, rely5], state)
        return state

    def _update_movies_AdC(self, sig_mov, rely1, rely2, rely3, rely4, rely5,
                           state):
        print('update movie AdC')
        sig_mov = json.loads(sig_mov)
        _initial = False
        _update = False
        if state:
            if not state['data'][0]['customdata'] == sig_mov:
                _update = True
        else:
            _initial = True
        if _initial or _update:
            print("updating AdC trace")
            trace = [
                go.Heatmap(
                    x=self.cur_AdC.coords['width'].values,
                    y=self.cur_AdC.coords['height'].values,
                    z=self.cur_AdC.values,
                    colorscale='Viridis',
                    colorbar=dict(x=1),
                    customdata=sig_mov,
                    hoverinfo='none')
            ]
            if _update:
                state.update(data=trace)
        if _initial:
            layout = go.Layout(
                title="A dot C (Units) at frame: {}".format(sig_mov['f']),
                xaxis=dict(
                    title='width',
                    range=[0, self._w],
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                    constrain='domain',
                    constraintoward='right'),
                yaxis=dict(
                    title='height',
                    range=[0, self._h],
                    scaleanchor='x',
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                    constrain='domain',
                    constraintoward='top'))
            state = go.Figure(data=trace, layout=layout)
        elif _update:
            state['layout']['title'] = "A dot C (Units) at frame: {}".format(
                sig_mov['f'])
        state = self._sync_zoom([rely1, rely2, rely3, rely4, rely5], state)
        return state

    def _update_movies_bdf(self, sig_mov, rely1, rely2, rely3, rely4, rely5,
                           state):
        print('update movie bdf')
        sig_mov = json.loads(sig_mov)
        _initial = False
        _update = False
        if state:
            if not state['data'][0]['customdata'] == sig_mov['f']:
                _update = True
        else:
            _initial = True
        if _initial or _update:
            print("updating bdf trace")
            trace = [
                go.Heatmap(
                    x=self.cur_bdf.coords['width'].values,
                    y=self.cur_bdf.coords['height'].values,
                    z=self.cur_bdf.values,
                    colorscale='Viridis',
                    colorbar=dict(x=1),
                    customdata=sig_mov,
                    hoverinfo='none')
            ]
            if _update:
                state.update(data=trace)
        if _initial:
            layout = go.Layout(
                title="b dot f (Background) at frame: {}".format(sig_mov['f']),
                xaxis=dict(
                    title='width',
                    range=[0, self._w],
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                    constrain='domain',
                    constraintoward='right'),
                yaxis=dict(
                    title='height',
                    range=[0, self._h],
                    scaleanchor='x',
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                    constrain='domain',
                    constraintoward='top'))
            state = go.Figure(data=trace, layout=layout)
        elif _update:
            state['layout'][
                'title'] = "b dot f (Background) at frame: {}".format(
                    sig_mov['f'])
        state = self._sync_zoom([rely1, rely2, rely3, rely4, rely5], state)
        return state

    def _update_movies_res(self, sig_mov, rely1, rely2, rely3, rely4, rely5,
                           state):
        print('update movie res')
        sig_mov = json.loads(sig_mov)
        _initial = False
        _update = False
        if state:
            if not state['data'][0]['customdata'] == sig_mov['f']:
                _update = True
        else:
            _initial = True
        if _initial or _update:
            print("updating res trace")
            trace = [
                go.Heatmap(
                    x=self.cur_res.coords['width'].values,
                    y=self.cur_res.coords['height'].values,
                    z=self.cur_res.values,
                    colorscale='Viridis',
                    colorbar=dict(x=1),
                    customdata=sig_mov,
                    hoverinfo='none')
            ]
            if _update:
                state.update(data=trace)
        if _initial:
            layout = go.Layout(
                title="Residual at frame: {}".format(sig_mov['f']),
                xaxis=dict(
                    title='width',
                    range=[0, self._w],
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                    constrain='domain',
                    constraintoward='right'),
                yaxis=dict(
                    title='height',
                    range=[0, self._h],
                    scaleanchor='x',
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                    constrain='domain',
                    constraintoward='top'))
            state = go.Figure(data=trace, layout=layout)
        elif _update:
            state['layout']['title'] = "Residual at frame: {}".format(
                sig_mov['f'])
        state = self._sync_zoom([rely1, rely2, rely3, rely4, rely5], state)
        return state

    def _sync_zoom(self, new_range_list, state, sync_x=True, sync_y=True):
        if sync_x:
            cur_x_range = state['layout']['xaxis']['range']
            new_x_range = []
        if sync_y:
            cur_y_range = state['layout']['yaxis']['range']
            new_y_range = []
        for rely in new_range_list:
            if rely:
                if sync_x and 'xaxis.range[0]' in rely:
                    if not (rely['xaxis.range[0]'] == cur_x_range[0]
                            and rely['xaxis.range[1]'] == cur_x_range[1]):
                        new_x_range = [
                            rely['xaxis.range[0]'], rely['xaxis.range[1]']
                        ]
                if sync_y and 'yaxis.range[0]' in rely:
                    if not (rely['yaxis.range[0]'] == cur_y_range[0]
                            and rely['yaxis.range[1]'] == cur_y_range[1]):
                        new_y_range = [
                            rely['yaxis.range[0]'], rely['yaxis.range[1]']
                        ]
        if sync_x and new_x_range:
            state['layout']['xaxis']['range'] = new_x_range
        if sync_y and new_y_range:
            state['layout']['yaxis']['range'] = new_y_range
        return state
