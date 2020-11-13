

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       varr_ref = varr_ref.chunk(chk)

                                    .. code:: CodeMirror-line

                                       varr_ref = save_minian(varr_ref.rename('org'), **param_save_minian)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    # motion correction

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: motion correction\ `¶ <#motion-correction>`__
               :name: motion-correction

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## load in from disk

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: load in from disk\ `¶ <#load-in-from-disk>`__
               :name: load-in-from-disk

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Here we load in the data we just saved. We use `'fname'` and `'backend'` from `param_save_minian` since they should be the same and you don't have to specify the same information twice.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Here we load in the data we just saved. We use ``'fname'``
            and ``'backend'`` from ``param_save_minian`` since they
            should be the same and you don't have to specify the same
            information twice.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       varr_ref = open_minian(dpath,

                                    .. code:: CodeMirror-line

                                                             fname=param_save_minian['fname'],

                                    .. code:: CodeMirror-line

                                                             backend=param_save_minian['backend'])['org']

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## estimate shifts

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: estimate shifts\ `¶ <#estimate-shifts>`__
               :name: estimate-shifts

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Recall the parameters for `estimate shifts`:

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ```python

                                 .. code:: CodeMirror-line

                                    param_estimate_shift = {

                                 .. code:: CodeMirror-line

                                        'dim': 'frame',

                                 .. code:: CodeMirror-line

                                        'max_sh': 20}

                                 .. code:: CodeMirror-line

                                    ```

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    The idea behind `estimate_shift_fft` is simple: for each frame it calculates a two-dimensional [cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation) between that frame and a template frame using [fft](https://en.wikipedia.org/wiki/Fast_Fourier_transform). The argument `'dim'` specifies along which dimension to run the shift estimation, and should always be set to `'frame'` for this pipeline. To properly calculate the correlation we have to zero-pad the input frame, otherwise our estimation will be biased towards zero shifts. The amount of zero-padding essentially determine the maximum amount of shifts that can be accounted for, and `max_sh` controls this quantity in pixels. The results from `estimate_shift_fft` are saved in a two dimensional `DataArray` called `shifts`, with two labels on the `variable` dimension, representing the shifts along `'height'` and `'width'` directions.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Recall the parameters for ``estimate shifts``:

            ::

               param_estimate_shift = {
                   'dim': 'frame',
                   'max_sh': 20}

            The idea behind ``estimate_shift_fft`` is simple: for each
            frame it calculates a two-dimensional
            `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`__
            between that frame and a template frame using
            `fft <https://en.wikipedia.org/wiki/Fast_Fourier_transform>`__.
            The argument ``'dim'`` specifies along which dimension to
            run the shift estimation, and should always be set to
            ``'frame'`` for this pipeline. To properly calculate the
            correlation we have to zero-pad the input frame, otherwise
            our estimation will be biased towards zero shifts. The
            amount of zero-padding essentially determine the maximum
            amount of shifts that can be accounted for, and ``max_sh``
            controls this quantity in pixels. The results from
            ``estimate_shift_fft`` are saved in a two dimensional
            ``DataArray`` called ``shifts``, with two labels on the
            ``variable`` dimension, representing the shifts along
            ``'height'`` and ``'width'`` directions.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       shifts = estimate_shifts(varr_ref.sel(subset_mc), **param_estimate_shift)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## save shifts

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: save shifts\ `¶ <#save-shifts>`__
               :name: save-shifts

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       shifts = shifts.chunk(dict(frame=chk['frame'])).rename('shifts')

                                    .. code:: CodeMirror-line

                                       shifts = save_minian(shifts, **param_save_minian)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## visualization of shifts

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: visualization of
               shifts\ `¶ <#visualization-of-shifts>`__
               :name: visualization-of-shifts

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Here, we visualize `shifts` as a fluctuating curve along `frame`s. This is the first time we explicitly use the package [holoviews](http://holoviews.org), which is a really nice package for visualizing data in an interactive manner, and it is highly recommended that you read through the holoviews tutorial to get familiar with its syntax.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Here, we visualize ``shifts`` as a fluctuating curve along
            ``frame``\ s. This is the first time we explicitly use the
            package `holoviews <http://holoviews.org>`__, which is a
            really nice package for visualizing data in an interactive
            manner, and it is highly recommended that you read through
            the holoviews tutorial to get familiar with its syntax.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%opts Curve [frame_width=500, tools=['hover'], aspect=2]

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           display(hv.NdOverlay(dict(width=hv.Curve(shifts.sel(variable='width')),

                                    .. code:: CodeMirror-line

                                                                     height=hv.Curve(shifts.sel(variable='height')))))

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## apply shifts

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: apply shifts\ `¶ <#apply-shifts>`__
               :name: apply-shifts

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    After determining what each frame's shift from the template is, we use the function `apply_shifts`, which takes as inputs our video (`varr_ref`) and (`shifts`) and returns the movie we want (`Y`). Notably, pixels that are shifted inside the field of view will result in NaN values (`np.nan`) along the edges of our video, and we have to decide what to do with these. The default is to fill them with 0. 

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            After determining what each frame's shift from the template
            is, we use the function ``apply_shifts``, which takes as
            inputs our video (``varr_ref``) and (``shifts``) and returns
            the movie we want (``Y``). Notably, pixels that are shifted
            inside the field of view will result in NaN values
            (``np.nan``) along the edges of our video, and we have to
            decide what to do with these. The default is to fill them
            with 0.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       Y = apply_shifts(varr_ref, shifts)

                                    .. code:: CodeMirror-line

                                       Y = Y.fillna(0).astype(varr_ref.dtype)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Alternatively you can leverage the [dropna](http://xarray.pydata.org/en/stable/generated/xarray.DataArray.dropna.html) function to drop them, or [fillna](http://xarray.pydata.org/en/stable/generated/xarray.DataArray.fillna.html) to fill them with a specific value (potentially `varr_mc.min()`)

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    For example, instead of filling the NaN pixels with the nearest available value, you drop these pixels with the following code:

                                 .. code:: CodeMirror-line

                                    ```python

                                 .. code:: CodeMirror-line

                                    varr_mc = varr_mc.where(varr_mc.isnull().sum('frame') == 0).dropna('height', how='all').dropna('width', how='all')

                                 .. code:: CodeMirror-line

                                    ```

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Alternatively you can leverage the
            `dropna <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.dropna.html>`__
            function to drop them, or
            `fillna <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.fillna.html>`__
            to fill them with a specific value (potentially
            ``varr_mc.min()``)

            For example, instead of filling the NaN pixels with the
            nearest available value, you drop these pixels with the
            following code:

            ::

               varr_mc = varr_mc.where(varr_mc.isnull().sum('frame') == 0).dropna('height', how='all').dropna('width', how='all')

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## visualization of motion-correction

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: visualization of
               motion-correction\ `¶ <#visualization-of-motion-correction>`__
               :name: visualization-of-motion-correction

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Here we visualize the final result of motion correction (`varr_mc`) with `VArrayViewer`. The optional argument `framerate` only controls how the frame slider behaves, not how the data is handled.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Here we visualize the final result of motion correction
            (``varr_mc``) with ``VArrayViewer``. The optional argument
            ``framerate`` only controls how the frame slider behaves,
            not how the data is handled.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           vaviewer = VArrayViewer(

                                    .. code:: CodeMirror-line

                                               [varr_ref.rename('before_mc'), Y.rename('after_mc')],

                                    .. code:: CodeMirror-line

                                               framerate=5,

                                    .. code:: CodeMirror-line

                                               summary=None,

                                    .. code:: CodeMirror-line

                                               layout=True)

                                    .. code:: CodeMirror-line

                                           display(vaviewer.show())

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## save result

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: save result\ `¶ <#save-result>`__
               :name: save-result

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       Y = Y.chunk(chk)

                                    .. code:: CodeMirror-line

                                       Y = save_minian(Y.rename('Y'), **param_save_minian)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## generate video for motion-correction

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: generate video for
               motion-correction\ `¶ <#generate-video-for-motion-correction>`__
               :name: generate-video-for-motion-correction

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Here we have some additional visualizations for motion correction. We can generate a video and play it to quickly go through the results. In addition we can look at the max projection before and after motion correction. If there were a lot of translational motion presented in the raw video, we expect the border of cells are much more well-defined, and even some "different" cells (due to motion) are "merged" together in the max projection.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Here we have some additional visualizations for motion
            correction. We can generate a video and play it to quickly
            go through the results. In addition we can look at the max
            projection before and after motion correction. If there were
            a lot of translational motion presented in the raw video, we
            expect the border of cells are much more well-defined, and
            even some "different" cells (due to motion) are "merged"
            together in the max projection.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       vid_arr = xr.concat([varr_ref, Y], 'width').chunk(dict(height=-1, width=-1))

                                    .. code:: CodeMirror-line

                                       vmax = varr_ref.max().compute().values

                                    .. code:: CodeMirror-line

                                       write_video(vid_arr / vmax * 255, 'minian_mc.mp4', dpath)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       im_opts = dict(frame_width=500, aspect=752/480, cmap='Viridis', colorbar=True)

                                    .. code:: CodeMirror-line

                                       (regrid(hv.Image(varr_ref.max('frame').compute(), ['width', 'height'], label='before_mc')).opts(**im_opts)

                                    .. code:: CodeMirror-line

                                        + regrid(hv.Image(Y.max('frame').compute(), ['width', 'height'], label='after_mc')).opts(**im_opts))

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    # initialization

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: initialization\ `¶ <#initialization>`__
               :name: initialization

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    In order to run CNMF, we first need to generate an initial estimate of where our cells are likely to be and what their temporal activity is likely to look like. The whole initialization section is adapted from the [MIN1PIPE](https://github.com/JinghaoLu/MIN1PIPE) package. See their [paper](https://www.cell.com/cell-reports/fulltext/S2211-1247(18)30826-X) for full details about the theory. Here we only give enough information so that we can select parameters.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            In order to run CNMF, we first need to generate an initial
            estimate of where our cells are likely to be and what their
            temporal activity is likely to look like. The whole
            initialization section is adapted from the
            `MIN1PIPE <https://github.com/JinghaoLu/MIN1PIPE>`__
            package. See their
            `paper <https://www.cell.com/cell-reports/fulltext/S2211-1247(18)30826-X>`__
            for full details about the theory. Here we only give enough
            information so that we can select parameters.

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## load in from disk

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: load in from disk\ `¶ <#load-in-from-disk>`__
               :name: load-in-from-disk

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    The first thing we want to do is open up the dataset we just saved.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            The first thing we want to do is open up the dataset we just
            saved.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       minian = open_minian(dpath,

                                    .. code:: CodeMirror-line

                                                            fname=param_save_minian['fname'],

                                    .. code:: CodeMirror-line

                                                            backend=param_save_minian['backend'])

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Here we get the movie (`Y`) from the dataset, calculate a max projection that will be used later, and generate a flattened version of our video (`Y_flt`), where the original dimemsions `'height'` and `'width'` are flattened as one dimension `spatial`.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Here we get the movie (``Y``) from the dataset, calculate a
            max projection that will be used later, and generate a
            flattened version of our video (``Y_flt``), where the
            original dimemsions ``'height'`` and ``'width'`` are
            flattened as one dimension ``spatial``.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       Y = minian['Y'].astype(np.float)

                                    .. code:: CodeMirror-line

                                       max_proj = Y.max('frame').compute()

                                    .. code:: CodeMirror-line

                                       Y_flt = Y.stack(spatial=['height', 'width'])

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## generating over-complete set of seeds

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: generating over-complete set of
               seeds\ `¶ <#generating-over-complete-set-of-seeds>`__
               :name: generating-over-complete-set-of-seeds

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    The first step is to initialize the **seeds**. Recall the parameters:

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ```python

                                 .. code:: CodeMirror-line

                                    param_seeds_init = {

                                 .. code:: CodeMirror-line

                                        'wnd_size': 2000,

                                 .. code:: CodeMirror-line

                                        'method': 'rolling',

                                 .. code:: CodeMirror-line

                                        'stp_size': 1000,

                                 .. code:: CodeMirror-line

                                        'nchunk': 100,

                                 .. code:: CodeMirror-line

                                        'max_wnd': 15,

                                 .. code:: CodeMirror-line

                                        'diff_thres': 3}

                                 .. code:: CodeMirror-line

                                    ```

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    The idea is that we select some subset of frames, compute a max projection of those frames, and find the local maxima of that max projection. We keep repeating this process and putting together all the local maxima we get along the way until we get an overly-complete set of local maxima/bright-spots, which are the potential locations of cells. We call them **seeds**. The assumption here is that the center of cells are brighter than their surroundings on some, but not necessarily all, frames. The first and only required argument `seeds_init` takes is the video array we want to process (here, `Y`). There are four additional arguments controlling how we subset the frames: `wnd_size` controls the window size of each chunk (*i.e* the number of frames in each chunk); `method` can be either `'rolling'` or `'random'`. For `method='rolling'`, the moving window will roll along `frame`, whereas for `method='random'`, chunks with `wnd_size` number of frames will be randomly selected; `stp_size` is only used if `method='rolling'`, and is the step-size of the rolling window, or in other words, the distance between the **center** of each rolling window. For example, if `wnd_size=100` and `stp_size=200`, the windows will be as follows: **(0, 100)**, **(200, 300)**, **(400, 500)** *etc.* Obviously that was a **bad** choice since you probably want the windows to overlap or you will miss cells. `nchunk` is only used if `method='random'`, and is the number of random chunks we will draw. Additionally we have two parameters controlling how the local maxima are found. `'max_wnd'` controls the window size within which a single pixel will be choosen as local maxima. In order to capture cells with all sizes, we actually find local maximas with different window size and merge all of them, starting from **2** all the way up to `'max_wnd'`. Hence `'max_wnd'` should be the radius of the **largest** cell you want to detect. Finally in order to get rid of local maxima with very little fluctuation, we set a `'diff_thres'` which is the minimal fluorescent diffrence of a seed across `frame`s. Since the linear scale of the raw data is preserved, we can set this threshold emprically.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-info">  

                                 .. code:: CodeMirror-line

                                    The default values of `seeds_init` usually work fairly well for a dense region like CA1. If you are working with deep brain region with sparse cells, try to increase wnd_size and stp_size to make the following <strong>seeds</strong> cleaning steps faster and cleaner.

                                 .. code:: CodeMirror-line

                                    </div> 

                                 .. code:: CodeMirror-line

                                    ​

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            The first step is to initialize the **seeds**. Recall the
            parameters:

            ::

               param_seeds_init = {
                   'wnd_size': 2000,
                   'method': 'rolling',
                   'stp_size': 1000,
                   'nchunk': 100,
                   'max_wnd': 15,
                   'diff_thres': 3}

            The idea is that we select some subset of frames, compute a
            max projection of those frames, and find the local maxima of
            that max projection. We keep repeating this process and
            putting together all the local maxima we get along the way
            until we get an overly-complete set of local
            maxima/bright-spots, which are the potential locations of
            cells. We call them **seeds**. The assumption here is that
            the center of cells are brighter than their surroundings on
            some, but not necessarily all, frames. The first and only
            required argument ``seeds_init`` takes is the video array we
            want to process (here, ``Y``). There are four additional
            arguments controlling how we subset the frames: ``wnd_size``
            controls the window size of each chunk (*i.e* the number of
            frames in each chunk); ``method`` can be either
            ``'rolling'`` or ``'random'``. For ``method='rolling'``, the
            moving window will roll along ``frame``, whereas for
            ``method='random'``, chunks with ``wnd_size`` number of
            frames will be randomly selected; ``stp_size`` is only used
            if ``method='rolling'``, and is the step-size of the rolling
            window, or in other words, the distance between the
            **center** of each rolling window. For example, if
            ``wnd_size=100`` and ``stp_size=200``, the windows will be
            as follows: **(0, 100)**, **(200, 300)**, **(400, 500)**
            *etc.* Obviously that was a **bad** choice since you
            probably want the windows to overlap or you will miss cells.
            ``nchunk`` is only used if ``method='random'``, and is the
            number of random chunks we will draw. Additionally we have
            two parameters controlling how the local maxima are found.
            ``'max_wnd'`` controls the window size within which a single
            pixel will be choosen as local maxima. In order to capture
            cells with all sizes, we actually find local maximas with
            different window size and merge all of them, starting from
            **2** all the way up to ``'max_wnd'``. Hence ``'max_wnd'``
            should be the radius of the **largest** cell you want to
            detect. Finally in order to get rid of local maxima with
            very little fluctuation, we set a ``'diff_thres'`` which is
            the minimal fluorescent diffrence of a seed across
            ``frame``\ s. Since the linear scale of the raw data is
            preserved, we can set this threshold emprically.

            .. container:: alert alert-info

               The default values of \`seeds_init\` usually work fairly
               well for a dense region like CA1. If you are working with
               deep brain region with sparse cells, try to increase
               wnd_size and stp_size to make the following **seeds**
               cleaning steps faster and cleaner.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       seeds = seeds_init(Y, **param_seeds_init)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    We can visualize the seeds as points overlaid on top of the `max_proj` image. Each white dot is a seed and could potentially be the location of a cell. 

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            We can visualize the seeds as points overlaid on top of the
            ``max_proj`` image. Each white dot is a seed and could
            potentially be the location of a cell.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       visualize_seeds(max_proj, seeds)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## peak-noise-ratio refine

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: peak-noise-ratio
               refine\ `¶ <#peak-noise-ratio-refine>`__
               :name: peak-noise-ratio-refine

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    We further refine seeds based upon their temporal activity.  This requires that we separate our signal based upon frequency, and this also brings us to the most powerful and important aspect of this pipeline -- parameter exploring.  We are going to take a few example seeds and separate their activity based upon a few frequencies, and we will then view the results and select a frequency which we beleive best separates signal from noise. 

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    This will seem to be the most complicated chunk of code so far, but it is important to read through, since we will see similar things later and it is a very powerful piece of code that can help you visualize a lot. The basic idea is we run some function on a small subset of data using different parameters within a `for` loop, and after that visualize the results using `holoviews`. Note that interactive mode needs to be set as `True` for parameter exploring steps like this to work.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    The goal of this specific piece of code is to determine the "frequency" at which we can best seperate our signal from noise, which is an important parameter used at various places below. We will go line by line: First we create a `list` of frequencies we want to try out -- `noise_freq_list`. The "frequency" values here are a proportion of your **sampling rate**. Note that if you have temporally downsampled, the proportion here is relative to the downsampled rate. Then we randomly select 6 seeds from `seeds_gmm` and call them `example_seeds`, which in turn help us pull out the temporal traces from the movie `Y_flt`.  The traces of the `example_seeds` are assigned to `example_trace`. We then create an empty dictionary `smooth_dict` to store the resulting visualizations. After initializing these variables, we use a `for` loop to iterate through `noise_freq_list`, with one of the values from the list as `freq` during each iteration. Within the loop, we run `smooth_sig` twice on `example_trace` with the current `freq` we are testing out. The low-passed result is assigned to `trace_smth_low`, while the high-pass result is assigned to `trace_smth_high`. Then we make sure to actually carry-out the computation by calling the `compute` method on the resulting `DataArray`s. Finally, we turn the two traces into visualizations: we construct interactive line plots ([hv.Curve](http://holoviews.org/reference/elements/bokeh/Curve.html)s) from them and put them in a container called a [hv.HoloMap](http://holoviews.org/reference/containers/bokeh/HoloMap.html). Again if you are confused about how the visualization works, you can check out [the tutorial](http://holoviews.org/getting_started/Introduction.html). After that we store the whole visualization in `smooth_dict`, with the keys being the `freq` and values corresponding to the result of this iteration.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-info"> 

                                 .. code:: CodeMirror-line

                                    Here you can edit the values that you want to test in the noise_freq_list.

                                 .. code:: CodeMirror-line

                                    </div>

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            We further refine seeds based upon their temporal activity.
            This requires that we separate our signal based upon
            frequency, and this also brings us to the most powerful and
            important aspect of this pipeline -- parameter exploring. We
            are going to take a few example seeds and separate their
            activity based upon a few frequencies, and we will then view
            the results and select a frequency which we beleive best
            separates signal from noise.

            This will seem to be the most complicated chunk of code so
            far, but it is important to read through, since we will see
            similar things later and it is a very powerful piece of code
            that can help you visualize a lot. The basic idea is we run
            some function on a small subset of data using different
            parameters within a ``for`` loop, and after that visualize
            the results using ``holoviews``. Note that interactive mode
            needs to be set as ``True`` for parameter exploring steps
            like this to work.

            The goal of this specific piece of code is to determine the
            "frequency" at which we can best seperate our signal from
            noise, which is an important parameter used at various
            places below. We will go line by line: First we create a
            ``list`` of frequencies we want to try out --
            ``noise_freq_list``. The "frequency" values here are a
            proportion of your **sampling rate**. Note that if you have
            temporally downsampled, the proportion here is relative to
            the downsampled rate. Then we randomly select 6 seeds from
            ``seeds_gmm`` and call them ``example_seeds``, which in turn
            help us pull out the temporal traces from the movie
            ``Y_flt``. The traces of the ``example_seeds`` are assigned
            to ``example_trace``. We then create an empty dictionary
            ``smooth_dict`` to store the resulting visualizations. After
            initializing these variables, we use a ``for`` loop to
            iterate through ``noise_freq_list``, with one of the values
            from the list as ``freq`` during each iteration. Within the
            loop, we run ``smooth_sig`` twice on ``example_trace`` with
            the current ``freq`` we are testing out. The low-passed
            result is assigned to ``trace_smth_low``, while the
            high-pass result is assigned to ``trace_smth_high``. Then we
            make sure to actually carry-out the computation by calling
            the ``compute`` method on the resulting ``DataArray``\ s.
            Finally, we turn the two traces into visualizations: we
            construct interactive line plots
            (`hv.Curve <http://holoviews.org/reference/elements/bokeh/Curve.html>`__\ s)
            from them and put them in a container called a
            `hv.HoloMap <http://holoviews.org/reference/containers/bokeh/HoloMap.html>`__.
            Again if you are confused about how the visualization works,
            you can check out `the
            tutorial <http://holoviews.org/getting_started/Introduction.html>`__.
            After that we store the whole visualization in
            ``smooth_dict``, with the keys being the ``freq`` and values
            corresponding to the result of this iteration.

            .. container:: alert alert-info

               Here you can edit the values that you want to test in the
               noise_freq_list.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           noise_freq_list = [0.005, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3, 0.45]

                                    .. code:: CodeMirror-line

                                           example_seeds = seeds.sample(6, axis='rows')

                                    .. code:: CodeMirror-line

                                           example_trace = (Y_flt

                                    .. code:: CodeMirror-line

                                                            .sel(spatial=[tuple(hw) for hw in example_seeds[['height', 'width']].values])

                                    .. code:: CodeMirror-line

                                                            .assign_coords(spatial=np.arange(6))

                                    .. code:: CodeMirror-line

                                                            .rename(dict(spatial='seed')))

                                    .. code:: CodeMirror-line

                                           smooth_dict = dict()

                                    .. code:: CodeMirror-line

                                           for freq in noise_freq_list:

                                    .. code:: CodeMirror-line

                                               trace_smth_low = smooth_sig(example_trace, freq)

                                    .. code:: CodeMirror-line

                                               trace_smth_high = smooth_sig(example_trace, freq, btype='high')

                                    .. code:: CodeMirror-line

                                               trace_smth_low = trace_smth_low.compute()

                                    .. code:: CodeMirror-line

                                               trace_smth_high = trace_smth_high.compute()

                                    .. code:: CodeMirror-line

                                               hv_trace = hv.HoloMap({

                                    .. code:: CodeMirror-line

                                                   'signal': (hv.Dataset(trace_smth_low)

                                    .. code:: CodeMirror-line

                                                              .to(hv.Curve, kdims=['frame'])

                                    .. code:: CodeMirror-line

                                                              .opts(frame_width=300, aspect=2, ylabel='Signal (A.U.)')),

                                    .. code:: CodeMirror-line

                                                   'noise': (hv.Dataset(trace_smth_high)

                                    .. code:: CodeMirror-line

                                                             .to(hv.Curve, kdims=['frame'])

                                    .. code:: CodeMirror-line

                                                             .opts(frame_width=300, aspect=2, ylabel='Signal (A.U.)'))

                                    .. code:: CodeMirror-line

                                               }, kdims='trace').collate()

                                    .. code:: CodeMirror-line

                                               smooth_dict[freq] = hv_trace

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    After all the loops are done, we put together a holoviews plot (`hv.HoloMap`) from `smooth_dict`, and we specify that we want our traces to `overlay` each other along the `'trace'` dimension while being laid out along the `'spatial'` dimension. The result turns into a nicely animated interactive plot, from which we can determine the frequency that best separates noise and signal.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            After all the loops are done, we put together a holoviews
            plot (``hv.HoloMap``) from ``smooth_dict``, and we specify
            that we want our traces to ``overlay`` each other along the
            ``'trace'`` dimension while being laid out along the
            ``'spatial'`` dimension. The result turns into a nicely
            animated interactive plot, from which we can determine the
            frequency that best separates noise and signal.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           hv_res = (hv.HoloMap(smooth_dict, kdims=['noise_freq']).collate().opts(aspect=2)

                                    .. code:: CodeMirror-line

                                                     .overlay('trace').layout('seed').cols(3))

                                    .. code:: CodeMirror-line

                                           display(hv_res)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Having determined the frequency that best separates signal from noise, we move on the next step of seeds refining. Recall the parameters:

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ```python

                                 .. code:: CodeMirror-line

                                    param_pnr_refine = {

                                 .. code:: CodeMirror-line

                                        'noise_freq': 0.06,

                                 .. code:: CodeMirror-line

                                        'thres': 1,

                                 .. code:: CodeMirror-line

                                        'med_wnd': None}

                                 .. code:: CodeMirror-line

                                    ```

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    `pnr_refine` stands for "peak-to-noise ratio" refine. The "peak" and "noise" here are defined differently from before. First we seperate/filter the temporal signal for each seed based on frequency -- the signals composed of the lower half of the frequency are regarded as **real** signals, while the higher half of the frequencies is presumably **noise** ("half" being relative to [Nyquist frequency](https://en.wikipedia.org/wiki/Nyquist_frequency)). Then we take the peak-to-valley value (really just **max** minus **min**, or, [np.ptp](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.ptp.html)) for both the **real** signal and **noise** signal. Then, "peak-to-noise ratio" is the ratio between the `np.ptp` values of **real** and **noise** signals. So, the critical assumption here is that real cell activity is of lower frequency while noise is of a higher frequency, and they seperate at approximately half the Nyquist frequency, or, one-fourth of the sampling frequency of the video.  Moreover, we don't want those "seeds" whose **real** signals are buried in **noise**. If these assumptions does not suit your recordings - for example, if you have a really low sampling rate, or if your video are unavoidably noisy - consider skipping this step. The function `pnr_refine` takes in `varr` and `seeds` as its first two arguments; the `noise_freq` that best separates signal and noise, which hopefully has been determined from the previous cell; and `thres`, a threshold for "peak-to-noise ratios" below which seeds will be discarded. Pragmatically `thres=1` works fine and makes sense. You can also use `thres='auto'`, where a gaussian mixture model with 2 components will be run on the peak-to-noise ratios and seeds will be selected if they belong to the "higher" gaussian. `med_wnd` is the window size of the median filter that gets passed in as `size` in [`scipy.ndimage.filters.median_filter`](https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.filters.median_filter.html). This is only useful in rare cases where the signal of some seeds assume a huge change in baseline fluorescence and it is not desirable to keep such seeds. In this case the median-filtered signal is subtracted from the original signal to get rid of the artifact. In other cases `'med_wnd'` should be left to `None`.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    Now we can use the previous visualization result to pick the best frequency!

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ![pnr_param](img/pnr_param_v2.png)

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-success" role="alert">

                                 .. code:: CodeMirror-line

                                    What we are looking for here is the frequency that can seperate <strong>real</strong> signal and <strong>noise</strong> the best, which means the left panel in the example trace, with the `noise_freq` = 0.005, is not ideal. In the mean time, we also don't want the signal bands to be overly thick which is showing in the right panel with the `noise_freq` = 0.45. Thus, the middle trace with `noise_freq` = 0.05 best suits the needs! 

                                 .. code:: CodeMirror-line

                                    </div>

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-info">  

                                 .. code:: CodeMirror-line

                                    Now, say you already found your parameters, it's time now to pass them in! Either go back to initial parameters setting step and modify them there, or call the parameter here and change its value/s accordingly. 

                                 .. code:: CodeMirror-line

                                    </div>

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    For example, if you want to change `noise_freq` to 0.05, and start using median filter equal to 501 here:

                                 .. code:: CodeMirror-line

                                    ```python

                                 .. code:: CodeMirror-line

                                    param_pnr_refine['noise_freq'] = 0.05

                                 .. code:: CodeMirror-line

                                    param_pnr_refine['med_wnd'] = 501

                                 .. code:: CodeMirror-line

                                    ```

                                 .. code:: CodeMirror-line

                                    Finally, run the following code cell to further clean the seeds:

                                 .. code:: CodeMirror-line

                                    ​

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Having determined the frequency that best separates signal
            from noise, we move on the next step of seeds refining.
            Recall the parameters:

            ::

               param_pnr_refine = {
                   'noise_freq': 0.06,
                   'thres': 1,
                   'med_wnd': None}

            ``pnr_refine`` stands for "peak-to-noise ratio" refine. The
            "peak" and "noise" here are defined differently from before.
            First we seperate/filter the temporal signal for each seed
            based on frequency -- the signals composed of the lower half
            of the frequency are regarded as **real** signals, while the
            higher half of the frequencies is presumably **noise**
            ("half" being relative to `Nyquist
            frequency <https://en.wikipedia.org/wiki/Nyquist_frequency>`__).
            Then we take the peak-to-valley value (really just **max**
            minus **min**, or,
            `np.ptp <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.ptp.html>`__)
            for both the **real** signal and **noise** signal. Then,
            "peak-to-noise ratio" is the ratio between the ``np.ptp``
            values of **real** and **noise** signals. So, the critical
            assumption here is that real cell activity is of lower
            frequency while noise is of a higher frequency, and they
            seperate at approximately half the Nyquist frequency, or,
            one-fourth of the sampling frequency of the video. Moreover,
            we don't want those "seeds" whose **real** signals are
            buried in **noise**. If these assumptions does not suit your
            recordings - for example, if you have a really low sampling
            rate, or if your video are unavoidably noisy - consider
            skipping this step. The function ``pnr_refine`` takes in
            ``varr`` and ``seeds`` as its first two arguments; the
            ``noise_freq`` that best separates signal and noise, which
            hopefully has been determined from the previous cell; and
            ``thres``, a threshold for "peak-to-noise ratios" below
            which seeds will be discarded. Pragmatically ``thres=1``
            works fine and makes sense. You can also use
            ``thres='auto'``, where a gaussian mixture model with 2
            components will be run on the peak-to-noise ratios and seeds
            will be selected if they belong to the "higher" gaussian.
            ``med_wnd`` is the window size of the median filter that
            gets passed in as ``size`` in
            ```scipy.ndimage.filters.median_filter`` <https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.filters.median_filter.html>`__.
            This is only useful in rare cases where the signal of some
            seeds assume a huge change in baseline fluorescence and it
            is not desirable to keep such seeds. In this case the
            median-filtered signal is subtracted from the original
            signal to get rid of the artifact. In other cases
            ``'med_wnd'`` should be left to ``None``.

            Now we can use the previous visualization result to pick the
            best frequency!

            |pnr_param|

            .. container:: alert alert-success

               What we are looking for here is the frequency that can
               seperate **real** signal and **noise** the best, which
               means the left panel in the example trace, with the
               \`noise_freq\` = 0.005, is not ideal. In the mean time,
               we also don't want the signal bands to be overly thick
               which is showing in the right panel with the
               \`noise_freq\` = 0.45. Thus, the middle trace with
               \`noise_freq\` = 0.05 best suits the needs!

            .. container:: alert alert-info

               Now, say you already found your parameters, it's time now
               to pass them in! Either go back to initial parameters
               setting step and modify them there, or call the parameter
               here and change its value/s accordingly.

            For example, if you want to change ``noise_freq`` to 0.05,
            and start using median filter equal to 501 here:

            ::

               param_pnr_refine['noise_freq'] = 0.05
               param_pnr_refine['med_wnd'] = 501

            Finally, run the following code cell to further clean the
            seeds:

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       seeds, pnr, gmm = pnr_refine(Y_flt, seeds.copy(), **param_pnr_refine)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Here in the belowing code cell we will visualize the gmm fit, but **only** when you chose `thres='auto'` before. The x axis here is pnr ratio value, and the x value of the intersection of blue and red curve is the auto chose threshold, everything below this threshold will be seen as noise.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Here in the belowing code cell we will visualize the gmm
            fit, but **only** when you chose ``thres='auto'`` before.
            The x axis here is pnr ratio value, and the x value of the
            intersection of blue and red curve is the auto chose
            threshold, everything below this threshold will be seen as
            noise.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       if gmm:

                                    .. code:: CodeMirror-line

                                           display(visualize_gmm_fit(pnr, gmm, 100))

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    And again we can visualize seeds that's taken out during this step.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            And again we can visualize seeds that's taken out during
            this step.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       visualize_seeds(max_proj, seeds, 'mask_pnr')

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Still, white dots are accepted seeds and red dots are taken out. 

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-info"> 

                                 .. code:: CodeMirror-line

                                    if you see seeds that you believe should be cells have been taken out here, either skip this step or try lower the threshold a bit. You can also use the individual trace ploting method we discussed at the end of gmm_refine part to look into specific seed. 

                                 .. code:: CodeMirror-line

                                    </div>

                                 .. code:: CodeMirror-line

                                    ​

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Still, white dots are accepted seeds and red dots are taken
            out.

            .. container:: alert alert-info

               if you see seeds that you believe should be cells have
               been taken out here, either skip this step or try lower
               the threshold a bit. You can also use the individual
               trace ploting method we discussed at the end of
               gmm_refine part to look into specific seed.

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## ks refine

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: ks refine\ `¶ <#ks-refine>`__
               :name: ks-refine

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    `ks_refine` refines the seeds using [Kolmogorov-Smirnov test](https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test). Recall the parameters:

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ```python

                                 .. code:: CodeMirror-line

                                    param_ks_refine = {

                                 .. code:: CodeMirror-line

                                        'sig': 0.05}

                                 .. code:: CodeMirror-line

                                    ```

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    The idea is simple: if a seed corresponds to a cell, its fluorescence intensity across frames should be somewhat [bimodal](https://en.wikipedia.org/wiki/Multimodal_distribution), with a large normal distribution representing silence/little activity, and another peak representing when the seed/cell is active. Thus, we can carry out KS test on the intensity distribution of each seed, and keep only the seeds where the null hypothesis (that the fluoresence is simply a normal distribution) is rejected. `ks_refine` takes in `varr` and `seeds` as its first two arguments, then a `sig` which is the significance level at which the null hypothesis is rejected (defaulted to **0.05**). 

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-info"> 

                                 .. code:: CodeMirror-line

                                    In practice, we have found this step tends to take away real cells when video are very short (for example, the one that comes with this package under "./demo_movies"). This is likely because the number of "active" frames is too small. Feel free to skip this step if you encounter the same situation.

                                 .. code:: CodeMirror-line

                                        </div>

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            ``ks_refine`` refines the seeds using `Kolmogorov-Smirnov
            test <https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test>`__.
            Recall the parameters:

            ::

               param_ks_refine = {
                   'sig': 0.05}

            The idea is simple: if a seed corresponds to a cell, its
            fluorescence intensity across frames should be somewhat
            `bimodal <https://en.wikipedia.org/wiki/Multimodal_distribution>`__,
            with a large normal distribution representing silence/little
            activity, and another peak representing when the seed/cell
            is active. Thus, we can carry out KS test on the intensity
            distribution of each seed, and keep only the seeds where the
            null hypothesis (that the fluoresence is simply a normal
            distribution) is rejected. ``ks_refine`` takes in ``varr``
            and ``seeds`` as its first two arguments, then a ``sig``
            which is the significance level at which the null hypothesis
            is rejected (defaulted to **0.05**).

            .. container:: alert alert-info

               In practice, we have found this step tends to take away
               real cells when video are very short (for example, the
               one that comes with this package under "./demo_movies").
               This is likely because the number of "active" frames is
               too small. Feel free to skip this step if you encounter
               the same situation.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       seeds = ks_refine(Y_flt, seeds[seeds['mask_pnr']], **param_ks_refine)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       visualize_seeds(max_proj, seeds, 'mask_ks')

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## merge seeds

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: merge seeds\ `¶ <#merge-seeds>`__
               :name: merge-seeds

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    At this point, much of our refined seeds likely reflect the position of an actual cell.  However, we are likely to still have multiple seeds per cell, which we want to avoid.  Here we discard redudant seeds through a process of merging.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    Recall the parameters:

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ```python

                                 .. code:: CodeMirror-line

                                    param_seeds_merge = {

                                 .. code:: CodeMirror-line

                                        'thres_dist': 5,

                                 .. code:: CodeMirror-line

                                        'thres_corr': 0.7,

                                 .. code:: CodeMirror-line

                                        'noise_freq': .06'}

                                 .. code:: CodeMirror-line

                                    ```

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    The function `seeds_merge` attempts to merge seeds together which potentially come from the same cell, based upon their spatial distance and temporal correlation. Specifically, `thres_dist` is the threshold for euclidean distance between pairs of seeds, in pixels, and `thres_corr` is the threshold for pearson correlation between pairs of seeds. In addition, it's very beneficial to smooth the signals before running the correlation, and again `noise_freq` determines how smoothing should be done. In addition to feeding in a number, such as the noise frequency you defined earlier during `seeds_refine_pnr`, you can also use `noise_freq='envelope'`.  When `noise_freq='envelope'`, a hilbert transform will be run on the temporal traces of each seed and the correlation will be calculated on the envelope signal. Any pair of seeds that are within `thres_dist` **and** has a correlation higher than `thres_corr` will be merged together, such that only the seed with maximum intensity in the max projection of the video will be kept. Thus `thres_dist` should be the expected size of cells and `thres_corr` should be relatively high to avoid over-merging.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-info"> 

                                 .. code:: CodeMirror-line

                                    Potentially we could pick out multiple seeds that are actually within one cell, but we want to avoid that as much as possible to have a clean start for CNMF later, you can try lower the thres_corr or raise up the thres_dist to merge more cells. Ideally, you want to see only one accepted seed (white dot) within each cell. 

                                 .. code:: CodeMirror-line

                                    </div>

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            At this point, much of our refined seeds likely reflect the
            position of an actual cell. However, we are likely to still
            have multiple seeds per cell, which we want to avoid. Here
            we discard redudant seeds through a process of merging.

            Recall the parameters:

            ::

               param_seeds_merge = {
                   'thres_dist': 5,
                   'thres_corr': 0.7,
                   'noise_freq': .06'}

            The function ``seeds_merge`` attempts to merge seeds
            together which potentially come from the same cell, based
            upon their spatial distance and temporal correlation.
            Specifically, ``thres_dist`` is the threshold for euclidean
            distance between pairs of seeds, in pixels, and
            ``thres_corr`` is the threshold for pearson correlation
            between pairs of seeds. In addition, it's very beneficial to
            smooth the signals before running the correlation, and again
            ``noise_freq`` determines how smoothing should be done. In
            addition to feeding in a number, such as the noise frequency
            you defined earlier during ``seeds_refine_pnr``, you can
            also use ``noise_freq='envelope'``. When
            ``noise_freq='envelope'``, a hilbert transform will be run
            on the temporal traces of each seed and the correlation will
            be calculated on the envelope signal. Any pair of seeds that
            are within ``thres_dist`` **and** has a correlation higher
            than ``thres_corr`` will be merged together, such that only
            the seed with maximum intensity in the max projection of the
            video will be kept. Thus ``thres_dist`` should be the
            expected size of cells and ``thres_corr`` should be
            relatively high to avoid over-merging.

            .. container:: alert alert-info

               Potentially we could pick out multiple seeds that are
               actually within one cell, but we want to avoid that as
               much as possible to have a clean start for CNMF later,
               you can try lower the thres_corr or raise up the
               thres_dist to merge more cells. Ideally, you want to see
               only one accepted seed (white dot) within each cell.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       seeds_final = seeds[seeds['mask_ks']].reset_index(drop=True)

                                    .. code:: CodeMirror-line

                                       seeds_mrg = seeds_merge(Y_flt, seeds_final, **param_seeds_merge)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       visualize_seeds(max_proj, seeds_mrg, 'mask_mrg')

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## initialize spatial and temporal matrices from seeds

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: initialize spatial and temporal matrices from
               seeds\ `¶ <#initialize-spatial-and-temporal-matrices-from-seeds>`__
               :name: initialize-spatial-and-temporal-matrices-from-seeds

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Up till now, the seeds we have are only one-pixel dots. In order to kick start CNMF we need something more like the spatial footprint (`A`) and temporal activities (`C`) of real cells. Thus we need to `initilalize` `A` and `C` from the seeds we have (`seeds_mrg`). Recall the parameters:

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ```python

                                 .. code:: CodeMirror-line

                                    param_initialize = {

                                 .. code:: CodeMirror-line

                                        'thres_corr': 0.8,

                                 .. code:: CodeMirror-line

                                        'wnd': 10,

                                 .. code:: CodeMirror-line

                                        'noise_freq': .06

                                 .. code:: CodeMirror-line

                                    }

                                 .. code:: CodeMirror-line

                                    ```

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    To obtain the initial spatial matrix `A`, for each seed, we simply use a Pearson correlation between the seed and surrounding pixels. Apparantly cacluating correlation with all other pixels for every seed is time-consuming and unnecessary. `'wnd'` controls the window size for calculating the correlation, and thus is the maximum possible size of any spatial footprint in the initial spatial matrix. At the same time we do not want pixels with low correlation value to influence our estimation of temporal signals, thus a `'thres_corr'` is also implemented where only pixels with correlation above this threshold are kept. After generating `A`, for each seed, we calculate a weighted average of pixels around the seed, where the weight are the initial spatial footprints in `A` we just generated. We use this weighted average as the initial estimation of temporal activities for each units in `C`. Finally, we need two more terms: `b` and `f`, representing the spatial footprint and temporal dynamics of the **background**, respectively. Since usually the backgrounds are already removed at this stage, we provide a very simple estimation of remaining background -- we simply mask `Y` with the spatial footprints of units in `A`, that is, we only keep pixels that does not appear in the spatial foorprints of any units. We calculate a mean projection across `frame`s and use as `b`, and we calculate mean fluorescence along `frame`s and use as `f`.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Up till now, the seeds we have are only one-pixel dots. In
            order to kick start CNMF we need something more like the
            spatial footprint (``A``) and temporal activities (``C``) of
            real cells. Thus we need to ``initilalize`` ``A`` and ``C``
            from the seeds we have (``seeds_mrg``). Recall the
            parameters:

            ::

               param_initialize = {
                   'thres_corr': 0.8,
                   'wnd': 10,
                   'noise_freq': .06
               }

            To obtain the initial spatial matrix ``A``, for each seed,
            we simply use a Pearson correlation between the seed and
            surrounding pixels. Apparantly cacluating correlation with
            all other pixels for every seed is time-consuming and
            unnecessary. ``'wnd'`` controls the window size for
            calculating the correlation, and thus is the maximum
            possible size of any spatial footprint in the initial
            spatial matrix. At the same time we do not want pixels with
            low correlation value to influence our estimation of
            temporal signals, thus a ``'thres_corr'`` is also
            implemented where only pixels with correlation above this
            threshold are kept. After generating ``A``, for each seed,
            we calculate a weighted average of pixels around the seed,
            where the weight are the initial spatial footprints in ``A``
            we just generated. We use this weighted average as the
            initial estimation of temporal activities for each units in
            ``C``. Finally, we need two more terms: ``b`` and ``f``,
            representing the spatial footprint and temporal dynamics of
            the **background**, respectively. Since usually the
            backgrounds are already removed at this stage, we provide a
            very simple estimation of remaining background -- we simply
            mask ``Y`` with the spatial footprints of units in ``A``,
            that is, we only keep pixels that does not appear in the
            spatial foorprints of any units. We calculate a mean
            projection across ``frame``\ s and use as ``b``, and we
            calculate mean fluorescence along ``frame``\ s and use as
            ``f``.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       A, C, b, f = initialize(Y, seeds_mrg[seeds_mrg['mask_mrg']], **param_initialize)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Finally we visualize the result of our initialization by plotting a projection of the spatial matrix `A`, a raster of the temporal matrix `C`, as well as background terms `b` and `f`.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Finally we visualize the result of our initialization by
            plotting a projection of the spatial matrix ``A``, a raster
            of the temporal matrix ``C``, as well as background terms
            ``b`` and ``f``.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       im_opts = dict(frame_width=500, aspect=A.sizes['width']/A.sizes['height'], cmap='Viridis', colorbar=True)

                                    .. code:: CodeMirror-line

                                       cr_opts = dict(frame_width=750, aspect=1.5*A.sizes['width']/A.sizes['height'])

                                    .. code:: CodeMirror-line

                                       (regrid(hv.Image(A.sum('unit_id').rename('A').compute(), kdims=['width', 'height'])).opts(**im_opts)

                                    .. code:: CodeMirror-line

                                        + regrid(hv.Image(C.rename('C').compute(), kdims=['frame', 'unit_id'])).opts(cmap='viridis', colorbar=True, **cr_opts)

                                    .. code:: CodeMirror-line

                                         + regrid(hv.Image(b.rename('b').compute(), kdims=['width', 'height'])).opts(**im_opts)

                                    .. code:: CodeMirror-line

                                        + datashade(hv.Curve(f.rename('f').compute(), kdims=['frame']), min_alpha=200).opts(**cr_opts)

                                    .. code:: CodeMirror-line

                                       ).cols(2)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## save results

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: save results\ `¶ <#save-results>`__
               :name: save-results

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Then we save the results in the dataset. Note here that we change the name of a dimension by writing `rename(unit_id='unit_id_init')`. The name of this dimension is changed as a precaution, since the size of the dimension `unit_id` will likely change in the next section **CNMF**. During CNMF, most likely units will be merged, and there will be conflicts if we save other variables with dimension `unit_id` that have different coordinates.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Then we save the results in the dataset. Note here that we
            change the name of a dimension by writing
            ``rename(unit_id='unit_id_init')``. The name of this
            dimension is changed as a precaution, since the size of the
            dimension ``unit_id`` will likely change in the next section
            **CNMF**. During CNMF, most likely units will be merged, and
            there will be conflicts if we save other variables with
            dimension ``unit_id`` that have different coordinates.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       A = save_minian(A.rename('A_init').rename(unit_id='unit_id_init'), **param_save_minian)

                                    .. code:: CodeMirror-line

                                       C = save_minian(C.rename('C_init').rename(unit_id='unit_id_init'), **param_save_minian)

                                    .. code:: CodeMirror-line

                                       b = save_minian(b.rename('b_init'), **param_save_minian)

                                    .. code:: CodeMirror-line

                                       f = save_minian(f.rename('f_init'), **param_save_minian)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    # CNMF

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: CNMF\ `¶ <#CNMF>`__
               :name: CNMF

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    This section assume you already have some knowledge about using CNMF as a method of extracting neural activities from video. If not, it is recommended that you read [the paper](https://www.sciencedirect.com/science/article/pii/S0896627315010843), to get a broad understanding of the problem and proposed solution.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    As a quick reminder, here is the essential idea of CNMF: We believe our movie, `Y`, with dimensions `height`, `width` and `frame`, can be written in (and thus broken down as) the following equation: $$\mathbf{Y} = \mathbf{A} \cdot \mathbf{C} + \mathbf{b} \cdot \mathbf{f} + \epsilon$$ where `A` is the spatial footprint of each unit, with dimension `height`, `width` and `unit_id`; `C` is the temporal activities of each unit, with dimension `unit_id` and `frame`; `b` and `f` are the spatial footprint and temporal activities of some background, respectively; and $\epsilon$ is the noise. Note that strictly speaking, matrix multiplication is usually only defined for two dimensional matrices, but our `A` here has three dimensions, so in fact we are taking the [tensor product](https://en.wikipedia.org/wiki/Tensor_product) of `A` and `C`, reducing the dimension `unit_id`. This might seem to complicate things (compared to just treating `height` and `width` as one flattened `spatial` dimension), but it ends up making some sense. When you take a dot product of any two "matrices" on a certain **dimension**, all that is happening is a **product** followed by a **sum** -- you take the product for all pairs of matching numbers coming from the two "matrices", where "match" is defined by their index along said dimension, and then you take the sum of all those products along the dimension. Thus when we take the tensor product of `A` and `C`, we are actually multiplying all those numbers in dimension `height`, `width` and `frame`, matched by `unit_id`, and then take the sum. Conceptually, for each unit, we are weighting the spatial footprint (`height` and `width`) by the fluorecense of that unit on given `frame`, which is the **product**, and then we are overlaying all units together, which is the **sum**. With that, the equation above is trying to say that our movie is made up of a weighted sum of the spatial footprint and temporal activities of all units, plus some background and noise.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    Now, there is another rule about `C` that separates it from background and noise, and saves it from being just some random matrix that happens to fit well with the data (`Y`) without having any biological meaning. This rule is the second essential idea of CNMF: each "row" of `C`, which is the temporal trace for each unit, should be described as an [autoregressive process](https://en.wikipedia.org/wiki/Autoregressive_model) (AR process), with a parameter `p` defining the **order** of the AR process: $$ c(t) = \sum_{i=0}^{p}\gamma_i c(t-i) + s(t) + \epsilon$$ where $c(t)$ is the calcium concentration at time (`frame`) $t$, $s(t)$ is spike/firing rate at time $t$ (what we actually care about), and $\epsilon$ is noise. Basically, this equation is trying to say that at any given time $t$, the calcium concentration at that moment $c(t)$ depends on the spike at that moment $s(t)$, as well as its own history up to `p` time-steps back $c(t-i)$, scaled by some parameters $\gamma_i$s, plus some noise $\epsilon$. Another intuition of this equation comes from looking at different `p`s: when `p=0`, the calcium concentration is an exact copy of the spiking activities, which is probably not true; when `p=1`, the calcium concentration has an instant rise in response to a spike followed by an exponential decay; when `p=2`, calcium concentration has some rise time following a spike and an exponential decay; when `p>2`, more convoluted waveforms start to emerge.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    With all this in mind, CNMF tries to find the spatial matrix (`A`) and temporal activity (`C`) (along with `b` and `f`) that best describe `Y`. There are a few more important practical concerns: Firstly we cannot solve this problem in one shot -- we need to iteratively and separately update `A` and `C` to approach the true solution -- and  we need something to start with (that is what **initilization** section is about). Surprisingly often times 2 iterative steps after our initialization seem to give good enough results, but you can always add more iterations (and you should be able to easily do that after reading the comments). Secondly, by intuition you may define "best describe `Y`" as the results that minimize the noise $\epsilon$ (or residuals, if you will). However we have to control for the [sparsity](https://en.wikipedia.org/wiki/Sparse_matrix) of our model as well, since we do not want every little random pixel that happens to correlate with a cell to be counted as part of the spatial footprint of the cell (non-sparse `A`), nor do we want a tiny spike at every frame trying to explain every noisy peak we observe (non-sparse `C`). Thus, the balance between fidelity (minimizing error) and sparsity (minimizing non-zero entries) is an important idea for both the spatial and temporal update.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            This section assume you already have some knowledge about
            using CNMF as a method of extracting neural activities from
            video. If not, it is recommended that you read `the
            paper <https://www.sciencedirect.com/science/article/pii/S0896627315010843>`__,
            to get a broad understanding of the problem and proposed
            solution.

            As a quick reminder, here is the essential idea of CNMF: We
            believe our movie, ``Y``, with dimensions ``height``,
            ``width`` and ``frame``, can be written in (and thus broken
            down as) the following equation:

            .. container:: MathJax_Display

               :presentation:`𝐘=𝐀⋅𝐂+𝐛⋅𝐟+𝜖\ 

               .. math:: \mathbf{Y} = \mathbf{A} \cdot \mathbf{C} + \mathbf{b} \cdot \mathbf{f} + \epsilon

               `

            .. math:: \mathbf{Y} = \mathbf{A} \cdot \mathbf{C} + \mathbf{b} \cdot \mathbf{f} + \epsilon

            \ where ``A`` is the spatial footprint of each unit, with
            dimension ``height``, ``width`` and ``unit_id``; ``C`` is
            the temporal activities of each unit, with dimension
            ``unit_id`` and ``frame``; ``b`` and ``f`` are the spatial
            footprint and temporal activities of some background,
            respectively; and
            :presentation:`𝜖\ :math:`\epsilon``\ :math:`\epsilon` is the
            noise. Note that strictly speaking, matrix multiplication is
            usually only defined for two dimensional matrices, but our
            ``A`` here has three dimensions, so in fact we are taking
            the `tensor
            product <https://en.wikipedia.org/wiki/Tensor_product>`__ of
            ``A`` and ``C``, reducing the dimension ``unit_id``. This
            might seem to complicate things (compared to just treating
            ``height`` and ``width`` as one flattened ``spatial``
            dimension), but it ends up making some sense. When you take
            a dot product of any two "matrices" on a certain
            **dimension**, all that is happening is a **product**
            followed by a **sum** -- you take the product for all pairs
            of matching numbers coming from the two "matrices", where
            "match" is defined by their index along said dimension, and
            then you take the sum of all those products along the
            dimension. Thus when we take the tensor product of ``A`` and
            ``C``, we are actually multiplying all those numbers in
            dimension ``height``, ``width`` and ``frame``, matched by
            ``unit_id``, and then take the sum. Conceptually, for each
            unit, we are weighting the spatial footprint (``height`` and
            ``width``) by the fluorecense of that unit on given
            ``frame``, which is the **product**, and then we are
            overlaying all units together, which is the **sum**. With
            that, the equation above is trying to say that our movie is
            made up of a weighted sum of the spatial footprint and
            temporal activities of all units, plus some background and
            noise.
            Now, there is another rule about ``C`` that separates it
            from background and noise, and saves it from being just some
            random matrix that happens to fit well with the data (``Y``)
            without having any biological meaning. This rule is the
            second essential idea of CNMF: each "row" of ``C``, which is
            the temporal trace for each unit, should be described as an
            `autoregressive
            process <https://en.wikipedia.org/wiki/Autoregressive_model>`__
            (AR process), with a parameter ``p`` defining the **order**
            of the AR process:

            .. container:: MathJax_Display

               :presentation:`𝑐(𝑡)=∑𝑖=0𝑝𝛾𝑖𝑐(𝑡−𝑖)+𝑠(𝑡)+𝜖\ 

               .. math:: c(t) = \sum\limits_{i = 0}^{p}\gamma_{i}c(t - i) + s(t) + \epsilon

               `

            .. math::  c(t) = \sum_{i=0}^{p}\gamma_i c(t-i) + s(t) + \epsilon

            \ where :presentation:`𝑐(𝑡)\ :math:`c(t)``\ :math:`c(t)` is
            the calcium concentration at time (``frame``)
            :presentation:`𝑡\ :math:`t``\ :math:`t`,
            :presentation:`𝑠(𝑡)\ :math:`s(t)``\ :math:`s(t)` is
            spike/firing rate at time
            :presentation:`𝑡\ :math:`t``\ :math:`t` (what we actually
            care about), and
            :presentation:`𝜖\ :math:`\epsilon``\ :math:`\epsilon` is
            noise. Basically, this equation is trying to say that at any
            given time :presentation:`𝑡\ :math:`t``\ :math:`t`, the
            calcium concentration at that moment
            :presentation:`𝑐(𝑡)\ :math:`c(t)``\ :math:`c(t)` depends on
            the spike at that moment
            :presentation:`𝑠(𝑡)\ :math:`s(t)``\ :math:`s(t)`, as well as
            its own history up to ``p`` time-steps back
            :presentation:`𝑐(𝑡−𝑖)\ :math:`c(t - i)``\ :math:`c(t-i)`,
            scaled by some parameters
            :presentation:`𝛾𝑖\ :math:`\gamma_{i}``\ :math:`\gamma_i`\ s,
            plus some noise
            :presentation:`𝜖\ :math:`\epsilon``\ :math:`\epsilon`.
            Another intuition of this equation comes from looking at
            different ``p``\ s: when ``p=0``, the calcium concentration
            is an exact copy of the spiking activities, which is
            probably not true; when ``p=1``, the calcium concentration
            has an instant rise in response to a spike followed by an
            exponential decay; when ``p=2``, calcium concentration has
            some rise time following a spike and an exponential decay;
            when ``p>2``, more convoluted waveforms start to emerge.
            With all this in mind, CNMF tries to find the spatial matrix
            (``A``) and temporal activity (``C``) (along with ``b`` and
            ``f``) that best describe ``Y``. There are a few more
            important practical concerns: Firstly we cannot solve this
            problem in one shot -- we need to iteratively and separately
            update ``A`` and ``C`` to approach the true solution -- and
            we need something to start with (that is what
            **initilization** section is about). Surprisingly often
            times 2 iterative steps after our initialization seem to
            give good enough results, but you can always add more
            iterations (and you should be able to easily do that after
            reading the comments). Secondly, by intuition you may define
            "best describe ``Y``" as the results that minimize the noise
            :presentation:`𝜖\ :math:`\epsilon``\ :math:`\epsilon` (or
            residuals, if you will). However we have to control for the
            `sparsity <https://en.wikipedia.org/wiki/Sparse_matrix>`__
            of our model as well, since we do not want every little
            random pixel that happens to correlate with a cell to be
            counted as part of the spatial footprint of the cell
            (non-sparse ``A``), nor do we want a tiny spike at every
            frame trying to explain every noisy peak we observe
            (non-sparse ``C``). Thus, the balance between fidelity
            (minimizing error) and sparsity (minimizing non-zero
            entries) is an important idea for both the spatial and
            temporal update.

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## loading data

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: loading data\ `¶ <#loading-data>`__
               :name: loading-data

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    First we load in our data from previous steps. `'unit_id'` is renamed as a precaution, mentioned at the end of the **initialization** section.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            First we load in our data from previous steps. ``'unit_id'``
            is renamed as a precaution, mentioned at the end of the
            **initialization** section.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       minian = open_minian(dpath,

                                    .. code:: CodeMirror-line

                                                            fname=param_save_minian['fname'],

                                    .. code:: CodeMirror-line

                                                            backend=param_save_minian['backend'])

                                    .. code:: CodeMirror-line

                                       Y = minian['Y'].astype(np.float)

                                    .. code:: CodeMirror-line

                                       A_init = minian['A_init'].rename(unit_id_init='unit_id')

                                    .. code:: CodeMirror-line

                                       C_init = minian['C_init'].rename(unit_id_init='unit_id')

                                    .. code:: CodeMirror-line

                                       b_init = minian['b_init']

                                    .. code:: CodeMirror-line

                                       f_init = minian['f_init']

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## estimate spatial noise

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: estimate spatial
               noise\ `¶ <#estimate-spatial-noise>`__
               :name: estimate-spatial-noise

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Prior to performing CNMF's first spatial update, we need to get a sense of how much noise is expected, which we will then feed into CNMF. To do so, we compute an fft-transform for every pixel independently, and estimate noise from its [power spectral density](https://en.wikipedia.org/wiki/Spectral_density). Recall the parameters:

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ```python

                                 .. code:: CodeMirror-line

                                    param_get_noise = {

                                 .. code:: CodeMirror-line

                                        'noise_range': (0.06, 0.5),

                                 .. code:: CodeMirror-line

                                        'noise_method': 'logmexp'}

                                 .. code:: CodeMirror-line

                                    ```

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    Note that the number in `noise_range` is relative to the sampling frequency, so **0.5** actually represents the Nyquist frequency and is the highest you can go as far as fft is concerned. Thus **(0.25, 0.5)** is the higher frequency half of the signal. After choosing `noise_range`, we have to decide how to collapse across different frequencies to get a single number of noise power for each pixel. Three `noise_method`s are availabe: `noise_method='mean'` and `noise_method='median'` will use the mean and median across all `freq` as the estimation of noise for each pixel. `noise_method='logmexp'`is a bit more complicated -- the equation is as follows: $sn = \exp( \operatorname{\mathbb{E}}[\log psd] )$ where $\exp$ is the [exponential function](Exponential_function), $\operatorname{\mathbb{E}}$ is the [expectation operator](https://en.wikipedia.org/wiki/Expected_value) (mean), $\log$ is [natural logarithm](https://en.wikipedia.org/wiki/Natural_logarithm), $psd$ is the spectral density of noise for any pixel, and $sn$ is the resulting estimation of noise power. It is recommended to keep `noise_method='logmexp'` since this is the default behavior of the [CaImAn](https://github.com/flatironinstitute/CaImAn) package.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-info">  

                                 .. code:: CodeMirror-line

                                    In order to define the lower bound of <strong>noise_range</strong> (the upper bound can be left equal to 0.5), examine the PSD plot and define the frequency value (again, this is actually a proportion of your sampling rate), where power has dropped off across all pixels (i.e., <strong>spatial</strong>).

                                 .. code:: CodeMirror-line

                                    </div>

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Prior to performing CNMF's first spatial update, we need to
            get a sense of how much noise is expected, which we will
            then feed into CNMF. To do so, we compute an fft-transform
            for every pixel independently, and estimate noise from its
            `power spectral
            density <https://en.wikipedia.org/wiki/Spectral_density>`__.
            Recall the parameters:

            ::

               param_get_noise = {
                   'noise_range': (0.06, 0.5),
                   'noise_method': 'logmexp'}

            Note that the number in ``noise_range`` is relative to the
            sampling frequency, so **0.5** actually represents the
            Nyquist frequency and is the highest you can go as far as
            fft is concerned. Thus **(0.25, 0.5)** is the higher
            frequency half of the signal. After choosing
            ``noise_range``, we have to decide how to collapse across
            different frequencies to get a single number of noise power
            for each pixel. Three ``noise_method``\ s are availabe:
            ``noise_method='mean'`` and ``noise_method='median'`` will
            use the mean and median across all ``freq`` as the
            estimation of noise for each pixel.
            ``noise_method='logmexp'``\ is a bit more complicated -- the
            equation is as follows:
            :presentation:`𝑠𝑛=exp(𝔼[log𝑝𝑠𝑑])\ :math:`sn = \exp(\mathbb{E}\lbrack\log psd\rbrack)``\ :math:`sn = \exp( \operatorname{\mathbb{E}}[\log psd] )`
            where :presentation:`exp\ :math:`\exp``\ :math:`\exp` is the
            `exponential function <Exponential_function>`__,
            :presentation:`𝔼\ :math:`\mathbb{E}``\ :math:`\operatorname{\mathbb{E}}`
            is the `expectation
            operator <https://en.wikipedia.org/wiki/Expected_value>`__
            (mean), :presentation:`log\ :math:`\log``\ :math:`\log` is
            `natural
            logarithm <https://en.wikipedia.org/wiki/Natural_logarithm>`__,
            :presentation:`𝑝𝑠𝑑\ :math:`psd``\ :math:`psd` is the
            spectral density of noise for any pixel, and
            :presentation:`𝑠𝑛\ :math:`sn``\ :math:`sn` is the resulting
            estimation of noise power. It is recommended to keep
            ``noise_method='logmexp'`` since this is the default
            behavior of the
            `CaImAn <https://github.com/flatironinstitute/CaImAn>`__
            package.

            .. container:: alert alert-info

               In order to define the lower bound of **noise_range**
               (the upper bound can be left equal to 0.5), examine the
               PSD plot and define the frequency value (again, this is
               actually a proportion of your sampling rate), where power
               has dropped off across all pixels (i.e., **spatial**).

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       sn_spatial = get_noise_fft(Y, **param_get_noise).persist()

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## test parameters for spatial update

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: test parameters for spatial
               update\ `¶ <#test-parameters-for-spatial-update>`__
               :name: test-parameters-for-spatial-update

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    We will now do some parameter exploring before actually performing the first spatial update.  We do this because we do not want to do a 10-minute spatial update only to find the selected parameters do not produce nice results. For parameter exploration, we will analyze a very small subset of data so that we can quickly examine the influence of various paramater values. Here, we randomly select 10 units from `A_init.coords['unit_id']` with the help of [`np.random.choice`](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.choice.html).

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            We will now do some parameter exploring before actually
            performing the first spatial update. We do this because we
            do not want to do a 10-minute spatial update only to find
            the selected parameters do not produce nice results. For
            parameter exploration, we will analyze a very small subset
            of data so that we can quickly examine the influence of
            various paramater values. Here, we randomly select 10 units
            from ``A_init.coords['unit_id']`` with the help of
            ```np.random.choice`` <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.choice.html>`__.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           units = np.random.choice(A_init.coords['unit_id'], 10, replace=False)

                                    .. code:: CodeMirror-line

                                           units.sort()

                                    .. code:: CodeMirror-line

                                           A_sub = A_init.sel(unit_id=units).persist()

                                    .. code:: CodeMirror-line

                                           C_sub = C_init.sel(unit_id=units).persist()

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Here, we again perform parameter exploration using a `for` loop and visualization with help of `dict` and `holoviews`, only this time we use a convenient function, `visualize_spatial_update` from `minian`, to handle all the visualization details. For now, the sparseness penalty (`sparse_penal`) is only one parameter in `update_spatial` that we are interested in playing with, but there is nothing stopping you from adding more. Discussion of all the parameters for `update_spatial` will follow soon.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-info">  

                                 .. code:: CodeMirror-line

                                    Here, you can simply <strong>add</strong> the values that you want to test or <strong>delete</strong> the values you are not interested in from <strong>spar_ls</strong>. Pragmatically, the range of 0.05 to 1  is reasonable.

                                 .. code:: CodeMirror-line

                                    </div>

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Here, we again perform parameter exploration using a ``for``
            loop and visualization with help of ``dict`` and
            ``holoviews``, only this time we use a convenient function,
            ``visualize_spatial_update`` from ``minian``, to handle all
            the visualization details. For now, the sparseness penalty
            (``sparse_penal``) is only one parameter in
            ``update_spatial`` that we are interested in playing with,
            but there is nothing stopping you from adding more.
            Discussion of all the parameters for ``update_spatial`` will
            follow soon.

            .. container:: alert alert-info

               Here, you can simply **add** the values that you want to
               test or **delete** the values you are not interested in
               from **spar_ls**. Pragmatically, the range of 0.05 to 1
               is reasonable.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           sprs_ls = [0.05, 0.1, 0.5]

                                    .. code:: CodeMirror-line

                                           A_dict = dict()

                                    .. code:: CodeMirror-line

                                           C_dict = dict()

                                    .. code:: CodeMirror-line

                                           for cur_sprs in sprs_ls:

                                    .. code:: CodeMirror-line

                                               cur_A, cur_b, cur_C, cur_f = update_spatial(

                                    .. code:: CodeMirror-line

                                                   Y, A_sub, b_init, C_sub, f_init,

                                    .. code:: CodeMirror-line

                                                   sn_spatial, dl_wnd=param_first_spatial['dl_wnd'], sparse_penal=cur_sprs)

                                    .. code:: CodeMirror-line

                                               if cur_A.sizes['unit_id']:

                                    .. code:: CodeMirror-line

                                                   A_dict[cur_sprs] = cur_A.compute()

                                    .. code:: CodeMirror-line

                                                   C_dict[cur_sprs] = cur_C.compute()

                                    .. code:: CodeMirror-line

                                           hv_res = visualize_spatial_update(A_dict, C_dict, kdims=['sparse penalty'])

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Finally, we actually plot the visualization `hv_res`. What you should expect here will be explained later along with what `sparse_penal` actually does.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Finally, we actually plot the visualization ``hv_res``. What
            you should expect here will be explained later along with
            what ``sparse_penal`` actually does.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           display(hv_res)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## first spatial update

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: first spatial
               update\ `¶ <#first-spatial-update>`__
               :name: first-spatial-update

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Here is the idea behind `update_spatial`. Recall the parameters:

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ```python

                                 .. code:: CodeMirror-line

                                    param_first_spatial = {

                                 .. code:: CodeMirror-line

                                        'dl_wnd': 10,

                                 .. code:: CodeMirror-line

                                        'sparse_penal': 0.01,

                                 .. code:: CodeMirror-line

                                        'update_background': True,

                                 .. code:: CodeMirror-line

                                        'normalize': True,

                                 .. code:: CodeMirror-line

                                        'zero_thres': 'eps'}

                                 .. code:: CodeMirror-line

                                    ```

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    To reiterate, the big picture is that given the data (`Y`) and our units' activity (`C`) from previous the update (which is `C_init`), we want to find the spatial footprints (`A`) such that 1. the **error** `Y - A.dot(C, 'unit_id')` is as small as possible, and 2. the [**l1-norm**](http://mathworld.wolfram.com/L1-Norm.html) of `A` is as small as possible. Here the **l1-norm** is a proxy to control for the sparsity of `A`. Ideally to promote sparsity we want to control for the number of non-zero entries in `A`, which is the [l0-norm](https://en.wikipedia.org/wiki/Lp_space#When_p_=_0). However optimizing for the l0-norm is typically [computationally hard to do](https://stats.stackexchange.com/questions/269298/why-do-we-only-see-l-1-and-l-2-regularization-but-not-other-norms), and it is usually good enough to use **l1-norm** instead as a proxy.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    Now, in theory we want to update every entry in  `A` iteratively with the above two goals in mind. However, updating that amount of numbers in `A` is still computationally very demanding, and it is much better if we can breakdown our big problem into smaller chunks that can be parallelized (making things much faster). **CNMF** is all about solving the issues caused by overlapping neurons, so it is best to keep the dependency across units (along dimension `unit_id`) and update these entries together. However, it should be fine to treat each pixel as independent and update different pixels separately (in parallel). Thus, our new, "smaller" problem is: for each pixel, find the corresponding pixel in `A`, across all `unit_id`, that give us smallest **l1-norm** as well as smallest **error** when multiplied by `C`. In equation form, this is:

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    $$\begin{equation*}

                                 .. code:: CodeMirror-line

                                    \begin{aligned}

                                 .. code:: CodeMirror-line

                                    & \underset{A_{ij}}{\text{minimize}}

                                 .. code:: CodeMirror-line

                                    & & \left \lVert Y_{ij} - A_{ij} \cdot C \right \rVert ^2 + \alpha \left \lvert A_{ij} \right \rvert \\

                                 .. code:: CodeMirror-line

                                    & \text{subject to}

                                 .. code:: CodeMirror-line

                                    & & A_{ij} \geq 0 

                                 .. code:: CodeMirror-line

                                    \end{aligned}

                                 .. code:: CodeMirror-line

                                    \end{equation*}$$

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    where we use $A_{ij}$ to represent one pixel in `A`, like `A.sel(height=i, width=j)`, which will only have one dimension left: `unit_id`. Similarly $Y_{ij}$ is the corresponding pixel in `Y` which will only have the dimension `frame` left. Thus, $\left \lVert Y_{ij} - A_{ij} \cdot C \right \rVert ^2$ is our **error** term and $\left \lvert A_{ij} \right \rvert$ is our **l1-norm**. Moreover, we put these two terms together as a unitary target function/common goal to be minimized, with $\alpha$ controlling the balance between them. This balance can be seen by considering the impact of $\alpha$: the higher the value of $\alpha$, the greater the contribution the **l1-norm** term makes to the common goal (target function), the more penalty/emphasis you place on sparsity, and as a result, the more sparse `A` will be. The determination of the exact value of $\alpha$ is rather complicated, but the parameter we have for `update_spatial` is relative, where `alpha=1` corresponds to the default behavior of **CaImAn** package, and is usually a good place to start testing. 

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-success" role="alert">

                                 .. code:: CodeMirror-line

                                    Here is a good place to bring back the parameter exploring visualization results from the previous step and make sense of them! Pragmatically, relatively small values of <strong>sparse_penal</strong> have very little impact on the resulting <strong>A</strong>, but once you hit a large enough value, you will start to see units getting dimmer, sometimes completely disappearing. You might think this is the sparsity penalty in action, but from experience this is usually a case you want to <strong>avoid</strong>. After all, <strong>update_spatial</strong> has no way to differentiate noise from cells other than their corresponding temporal trace. Thus, you do not want <strong>update_spatial</strong> to take out cells for you unless you strongly trust the temporal traces (which you shouldn't for now since it's the first update and the temporal traces we have are merely weighted means of the original movie). If you are still puzzled about how to pick the right <strong>sparse_panel</strong> from the previous parameter exploring step, below we provide an illustrative example.

                                 .. code:: CodeMirror-line

                                    </div>    

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ![1st spatial update param exploring](img/sparse_panel_spatial_update.PNG)

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-success" role="alert">

                                 .. code:: CodeMirror-line

                                    What you are seeing here is parameter testing of the first spatial update. The left panel is the result with <strong>sparse_penal = 0.01</strong>, the middle panel the results with <strong>sparse_penal = 0.3</strong>, and the right the results with <strong>sparse_penal = 1</strong>. Ideally, we want the <strong>Binary Spatial Matrix</strong> to best mimic the real spatial footprint, which also means, they should be shaped like a cell. Thus, in this specific example, <strong>sparse_panel = 0.01</strong> (left penal) is not a good choice. Secondly, we also don't want to actually get rid of cells by using a high sparse panelty value at this step, which means <strong>sparse_panel = 1</strong> (right penal) is not good as well. Thus, <strong>sparse_panel = 0.3</strong> (middle panel) is a fairly good parameter to choose here.

                                 .. code:: CodeMirror-line

                                    </div>

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    There is yet another parameter, `dl_wnd`, that is relevant to practical consideration. Recall that we are updating $A_{ij}$ for our "small" problem, which has the dimension `unit_id` and has `A.sizes['unit_id]` number of entries (that is, the number of units). This is computationally feasible, but still a lot, especially when you do this for all pixels. One way to reduce computational demand is to leave out certain units when updating certain pixels -- in particular, it does not make sense to consider a unit that is supposed to be at the top left corner of the field of view when we update a pixel in the bottom right corner. In other words, for each pixel, we solve the "small" problem with only a subset of all potential units, thus hugely increasing the speed of `update_spatial`. This is where `A_init` comes into play (actually the only place it is used -- we do not need `A` at all for the update itself). We compute a morphological dilation, like that used during [background removal](#background-removal), on `A_init`, unit by unit, with window size `dl_wnd`, and we use the result as a **masking matrix**. Then, during the actual update of any given pixel, only units that have a non-zero value at the corresponding pixel in the **masking matrix** will be considered for update. In other words, we are allowing each unit to expand from `A_init` up to a distance of `dl_wnd`, and killing off any possibility beyond that range. The rationale of using `dl_wnd` here is that even if for some reason we have only one non-zero pixel representing the center of a certain unit in `A_init`, that unit can potentially expand to a full size cell, but anything beyond that would probably be either part of other cells or random noise. Thus, we want to set `dl_wnd`to approximately the radius of the largest cell to help ensure we get a clean footprint for all cells.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    Then we have a boolean parameter, `update_background`, controlling whether we want to update the background in this step. This is the only place in the pipeline that the background will be updated, and the way it is updated is by essentially treating `b` as another `unit` and updating it according to the temporal activity `f`. Pragmatically since the morphology-based [background removal](#backgroun-removal) works so well at cleaning the backgrounds, this updating has little impact on the result.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    Due to the actual implementation of the optimization method, it is hard for the computer to set some variables to absolutely zero. Instead, we usually have a very small float numbers in place of zeros. `zero_thres` solves this by thresholding all the values and setting anything below `zero_thres` to zero. You want to use a very small number for `zero_thres`. Setting `zero_thres='eps'` will use the [machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon)(the smallest non-negative number a machine can represent) of current datatype.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    Finally, we have an additional step after everything: normalization so that the spatial footprint of each unit has unit-norm. In practice we found that normalizing the result helps promoting the numerical stability of the algorithm, and enable us to interpret the spatial footprints as "weights" on each pixel so that the temporal activities are in the same scale space across units and can be compared. However normlizing spatial footprint for each unit does not preserve the relationship between overlapping cells in terms of their relative contribution to the activities of shared pixels. If such interpretation is critical for your downstream analysis, consider turning this off.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    `update_spatial` takes in the original data (`Y`), the initial spatial footprint for units and background (`A` and `b`, respectively), the initial temporal trace for units and background (`C` and `f`, respectively), and the estimated noise on each pixel (`sn`), in that order. Optional arguments are `sparse_penal`, `dl_wnd`, `update_background`, `post_scal` and `zero_thres`.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Here is the idea behind ``update_spatial``. Recall the
            parameters:

            ::

               param_first_spatial = {
                   'dl_wnd': 10,
                   'sparse_penal': 0.01,
                   'update_background': True,
                   'normalize': True,
                   'zero_thres': 'eps'}

            To reiterate, the big picture is that given the data (``Y``)
            and our units' activity (``C``) from previous the update
            (which is ``C_init``), we want to find the spatial
            footprints (``A``) such that 1. the **error**
            ``Y - A.dot(C, 'unit_id')`` is as small as possible, and 2.
            the `l1-norm <http://mathworld.wolfram.com/L1-Norm.html>`__
            of ``A`` is as small as possible. Here the **l1-norm** is a
            proxy to control for the sparsity of ``A``. Ideally to
            promote sparsity we want to control for the number of
            non-zero entries in ``A``, which is the
            `l0-norm <https://en.wikipedia.org/wiki/Lp_space#When_p_=_0>`__.
            However optimizing for the l0-norm is typically
            `computationally hard to
            do <https://stats.stackexchange.com/questions/269298/why-do-we-only-see-l-1-and-l-2-regularization-but-not-other-norms>`__,
            and it is usually good enough to use **l1-norm** instead as
            a proxy.

            Now, in theory we want to update every entry in ``A``
            iteratively with the above two goals in mind. However,
            updating that amount of numbers in ``A`` is still
            computationally very demanding, and it is much better if we
            can breakdown our big problem into smaller chunks that can
            be parallelized (making things much faster). **CNMF** is all
            about solving the issues caused by overlapping neurons, so
            it is best to keep the dependency across units (along
            dimension ``unit_id``) and update these entries together.
            However, it should be fine to treat each pixel as
            independent and update different pixels separately (in
            parallel). Thus, our new, "smaller" problem is: for each
            pixel, find the corresponding pixel in ``A``, across all
            ``unit_id``, that give us smallest **l1-norm** as well as
            smallest **error** when multiplied by ``C``. In equation
            form, this is:

            .. container:: MathJax_Display

               :presentation:`minimize𝐴𝑖𝑗subject
               to‖‖𝑌𝑖𝑗−𝐴𝑖𝑗⋅𝐶‖‖2+𝛼\|\|𝐴𝑖𝑗\|\|𝐴𝑖𝑗≥0\ 

               .. math::

                  \begin{matrix}
                   & \underset{A_{ij}}{\text{minimize}} & & {\left\| Y_{ij} - A_{ij} \cdot C \right\|^{2} + \alpha\left| A_{ij} \right|} \\
                   & \text{subject\ to} & & {A_{ij} \geq 0} \\
                  \end{matrix}

               `

            .. math::

               \begin{equation*}
               \begin{aligned}
               & \underset{A_{ij}}{\text{minimize}}
               & & \left \lVert Y_{ij} - A_{ij} \cdot C \right \rVert ^2 + \alpha \left \lvert A_{ij} \right \rvert \\
               & \text{subject to}
               & & A_{ij} \geq 0 
               \end{aligned}
               \end{equation*}

            where we use
            :presentation:`𝐴𝑖𝑗\ :math:`A_{ij}``\ :math:`A_{ij}` to
            represent one pixel in ``A``, like
            ``A.sel(height=i, width=j)``, which will only have one
            dimension left: ``unit_id``. Similarly
            :presentation:`𝑌𝑖𝑗\ :math:`Y_{ij}``\ :math:`Y_{ij}` is the
            corresponding pixel in ``Y`` which will only have the
            dimension ``frame`` left. Thus,
            :presentation:`‖‖𝑌𝑖𝑗−𝐴𝑖𝑗⋅𝐶‖‖2\ :math:`\left\| Y_{ij} - A_{ij} \cdot C \right\|^{2}``\ :math:`\left \lVert Y_{ij} - A_{ij} \cdot C \right \rVert ^2`
            is our **error** term and
            :presentation:`\|\|𝐴𝑖𝑗\|\|\ :math:`\left| A_{ij} \right|``\ :math:`\left \lvert A_{ij} \right \rvert`
            is our **l1-norm**. Moreover, we put these two terms
            together as a unitary target function/common goal to be
            minimized, with
            :presentation:`𝛼\ :math:`\alpha``\ :math:`\alpha`
            controlling the balance between them. This balance can be
            seen by considering the impact of
            :presentation:`𝛼\ :math:`\alpha``\ :math:`\alpha`: the
            higher the value of
            :presentation:`𝛼\ :math:`\alpha``\ :math:`\alpha`, the
            greater the contribution the **l1-norm** term makes to the
            common goal (target function), the more penalty/emphasis you
            place on sparsity, and as a result, the more sparse ``A``
            will be. The determination of the exact value of
            :presentation:`𝛼\ :math:`\alpha``\ :math:`\alpha` is rather
            complicated, but the parameter we have for
            ``update_spatial`` is relative, where ``alpha=1``
            corresponds to the default behavior of **CaImAn** package,
            and is usually a good place to start testing.

            .. container:: alert alert-success

               Here is a good place to bring back the parameter
               exploring visualization results from the previous step
               and make sense of them! Pragmatically, relatively small
               values of **sparse_penal** have very little impact on the
               resulting **A**, but once you hit a large enough value,
               you will start to see units getting dimmer, sometimes
               completely disappearing. You might think this is the
               sparsity penalty in action, but from experience this is
               usually a case you want to **avoid**. After all,
               **update_spatial** has no way to differentiate noise from
               cells other than their corresponding temporal trace.
               Thus, you do not want **update_spatial** to take out
               cells for you unless you strongly trust the temporal
               traces (which you shouldn't for now since it's the first
               update and the temporal traces we have are merely
               weighted means of the original movie). If you are still
               puzzled about how to pick the right **sparse_panel** from
               the previous parameter exploring step, below we provide
               an illustrative example.

            |1st spatial update param exploring|

            .. container:: alert alert-success

               What you are seeing here is parameter testing of the
               first spatial update. The left panel is the result with
               **sparse_penal = 0.01**, the middle panel the results
               with **sparse_penal = 0.3**, and the right the results
               with **sparse_penal = 1**. Ideally, we want the **Binary
               Spatial Matrix** to best mimic the real spatial
               footprint, which also means, they should be shaped like a
               cell. Thus, in this specific example, **sparse_panel =
               0.01** (left penal) is not a good choice. Secondly, we
               also don't want to actually get rid of cells by using a
               high sparse panelty value at this step, which means
               **sparse_panel = 1** (right penal) is not good as well.
               Thus, **sparse_panel = 0.3** (middle panel) is a fairly
               good parameter to choose here.

            There is yet another parameter, ``dl_wnd``, that is relevant
            to practical consideration. Recall that we are updating
            :presentation:`𝐴𝑖𝑗\ :math:`A_{ij}``\ :math:`A_{ij}` for our
            "small" problem, which has the dimension ``unit_id`` and has
            ``A.sizes['unit_id]`` number of entries (that is, the number
            of units). This is computationally feasible, but still a
            lot, especially when you do this for all pixels. One way to
            reduce computational demand is to leave out certain units
            when updating certain pixels -- in particular, it does not
            make sense to consider a unit that is supposed to be at the
            top left corner of the field of view when we update a pixel
            in the bottom right corner. In other words, for each pixel,
            we solve the "small" problem with only a subset of all
            potential units, thus hugely increasing the speed of
            ``update_spatial``. This is where ``A_init`` comes into play
            (actually the only place it is used -- we do not need ``A``
            at all for the update itself). We compute a morphological
            dilation, like that used during `background
            removal <#background-removal>`__, on ``A_init``, unit by
            unit, with window size ``dl_wnd``, and we use the result as
            a **masking matrix**. Then, during the actual update of any
            given pixel, only units that have a non-zero value at the
            corresponding pixel in the **masking matrix** will be
            considered for update. In other words, we are allowing each
            unit to expand from ``A_init`` up to a distance of
            ``dl_wnd``, and killing off any possibility beyond that
            range. The rationale of using ``dl_wnd`` here is that even
            if for some reason we have only one non-zero pixel
            representing the center of a certain unit in ``A_init``,
            that unit can potentially expand to a full size cell, but
            anything beyond that would probably be either part of other
            cells or random noise. Thus, we want to set ``dl_wnd``\ to
            approximately the radius of the largest cell to help ensure
            we get a clean footprint for all cells.

            Then we have a boolean parameter, ``update_background``,
            controlling whether we want to update the background in this
            step. This is the only place in the pipeline that the
            background will be updated, and the way it is updated is by
            essentially treating ``b`` as another ``unit`` and updating
            it according to the temporal activity ``f``. Pragmatically
            since the morphology-based `background
            removal <#backgroun-removal>`__ works so well at cleaning
            the backgrounds, this updating has little impact on the
            result.

            Due to the actual implementation of the optimization method,
            it is hard for the computer to set some variables to
            absolutely zero. Instead, we usually have a very small float
            numbers in place of zeros. ``zero_thres`` solves this by
            thresholding all the values and setting anything below
            ``zero_thres`` to zero. You want to use a very small number
            for ``zero_thres``. Setting ``zero_thres='eps'`` will use
            the `machine
            epsilon <https://en.wikipedia.org/wiki/Machine_epsilon>`__\ (the
            smallest non-negative number a machine can represent) of
            current datatype.

            Finally, we have an additional step after everything:
            normalization so that the spatial footprint of each unit has
            unit-norm. In practice we found that normalizing the result
            helps promoting the numerical stability of the algorithm,
            and enable us to interpret the spatial footprints as
            "weights" on each pixel so that the temporal activities are
            in the same scale space across units and can be compared.
            However normlizing spatial footprint for each unit does not
            preserve the relationship between overlapping cells in terms
            of their relative contribution to the activities of shared
            pixels. If such interpretation is critical for your
            downstream analysis, consider turning this off.

            ``update_spatial`` takes in the original data (``Y``), the
            initial spatial footprint for units and background (``A``
            and ``b``, respectively), the initial temporal trace for
            units and background (``C`` and ``f``, respectively), and
            the estimated noise on each pixel (``sn``), in that order.
            Optional arguments are ``sparse_penal``, ``dl_wnd``,
            ``update_background``, ``post_scal`` and ``zero_thres``.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       A_spatial, b_spatial, C_spatial, f_spatial = update_spatial(

                                    .. code:: CodeMirror-line

                                           Y, A_init, b_init, C_init, f_init, sn_spatial, **param_first_spatial)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       opts = dict(plot=dict(height=A_init.sizes['height'], width=A_init.sizes['width'], colorbar=True), style=dict(cmap='Viridis'))

                                    .. code:: CodeMirror-line

                                       (regrid(hv.Image(A_init.sum('unit_id').compute().rename('A'), kdims=['width', 'height'])).opts(**opts).relabel("Spatial Footprints Initial")

                                    .. code:: CodeMirror-line

                                       + regrid(hv.Image((A_init.fillna(0) > 0).sum('unit_id').compute().rename('A'), kdims=['width', 'height']), aggregator='max').opts(**opts).relabel("Binary Spatial Footprints Initial")

                                    .. code:: CodeMirror-line

                                       + regrid(hv.Image(A_spatial.sum('unit_id').compute().rename('A'), kdims=['width', 'height'])).opts(**opts).relabel("Spatial Footprints First Update")

                                    .. code:: CodeMirror-line

                                       + regrid(hv.Image((A_spatial > 0).sum('unit_id').compute().rename('A'), kdims=['width', 'height']), aggregator='max').opts(**opts).relabel("Binary Spatial Footprints First Update")).cols(2)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       opts_im = dict(plot=dict(height=b_init.sizes['height'], width=b_init.sizes['width'], colorbar=True), style=dict(cmap='Viridis'))

                                    .. code:: CodeMirror-line

                                       opts_cr = dict(plot=dict(height=b_init.sizes['height'], width=b_init.sizes['height'] * 2))

                                    .. code:: CodeMirror-line

                                       (regrid(hv.Image(b_init.compute(), kdims=['width', 'height'])).opts(**opts_im).relabel('Background Spatial Initial')

                                    .. code:: CodeMirror-line

                                        + datashade(hv.Curve(f_init.compute(), kdims=['frame'])).opts(**opts_cr).relabel('Background Temporal Initial')

                                    .. code:: CodeMirror-line

                                        + regrid(hv.Image(b_spatial.compute(), kdims=['width', 'height'])).opts(**opts_im).relabel('Background Spatial First Update')

                                    .. code:: CodeMirror-line

                                        + datashade(hv.Curve(f_spatial.compute(), kdims=['frame'])).opts(**opts_cr).relabel('Background Temporal First Update')

                                    .. code:: CodeMirror-line

                                       ).cols(2)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## test parameters for temporal update

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: test parameters for temporal
               update\ `¶ <#test-parameters-for-temporal-update>`__
               :name: test-parameters-for-temporal-update

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    First off we select some `units` to do parameter exploring.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            First off we select some ``units`` to do parameter
            exploring.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           units = np.random.choice(A_spatial.coords['unit_id'], 10, replace=False)

                                    .. code:: CodeMirror-line

                                           units.sort()

                                    .. code:: CodeMirror-line

                                           A_sub = A_spatial.sel(unit_id=units).persist()

                                    .. code:: CodeMirror-line

                                           C_sub = C_spatial.sel(unit_id=units).persist()

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Now we move on to the parameter exploring of temporal update. Here we use the same idea we have before, only this time there is much more parameters to play with for temporal update, and we now have four `list`s of potential parameters: `p_ls`, `sprs_ls`, `add_ls`, and `noise_ls`. We use [`itertools.product`](https://docs.python.org/3.7/library/itertools.html#itertools.product) to iterate through all possible combinations of the potential values and save us from nested `for` loops.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Now we move on to the parameter exploring of temporal
            update. Here we use the same idea we have before, only this
            time there is much more parameters to play with for temporal
            update, and we now have four ``list``\ s of potential
            parameters: ``p_ls``, ``sprs_ls``, ``add_ls``, and
            ``noise_ls``. We use
            ```itertools.product`` <https://docs.python.org/3.7/library/itertools.html#itertools.product>`__
            to iterate through all possible combinations of the
            potential values and save us from nested ``for`` loops.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           p_ls = [1]

                                    .. code:: CodeMirror-line

                                           sprs_ls = [0.01, 0.05, 0.1, 2]

                                    .. code:: CodeMirror-line

                                           add_ls = [20]

                                    .. code:: CodeMirror-line

                                           noise_ls = [0.06]

                                    .. code:: CodeMirror-line

                                           YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict = [dict() for _ in range(6)]

                                    .. code:: CodeMirror-line

                                           YrA = compute_trace(Y, A_sub, b_spatial, C_sub, f_spatial).persist()

                                    .. code:: CodeMirror-line

                                           for cur_p, cur_sprs, cur_add, cur_noise in itt.product(p_ls, sprs_ls, add_ls, noise_ls):

                                    .. code:: CodeMirror-line

                                               ks = (cur_p, cur_sprs, cur_add, cur_noise)

                                    .. code:: CodeMirror-line

                                               print("p:{}, sparse penalty:{}, additional lag:{}, noise frequency:{}"

                                    .. code:: CodeMirror-line

                                                     .format(cur_p, cur_sprs, cur_add, cur_noise))

                                    .. code:: CodeMirror-line

                                               YrA, cur_C, cur_S, cur_B, cur_C0, cur_sig, cur_g, cur_scal = update_temporal(

                                    .. code:: CodeMirror-line

                                                   Y, A_sub, b_spatial, C_sub, f_spatial, sn_spatial, YrA=YrA,

                                    .. code:: CodeMirror-line

                                                   sparse_penal=cur_sprs, p=cur_p, use_spatial=False, use_smooth=True,

                                    .. code:: CodeMirror-line

                                                   add_lag = cur_add, noise_freq=cur_noise)

                                    .. code:: CodeMirror-line

                                               YA_dict[ks], C_dict[ks], S_dict[ks], g_dict[ks], sig_dict[ks], A_dict[ks] = (

                                    .. code:: CodeMirror-line

                                                   YrA.compute(), cur_C.compute(), cur_S.compute(), cur_g.compute(), cur_sig.compute(), A_sub.compute())

                                    .. code:: CodeMirror-line

                                           hv_res = visualize_temporal_update(

                                    .. code:: CodeMirror-line

                                               YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict,

                                    .. code:: CodeMirror-line

                                               kdims=['p', 'sparse penalty', 'additional lag', 'noise frequency'])

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    A piece of useful infomation after you run this cell is that under what testing parameter, which sample units got dropped because of poor fit:

                                 .. code:: CodeMirror-line

                                    ![dropped sample units](img/first_tem_drop_v2.PNG)

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-success">  

                                 .. code:: CodeMirror-line

                                    Cross compare this with the raw trace plot, find the most reasonable parameters that drop the right sample cells.

                                 .. code:: CodeMirror-line

                                    </div>

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    Then,  we plot the visualization `hv_res` of the 10 ramdom units we just generated at the belowing code cell. Don't worry if each parameter doesn't make much sense now, What you should expect here will be explained later in <strong>first temporal update</strong> along with what `param_first_temporal` actually does (Look for the green tips box)!

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            A piece of useful infomation after you run this cell is that
            under what testing parameter, which sample units got dropped
            because of poor fit: |dropped sample units|

            .. container:: alert alert-success

               Cross compare this with the raw trace plot, find the most
               reasonable parameters that drop the right sample cells.

            Then, we plot the visualization ``hv_res`` of the 10 ramdom
            units we just generated at the belowing code cell. Don't
            worry if each parameter doesn't make much sense now, What
            you should expect here will be explained later in **first
            temporal update** along with what ``param_first_temporal``
            actually does (Look for the green tips box)!

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           display(hv_res)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## first temporal update

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: first temporal
               update\ `¶ <#first-temporal-update>`__
               :name: first-temporal-update

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Here is the idea for temporal update: Recall tha parameters:

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ```python

                                 .. code:: CodeMirror-line

                                    param_first_temporal = {

                                 .. code:: CodeMirror-line

                                        'noise_freq': 0.06,

                                 .. code:: CodeMirror-line

                                        'sparse_penal': 0.1,

                                 .. code:: CodeMirror-line

                                        'p': 1,

                                 .. code:: CodeMirror-line

                                        'add_lag': 20,

                                 .. code:: CodeMirror-line

                                        'use_spatial': False,

                                 .. code:: CodeMirror-line

                                        'jac_thres': 0.2,

                                 .. code:: CodeMirror-line

                                        'zero_thres': 1e-8,

                                 .. code:: CodeMirror-line

                                        'max_iters': 200,

                                 .. code:: CodeMirror-line

                                        'use_smooth': True,

                                 .. code:: CodeMirror-line

                                        'scs_fallback': False,

                                 .. code:: CodeMirror-line

                                        'post_scal': True}

                                 .. code:: CodeMirror-line

                                    ```

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    Similar to the spatial update, given  the spatial footprint of each unit (`A`), our goal is now to find the activity of each unit (`C`) that minimizes both the **error** (`Y - A.dot(C, 'unit_id')`) and the **l1-norm** of `C`. However there is an additional constraint: the trace of each unit in `C` must follow an autoregressive process. Due to this additional layer of complexity, things becomes more computationaly expensive.  To reduce computatioinal cost, first observe that `A` is usually much larger than `C` (you usually have more total pixels than `frame`s), and performing the dot product, `A.dot(C, 'unit_id')`, everytime you try a different number in `C`,  is infeasible. Thus, we convert our **error** term to something like $\mathbf{A}^{-1} \cdot \mathbf{Y} - \mathbf{C}$, where $\mathbf{A}^{-1}$ represents a matrix that can "undo" what `A` usually does to `C` -- instead of weighting the temporal activity of each unit by its spatial footprint (converting a matrix with dimension `unit_id` and `frame` into one with dimensions `height`, `width` and `frame`), $\mathbf{A}^{-1}$ "extracts" the temporal activity of each unit based upon their spatial footprint (converting a matrix with dimension `height`, `width` and `frame` into one with dimensions `unit_id` and `frame`). In other words, $\mathbf{A}^{-1}$ is like an [inverse](https://en.wikipedia.org/wiki/Moore–Penrose_inverse) of `A`. This way, we only need to calculate $\mathbf{A}^{-1} \cdot \mathbf{Y}$ once and be done -- we can use that result everytime we update `C`. The calculation of $\mathbf{A}^{-1} \cdot \mathbf{Y}$ is rather complicated and not strictly mathematically accurate, but it provides a good approximation with huge computational benefit, and is the default behavior of CaImAn. You can turn this off by supplying `use_spatial=True` -- however that is usually too computationally demanding to do. We will assume `use_spatial=False` in the following discussion and call the $\mathbf{A}^{-1} \cdot \mathbf{Y}$ term `YrA`, as in the code. The second thing to observe is that we cannot keep the `unit_id` dimension and chop up the `frame` dimension for parallel processing (like how we chopped up pixels during the spatial update), since we have to check whether each trace along the `frame` dimension follows an autoregressive process. Instead, we turn to the `unit_id` dimension to make our problem "smaller". Since we have a relatively good `A` now, it should be OK to update units that are not spatially overlapping independently. This idea should work if you have a relatively sparse distribution of cells. However if your field-of-view is packed with cells, if we were to consider cells overlapping if they share only one pixel, we would likely end up having to update `C` altogether, since every cell is transitively overlapping with every other cell. Instead, we put a threshold on how we define "overlap", and that is what `jac_thres` is for -- only cells that have an area of their spatial footprint overlapping that is more than this threshold (ranging from 0 to 1) will be considered "overlapping". (The "proportion of overlapping area" has a formal name: [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index), hence the name `jac_thres`). Pragamatically `jac_thres=0.2` works for data that is very compact in cells.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    We now turn to the "other layer of complexity," which is the autoregressive process. Recall that the temporal trace of each unit should be fitted by the following equation: $$c(t) = \sum_{i=0}^{p}\gamma_i c(t-i) + s(t) + \epsilon$$ The first thing we want to determine is `p`. As discussed before, `p=2` is a good choice if your calcium transients have an observable rise-time. `p=1` might work better if the rise-time of your signal is faster than your sampling rate and you thus don't need to explicitly model it. Notably, `p>2` could result in [over-fitting](https://en.wikipedia.org/wiki/Overfitting) and is not recomended unless you are certain that your calcium traces have a more complicated waveform. Next, notice that we have several $\gamma_i$s unaccounted for (though usually not too many if `p` is small). Luckily, we do not have to iteratively update these -- it turns out that the $\gamma_i$s of an autoregressive process are related to the [autocovariance](https://en.wikipedia.org/wiki/Autocovariance) of the signal at different lags, which can be readily computed from `YrA`. For full derivation of these relationships, please refer to the [original CNMF paper](https://www.sciencedirect.com/science/article/pii/S0896627315010843?via%3Dihub). Here, we will merely assume that the parameters that affect how much a signal depends on its own history are related to the covariance of the signal when you shift it by different temporal lags. In this way, $\gamma_i$s can be computed rather deterministicly. Say you set `p=2` and thus you have two $\gamma_i$s to be estimated -- you would need exactly two equations involving the autocovariance function up to 2 time-step lags to give you the two $\gamma_i$s. However, you can add additional equations using different lags to better model the propogation of signal, since the impact of $\gamma_i$s can theoretically extend infinitely back in time, and should be reflected in the autocovariance function at any additional lag. In practice, we use a finite number of equations, solved with [least squares](https://en.wikipedia.org/wiki/Least_squares). Thus it is important to choose an appropriate number of **additional** equations, which is what `add_lag` controls. An `add_lag` that is too small like `add_lag=0` will leave everything to the first `p` number of equations and autocovariance functions, which might not be reliable. Pragmatically, smaller `add_lag` values tend to bias the $\gamma_i$s to give a much faster decay, whereas larger `add_lag` values tend to give a longer decay. **As a rule of thumb, it is usually good to set `add_lag` to approximately the decay time of your signal (in frames).** 

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    Once we have estimated the $\gamma_i$s, the calcium traces, $c(t)$, and spikes, $s(t)$, are essentially **one thing** -- given calcium traces and how they rise/decay in response to spikes, we can deduce where the spikes happen, and *vice versa*. We can express this determined relationship with a matrix $\mathbf{G}$ where $s(t) = \mathbf{G} \cdot c(t)$. In other words, $\mathbf{G}$ is the matrix that "undoes" what $\gamma_i$s do to $s(t)$. With all these parameters sorted out, we finally come to the actual optimization problem:

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    $$\begin{equation*}

                                 .. code:: CodeMirror-line

                                    \begin{aligned}

                                 .. code:: CodeMirror-line

                                    & \underset{C_{i}}{\text{minimize}}

                                 .. code:: CodeMirror-line

                                    & & \left \lVert \mathbf{YrA}_{i} - \mathbf{C}_{i} \right \rVert ^2 + \alpha \left \lvert \mathbf{G}_{i} \cdot \mathbf{C}_{i} \right \rvert \\

                                 .. code:: CodeMirror-line

                                    & \text{subject to}

                                 .. code:: CodeMirror-line

                                    & & \mathbf{C}_{i} \geq 0, \; \mathbf{G}_{i} \cdot \mathbf{C}_{i} \geq 0 

                                 .. code:: CodeMirror-line

                                    \end{aligned}

                                 .. code:: CodeMirror-line

                                    \end{equation*}$$

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    Just as during the spatial update, we select some units ($i$), and update their calcium dynamics ($\mathbf{C}_i$) based on the **error** and the **l1-norm** of the **spikes** ($\mathbf{G}_i \cdot \mathbf{C}_i$). Again, it does not make sense to have negative calcium dynamics or spikes, so that is a constraint on the problem. Moreover, we need an $\alpha$ to provide balance between fidelity and sparsity, which can be scaled up and down with `sparse_penal` (`sparse_penal=1` is equivalent to the default behavior of CaImAn). Furthermore, $\alpha$ should depend on the expected level of noise. Note that we cannot use `sn_spatial` since that was the noise for each pixel, and we need the noise for each unit. The function `update_temporal` estimates the noise of each unit for you -- you just have to tell it the `noise_freq`uency. Like before, **0.5** is the highest you can go. With the default, `noise_freq=0.25`, the higher frequency half of the signal will be considered noise. In addition to affecting the estimation of noise power, `noise_freq` affects another smoothing process: when estimating $\gamma_i$s, it is usually helpful to run a filter on the signal to get rid of high freqeuency noise, particularly when you don't have a large `add_lag`. The parameter, `noise_freq` is the cut-off frequency of the low-pass filter run on the temporal trace for each unit.  Additionally, you can set the value of `use_smooth` to control whether the filtering is done at all. Even with this careful design, however, it is sometimes hard to approach the true solution to the problem. When that happens, `update_temporal` will warn you by saying something like "problem solved sub-optimally". Usually, a few of these warnings is OK, but if you see this warning a lot it either means your parameters are unreasonable or you need more iterations to approach the real answer. You can use `max_iters` to control how many iterations to run for each small problem before the computer gives up and throws a warning. Furthermore, in some very, very rare cases, the default [ecos solver](https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver) (the algorithm that does all the heavy-lifting) can fail and throw a "problem infeasible" warning, and it's worth trying a different solver, namely [scs](https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver).  Be aware that scs produces results with very, very slow performance. The boolean parameter `scs_fallback` controls whether the scs attempt should be made before giving up. Importantly, both increasing `max_iters` and using `scs_fallback` will significantly increase the computation time and will not help at all if the parameters you provided are unreasonable to begin with, so try to use this only as a last resort.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    Finally, after the optimization is done, and just like [`update_spatial`](#first-spatial-update), we have a `zero_thres` to get rid of the small numbers, after which we can do a `post_scal` to counter the artifacts introduced by the **l1-norm** penalty.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    `update_temporal` takes in `Y`, `A`, `b`, `C`, `f`, and `sn_spatial` (even if we won't need it by default), in that order. Optionally you can pass in `noise_freq`, `p`, `add_lag`, `jac_thres`, `use_spatial`, `sparse_penal`, `max_iters`, `use_smooth`, `scs_fallback`, `zero_thres` and `post_scal`, as we have discussed. `update_temporal` returns much more than we expected -- in addition to `C_temporal` and `S_temporal`, which are the results we care most about, it also returns `YrA`, and `g_temporal` (the $\mathbf{G}$ matrix for each unit). Moreover, it returns `B_temporal`, `C0_temporal` and `sig_temporal`, representing the final layer of complexity: when we update the temporal trace, there might be a global baseline calcium concentration, which is modeled by $b$ and returned in `B_temporal`. A spike may also have happened right before recording starts and the resulting calcium transient could still be decaying in the first few seconds, so we model this with an initial calcium concentration, $c_0$, that follows the same decaying pattern defined by $\gamma_i$s, and is returned in `C0_temporal`. Both $b$ and $c_0$ are single numbers that get updated along with the calcium dynamics for each unit. Finally there is `sig_temporal` which is the combination of all the signals, that is: `C_temporal + C0_temporal + B_temporal`

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-success">

                                 .. code:: CodeMirror-line

                                    You should now have an idea of what each parameter is doing in `update_temporal`, and be able to make sense of the visualization results of the parameter exploring steps.

                                 .. code:: CodeMirror-line

                                        

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    - As was briefly mentioned before, minian's output of <strong>dropped sample units</strong> information and visualization of their <strong>raw traces</strong> is useful after the first temporal update. Since one of the main purposes of the <strong>first temporal update</strong> is to get rid of trash cells and cells with noisy signal, successful parameter selection is evidenced by dropped units with raw traces that look like noise (no clear bursts of activity). Alternatively, if cell-like activity is seen in the raw trace of a dropped unit, this may indicate that the selected parameters are too conservative. 

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    - When reading the temporal trace plot, "fitted spikes" (green), "fitted signal" (orange), and "fitted calcium trace" (blue), are all alligned to the "raw signal" based upon the model. Ideally, we want only one spike for each burst of signal, with "fitted signal" and "fitted calcium trace" decaying in a manner that follows the raw signal. Below is the temporal plot of an example unit using different <strong>sparse_panel</strong>:

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ### Example Temporal Traces

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ![example temporal traces](img/first_tem_param.png)

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    Here, the top trace is when <strong>sparse_panel</strong> = 1, and we can see that there are lots of small spikes at the bottom, indicating we may want to increase the <strong>sparse_panel</strong> to get rid of them. However, when we are using <strong>sparse_panel</strong> = 10 (bottom panel), it's clear that we are missing real spikes from raw signal. Thus, the middle panel with <strong>sparse_panel</strong> = 3 fits the raw signal the best here.

                                 .. code:: CodeMirror-line

                                    </div>

                                 .. code:: CodeMirror-line

                                    ​

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Here is the idea for temporal update: Recall tha parameters:

            ::

               param_first_temporal = {
                   'noise_freq': 0.06,
                   'sparse_penal': 0.1,
                   'p': 1,
                   'add_lag': 20,
                   'use_spatial': False,
                   'jac_thres': 0.2,
                   'zero_thres': 1e-8,
                   'max_iters': 200,
                   'use_smooth': True,
                   'scs_fallback': False,
                   'post_scal': True}

            Similar to the spatial update, given the spatial footprint
            of each unit (``A``), our goal is now to find the activity
            of each unit (``C``) that minimizes both the **error**
            (``Y - A.dot(C, 'unit_id')``) and the **l1-norm** of ``C``.
            However there is an additional constraint: the trace of each
            unit in ``C`` must follow an autoregressive process. Due to
            this additional layer of complexity, things becomes more
            computationaly expensive. To reduce computatioinal cost,
            first observe that ``A`` is usually much larger than ``C``
            (you usually have more total pixels than ``frame``\ s), and
            performing the dot product, ``A.dot(C, 'unit_id')``,
            everytime you try a different number in ``C``, is
            infeasible. Thus, we convert our **error** term to something
            like
            :presentation:`𝐀−1⋅𝐘−𝐂\ :math:`\mathbf{A}^{- 1} \cdot \mathbf{Y} - \mathbf{C}``\ :math:`\mathbf{A}^{-1} \cdot \mathbf{Y} - \mathbf{C}`,
            where
            :presentation:`𝐀−1\ :math:`\mathbf{A}^{- 1}``\ :math:`\mathbf{A}^{-1}`
            represents a matrix that can "undo" what ``A`` usually does
            to ``C`` -- instead of weighting the temporal activity of
            each unit by its spatial footprint (converting a matrix with
            dimension ``unit_id`` and ``frame`` into one with dimensions
            ``height``, ``width`` and ``frame``),
            :presentation:`𝐀−1\ :math:`\mathbf{A}^{- 1}``\ :math:`\mathbf{A}^{-1}`
            "extracts" the temporal activity of each unit based upon
            their spatial footprint (converting a matrix with dimension
            ``height``, ``width`` and ``frame`` into one with dimensions
            ``unit_id`` and ``frame``). In other words,
            :presentation:`𝐀−1\ :math:`\mathbf{A}^{- 1}``\ :math:`\mathbf{A}^{-1}`
            is like an
            `inverse <https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse>`__
            of ``A``. This way, we only need to calculate
            :presentation:`𝐀−1⋅𝐘\ :math:`\mathbf{A}^{- 1} \cdot \mathbf{Y}``\ :math:`\mathbf{A}^{-1} \cdot \mathbf{Y}`
            once and be done -- we can use that result everytime we
            update ``C``. The calculation of
            :presentation:`𝐀−1⋅𝐘\ :math:`\mathbf{A}^{- 1} \cdot \mathbf{Y}``\ :math:`\mathbf{A}^{-1} \cdot \mathbf{Y}`
            is rather complicated and not strictly mathematically
            accurate, but it provides a good approximation with huge
            computational benefit, and is the default behavior of
            CaImAn. You can turn this off by supplying
            ``use_spatial=True`` -- however that is usually too
            computationally demanding to do. We will assume
            ``use_spatial=False`` in the following discussion and call
            the
            :presentation:`𝐀−1⋅𝐘\ :math:`\mathbf{A}^{- 1} \cdot \mathbf{Y}``\ :math:`\mathbf{A}^{-1} \cdot \mathbf{Y}`
            term ``YrA``, as in the code. The second thing to observe is
            that we cannot keep the ``unit_id`` dimension and chop up
            the ``frame`` dimension for parallel processing (like how we
            chopped up pixels during the spatial update), since we have
            to check whether each trace along the ``frame`` dimension
            follows an autoregressive process. Instead, we turn to the
            ``unit_id`` dimension to make our problem "smaller". Since
            we have a relatively good ``A`` now, it should be OK to
            update units that are not spatially overlapping
            independently. This idea should work if you have a
            relatively sparse distribution of cells. However if your
            field-of-view is packed with cells, if we were to consider
            cells overlapping if they share only one pixel, we would
            likely end up having to update ``C`` altogether, since every
            cell is transitively overlapping with every other cell.
            Instead, we put a threshold on how we define "overlap", and
            that is what ``jac_thres`` is for -- only cells that have an
            area of their spatial footprint overlapping that is more
            than this threshold (ranging from 0 to 1) will be considered
            "overlapping". (The "proportion of overlapping area" has a
            formal name: `Jaccard
            index <https://en.wikipedia.org/wiki/Jaccard_index>`__,
            hence the name ``jac_thres``). Pragamatically
            ``jac_thres=0.2`` works for data that is very compact in
            cells.

            We now turn to the "other layer of complexity," which is the
            autoregressive process. Recall that the temporal trace of
            each unit should be fitted by the following equation:

            .. container:: MathJax_Display

               :presentation:`𝑐(𝑡)=∑𝑖=0𝑝𝛾𝑖𝑐(𝑡−𝑖)+𝑠(𝑡)+𝜖\ 

               .. math:: c(t) = \sum\limits_{i = 0}^{p}\gamma_{i}c(t - i) + s(t) + \epsilon

               `

            .. math:: c(t) = \sum_{i=0}^{p}\gamma_i c(t-i) + s(t) + \epsilon

            \ The first thing we want to determine is ``p``. As
            discussed before, ``p=2`` is a good choice if your calcium
            transients have an observable rise-time. ``p=1`` might work
            better if the rise-time of your signal is faster than your
            sampling rate and you thus don't need to explicitly model
            it. Notably, ``p>2`` could result in
            `over-fitting <https://en.wikipedia.org/wiki/Overfitting>`__
            and is not recomended unless you are certain that your
            calcium traces have a more complicated waveform. Next,
            notice that we have several
            :presentation:`𝛾𝑖\ :math:`\gamma_{i}``\ :math:`\gamma_i`\ s
            unaccounted for (though usually not too many if ``p`` is
            small). Luckily, we do not have to iteratively update these
            -- it turns out that the
            :presentation:`𝛾𝑖\ :math:`\gamma_{i}``\ :math:`\gamma_i`\ s
            of an autoregressive process are related to the
            `autocovariance <https://en.wikipedia.org/wiki/Autocovariance>`__
            of the signal at different lags, which can be readily
            computed from ``YrA``. For full derivation of these
            relationships, please refer to the `original CNMF
            paper <https://www.sciencedirect.com/science/article/pii/S0896627315010843?via%3Dihub>`__.
            Here, we will merely assume that the parameters that affect
            how much a signal depends on its own history are related to
            the covariance of the signal when you shift it by different
            temporal lags. In this way,
            :presentation:`𝛾𝑖\ :math:`\gamma_{i}``\ :math:`\gamma_i`\ s
            can be computed rather deterministicly. Say you set ``p=2``
            and thus you have two
            :presentation:`𝛾𝑖\ :math:`\gamma_{i}``\ :math:`\gamma_i`\ s
            to be estimated -- you would need exactly two equations
            involving the autocovariance function up to 2 time-step lags
            to give you the two
            :presentation:`𝛾𝑖\ :math:`\gamma_{i}``\ :math:`\gamma_i`\ s.
            However, you can add additional equations using different
            lags to better model the propogation of signal, since the
            impact of
            :presentation:`𝛾𝑖\ :math:`\gamma_{i}``\ :math:`\gamma_i`\ s
            can theoretically extend infinitely back in time, and should
            be reflected in the autocovariance function at any
            additional lag. In practice, we use a finite number of
            equations, solved with `least
            squares <https://en.wikipedia.org/wiki/Least_squares>`__.
            Thus it is important to choose an appropriate number of
            **additional** equations, which is what ``add_lag``
            controls. An ``add_lag`` that is too small like
            ``add_lag=0`` will leave everything to the first ``p``
            number of equations and autocovariance functions, which
            might not be reliable. Pragmatically, smaller ``add_lag``
            values tend to bias the
            :presentation:`𝛾𝑖\ :math:`\gamma_{i}``\ :math:`\gamma_i`\ s
            to give a much faster decay, whereas larger ``add_lag``
            values tend to give a longer decay. **As a rule of thumb, it
            is usually good to set ``add_lag`` to approximately the
            decay time of your signal (in frames).**
            Once we have estimated the
            :presentation:`𝛾𝑖\ :math:`\gamma_{i}``\ :math:`\gamma_i`\ s,
            the calcium traces,
            :presentation:`𝑐(𝑡)\ :math:`c(t)``\ :math:`c(t)`, and
            spikes, :presentation:`𝑠(𝑡)\ :math:`s(t)``\ :math:`s(t)`,
            are essentially **one thing** -- given calcium traces and
            how they rise/decay in response to spikes, we can deduce
            where the spikes happen, and *vice versa*. We can express
            this determined relationship with a matrix
            :presentation:`𝐆\ :math:`\mathbf{G}``\ :math:`\mathbf{G}`
            where
            :presentation:`𝑠(𝑡)=𝐆⋅𝑐(𝑡)\ :math:`s(t) = \mathbf{G} \cdot c(t)``\ :math:`s(t) = \mathbf{G} \cdot c(t)`.
            In other words,
            :presentation:`𝐆\ :math:`\mathbf{G}``\ :math:`\mathbf{G}` is
            the matrix that "undoes" what
            :presentation:`𝛾𝑖\ :math:`\gamma_{i}``\ :math:`\gamma_i`\ s
            do to :presentation:`𝑠(𝑡)\ :math:`s(t)``\ :math:`s(t)`. With
            all these parameters sorted out, we finally come to the
            actual optimization problem:

            .. container:: MathJax_Display

               :presentation:`minimize𝐶𝑖subject
               to‖𝐘𝐫𝐀𝑖−𝐂𝑖‖2+𝛼\|\|𝐆𝑖⋅𝐂𝑖\|\|𝐂𝑖≥0,𝐆𝑖⋅𝐂𝑖≥0\ 

               .. math::

                  \begin{matrix}
                   & \underset{C_{i}}{\text{minimize}} & & {\left\| {\mathbf{Y}\mathbf{r}\mathbf{A}}_{i} - \mathbf{C}_{i} \right\|^{2} + \alpha\left| \mathbf{G}_{i} \cdot \mathbf{C}_{i} \right|} \\
                   & \text{subject\ to} & & {\mathbf{C}_{i} \geq 0,\;\mathbf{G}_{i} \cdot \mathbf{C}_{i} \geq 0} \\
                  \end{matrix}

               `

            .. math::

               \begin{equation*}
               \begin{aligned}
               & \underset{C_{i}}{\text{minimize}}
               & & \left \lVert \mathbf{YrA}_{i} - \mathbf{C}_{i} \right \rVert ^2 + \alpha \left \lvert \mathbf{G}_{i} \cdot \mathbf{C}_{i} \right \rvert \\
               & \text{subject to}
               & & \mathbf{C}_{i} \geq 0, \; \mathbf{G}_{i} \cdot \mathbf{C}_{i} \geq 0 
               \end{aligned}
               \end{equation*}

            Just as during the spatial update, we select some units
            (:presentation:`𝑖\ :math:`i``\ :math:`i`), and update their
            calcium dynamics
            (:presentation:`𝐂𝑖\ :math:`\mathbf{C}_{i}``\ :math:`\mathbf{C}_i`)
            based on the **error** and the **l1-norm** of the **spikes**
            (:presentation:`𝐆𝑖⋅𝐂𝑖\ :math:`\mathbf{G}_{i} \cdot \mathbf{C}_{i}``\ :math:`\mathbf{G}_i \cdot \mathbf{C}_i`).
            Again, it does not make sense to have negative calcium
            dynamics or spikes, so that is a constraint on the problem.
            Moreover, we need an
            :presentation:`𝛼\ :math:`\alpha``\ :math:`\alpha` to provide
            balance between fidelity and sparsity, which can be scaled
            up and down with ``sparse_penal`` (``sparse_penal=1`` is
            equivalent to the default behavior of CaImAn). Furthermore,
            :presentation:`𝛼\ :math:`\alpha``\ :math:`\alpha` should
            depend on the expected level of noise. Note that we cannot
            use ``sn_spatial`` since that was the noise for each pixel,
            and we need the noise for each unit. The function
            ``update_temporal`` estimates the noise of each unit for you
            -- you just have to tell it the ``noise_freq``\ uency. Like
            before, **0.5** is the highest you can go. With the default,
            ``noise_freq=0.25``, the higher frequency half of the signal
            will be considered noise. In addition to affecting the
            estimation of noise power, ``noise_freq`` affects another
            smoothing process: when estimating
            :presentation:`𝛾𝑖\ :math:`\gamma_{i}``\ :math:`\gamma_i`\ s,
            it is usually helpful to run a filter on the signal to get
            rid of high freqeuency noise, particularly when you don't
            have a large ``add_lag``. The parameter, ``noise_freq`` is
            the cut-off frequency of the low-pass filter run on the
            temporal trace for each unit. Additionally, you can set the
            value of ``use_smooth`` to control whether the filtering is
            done at all. Even with this careful design, however, it is
            sometimes hard to approach the true solution to the problem.
            When that happens, ``update_temporal`` will warn you by
            saying something like "problem solved sub-optimally".
            Usually, a few of these warnings is OK, but if you see this
            warning a lot it either means your parameters are
            unreasonable or you need more iterations to approach the
            real answer. You can use ``max_iters`` to control how many
            iterations to run for each small problem before the computer
            gives up and throws a warning. Furthermore, in some very,
            very rare cases, the default `ecos
            solver <https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver>`__
            (the algorithm that does all the heavy-lifting) can fail and
            throw a "problem infeasible" warning, and it's worth trying
            a different solver, namely
            `scs <https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver>`__.
            Be aware that scs produces results with very, very slow
            performance. The boolean parameter ``scs_fallback`` controls
            whether the scs attempt should be made before giving up.
            Importantly, both increasing ``max_iters`` and using
            ``scs_fallback`` will significantly increase the computation
            time and will not help at all if the parameters you provided
            are unreasonable to begin with, so try to use this only as a
            last resort.

            Finally, after the optimization is done, and just like
            ```update_spatial`` <#first-spatial-update>`__, we have a
            ``zero_thres`` to get rid of the small numbers, after which
            we can do a ``post_scal`` to counter the artifacts
            introduced by the **l1-norm** penalty.

            ``update_temporal`` takes in ``Y``, ``A``, ``b``, ``C``,
            ``f``, and ``sn_spatial`` (even if we won't need it by
            default), in that order. Optionally you can pass in
            ``noise_freq``, ``p``, ``add_lag``, ``jac_thres``,
            ``use_spatial``, ``sparse_penal``, ``max_iters``,
            ``use_smooth``, ``scs_fallback``, ``zero_thres`` and
            ``post_scal``, as we have discussed. ``update_temporal``
            returns much more than we expected -- in addition to
            ``C_temporal`` and ``S_temporal``, which are the results we
            care most about, it also returns ``YrA``, and ``g_temporal``
            (the
            :presentation:`𝐆\ :math:`\mathbf{G}``\ :math:`\mathbf{G}`
            matrix for each unit). Moreover, it returns ``B_temporal``,
            ``C0_temporal`` and ``sig_temporal``, representing the final
            layer of complexity: when we update the temporal trace,
            there might be a global baseline calcium concentration,
            which is modeled by :presentation:`𝑏\ :math:`b``\ :math:`b`
            and returned in ``B_temporal``. A spike may also have
            happened right before recording starts and the resulting
            calcium transient could still be decaying in the first few
            seconds, so we model this with an initial calcium
            concentration,
            :presentation:`𝑐0\ :math:`c_{0}``\ :math:`c_0`, that follows
            the same decaying pattern defined by
            :presentation:`𝛾𝑖\ :math:`\gamma_{i}``\ :math:`\gamma_i`\ s,
            and is returned in ``C0_temporal``. Both
            :presentation:`𝑏\ :math:`b``\ :math:`b` and
            :presentation:`𝑐0\ :math:`c_{0}``\ :math:`c_0` are single
            numbers that get updated along with the calcium dynamics for
            each unit. Finally there is ``sig_temporal`` which is the
            combination of all the signals, that is:
            ``C_temporal + C0_temporal + B_temporal``

            .. container:: alert alert-success

               You should now have an idea of what each parameter is
               doing in \`update_temporal`, and be able to make sense of
               the visualization results of the parameter exploring
               steps.

               -  As was briefly mentioned before, minian's output of
                  **dropped sample units** information and visualization
                  of their **raw traces** is useful after the first
                  temporal update. Since one of the main purposes of the
                  **first temporal update** is to get rid of trash cells
                  and cells with noisy signal, successful parameter
                  selection is evidenced by dropped units with raw
                  traces that look like noise (no clear bursts of
                  activity). Alternatively, if cell-like activity is
                  seen in the raw trace of a dropped unit, this may
                  indicate that the selected parameters are too
                  conservative.

               -  When reading the temporal trace plot, "fitted spikes"
                  (green), "fitted signal" (orange), and "fitted calcium
                  trace" (blue), are all alligned to the "raw signal"
                  based upon the model. Ideally, we want only one spike
                  for each burst of signal, with "fitted signal" and
                  "fitted calcium trace" decaying in a manner that
                  follows the raw signal. Below is the temporal plot of
                  an example unit using different **sparse_panel**:

               .. rubric:: Example Temporal
                  Traces\ `¶ <#Example-Temporal-Traces>`__
                  :name: Example-Temporal-Traces

               |example temporal traces|

               Here, the top trace is when **sparse_panel** = 1, and we
               can see that there are lots of small spikes at the
               bottom, indicating we may want to increase the
               **sparse_panel** to get rid of them. However, when we are
               using **sparse_panel** = 10 (bottom panel), it's clear
               that we are missing real spikes from raw signal. Thus,
               the middle panel with **sparse_panel** = 3 fits the raw
               signal the best here.

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    The code below produces plots of temporal traces and spikes after the first temporal update and allows us to compare them to the signal originiating from the initialization step.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            The code below produces plots of temporal traces and spikes
            after the first temporal update and allows us to compare
            them to the signal originiating from the initialization
            step.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       YrA, C_temporal, S_temporal, B_temporal, C0_temporal, sig_temporal, g_temporal, scale = update_temporal(

                                    .. code:: CodeMirror-line

                                           Y, A_spatial, b_spatial, C_spatial, f_spatial, sn_spatial, **param_first_temporal)

                                    .. code:: CodeMirror-line

                                       A_temporal = A_spatial.sel(unit_id = C_temporal.coords['unit_id'])

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       opts_im = dict(frame_width=500, aspect=2, colorbar=True, cmap='Viridis', logz=True)

                                    .. code:: CodeMirror-line

                                       (regrid(hv.Image(C_init.compute().rename('ci'), kdims=['frame', 'unit_id'])).opts(**opts_im).relabel("Temporal Trace Initial")

                                    .. code:: CodeMirror-line

                                        + hv.Div('')

                                    .. code:: CodeMirror-line

                                        + regrid(hv.Image(C_temporal.compute().rename('c1'), kdims=['frame', 'unit_id'])).opts(**opts_im).relabel("Temporal Trace First Update")

                                    .. code:: CodeMirror-line

                                        + regrid(hv.Image(S_temporal.compute().rename('s1'), kdims=['frame', 'unit_id'])).opts(**opts_im).relabel("Spikes First Update")

                                    .. code:: CodeMirror-line

                                       ).cols(2)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    The following cell of code allows us to visualize units that were dropped during the first temporal update.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            The following cell of code allows us to visualize units that
            were dropped during the first temporal update.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           h, w = A_spatial.sizes['height'], A_spatial.sizes['width']

                                    .. code:: CodeMirror-line

                                           im_opts = dict(aspect=w/h, frame_width=500, cmap='Viridis')

                                    .. code:: CodeMirror-line

                                           cr_opts = dict(aspect=3, frame_width=1000)

                                    .. code:: CodeMirror-line

                                           bad_units = list(set(A_spatial.coords['unit_id'].values) - set(A_temporal.coords['unit_id'].values))

                                    .. code:: CodeMirror-line

                                           bad_units.sort()

                                    .. code:: CodeMirror-line

                                           if len(bad_units)>0:

                                    .. code:: CodeMirror-line

                                               hv_res = (hv.NdLayout({

                                    .. code:: CodeMirror-line

                                                   "Spatial Footprin": regrid(hv.Dataset(A_spatial.sel(unit_id=bad_units).compute().rename('A'))

                                    .. code:: CodeMirror-line

                                                                              .to(hv.Image, kdims=['width', 'height'])).opts(**im_opts),

                                    .. code:: CodeMirror-line

                                                   "Spatial Footprints of Accepted Units": regrid(hv.Image(A_temporal.sum('unit_id').compute().rename('A'), kdims=['width', 'height'])).opts(**im_opts)

                                    .. code:: CodeMirror-line

                                               })

                                    .. code:: CodeMirror-line

                                                         + datashade(hv.Dataset(YrA.sel(unit_id=bad_units).rename('raw'))

                                    .. code:: CodeMirror-line

                                                                     .to(hv.Curve, kdims=['frame'])).opts(**cr_opts).relabel("Temporal Trace")).cols(1)

                                    .. code:: CodeMirror-line

                                               display(hv_res)

                                    .. code:: CodeMirror-line

                                           else:

                                    .. code:: CodeMirror-line

                                               print("No rejected units to display")

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Lastly, we can visualize the activity of each unit. There are four traces in the top plot: "Raw Signal" corresponds to `YrA`, "Fitted Spikes" to `S_temporal`, "Fitted Calcium Trace" to `C_temporal` and "Fitted Signal" to `sig_temporal`. The latter two traces usually overlap with each other since `B_temporal` and `C0_temporal` are often equal **0**. Sadly, due to large number of frames and the limitation of our browser, it is usually only possible to visualize 50 units at a time, hence `select(unit_id=slice(0, 50))`. Nevertheless it gives us an idea of how things went. Put in other numbers if you want to see other units. 

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Lastly, we can visualize the activity of each unit. There
            are four traces in the top plot: "Raw Signal" corresponds to
            ``YrA``, "Fitted Spikes" to ``S_temporal``, "Fitted Calcium
            Trace" to ``C_temporal`` and "Fitted Signal" to
            ``sig_temporal``. The latter two traces usually overlap with
            each other since ``B_temporal`` and ``C0_temporal`` are
            often equal **0**. Sadly, due to large number of frames and
            the limitation of our browser, it is usually only possible
            to visualize 50 units at a time, hence
            ``select(unit_id=slice(0, 50))``. Nevertheless it gives us
            an idea of how things went. Put in other numbers if you want
            to see other units.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           display(visualize_temporal_update(YrA.compute(), C_temporal.compute(), S_temporal.compute(),

                                    .. code:: CodeMirror-line

                                                                             g_temporal.compute(), sig_temporal.compute(), A_temporal.compute()))

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## merge units

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: merge units\ `¶ <#merge-units>`__
               :name: merge-units

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    One thing CNMF cannot do is merge together units that belong to the same cell. Even though we tried something similar during [initialization](#initialization), we might miss some, and it is better to do it here again. Recall the parameters:

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ```python

                                 .. code:: CodeMirror-line

                                    param_first_merge = {

                                 .. code:: CodeMirror-line

                                        'thres_corr': 0.9}

                                 .. code:: CodeMirror-line

                                    ```

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    The idea is straight-forward and based purely on pearson correlation of temporal activities. Any units whose spatial footprints share at least one pixel are considered potential targets for merging, and any of these units that have a pearson correlation of temporal activities higher than `thres_corr` will be merged. 

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            One thing CNMF cannot do is merge together units that belong
            to the same cell. Even though we tried something similar
            during `initialization <#initialization>`__, we might miss
            some, and it is better to do it here again. Recall the
            parameters:

            ::

               param_first_merge = {
                   'thres_corr': 0.9}

            The idea is straight-forward and based purely on pearson
            correlation of temporal activities. Any units whose spatial
            footprints share at least one pixel are considered potential
            targets for merging, and any of these units that have a
            pearson correlation of temporal activities higher than
            ``thres_corr`` will be merged.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       A_mrg, sig_mrg, add_list = unit_merge(A_temporal, sig_temporal, [S_temporal, C_temporal], **param_first_merge)

                                    .. code:: CodeMirror-line

                                       S_mrg, C_mrg = add_list[:]

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Now you can visualize the results of unit merging. The left panel shows the original temporal signal, while the right panel shows the temporal signal after merging.

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-info">

                                 .. code:: CodeMirror-line

                                    Ideally, you want to see units in the left panel with <strong>too</strong> similar of signals, merged in the right penal. Adjust the <strong>thres_corr</strong> in <strong>param_first_merge</strong> accordingly.

                                 .. code:: CodeMirror-line

                                    </div>    

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Now you can visualize the results of unit merging. The left
            panel shows the original temporal signal, while the right
            panel shows the temporal signal after merging.

            .. container:: alert alert-info

               Ideally, you want to see units in the left panel with
               **too** similar of signals, merged in the right penal.
               Adjust the **thres_corr** in **param_first_merge**
               accordingly.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       opts_im = dict(frame_width=500, aspect=2, colorbar=True, cmap='Viridis', logz=True)

                                    .. code:: CodeMirror-line

                                       (regrid(hv.Image(sig_temporal.compute().rename('c1'), kdims=['frame', 'unit_id'])).relabel("Temporal Signals Before Merge").opts(**opts_im) +

                                    .. code:: CodeMirror-line

                                       regrid(hv.Image(sig_mrg.compute().rename('c2'), kdims=['frame', 'unit_id'])).relabel("Temporal Signals After Merge").opts(**opts_im))

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## test parameters for spatial update

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: test parameters for spatial
               update\ `¶ <#test-parameters-for-spatial-update>`__
               :name: test-parameters-for-spatial-update

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    This section is almost identical to the [first time](#test-parameters-for-first-spatial-update) we explore spatial parameters, except for changes in variable names.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            This section is almost identical to the `first
            time <#test-parameters-for-first-spatial-update>`__ we
            explore spatial parameters, except for changes in variable
            names.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           units = np.random.choice(A_mrg.coords['unit_id'], 10, replace=False)

                                    .. code:: CodeMirror-line

                                           units.sort()

                                    .. code:: CodeMirror-line

                                           A_sub = A_mrg.sel(unit_id=units).persist()

                                    .. code:: CodeMirror-line

                                           sig_sub = sig_mrg.sel(unit_id=units).persist()

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-info">  

                                 .. code:: CodeMirror-line

                                    Again, you can simply <strong>add</strong> the values that you want to test to <strong>sprs_ls</strong>. Pragmatically, it's generally fine to use the same <strong>sprs_ls</strong> from the first spatial update or one that is a little smaller.

                                 .. code:: CodeMirror-line

                                    </div>

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. container:: alert alert-info

               Again, you can simply **add** the values that you want to
               test to **sprs_ls**. Pragmatically, it's generally fine
               to use the same **sprs_ls** from the first spatial update
               or one that is a little smaller.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           sprs_ls = [0.001, 0.005, 0.01]

                                    .. code:: CodeMirror-line

                                           A_dict = dict()

                                    .. code:: CodeMirror-line

                                           C_dict = dict()

                                    .. code:: CodeMirror-line

                                           for cur_sprs in sprs_ls:

                                    .. code:: CodeMirror-line

                                               cur_A, cur_b, cur_C, cur_f = update_spatial(

                                    .. code:: CodeMirror-line

                                                   Y, A_sub, b_init, sig_sub, f_init,

                                    .. code:: CodeMirror-line

                                                   sn_spatial, dl_wnd=param_second_spatial['dl_wnd'], sparse_penal=cur_sprs)

                                    .. code:: CodeMirror-line

                                               if cur_A.sizes['unit_id']:

                                    .. code:: CodeMirror-line

                                                   A_dict[cur_sprs] = cur_A.compute()

                                    .. code:: CodeMirror-line

                                                   C_dict[cur_sprs] = cur_C.compute()

                                    .. code:: CodeMirror-line

                                           hv_res = visualize_spatial_update(A_dict, C_dict, kdims=['sparse penalty'])

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           display(hv_res)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-info" role="alert">

                                 .. code:: CodeMirror-line

                                    Again, use the visualization results here to help choose the <strong>sparse_panel</strong> and <strong>dl_wnd</strong>, to use in the next step.  Be sure to update the paramaters.

                                 .. code:: CodeMirror-line

                                    </div>

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. container:: alert alert-info

               Again, use the visualization results here to help choose
               the **sparse_panel** and **dl_wnd**, to use in the next
               step. Be sure to update the paramaters.

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## second spatial update

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: second spatial
               update\ `¶ <#second-spatial-update>`__
               :name: second-spatial-update

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Below is the second iteration of the spatial update. It is identical to [first spatial update](#first-spatial-update), with the exception of appending **it2**s after the variable names, standing for "iteration 2". From this, it should be apparentt that if you you can modify the code to have more cycles of spatial updates followed by temporal updates. Simply add more sections like this and [the section below](#second-temporal-update).

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Below is the second iteration of the spatial update. It is
            identical to `first spatial
            update <#first-spatial-update>`__, with the exception of
            appending **it2**\ s after the variable names, standing for
            "iteration 2". From this, it should be apparentt that if you
            you can modify the code to have more cycles of spatial
            updates followed by temporal updates. Simply add more
            sections like this and `the section
            below <#second-temporal-update>`__.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       A_spatial_it2, b_spatial_it2, C_spatial_it2, f_spatial_it2 = update_spatial(

                                    .. code:: CodeMirror-line

                                           Y, A_mrg, b_spatial, sig_mrg, f_spatial, sn_spatial, **param_second_spatial)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       opts = dict(aspect=A_spatial_it2.sizes['width']/A_spatial_it2.sizes['height'], frame_width=500, colorbar=True, cmap='Viridis')

                                    .. code:: CodeMirror-line

                                       (regrid(hv.Image(A_mrg.sum('unit_id').compute().rename('A'), kdims=['width', 'height'])).opts(**opts).relabel("Spatial Footprints First Update")

                                    .. code:: CodeMirror-line

                                       + regrid(hv.Image((A_mrg.fillna(0) > 0).sum('unit_id').compute().rename('A'), kdims=['width', 'height']), aggregator='max').opts(**opts).relabel("Binary Spatial Footprints First Update")

                                    .. code:: CodeMirror-line

                                       + regrid(hv.Image(A_spatial_it2.sum('unit_id').compute().rename('A'), kdims=['width', 'height'])).opts(**opts).relabel("Spatial Footprints Second Update")

                                    .. code:: CodeMirror-line

                                       + regrid(hv.Image((A_spatial_it2 > 0).sum('unit_id').compute().rename('A'), kdims=['width', 'height']), aggregator='max').opts(**opts).relabel("Binary Spatial Footprints Second Update")).cols(2)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Here again, visualize the result of second spatial update, if not satisfying with this, feel free to reset **param_second_spatial** and rerun this session.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Here again, visualize the result of second spatial update,
            if not satisfying with this, feel free to reset
            **param_second_spatial** and rerun this session.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       opts_im = dict(aspect=b_spatial_it2.sizes['width'] / b_spatial_it2.sizes['height'], frame_width=500, colorbar=True, cmap='Viridis')

                                    .. code:: CodeMirror-line

                                       opts_cr = dict(aspect=2, frame_height=int(500 * b_spatial_it2.sizes['height'] / b_spatial_it2.sizes['width']))

                                    .. code:: CodeMirror-line

                                       (regrid(hv.Image(b_spatial.compute(), kdims=['width', 'height'])).opts(**opts_im).relabel('Background Spatial First Update')

                                    .. code:: CodeMirror-line

                                        + datashade(hv.Curve(f_spatial.compute(), kdims=['frame'])).opts(**opts_cr).relabel('Background Temporal First Update')

                                    .. code:: CodeMirror-line

                                        + regrid(hv.Image(b_spatial_it2.compute(), kdims=['width', 'height'])).opts(**opts_im).relabel('Background Spatial Second Update')

                                    .. code:: CodeMirror-line

                                        + datashade(hv.Curve(f_spatial_it2.compute(), kdims=['frame'])).opts(**opts_cr).relabel('Background Temporal Second Update')

                                    .. code:: CodeMirror-line

                                       ).cols(2)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## test parameters for temporal update

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: test parameters for temporal
               update\ `¶ <#test-parameters-for-temporal-update>`__
               :name: test-parameters-for-temporal-update

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    This section is almost identical to the [first time](#test-parameters-for-first-temporal-update) except for variable names.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            This section is almost identical to the `first
            time <#test-parameters-for-first-temporal-update>`__ except
            for variable names.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           units = np.random.choice(A_spatial_it2.coords['unit_id'], 10, replace=False)

                                    .. code:: CodeMirror-line

                                           units.sort()

                                    .. code:: CodeMirror-line

                                           A_sub = A_spatial_it2.sel(unit_id=units).persist()

                                    .. code:: CodeMirror-line

                                           C_sub = C_spatial_it2.sel(unit_id=units).persist()

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-info" role="alert">

                                 .. code:: CodeMirror-line

                                    Generally, our aim here for the second temporal update is too refine the model and make the "fitted spikes", "fitted signal", and "fitted calcium trace" fit the "raw signal" better.

                                 .. code:: CodeMirror-line

                                    </div>

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. container:: alert alert-info

               Generally, our aim here for the second temporal update is
               too refine the model and make the "fitted spikes",
               "fitted signal", and "fitted calcium trace" fit the "raw
               signal" better.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           p_ls = [1]

                                    .. code:: CodeMirror-line

                                           sprs_ls = [0.01, 0.05, 0.1]

                                    .. code:: CodeMirror-line

                                           add_ls = [20]

                                    .. code:: CodeMirror-line

                                           noise_ls = [0.06]

                                    .. code:: CodeMirror-line

                                           YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict = [dict() for _ in range(6)]

                                    .. code:: CodeMirror-line

                                           YrA = compute_trace(Y, A_sub, b_spatial, C_sub, f_spatial).persist()

                                    .. code:: CodeMirror-line

                                           for cur_p, cur_sprs, cur_add, cur_noise in itt.product(p_ls, sprs_ls, add_ls, noise_ls):

                                    .. code:: CodeMirror-line

                                               ks = (cur_p, cur_sprs, cur_add, cur_noise)

                                    .. code:: CodeMirror-line

                                               print("p:{}, sparse penalty:{}, additional lag:{}, noise frequency:{}"

                                    .. code:: CodeMirror-line

                                                     .format(cur_p, cur_sprs, cur_add, cur_noise))

                                    .. code:: CodeMirror-line

                                               YrA, cur_C, cur_S, cur_B, cur_C0, cur_sig, cur_g, cur_scal = update_temporal(

                                    .. code:: CodeMirror-line

                                                   Y, A_sub, b_spatial, C_sub, f_spatial, sn_spatial, YrA=YrA,

                                    .. code:: CodeMirror-line

                                                   sparse_penal=cur_sprs, p=cur_p, use_spatial=False, use_smooth=True,

                                    .. code:: CodeMirror-line

                                                   add_lag = cur_add, noise_freq=cur_noise)

                                    .. code:: CodeMirror-line

                                               YA_dict[ks], C_dict[ks], S_dict[ks], g_dict[ks], sig_dict[ks], A_dict[ks] = (

                                    .. code:: CodeMirror-line

                                                   YrA.compute(), cur_C.compute(), cur_S.compute(), cur_g.compute(), cur_sig.compute(), A_sub.compute())

                                    .. code:: CodeMirror-line

                                           hv_res = visualize_temporal_update(

                                    .. code:: CodeMirror-line

                                               YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict,

                                    .. code:: CodeMirror-line

                                               kdims=['p', 'sparse penalty', 'additional lag', 'noise frequency'])

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           display(hv_res)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## second temporal update

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: second temporal
               update\ `¶ <#second-temporal-update>`__
               :name: second-temporal-update

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    This section is identical to the [first temporal update](#first-temporal-update) except for variable names.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            This section is identical to the `first temporal
            update <#first-temporal-update>`__ except for variable
            names.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       YrA, C_temporal_it2, S_temporal_it2, B_temporal_it2, C0_temporal_it2, sig_temporal_it2, g_temporal_it2, scale_temporal_it2 = update_temporal(

                                    .. code:: CodeMirror-line

                                           Y, A_spatial_it2, b_spatial_it2, C_spatial_it2, f_spatial_it2, sn_spatial, **param_second_temporal)

                                    .. code:: CodeMirror-line

                                       A_temporal_it2 = A_spatial_it2.sel(unit_id=C_temporal_it2.coords['unit_id'])

                                    .. code:: CodeMirror-line

                                       g_temporal_it2 = g_temporal_it2.sel(unit_id=C_temporal_it2.coords['unit_id'])

                                    .. code:: CodeMirror-line

                                       A_temporal_it2 = rechunk_like(A_temporal_it2, A_spatial_it2)

                                    .. code:: CodeMirror-line

                                       g_temporal_it2 = rechunk_like(g_temporal_it2, C_temporal_it2)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       opts_im = dict(frame_width=500, aspect=2, colorbar=True, cmap='Viridis', logz=True)

                                    .. code:: CodeMirror-line

                                       (regrid(hv.Image(C_mrg.compute().rename('c1'), kdims=['frame', 'unit_id'])).opts(**opts_im).relabel("Temporal Trace First Update")

                                    .. code:: CodeMirror-line

                                        + regrid(hv.Image(S_mrg.compute().rename('s1'), kdims=['frame', 'unit_id'])).opts(**opts_im).relabel("Spikes First Update")

                                    .. code:: CodeMirror-line

                                        + regrid(hv.Image(C_temporal_it2.compute().rename('c2').rename(unit_id='unit_id_it2'), kdims=['frame', 'unit_id_it2'])).opts(**opts_im).relabel("Temporal Trace Second Update")

                                    .. code:: CodeMirror-line

                                        + regrid(hv.Image(S_temporal_it2.compute().rename('s2').rename(unit_id='unit_id_it2'), kdims=['frame', 'unit_id_it2'])).opts(**opts_im).relabel("Spikes Second Update")).cols(2)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Here we visualize all the units that are dropped during this step.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Here we visualize all the units that are dropped during this
            step.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           h, w = A_spatial_it2.sizes['height'], A_spatial_it2.sizes['width']

                                    .. code:: CodeMirror-line

                                           im_opts = dict(aspect=w/h, frame_width=500, cmap='Viridis')

                                    .. code:: CodeMirror-line

                                           cr_opts = dict(aspect=3, frame_width=1000)

                                    .. code:: CodeMirror-line

                                           bad_units = list(set(A_spatial_it2.coords['unit_id'].values) - set(A_temporal_it2.coords['unit_id'].values))

                                    .. code:: CodeMirror-line

                                           bad_units.sort()

                                    .. code:: CodeMirror-line

                                           if len(bad_units)>0:

                                    .. code:: CodeMirror-line

                                               hv_res = (hv.NdLayout({

                                    .. code:: CodeMirror-line

                                                   "Spatial Footprin": regrid(hv.Dataset(A_spatial_it2.sel(unit_id=bad_units).compute().rename('A'))

                                    .. code:: CodeMirror-line

                                                                              .to(hv.Image, kdims=['width', 'height'])).opts(**im_opts),

                                    .. code:: CodeMirror-line

                                                   "Spatial Footprints of Accepted Units": regrid(hv.Image(A_temporal_it2.sum('unit_id').compute().rename('A'), kdims=['width', 'height'])).opts(**im_opts)

                                    .. code:: CodeMirror-line

                                               })

                                    .. code:: CodeMirror-line

                                                         + datashade(hv.Dataset(YrA.sel(unit_id=bad_units).compute().rename('raw'))

                                    .. code:: CodeMirror-line

                                                                     .to(hv.Curve, kdims=['frame'])).opts(**cr_opts).relabel("Temporal Trace")).cols(1)

                                    .. code:: CodeMirror-line

                                               display(hv_res)

                                    .. code:: CodeMirror-line

                                           else:

                                    .. code:: CodeMirror-line

                                               print("No rejected units to display")

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           display(visualize_temporal_update(YrA.compute(), C_temporal_it2.compute(), S_temporal_it2.compute(),

                                    .. code:: CodeMirror-line

                                                                             g_temporal_it2.compute(), sig_temporal_it2.compute(), A_temporal_it2.compute()))

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## save results

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: save results\ `¶ <#save-results>`__
               :name: save-results

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Finally, we save our results in the `minian` dataset. Note that you can save any other variables by calling `save_minian` and using the code below as a reference. For example, you might want to consider using `sig_temporal` instead of `C_temporal` for your subsequent analysis. Also, you are not restricted to use the [netcdf](https://www.unidata.ucar.edu/software/netcdf/) format, though it is recommended. [Explore the xarray documentation](http://xarray.pydata.org/en/stable/io.html) for all IO options, and moreover, [numpy IO capabilities](https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.io.html), since `xarray` is built on top of `numpy`.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Finally, we save our results in the ``minian`` dataset. Note
            that you can save any other variables by calling
            ``save_minian`` and using the code below as a reference. For
            example, you might want to consider using ``sig_temporal``
            instead of ``C_temporal`` for your subsequent analysis.
            Also, you are not restricted to use the
            `netcdf <https://www.unidata.ucar.edu/software/netcdf/>`__
            format, though it is recommended. `Explore the xarray
            documentation <http://xarray.pydata.org/en/stable/io.html>`__
            for all IO options, and moreover, `numpy IO
            capabilities <https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.io.html>`__,
            since ``xarray`` is built on top of ``numpy``.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       A_temporal_it2 = save_minian(A_temporal_it2.rename('A'), **param_save_minian)

                                    .. code:: CodeMirror-line

                                       C_temporal_it2 = save_minian(C_temporal_it2.rename('C'), **param_save_minian)

                                    .. code:: CodeMirror-line

                                       S_temporal_it2 = save_minian(S_temporal_it2.rename('S'), **param_save_minian)

                                    .. code:: CodeMirror-line

                                       g_temporal_it2 = save_minian(g_temporal_it2.rename('g'), **param_save_minian)

                                    .. code:: CodeMirror-line

                                       C0_temporal_it2 = save_minian(C0_temporal_it2.rename('C0'), **param_save_minian)

                                    .. code:: CodeMirror-line

                                       B_temporal_it2 = save_minian(B_temporal_it2.rename('B'), **param_save_minian)

                                    .. code:: CodeMirror-line

                                       b_spatial_it2 = save_minian(b_spatial_it2.rename('b'), **param_save_minian)

                                    .. code:: CodeMirror-line

                                       f_spatial_it2 = save_minian(f_spatial_it2.rename('f'), **param_save_minian)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    ## visualization

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            .. rubric:: visualization\ `¶ <#visualization>`__
               :name: visualization

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Here we load the data we just saved for visualization purposes.

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Here we load the data we just saved for visualization
            purposes.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       minian = open_minian(dpath,

                                    .. code:: CodeMirror-line

                                                            fname=param_save_minian['fname'],

                                    .. code:: CodeMirror-line

                                                            backend=param_save_minian['backend'])

                                    .. code:: CodeMirror-line

                                       varr = load_videos(dpath, **param_load_videos)

                                    .. code:: CodeMirror-line

                                       chk = get_optimal_chk(varr.astype(float), dim_grp=[('frame',), ('height', 'width')])

                                    .. code:: CodeMirror-line

                                       varr = varr.chunk(dict(frame=chk['frame']))

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    The following cell calls `generate_videos` to create a video that can help us quickly visualize the results. Under default settings, this video will be saved in your data folder. `generate_videos` takes in the dataset that contains cnmf results, an array representation of the raw video, the full path to the output video file, and a `dict` specifying chunks for performance. The resulting video will have four parts - Top left is the **Raw Video** after pre-processing and motion correction `minian['org']`; Top right is the **Processed Video** `minian['Y']` (that is, after pre-processing and motion correction); Bottom left is the **Residule**, that is **Raw Video** - **Units**. Bottom right is the **Units** from CNMF `minian['A'].dot(minian['C'], 'unit_id')`;

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            The following cell calls ``generate_videos`` to create a
            video that can help us quickly visualize the results. Under
            default settings, this video will be saved in your data
            folder. ``generate_videos`` takes in the dataset that
            contains cnmf results, an array representation of the raw
            video, the full path to the output video file, and a
            ``dict`` specifying chunks for performance. The resulting
            video will have four parts - Top left is the **Raw Video**
            after pre-processing and motion correction
            ``minian['org']``; Top right is the **Processed Video**
            ``minian['Y']`` (that is, after pre-processing and motion
            correction); Bottom left is the **Residule**, that is **Raw
            Video** - **Units**. Bottom right is the **Units** from CNMF
            ``minian['A'].dot(minian['C'], 'unit_id')``;

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       generate_videos(

                                    .. code:: CodeMirror-line

                                           minian, varr, dpath, param_save_minian['fname'] + ".mp4", scale='auto')

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    Here we have a `CNMFViewer` to visualize the final results. 

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-info">

                                 .. code:: CodeMirror-line

                                    <strong>Top Left panel</strong>-- spatial footprints of all cells (a sum projection).   

                                 .. code:: CodeMirror-line

                                        

                                 .. code:: CodeMirror-line

                                    <strong>Top Middle panel</strong> `if UseAC` -- the dot product of A (spatial footprint) and C (temporal activities) matrix of selected neurons.   

                                 .. code:: CodeMirror-line

                                                                  `if not UseAC` -- the spatial footprints of selected neurons (a sum projection).

                                 .. code:: CodeMirror-line

                                                 

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <strong>Top Right panel</strong>-- raw video after pre-processing and motion correction, which is the movie that's fed in as `org` to `CNMFViewer`, if nothing is fed in it's `minian['org']`.

                                 .. code:: CodeMirror-line

                                    </div>

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-info">

                                 .. code:: CodeMirror-line

                                     

                                 .. code:: CodeMirror-line

                                    The <strong>Bottom Left Controller Panel</strong> has several useful features:

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <strong>Refresh</strong> -- refreshes the data when you switch to a new group of units and it is not loading properly.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <strong>Load Data</strong> -- loads the data into memory, which will take some time by itself, but will make the later visualization faster.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <strong>UseAC</strong> check box -- choose whether or not you want the middle panel to be the dot product of A (spatial footprint) and C (temporal activities) matrix of selected neurons. Note that this will make visualization process slower.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <strong>Normalize</strong> -- normalizes the bottom middle trace and spike plot for each unit to itself.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <strong>ShowC</strong> -- shows calcium traces for each unit across time in the bottom middle plot.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <strong>ShowS</strong> -- shows spikes for each unit across time in the bottom middle plot.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <strong>Previous Group</strong> and <strong>Next Group</strong> buttons -- allow you to easily go backward/forward to another group of units.

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <strong>Video Play Panel</strong> -- lets you play the top middle and right panel in real time.

                                 .. code:: CodeMirror-line

                                    </div>

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-info">

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    The <strong>Bottom Middle Panel</strong> contains plots of units along the time axis. Each group will have 4-5 units showing in the plot. Combine the plot with the videos to check the quality of your CNMF results. 

                                 .. code:: CodeMirror-line

                                    </div>

                                 .. code:: CodeMirror-line

                                    ​

                                 .. code:: CodeMirror-line

                                    <div class="alert alert-info">

                                 .. code:: CodeMirror-line

                                        

                                 .. code:: CodeMirror-line

                                    The <strong>Bottom Right panel</strong>. is a labeling tool for you to manually kick out "bad" units by labelling them (they will be demarcated with a `-1`). You can also flag units to be merged if their temporal activities and spatial footprint suggest they should be.

                                 .. code:: CodeMirror-line

                                    </div>

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            Here we have a ``CNMFViewer`` to visualize the final
            results.

            .. container:: alert alert-info

               **Top Left panel**-- spatial footprints of all cells (a
               sum projection).
               | **Top Middle panel** ``if UseAC`` -- the dot product of
                 A (spatial footprint) and C (temporal activities)
                 matrix of selected neurons.
               | ``if not UseAC`` -- the spatial footprints of selected
                 neurons (a sum projection).

               **Top Right panel**-- raw video after pre-processing and
               motion correction, which is the movie that's fed in as
               ``org`` to ``CNMFViewer``, if nothing is fed in it's
               ``minian['org']``.

            .. container:: alert alert-info

               The **Bottom Left Controller Panel** has several useful
               features:

               **Refresh** -- refreshes the data when you switch to a
               new group of units and it is not loading properly.

               **Load Data** -- loads the data into memory, which will
               take some time by itself, but will make the later
               visualization faster.

               **UseAC** check box -- choose whether or not you want the
               middle panel to be the dot product of A (spatial
               footprint) and C (temporal activities) matrix of selected
               neurons. Note that this will make visualization process
               slower.

               **Normalize** -- normalizes the bottom middle trace and
               spike plot for each unit to itself.

               **ShowC** -- shows calcium traces for each unit across
               time in the bottom middle plot.

               **ShowS** -- shows spikes for each unit across time in
               the bottom middle plot.

               **Previous Group** and **Next Group** buttons -- allow
               you to easily go backward/forward to another group of
               units.

               **Video Play Panel** -- lets you play the top middle and
               right panel in real time.

            .. container:: alert alert-info

               The **Bottom Middle Panel** contains plots of units along
               the time axis. Each group will have 4-5 units showing in
               the plot. Combine the plot with the videos to check the
               quality of your CNMF results.

            .. container:: alert alert-info

               The **Bottom Right panel**. is a labeling tool for you to
               manually kick out "bad" units by labelling them (they
               will be demarcated with a ``-1``). You can also flag
               units to be merged if their temporal activities and
               spatial footprint suggest they should be.

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       %%time

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           cnmfviewer = CNMFViewer(minian)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       hv.output(size=output_size)

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           display(cnmfviewer.show())

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

   .. container:: cell text_cell unselected rendered

      .. container:: prompt input_prompt

      .. container:: inner_cell

         .. container:: ctb_hideshow

            .. container:: celltoolbar

         .. container:: input_area

            .. container:: CodeMirror cm-s-default CodeMirror-wrap

               .. container::

               .. container:: CodeMirror-vscrollbar

                  .. container::

               .. container:: CodeMirror-hscrollbar

                  .. container::

               .. container:: CodeMirror-scrollbar-filler

               .. container:: CodeMirror-gutter-filler

               .. container:: CodeMirror-scroll

                  .. container:: CodeMirror-sizer

                     .. container::

                        .. container:: CodeMirror-lines

                           .. container::

                              .. container:: CodeMirror-measure

                              .. container:: CodeMirror-measure

                              .. container::

                              .. container:: CodeMirror-cursors

                                 .. container:: CodeMirror-cursor

                                     

                              .. container:: CodeMirror-code

                                 .. code:: CodeMirror-line

                                    The following code cell serves to save your manually changed labels

                  .. container::

                  .. container:: CodeMirror-gutters

         .. container:: text_cell_render rendered_html

            The following code cell serves to save your manually changed
            labels

   .. container:: cell code_cell unselected rendered

      .. container:: input

         .. container:: prompt_container

            .. container:: prompt input_prompt

               In [ ]:

            .. container:: run_this_cell

         .. container:: inner_cell

            .. container:: ctb_hideshow

               .. container:: celltoolbar

            .. container:: input_area

               .. container:: CodeMirror cm-s-ipython

                  .. container::

                  .. container:: CodeMirror-vscrollbar

                     .. container::

                  .. container:: CodeMirror-hscrollbar

                     .. container::

                  .. container:: CodeMirror-scrollbar-filler

                  .. container:: CodeMirror-gutter-filler

                  .. container:: CodeMirror-scroll

                     .. container:: CodeMirror-sizer

                        .. container::

                           .. container:: CodeMirror-lines

                              .. container::

                                 .. container:: CodeMirror-measure

                                    .. code:: CodeMirror-line-like

                                       xxxxxxxxxx

                                 .. container:: CodeMirror-measure

                                 .. container::

                                 .. container:: CodeMirror-cursors

                                    .. container:: CodeMirror-cursor

                                        

                                 .. container:: CodeMirror-code

                                    .. code:: CodeMirror-line

                                       if interactive:

                                    .. code:: CodeMirror-line

                                           save_minian(cnmfviewer.unit_labels, **param_save_minian)

                     .. container::

                     .. container:: CodeMirror-gutters

      .. container:: output_wrapper

         .. container:: out_prompt_overlay prompt

         .. container:: output

         .. container:: btn btn-default output_collapsed

            . . .

.. |workflow| image:: img/Workflow_v2.PNG
.. |Folder Structure| image:: img/folder_structure.png
.. |pnr_param| image:: img/pnr_param_v2.png
.. |1st spatial update param exploring| image:: img/sparse_panel_spatial_update.PNG
.. |dropped sample units| image:: img/first_tem_drop_v2.PNG
.. |example temporal traces| image:: img/first_tem_param.png
