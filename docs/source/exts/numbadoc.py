"""
This sphinx extension aims to improve the documentation of numba-decorated
functions. It it inspired by the design of Celery's sphinx extension
'celery.contrib.sphinx'.

Usage
-----

Add the extension to your :file:`conf.py` configuration module:

.. code-block:: python

    extensions = (...,
                  'numbadoc')

This extension adds two configuration fields, which determine the prefix
printed before jitted functions in the reference documentation.
Overwrite these values in your :file:`conf.py` to change this behaviour

.. code-block:: python

    #defaults
    numba_jit_prefix = '@numba.jit'
    numba_vectorize_prefix = '@numba.vectorize'

With the extension installed `autodoc` will automatically find
numba decorated objects and generate the correct docs.

If a vecotrized function with fixed signatures is found, these are injected
into the docstring.
"""
from typing import List, Iterator
from sphinx.domains.python import PyFunction
from sphinx.ext.autodoc import FunctionDocumenter

from numba.core.dispatcher import Dispatcher
from numba.np.ufunc.dufunc import DUFunc


class NumbaFunctionDocumenter(FunctionDocumenter):
    """Document numba decorated functions."""

    def import_object(self) -> bool:
        """Import the object given by *self.modname* and *self.objpath* and set
        it as *self.object*.

        Returns True if successful, False if an error occurred.
        """
        success = super().import_object()
        if success:
            # Store away numba wrapper
            self.jitobj = self.object
            # And bend references to underlying python function
            if hasattr(self.object, "py_func"):
                self.object = self.object.py_func
            elif hasattr(self.object, "_dispatcher") and hasattr(
                self.object._dispatcher, "py_func"
            ):
                self.object = self.object._dispatcher.py_func
            else:
                success = False
        return success

    def process_doc(self, docstrings: List[List[str]]) -> Iterator[str]:
        """Let the user process the docstrings before adding them."""
        # Essentially copied from FunctionDocumenter
        for docstringlines in docstrings:
            if self.env.app:
                # let extensions preprocess docstrings
                # need to manually set 'what' to FunctionDocumenter.objtype
                # to not confuse preprocessors like napoleon with an objtype
                # that they don't know
                self.env.app.emit(
                    "autodoc-process-docstring",
                    FunctionDocumenter.objtype,
                    self.fullname,
                    self.object,
                    self.options,
                    docstringlines,
                )
            # This block inserts information about precompiled signatures
            # if this is a precompiled vectorized function
            if getattr(self.jitobj, "types", []) and getattr(
                self.jitobj, "_frozen", False
            ):
                s = "| *Precompiled signatures:" + str(self.jitobj.types) + "*"
                docstringlines.insert(0, s)
            yield from docstringlines


class JitDocumenter(NumbaFunctionDocumenter):
    """Document jit/njit decorated functions."""

    objtype = "jitfun"

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, Dispatcher) and hasattr(member, "py_func")


class VectorizeDocumenter(NumbaFunctionDocumenter):
    """Document vectorize decorated functions."""

    objtype = "vecfun"

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return (
            isinstance(member, DUFunc)
            and hasattr(member, "_dispatcher")
            and hasattr(member._dispatcher, "py_func")
        )


class JitDirective(PyFunction):
    """Sphinx jitfun directive."""

    def get_signature_prefix(self, sig):
        return self.env.config.numba_jit_prefix


class VectorizeDirective(PyFunction):
    """Sphinx vecfun directive."""

    def get_signature_prefix(self, sig):
        return self.env.config.numba_vectorize_prefix


class CUDAJitDirective(PyFunction):
    """Sphinx jitfun directive."""

    def get_signature_prefix(self, sig):
        return self.env.config.numba_cuda_jit_prefix


def setup(app):
    """Setup Sphinx extension."""
    # Register the new documenters and directives (autojitfun, autovecfun)
    # Set the default prefix which is printed in front of the function signature
    app.setup_extension("sphinx.ext.autodoc")
    app.add_autodocumenter(JitDocumenter)
    app.add_directive_to_domain("py", "jitfun", JitDirective)
    app.add_config_value("numba_jit_prefix", "@numba.jit", True)
    app.add_autodocumenter(VectorizeDocumenter)
    app.add_directive_to_domain("py", "vecfun", VectorizeDirective)
    app.add_config_value("numba_vectorize_prefix", "@numba.vectorize", True)

    return {"parallel_read_safe": True}
