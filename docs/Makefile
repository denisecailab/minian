# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

cleanup:
	rm -rf ../minian/.ipynb_checkpoints

convert_noted:
	rm -rf build && \
	rm -f source/pipeline_noted.rst && \
	cd .. && \
	jupyter nbconvert --to rst pipeline_noted.ipynb && \
	mv pipeline_noted.rst docs/source && \
	jupyter nbconvert --to rst cross-registration.ipynb && \
	mv cross-registration.rst docs/source && \
	cd -

copy_images:
	mkdir -p build/html && \
	cp -r ../img build/html

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile cleanup convert_noted copy_images
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
