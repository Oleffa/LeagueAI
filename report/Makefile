# Makefile to create PDF documents from LaTeX-Files
PROJECT=report
TEX=pdflatex
BIBTEX=bibtex
PDFVIEWER=evince
PDF=$(PROJECT).pdf
BUILDTEX=$(TEX) $(PROJECT).tex


all:
	$(BUILDTEX)
	$(BIBTEX) $(PROJECT)
	$(BUILDTEX)
	$(BUILDTEX)
	$(MAKE) close-pdf
	$(PDFVIEWER) $(PDF) & echo $$! > $(PDFVIEWER).PID  # Store the process ID of PDF viewer
	$(MAKE) clean

clean-all:
	@rm -f *.dvi *.log *.bak *.aux *.bbl *.blg *.idx *.ps *.eps *.pdf *.toc *.out *~
	echo "Cleaned all binaries"

clean:
	@rm -f *.log *.bak *.aux *.bbl *.blg *.idx *.toc *.out *~ *.sta *.gz
	echo "Cleaned Binaries"


close-pdf:
	# If old pdf is already open, close it first then open a new one.
	if [ -e $(PDFVIEWER).PID ]; then \
	   	kill -TERM $$(cat $(PDFVIEWER).PID) || true; \
	fi;

.PHONY : all clean
