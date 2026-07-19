# Build the paper. Requires pdflatex + bibtex (MacTeX / TeX Live).
# acmart and its deps may live in the user tree (~/Library/texmf); pdflatex
# finds them automatically via kpathsea.

MAIN = main

.PHONY: all clean
all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib sections/*.tex
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).pdf sections/*.aux
