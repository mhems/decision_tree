TEX = pdflatex
TARGET = report

all:
	$(TEX) $(TARGET).tex
# compile twice to get references
	$(TEX) $(TARGET).tex
	rm -f $(TARGET).aux $(TARGET).log

clean:
	rm -f $(TARGET).pdf
