TEX = latex
PDF = dvipdf

TARGET = report

all:
	$(TEX) $(TARGET).tex
	$(PDF) $(TARGET).dvi

clean:
	rm -f $(TARGET).aux $(TARGET).dvi $(TARGET).log

cleanall:
	rm -f $(TARGET).aux $(TARGET).dvi $(TARGET).log
	rm -f $(TARGET).pdf
