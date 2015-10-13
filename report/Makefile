TEX = pdflatex
TARGET = report

all: $(TARGET).aux
	$(TEX) $(TARGET).tex

$(TARGET).aux:
	$(TEX) $(TARGET).tex

clean:
	rm -f $(TARGET).aux $(TARGET).log

cleanall:
	rm -f $(TARGET).pdf $(TARGET).aux $(TARGET).log
