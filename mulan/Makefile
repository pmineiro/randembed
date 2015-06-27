.SECONDARY:

all:
	@echo "*** run traintestall.m from matlab ***" 1>&2
	@echo "*** and go get a sandwich ***" 1>&2

rebuildmat: bibtex.mat corel5k.mat mediamill.mat yeast.mat
	make mostlyclean

%.mat: %-train.x.txt %-train.y.txt %-test.x.txt %-test.y.txt
	matlab -nodisplay -nojvm -r "txtz2mat('$*'); exit;"

%-train.x.txt %-train.y.txt %-test.x.txt %-test.y.txt: %-train.csv %-test.csv %.xml
	python xmlcsv2xy.py $*

%.csv: %.arff
	java -cp /usr/share/java/weka.jar weka.core.converters.CSVSaver -i $< -o $@

%-train.arff %-test.arff %.xml: %.rar
	unrar x -y -cl -inul $< $*.xml $*-train.arff $*-test.arff
	touch $*-train.arff $*-test.arff $*.xml

%.rar:
	wget http://sourceforge.net/projects/mulan/files/datasets/$@

mostlyclean:
	rm -f *.csv *.arff *.bak *.xml *.txt

clean: mostlyclean
	rm -f *.mat

.PHONY: clean mostlyclean
