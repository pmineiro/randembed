.SECONDARY:

.PHONY: all rebuildmat

all: 
	
rebuildmat: trainleft testleft
	@rm -f aloi.mat
	@echo "*** now run traintestall to rebuild matlab data ***" 1>&2

trainleft: aloi_train.vw.gz
	zcat $< | perl ./vw2matlab 3>trainleft 4>trainright

testleft: aloi_test.vw.gz
	zcat $< | perl ./vw2matlab 3>testleft 4>testright

oissrnd.m:
	wget -q http://www.mathworks.com/matlabcentral/fileexchange/1261-poissrnd/content/oissrnd.m

poissrnd.m: oissrnd.m
	@perl -MHTML::Entities -e '$$l=join "",<>; $$l=~m%<pre class="matlab-code">(.*?)</pre>%s; $$s=$$1; $$s=~s/break/return/; print decode_entities($$s)' $< > $@

