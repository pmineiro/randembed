.SECONDARY:

all: dict

train-remapped.zip:
	@echo 'wget http://www.kaggle.com/c/lshtc/download/$@' 1>&2
	@echo '  (you need to log into kaggle to get this file)  ' 1>&2
	@exit 1

dict: train-remapped.zip train2mat
	gunzip -dc $< | ./train2mat 3>left 4>right 5>dict

oissrnd.m:
	wget -q http://www.mathworks.com/matlabcentral/fileexchange/1261-poissrnd/content/oissrnd.m

poissrnd.m: oissrnd.m
	@perl -MHTML::Entities -e '$$l=join "",<>; $$l=~m%<pre class="matlab-code">(.*?)</pre>%s; $$s=$$1; $$s=~s/break/return/; print decode_entities($$s)' $< > $@
