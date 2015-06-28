aloi
==========
Reproduce [aloi](http://aloi.science.uva.nl/) dataset results from [this paper](http://arxiv.org/abs/1502.02710).

From a matlab (or octave) prompt, run 
> traintestall   
> ... (lots of hyperparameter tuning) ...   
> ... (you should go get a sandwich) ...

If you want to regenerate (or change) the matlab input data, you can try
> make rebuildmat  
> ... (builds matlab input files) ...  
 
but you will need matlab (not octave) to make that work.  Octave and txt2mat do not get along.

Thanks
----------
aloi\_train.vw.gz and aloi\_test.vw.gz were provided by John Langford.

Notice
----------
txt2mat is from [matlab central](http://www.mathworks.com/matlabcentral/fileexchange/18430-txt2mat) and is covered by the (BSD) license file [txt2mat.license](txt2mat.license).
