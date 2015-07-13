lshtc
==========
Reproduce [lshtc](https://www.kaggle.com/c/lshtc) dataset results from [this paper](http://arxiv.org/abs/1502.02710).  

This will only work under Matlab, and furthermore unless you have compiled the mex functions it will be intolerably slow.

If you lack a poissrnd function, you can try 
> make poissrnd.m

Otherwise, at a matlab prompt:
> &gt;&gt; traintestlshtc   
> ... (lots of number crunching) ...   

will eventually report test precision-at-1 for each epoch. It takes a while so you should go camping for the weekend.  (Building the embedding doesn't take much longer than lunch, but fitting the logistic is slow.)

Notice
----------
txt2mat is from [matlab central](http://www.mathworks.com/matlabcentral/fileexchange/18430-txt2mat) and is covered by the (BSD) license file [txt2mat.license](txt2mat.license).
