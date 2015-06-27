matlab
==========
This directory contains the matlab/octave code.

compiling the mex
-----------------
If you are using matlab, you should compile the mex to make everything go faster.  Hopefully you can just type
> make NUM_THREADS=6
 
and all the mex will be compiled for you.  Adjust NUM_THREADS based upon how much parallelism is appropriate for your setup.  (Sorry, it's not cool enough to auto-detect this).

If you lack a reasonable shell environment, you can execute the mex commands directly from matlab:

> &gt;&gt; mex OPTIMFLAGS="&#47;O2" -largeArrayDims -DNUM\_THREADS=2 -lmwblas -lmwlapack sparsequad.cpp  
> Building with 'Microsoft Visual C++ 2012'.  
> MEX completed successfully.  
> &gt;&gt; mex OPTIMFLAGS="&#47;O2" -largeArrayDims -DNUM\_THREADS=2 -lmwblas -lmwlapack dmsm.cpp  
> Building with 'Microsoft Visual C++ 2012'.  
> MEX completed successfully.  
> &gt;&gt; mex OPTIMFLAGS="&#47;O2" -largeArrayDims -DNUM\_THREADS=2 -lmwblas -lmwlapack sparseweightedsum.cpp  
> Building with 'Microsoft Visual C++ 2012'.  
> MEX completed successfully.
