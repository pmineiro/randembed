#include "mex.h"
#include "matrix.h"
#include <cmath>
#include <cstdint>
#include <thread>
#include <emmintrin.h>

/*
 * compilation, at matlab prompt: (adjust NUM_THREADS as appropriate)
 * 
 * == windows ==
 * 
 * mex OPTIMFLAGS="/O2" -largeArrayDims -DNUM_THREADS=4 dmsm.cpp
 * 
 * == linux ==
 * 
 * mex OPTIMFLAGS="-O3" -largeArrayDims -DNUM_THREADS=4 -v CXXFLAGS='$CXXFLAGS -std=c++0x -fPIC' dmsm.cpp
 */


#define DENSE_MATRIX_PARAMETER_IN     prhs[0]
#define SPARSE_MATRIX_PARAMETER_IN    prhs[1]
#define DIAG_VECTOR_PARAMETER_IN      prhs[2]
#define START_PARAMETER_IN            prhs[3]
#define END_PARAMETER_IN              prhs[4]

// X = D*S => X' = S'*D'

template<typename scalar>
static void
sparsetic_times_densetic (int nrhs, const mxArray* prhs[], mxArray* plhs[], size_t start, size_t end)
{
  mwIndex* ir = mxGetIr(SPARSE_MATRIX_PARAMETER_IN);  /* Row indexing      */
  mwIndex* jc = mxGetJc(SPARSE_MATRIX_PARAMETER_IN);  /* Column count      */
  double* s  = mxGetPr(SPARSE_MATRIX_PARAMETER_IN);   /* Non-zero elements */
  scalar* Btic = (scalar*) mxGetData(DENSE_MATRIX_PARAMETER_IN);
  mwSize Bcol = mxGetM(DENSE_MATRIX_PARAMETER_IN);
  scalar* Xtic = (scalar*) mxGetData(plhs[0]);
  mwSize Xcol = mxGetM(plhs[0]);

  size_t off = 0;

  if (nrhs > 3) {
    off = mxGetScalar(START_PARAMETER_IN) - 1;
  }

  if (nrhs == 2) {
    for (size_t i=start; i<end; ++i) {            /* Loop through rows of A (and X) */
      mwIndex stop = jc[off+i+1];
      for (mwIndex k=jc[off+i]; k<stop; ++k) {    /* Loop through non-zeros in ith row of A */
        double sk = s[k];
        scalar* Bticrow = Btic + ir[k] * Bcol;
        scalar* Xticrow = Xtic + i * Xcol;
        for (mwSize j=0; j<Xcol; ++j) {
          Xticrow[j] += sk * Bticrow[j];
        }
      }
    }
  }
  else {
    double* w = mxGetPr(DIAG_VECTOR_PARAMETER_IN);

    for (size_t i=start; i<end; ++i) {            /* Loop through rows of A (and X) */
      double wi = w[off+i];
      mwIndex stop = jc[off+i+1];
      for (mwIndex k=jc[off+i]; k<stop; ++k) {    /* Loop through non-zeros in ith row of A */
        double sk = wi * s[k];
        scalar* Bticrow = Btic + ir[k] * Bcol;
        scalar* Xticrow = Xtic + i * Xcol;
        for (mwSize j=0; j<Xcol; ++j) {
          Xticrow[j] += sk * Bticrow[j];
        }
      }
    }
  }
}

static int first = 1;

void mexFunction( int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray*prhs[] ) { 
  if (first) {
    mexPrintf("dmsm using NUM_THREADS=%u\n",NUM_THREADS);
    first=0;
  }

  switch (nrhs) {
    case 5:
      if (mxGetM(END_PARAMETER_IN) != 1 || mxGetN(END_PARAMETER_IN) != 1) {
        mexErrMsgTxt("End must be a scalar. Fail.");
        return;
      }

      // fall through

    case 4:
      if (mxGetM(START_PARAMETER_IN) != 1 || mxGetN(START_PARAMETER_IN) != 1) {
        mexErrMsgTxt("Start must be a scalar. Fail.");
        return;
      }

      // fall through

    case 3:
      if (mxIsSparse(DIAG_VECTOR_PARAMETER_IN)) {
        mexErrMsgTxt("Scale must be dense. Fail.");
        return;
      }

      if (mxGetM(DIAG_VECTOR_PARAMETER_IN) != 1 ||
          mxGetN(DIAG_VECTOR_PARAMETER_IN) != mxGetN(SPARSE_MATRIX_PARAMETER_IN)) {
        mexErrMsgTxt("Scale has incompatible shape. Fail.");
        return;
      }

      // fall through

    case 2:
      if (! mxIsSparse(SPARSE_MATRIX_PARAMETER_IN) || mxIsSparse(DENSE_MATRIX_PARAMETER_IN)) {
        mexErrMsgTxt("Require one sparse and one dense argument. Fail.");
        return;
      }
      if (mxGetM(SPARSE_MATRIX_PARAMETER_IN) != mxGetN(DENSE_MATRIX_PARAMETER_IN)) {
        mexErrMsgTxt("Arguments have incompatible shape. Fail.");
        return;
      }

      if (! mxIsSingle(DENSE_MATRIX_PARAMETER_IN) &&
          ! mxIsDouble(DENSE_MATRIX_PARAMETER_IN)) {
        mexErrMsgTxt("Dense argument must be double or single. Fail.");
        return;
      }

      break;
    default:
      mexErrMsgTxt("Wrong number of arguments. Fail.");
      return;
  }

  bool singlePrec = mxIsSingle(DENSE_MATRIX_PARAMETER_IN);

  void* Btic = mxGetData(DENSE_MATRIX_PARAMETER_IN);
  size_t Bcol = mxGetM(DENSE_MATRIX_PARAMETER_IN);

  size_t Arow = mxGetN(SPARSE_MATRIX_PARAMETER_IN);
  size_t start = 1;
  size_t end = Arow;

  switch (nrhs)
    {
      case 5:
        end = mxGetScalar(END_PARAMETER_IN);

        if (end > Arow) {
          mexErrMsgTxt("End is invalid.  Fail.");
          return;
        }

        // fall through
      case 4:
        start = mxGetScalar(START_PARAMETER_IN);

        if (start < 1) {
          mexErrMsgTxt("Start is invalid.  Fail.");
          return;
        }

        break;

      default:
        break;
    }

  plhs[0] = mxCreateNumericMatrix(Bcol, 1+end-start, (singlePrec) ? mxSINGLE_CLASS : mxDOUBLE_CLASS, mxREAL);

  std::thread t[NUM_THREADS];
  size_t quot = (1+end-start)/NUM_THREADS;

  for (size_t i = 0; i + 1 < NUM_THREADS; ++i) {
    if (singlePrec) {
      t[i] = std::thread(sparsetic_times_densetic<float>,
                         nrhs,
                         prhs,
                         plhs,
                         i * quot,
                         (i + 1) * quot);
    }
    else {
      t[i] = std::thread(sparsetic_times_densetic<double>,
                         nrhs,
                         prhs,
                         plhs,
                         i * quot,
                         (i + 1) * quot);
    }
  }

  if (singlePrec) {
    sparsetic_times_densetic<float> (nrhs, prhs, plhs, (NUM_THREADS - 1) * quot, 1 + end - start);
  }
  else {
    sparsetic_times_densetic<double> (nrhs, prhs, plhs, (NUM_THREADS - 1) * quot, 1 + end - start);
  }

  for (int i = 0; i + 1 < NUM_THREADS; ++i) {
    t[i].join ();
  }

  return;
}
