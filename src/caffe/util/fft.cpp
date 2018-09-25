#ifdef USE_CUDNN
#include "caffe/util/fft.hpp"

namespace caffe {
namespace cufft {

template <>
void fft(cufftHandle *plan, float *input, float *output) {
  CUFFT_CHECK(cufftExecR2C(*plan, input, (cuComplex *)output));
}

template <>
void fft(cufftHandle *plan, double *input, double *output) {
  CUFFT_CHECK(cufftExecD2Z(*plan, input, (cuDoubleComplex *)output));
}

template <>
void ifft(cufftHandle *plan, float *input, float *output) {
  CUFFT_CHECK(cufftExecC2R(*plan, (cuComplex *)input, output));
}

template <>
void ifft(cufftHandle *plan, double *input, double *output) {
  CUFFT_CHECK(cufftExecZ2D(*plan, (cuDoubleComplex *)input, output));
}

}  // namespace cufft
}  // namespace caffe
#endif
