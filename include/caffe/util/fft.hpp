#ifndef CAFFE_UTIL_FFT_H_
#define CAFFE_UTIL_FFT_H_

#include <cufft.h>
#include <cufftXt.h>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#define CUFFT_CHECK(condition) \
  do {\
    cufftResult status = condition; \
    CHECK_EQ(status, CUFFT_SUCCESS) << " "\
      << cufftGetErrorString(status); \
  } while(0)

inline const char* cufftGetErrorString(cufftResult status) {
  switch (status) {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";
    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";
    default:
      break;
  }
  return "Unknown cufft status";
}

namespace caffe {

namespace cufft {

enum FFTType {
  R2C = CUFFT_R2C, // Real to complex (interleaved)
  C2R = CUFFT_C2R, // Complex (interleaved) to real
  C2C = CUFFT_C2C, // Complex to complex (interleaved)
  D2Z = CUFFT_D2Z, // Double to double-complex (interleaved)
  Z2D = CUFFT_Z2D, // Double-complex (interleaved) to double
  Z2Z = CUFFT_Z2Z  // Double-complex to double-complex (interleaved)
};

enum FFTPlanType {
  FFT_1D = 0,
  FFT_2D,
  FFT_3D,
  FFT_MANY
};

enum FFTDirction {
  FFT = CUFFT_FORWARD,
  IFFT = CUFFT_INVERSE
};

inline void createFFTPlan(cufftHandle *plan) {
  CUFFT_CHECK(cufftCreate(plan));
}

inline size_t setFFTPlan(cufftHandle *plan, FFTPlanType plan_type, int nx, FFTType type, int batch) {
  size_t workspace_size = 0;
  switch (plan_type) {
    case FFT_1D:
      CUFFT_CHECK(cufftMakePlan1d(*plan, nx, (cufftType_t)type, batch, &workspace_size));
      LOG(INFO) << "FFT workspace size: " << workspace_size;
      break;
    default:
      LOG(FATAL) << "FFT plan type NOT supported";
  }
  return workspace_size;
}

template <typename Dtype>
void fft(cufftHandle *plan, Dtype *input, Dtype *output);

template <typename Dtype>
void ifft(cufftHandle *plan, Dtype *input, Dtype *output);

}  // namespace cufft

}  // namespace caffe

#endif  // CAFFE_UTIL_FFT_H_
