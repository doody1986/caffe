#ifndef CAFFE_UTIL_BCM_H_
#define CAFFE_UTIL_BCM_H_

#include <cuda_runtime.h>
#include <cmath>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
void BCMForward(Dtype *fft_w, Dtype *fft_x, Dtype *y,
                int n, int p, int q, int k);

template <typename Dtype>
void BCMBackwardWeight(Dtype *fft_dy, Dtype *fft_x, Dtype *dw,
                int n, int p, int q, int k);

template <typename Dtype>
void BCMBackwardData(Dtype *fft_dy, Dtype *fft_w, Dtype *dx,
                int n, int p, int q, int k);

template <typename Dtype>
inline bool IsPowerOf2(Dtype n) {
  return (ceil(log2(n)) == floor(log2(n)));
}

template <typename Dtype>
inline int FindNextPowerOf2(Dtype n) {
  int ret = (int)n;
  int power = 0;
  if (!IsPowerOf2(n)) {
    power = ceil(log2(n));
    ret = pow(2, power);
  }
  return ret;
}

}  // namespace caffe

#endif  // CAFFE_UTIL_FFT_H_
