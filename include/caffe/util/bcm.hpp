#ifndef CAFFE_UTIL_BCM_H_
#define CAFFE_UTIL_BCM_H_

#include <cuda_runtime.h>

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

}  // namespace caffe

#endif  // CAFFE_UTIL_FFT_H_
