#ifndef CAFFE_INNER_PRODUCT_BCM_LAYER_HPP_
#define CAFFE_INNER_PRODUCT_BCM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/fft.hpp"
#include "caffe/util/bcm.hpp"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 */
template <typename Dtype>
class InnerProductBCMLayer : public Layer<Dtype> {
 public:
  explicit InnerProductBCMLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~InnerProductBCMLayer();

  virtual inline const char* type() const { return "InnerProductBCM"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  // Number of output channels and input channels should be power of 2
  int bcm_N_;
  int bcm_K_;
  
  // BCM related
  int n_;
  int p_;
  int q_;
  int k_;
  int fft_k_;

  // FFT related
  cufftHandle w_plan_;
  cufftHandle x_plan_;
  cufftHandle y_plan_;
  cufftHandle ifft_forward_plan_;
  cufftHandle ifft_backward_weight_plan_;
  cufftHandle ifft_backward_data_plan_;

  // Temp workspace
  Dtype *fft_w_;
  Dtype *fft_x_;
  Dtype *fft_y_;
  Dtype *sum_w_;
  Dtype *sum_x_;
  Dtype *sum_y_;

  // Temp space for upsized data
  Dtype *upsized_bottom_;
  Dtype *upsized_top_diff_;

  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_BCM_LAYER_HPP_
