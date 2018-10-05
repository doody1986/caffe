#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_bcm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductBCMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();

  // Insert additional 0
  if (K_ != bcm_K_) {
    CUDA_CHECK(cudaMemset(upsized_bottom_, 0, bcm_K_ * M_ * sizeof(Dtype)));
    for (int m = 0; m < M_; m++) {
      CUDA_CHECK(cudaMemcpy(upsized_bottom_ + m * bcm_K_, bottom_data + m * K_,
                 K_ * sizeof(Dtype), cudaMemcpyDeviceToDevice));
    }
  }
  cufft::fft<Dtype>(&w_plan_, (Dtype *)weight, fft_w_);
  if (K_ != bcm_K_)
    cufft::fft<Dtype>(&x_plan_, (Dtype *)upsized_bottom_, fft_x_);
  else
    cufft::fft<Dtype>(&x_plan_, (Dtype *)bottom_data, fft_x_);
  CUDA_CHECK(cudaDeviceSynchronize());

  BCMForward<Dtype>(fft_w_,
             fft_x_,
             sum_y_,
             n_, p_, q_, fft_k_);
   
  CUDA_CHECK(cudaDeviceSynchronize());

  cufft::ifft<Dtype>(&ifft_forward_plan_, sum_y_, top_data);

  if (M_ == 1) {
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.gpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductBCMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (N_ != bcm_N_) {
    CUDA_CHECK(cudaMemset(upsized_top_diff_, 0, bcm_N_ * M_ * sizeof(Dtype)));
  }
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    if (N_ != bcm_N_) {
      for (int m = 0; m < M_; m++) {
       CUDA_CHECK(cudaMemcpy(upsized_top_diff_ + m * bcm_N_, top_diff + m * N_,
                  K_ * sizeof(Dtype), cudaMemcpyDeviceToDevice));
      }
    }
    // Gradient with respect to weight
    if (N_ != bcm_N_)
      cufft::fft<Dtype>(&y_plan_, (Dtype *)upsized_top_diff_, fft_y_);
    else
      cufft::fft<Dtype>(&y_plan_, (Dtype *)top_diff, fft_y_);
    // The FFT of x can be saved
    CUDA_CHECK(cudaDeviceSynchronize());
    BCMBackwardWeight<Dtype>(fft_y_,
                      fft_x_,
                      sum_w_,
                      n_, p_, q_, fft_k_);
    CUDA_CHECK(cudaDeviceSynchronize());
    cufft::ifft<Dtype>(&ifft_backward_weight_plan_,
      sum_w_, this->blobs_[0]->mutable_gpu_diff());
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    BCMBackwardData<Dtype>(fft_y_,
                    fft_w_,
                    sum_x_,
                    n_, p_, q_, fft_k_);
    CUDA_CHECK(cudaDeviceSynchronize());
    cufft::ifft<Dtype>(&ifft_backward_data_plan_,
      sum_x_, bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductBCMLayer);

}  // namespace caffe
