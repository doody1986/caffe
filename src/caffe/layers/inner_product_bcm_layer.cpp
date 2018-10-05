#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_bcm_layer.hpp"
#include "caffe/util/bcm.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductBCMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_bcm_param().num_output();
  bias_term_ = this->layer_param_.inner_product_bcm_param().bias_term();
  transpose_ = this->layer_param_.inner_product_bcm_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_bcm_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  n_ = M_; // Batch size
  // obtain BCM specific parameters for weights
  k_ = this->layer_param_.inner_product_bcm_param().block_size();
  fft_k_ = k_ / 2 + 1;

  // Block size and batch size limitation
  if (!IsPowerOf2((Dtype)k_)) LOG(FATAL) << "Block size is not a power of 2";
  if (K_ < k_ || M_ < k_ || N_ < k_) LOG(FATAL) << "Block size is not appropriate";
  if (M_ % k_ != 0) LOG(FATAL) << "Batch size is not power of 2";
  // dimension adjustment
  bcm_N_ = N_;
  bcm_K_ = K_;
  if (N_ % k_ != 0)
    bcm_N_ = FindNextPowerOf2((Dtype)N_);
  if (K_ % k_ != 0)
    bcm_K_ = FindNextPowerOf2((Dtype)K_);

  if (transpose_) {
    p_ = bcm_K_ / k_;
    q_ = bcm_N_ / k_;
  } else {
    p_ = bcm_N_ / k_;
    q_ = bcm_K_ / k_;
  }

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(3);
    weight_shape[0] = p_;
    weight_shape[1] = q_;
    weight_shape[2] = k_;

    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_bcm_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_bcm_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization

  this->param_propagate_down_.resize(this->blobs_.size(), true);
  // Set up FFT plan
  cufft::createFFTPlan(&w_plan_);
  cufft::setFFTPlan(&w_plan_, cufft::FFT_1D, k_, cufft::R2C, p_ * q_);
  cufft::createFFTPlan(&x_plan_);
  cufft::setFFTPlan(&x_plan_, cufft::FFT_1D, k_, cufft::R2C, n_ * q_);
  cufft::createFFTPlan(&y_plan_);
  cufft::setFFTPlan(&y_plan_, cufft::FFT_1D, k_, cufft::R2C, n_ * p_);
  cufft::createFFTPlan(&ifft_forward_plan_);
  cufft::setFFTPlan(&ifft_forward_plan_, cufft::FFT_1D, k_, cufft::C2R, n_ * p_);
  cufft::createFFTPlan(&ifft_backward_weight_plan_);
  cufft::setFFTPlan(&ifft_backward_weight_plan_, cufft::FFT_1D, k_, cufft::C2R, p_ * q_);
  cufft::createFFTPlan(&ifft_backward_data_plan_);
  cufft::setFFTPlan(&ifft_backward_data_plan_, cufft::FFT_1D, k_, cufft::C2R, n_ * q_);
  // Set up intermediate data
  cudaError_t err;
  err = cudaMalloc(&(this->fft_w_), p_ * q_ * fft_k_ * 2 * sizeof(Dtype));
  if (err != cudaSuccess) {
    LOG(FATAL) << "Not enough memory";
  }
  err = cudaMalloc(&(this->fft_x_), n_ * q_ * fft_k_ * 2 * sizeof(Dtype));
  if (err != cudaSuccess) {
    LOG(FATAL) << "Not enough memory";
  }
  err = cudaMalloc(&(this->fft_y_), n_ * p_ * fft_k_ * 2 * sizeof(Dtype));
  if (err != cudaSuccess) {
    LOG(FATAL) << "Not enough memory";
  }
  err = cudaMalloc(&(this->sum_w_), p_ * q_ * fft_k_ * 2 * sizeof(Dtype));
  if (err != cudaSuccess) {
    LOG(FATAL) << "Not enough memory";
  }
  err = cudaMalloc(&(this->sum_x_), n_ * q_ * fft_k_ * 2 * sizeof(Dtype));
  if (err != cudaSuccess) {
    LOG(FATAL) << "Not enough memory";
  }
  err = cudaMalloc(&(this->sum_y_), n_ * p_ * fft_k_ * 2 * sizeof(Dtype));
  if (err != cudaSuccess) {
    LOG(FATAL) << "Not enough memory";
  }

  if (K_ != bcm_K_) {
    CUDA_CHECK(cudaMalloc(&upsized_bottom_, bcm_K_ * M_ * sizeof(Dtype)));
  }
  if (N_ != bcm_N_) {
    CUDA_CHECK(cudaMalloc(&upsized_top_diff_, bcm_N_ * M_ * sizeof(Dtype)));
  }
}

template <typename Dtype>
void InnerProductBCMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_bcm_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";

  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void InnerProductBCMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
// Do nothing
}

template <typename Dtype>
void InnerProductBCMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
// Do nothing
}

template <typename Dtype>
InnerProductBCMLayer<Dtype>::~InnerProductBCMLayer() {
  cufft::destroyFFTPlan(&w_plan_);
  cufft::destroyFFTPlan(&x_plan_);
  cufft::destroyFFTPlan(&y_plan_);
  cufft::destroyFFTPlan(&ifft_forward_plan_);
  cufft::destroyFFTPlan(&ifft_backward_weight_plan_);
  cufft::destroyFFTPlan(&ifft_backward_data_plan_);
  cudaFree(fft_w_);
  cudaFree(fft_x_);
  cudaFree(fft_y_);
  cudaFree(sum_w_);
  cudaFree(sum_x_);
  cudaFree(sum_y_);
  // Free the temp space
  if (!upsized_bottom_)
    CUDA_CHECK(cudaFree(upsized_bottom_));
  if (!upsized_top_diff_)
    CUDA_CHECK(cudaFree(upsized_top_diff_));
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductBCMLayer);
REGISTER_LAYER_CLASS(InnerProductBCM);

}  // namespace caffe
