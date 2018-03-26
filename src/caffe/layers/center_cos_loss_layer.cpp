#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_cos_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CenterCosLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.center_cos_loss_param().num_output();  
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.center_cos_loss_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    vector<int> center_shape(2);
    center_shape[0] = N_;
    center_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(center_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > center_cos_filler(GetFiller<Dtype>(
        this->layer_param_.center_cos_loss_param().center_cos_filler()));
    center_cos_filler->Fill(this->blobs_[0].get());

  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void CenterCosLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  M_ = bottom[0]->num();
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  LossLayer<Dtype>::Reshape(bottom, top);
  //distance_cos_.ReshapeLike(*bottom[0]);
  //center_.ReshapeLike(*bottom[0]);
  center_norm_.ReshapeLike(*this->blobs_[0]); // dimension error?
  norm_x_.ReshapeLike(*bottom[0]);
  variation_sum_.ReshapeLike(*this->blobs_[0]);

  vector<int> loss_shape(2,M_);
  loss_shape[0] = 1;
  temp_loss_.Reshape(loss_shape);
}

template <typename Dtype>
void CenterCosLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  //const Dtype* center = this->blobs_[0]->cpu_data();
  const Dtype* center_data = this->blobs_[0]->cpu_data();
  Dtype* norm_x = norm_x_.mutable_cpu_data();
  //Dtype* distance_data = distance_cos_.mutable_cpu_data();
  Dtype* center_norm = center_norm_.mutable_cpu_data(); 

/********************************** compute center*************/
  for (int n = 0; n < N_; n++) {
    Dtype temp_norm = (Dtype)0.;
        // normalize ci
      caffe_copy(K_, center_data+n*K_, center_norm+n*K_); 
      temp_norm = caffe_cpu_dot(K_,center_norm + n*K_,center_norm+ n*K_);
      temp_norm = (Dtype)1./sqrt(temp_norm); 
      caffe_scal(K_,temp_norm,center_norm+n*K_);
      
  }
/**********************normalize  xi******************************/
  caffe_copy(M_ * K_, bottom_data, norm_x); 
  Dtype temp_norm_x ;
  for (int i =0; i<M_;i++){
    temp_norm_x = sqrt(caffe_cpu_dot(K_,norm_x + i*K_,norm_x+i*K_));
    temp_norm_x = (Dtype)1./temp_norm_x;
    caffe_scal(K_,temp_norm_x,norm_x+i*K_);
  }
  
  /*****************loss*****************/
  Dtype loss =0.;
  for (int i =0;i < M_;i++){
    const int label_value = static_cast<int>(label[i]);
    Dtype dot = caffe_cpu_dot(K_,center_norm+label_value*K_,norm_x+i*K_);///是否改变了change norm？ 
    loss-=dot;
  }
  //loss = loss/M_;
  top[0]->mutable_cpu_data()[0] = loss;

}

template <typename Dtype>
void CenterCosLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Blob<Dtype> temp_ ;
  vector<int> shape_x(1,K_);
  temp_.Reshape(shape_x);
  // Gradient with respect to centers
  if (this->param_propagate_down_[0]) {
    Dtype* temp_x = temp_.mutable_cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* variation_sum_data = variation_sum_.mutable_cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* center_data = this->blobs_[0]->cpu_data();

    // \sum_{y_i==j}
    caffe_set(N_ * K_, (Dtype)0., variation_sum_.mutable_cpu_data());

    for (int n = 0; n < N_; n++) {
      int count = 0;
      for (int m = 0; m < M_; m++) {
        const int label_value = static_cast<int>(label[m]);
        if (label_value == n) {
          count++;
		  caffe_set(K_,(Dtype)0.,temp_x);
		  caffe_copy(K_,bottom_data+m*K_,temp_x);
		  Dtype nc = sqrt(caffe_cpu_dot(K_,center_data + label_value*K_,center_data +label_value*K_));
		  Dtype nx = sqrt(caffe_cpu_dot(K_,bottom_data + m*K_,bottom_data +m*K_));
		  Dtype nxc =caffe_cpu_dot(K_,center_data + label_value*K_,bottom_data +m*K_);
		  Dtype dotc = nxc/nx/nc/nc/nc;
		  Dtype dotcx = (Dtype)1./nx/nc;
      caffe_cpu_axpby(K_, dotc, center_data+label_value*K_,-dotcx,temp_x);
		  caffe_sub(K_,variation_sum_data+n*K_,temp_x,variation_sum_data+n*K_);
        }
      }
      caffe_axpy(K_, (Dtype)1./(count + (Dtype)1.), variation_sum_data + n * K_, center_diff + n * K_);
    }
  }
  // Gradient with respect to bottom data 
  if (propagate_down[0]) {
    const Dtype* label = bottom[1]->cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* center_data = this->blobs_[0]->cpu_data();
    Dtype* diff = bottom[0]->mutable_cpu_diff();
    
    for (int i=0;i<M_;i++){
      const int label_value = static_cast<int>(label[i]);
      caffe_copy(K_,center_data+label_value*K_,diff+i*K_);
	  
	  Dtype nc = sqrt(caffe_cpu_dot(K_,center_data + label_value*K_,center_data +label_value*K_));
	  Dtype nx = sqrt(caffe_cpu_dot(K_,bottom_data + i*K_,bottom_data +i*K_));
	  Dtype nxc =caffe_cpu_dot(K_,center_data + label_value*K_,bottom_data +i*K_);
	  Dtype dotc = (Dtype)1./nx/nc;
	  Dtype dotcx = nxc/nx/nx/nx/nc;
	  
      
      caffe_cpu_axpby(K_,dotcx,bottom_data+i*K_,-dotc,diff+i*K_);
        
      }
    caffe_scal(M_ * K_,top[0]->cpu_diff()[0]/M_,diff);
    }

  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(CenterCosLossLayer);
#endif

INSTANTIATE_CLASS(CenterCosLossLayer);
REGISTER_LAYER_CLASS(CenterCosLoss);

}  // namespace caffe

