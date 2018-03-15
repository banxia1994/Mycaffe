#include <algorithm>
#include <vector>
#include <cfloat>

#include "caffe/layers/HoG_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void HoGLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  N_ = bottom[0]->channels();
  transpose_ = this->layer_param_.hog_param().transpose();
  block_size_ = this->layer_param_.hog_param().block_size();
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.hog_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis+1);
}

template <typename Dtype>
void HoGLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.hog_param().axis());
  const int new_K = bottom[0]->count(axis+1);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  switch (this->layer_param_.hog_param().hog()){
   case HoGParameter_HoGMethod_STEP:
     top_shape.resize(axis + 1);
     top_shape[axis] = N_*block_size_*block_size_;
     top[0]->Reshape(top_shape);
	 break;
   case HoGParameter_HoGMethod_POW:
     top_shape.resize(axis + 1);
     top_shape[axis] = N_*pow(2,block_size_-1)*pow(2,block_size_-1);
     top[0]->Reshape(top_shape);
	 break;   
   default:
     LOG(FATAL)<<"Unknoe HoG Method.";
  }
  vector<int> _shape(4,M_);
  _shape[1] = 1;
  _shape[2] = bottom[0]->height();
  _shape[3] = bottom[0]->width();
  maxMatrix_.Reshape(_shape);
  indexMatrix_.Reshape(_shape);
  top[1]->Reshape(_shape);
  top[2]->Reshape(_shape);
  //top[3]->Reshape(_shape);
}


template <typename Dtype>
void HoGLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* max_data = maxMatrix_.mutable_cpu_data();
  Dtype* index_data = indexMatrix_.mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* top_data1 = top[1]->mutable_cpu_data();
  Dtype* top_data2 = top[2]->mutable_cpu_data();
  const int top_count = top[1]->count();
  int H = bottom[0]->height();
  int W = bottom[0]->width();
  int tempBlock = 0;
  
  for(int i = 0; i < M_; i++){
     for (int k = 0; k < K_; k++){
	int tempIndex = 1;
	Dtype tempMax = bottom_data[i*N_*K_+k];
	for(int j = 1; j < N_; j++){
		if (tempMax < bottom_data[i*N_*K_+j*K_+k]){
			 tempMax = bottom_data[i*N_*K_+j*K_+k];
			 tempIndex = j+1;		
		}
         }
	max_data[i*K_+k] = tempMax;
	index_data[i*K_+k] = tempIndex;

    }
  }
  caffe_copy(top_count,max_data,top_data1);
  caffe_copy(top_count,index_data,top_data2);
switch (this->layer_param_.hog_param().hog()){
 case HoGParameter_HoGMethod_STEP:
    tempBlock = block_size_;
  for (int i = 0; i < M_;i++){
       //vector<int> cell1(48);
       //vector<int> cell2(48);
       //vector<int> cell3(48);
       //vector<int> cell4(48);
	   //vector<vector<int> > cell(ceil(float(H)/tempBlock)*ceil(float(W)*tempBlock),vector<int>(N_,0)); // when tempBlock represent blocksize
      vector<vector<int> > cell(tempBlock*tempBlock,vector<int>(N_,0));
	  for (int h = 0; h < H; h++){
           for(int w = 0; w < W; w++){
			 Dtype v_index = index_data[i*K_+h*W+w];
			 cell[(h/static_cast<int>(std::ceil((float)H/tempBlock)))*(tempBlock)+w/static_cast<int>(std::ceil((float)W/tempBlock))][v_index] += 1;
           }
         
       }
	   for (int s = 1; s < cell.size(); s++){
         cell[0].insert(cell[0].end(),cell[s].begin(),cell[s].end());
	   }
        // cell1.insert(cell1.end(),cell2.begin(),cell2.end());
         //cell1.insert(cell1.end(),cell3.begin(),cell3.end());
        // cell1.insert(cell1.end(),cell4.begin(),cell4.end());
    for (int j = 0; j < cell[0].size(); j++){
 	top_data[(i*cell[0].size())+j] = cell[0][j];
    }
  }
  break;
 case HoGParameter_HoGMethod_POW:
   tempBlock = pow(2,block_size_-1);
  for (int i = 0; i < M_;i++){
       //vector<int> cell1(48);
       //vector<int> cell2(48);
       //vector<int> cell3(48);
       //vector<int> cell4(48);
	   vector<vector<int> > cell(tempBlock*tempBlock,vector<int>(N_,0));
       for (int h = 0; h < H; h++){
           for(int w = 0; w < W; w++){
			 Dtype v_index = index_data[i*K_+h*W+w];
			 cell[(h/static_cast<int>(std::ceil((float)H/tempBlock)))*(tempBlock)+w/static_cast<int>(std::ceil((float)W/tempBlock))][v_index] += 1;
           }
         
       }
	   for (int s = 1; s < cell.size(); s++){
         cell[0].insert(cell[0].end(),cell[s].begin(),cell[s].end());
	   }
        // cell1.insert(cell1.end(),cell2.begin(),cell2.end());
         //cell1.insert(cell1.end(),cell3.begin(),cell3.end());
        // cell1.insert(cell1.end(),cell4.begin(),cell4.end());
    for (int j = 0; j < cell[0].size(); j++){
 	top_data[(i*cell[0].size())+j] = cell[0][j];
    }
  }
  break;
  default:
   LOG(FATAL) << "Unknown pooling method.";
 }
}

template <typename Dtype>
void HoGLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[1]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* index_data = top[2]->cpu_data();
    const int count = bottom[0]->count();
    caffe_set(count,Dtype(0),bottom_diff);
    for (int i = 0; i< M_; i++){
      for (int k = 0; k < K_; k++){
         int index = index_data[i*K_+k];
        bottom_diff[i*N_*K_+index*K_+k] += top_diff[i*K_+k];
      }
    }
  }
}



#ifdef CPU_ONLY
STUB_GPU(HoGLayer);
#endif

INSTANTIATE_CLASS(HoGLayer);
REGISTER_LAYER_CLASS(HoG);

} // namespace caffe
