#include <algorithm>
#include <vector>
#include <cfloat>

#include "caffe/layers/HoG_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

template <typename Dtype>
__global__ void CalMaxIndex(const int nthreads,const int K, const int N,const Dtype* bottom_data, Dtype* max_data, Dtype* index_data){
    CUDA_KERNEL_LOOP(index,nthreads){
      for (int k = 0; k < K; k++){
        int tempIndex = 1;
        Dtype tempMax = bottom_data[index*N*K+k];
        for (int j = 1; j < N; j++){
         if(tempMax < bottom_data[index*N*K+j*K+k]){
            tempMax = bottom_data[index*N*K+j*K+k];
            tempIndex = j+1;
         }
        }
        max_data[index*K+k] = tempMax;
        index_data[index*K+k] = tempIndex;
      }
      
    }
}
/*
template <typename Dtype>
__global__ void TopForward(const int nthreads,const int N,const int K,const int H,const int W,const Dtype* index_data, int tempBlock,Dtype* top_data){
    CUDA_KERNEL_LOOP(index,nthreads){
      vector<vector<int> > cell(tempBlock*tempBlock,vector<int>(N,0));
      for(int h = 0; h < H; h++){
        for(int w = 0; w < W; w++){
          Dtype v_index = index_data[index*K+h*W+w];
          cell[(h/static_cast<int>(std::ceil((float)H/tempBlock)))*(tempBlock)+w/static_cast<int>(std::ceil((float)W/tempBlock))][v_index] += 1;
        }
      }
      for(int s = 1; s < cell.size(); s++){
        cell[0].insert(cell[0].end(),cell[s].begin(),cell[s].end());
      }
      for (int j = 0; j < cell[0].size(); j++){
 	top_data[(index*cell[0].size())+j] = cell[0][j];
      }
    }
}
*/
template <typename Dtype>
void HoGLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* max_data = maxMatrix_.mutable_gpu_data();
    Dtype* index_data = indexMatrix_.mutable_gpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* top_data1 = top[1]->mutable_gpu_data();
    Dtype* top_data2 = top[2]->mutable_gpu_data();
    const int top_count = top[1]->count();
    int H = bottom[0]->height();
    int W = bottom[0]->width();
    int tempBlock = 0;
    int count = M_;
    CalMaxIndex<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(count,K_,N_,bottom_data,max_data,index_data);
    
    caffe_copy(top_count,max_data,top_data1);
    caffe_copy(top_count,index_data,top_data2);

    switch (this->layer_param_.hog_param().hog()){
        case HoGParameter_HoGMethod_STEP:
         tempBlock = block_size_;
          for (int i = 0; i < M_;i++){
	   vector<vector<float> > cell(tempBlock*tempBlock,vector<float>(N_,0));
           for (int h = 0; h < H; h++){
             for(int w = 0; w < W; w++){
			 Dtype v_index = indexMatrix_.cpu_data()[i*K_+h*W+w];
                         Dtype v_max = maxMatrix_.cpu_data()[i*K_+h*W+w];
			 cell[(h/static_cast<int>(std::ceil((float)H/tempBlock)))*(tempBlock)+w/static_cast<int>(std::ceil((float)W/tempBlock))][v_index] += 1;//v_max
             }
         
           }
          //　normalize hist
	  for (int c = 0; c < cell.size(); c++){
              float sum = 0.000001;
	      for (int ch = 0; ch < cell[c].size(); ch++){
		   sum += cell[c][ch]*cell[c][ch];
	      }
	      sum = sqrt(sum);
	      for (int ch = 0; ch < cell[c].size(); ch++)
	          cell[c][ch] /= sum;
	      }
	   for (int s = 1; s < cell.size(); s++){
             cell[0].insert(cell[0].end(),cell[s].begin(),cell[s].end());
	   }
           for (int j = 0; j < cell[0].size(); j++){
 	     top_data[(i*cell[0].size())+j] = cell[0][j];
          }
        }       
         //STEPTopForward<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(count,M_,K_,H,W,tempBlock,top_data);
         break;
      
          case HoGParameter_HoGMethod_POW:
           for (int i = 0; i < M_;i++){
             int record = 0;
             for(int b = 0; b <= block_size_; b++){  //add
               tempBlock = pow(2,b);
               vector<vector<int> > cell(tempBlock*tempBlock,vector<int>(N_,0));
               for (int h = 0; h < H; h++){
                 for(int w = 0; w < W; w++){
			 Dtype v_index = indexMatrix_.cpu_data()[i*K_+h*W+w];
                         Dtype v_max = maxMatrix_.cpu_data()[i*K_+h*W+w];
			 cell[(h/static_cast<int>(std::ceil((float)H/tempBlock)))*(tempBlock)+w/static_cast<int>(std::ceil((float)W/tempBlock))][v_index] += 1;//v_max
                 }
         
               }
          //　normalize hist
	   for (int c = 0; c < cell.size(); c++){
              float sum = 0.000001;
	      for (int ch = 0; ch < cell[c].size(); ch++){
		   sum += cell[c][ch]*cell[c][ch];
	      }
	      sum = sqrt(sum);
	      for (int ch = 0; ch < cell[c].size(); ch++)
	          cell[c][ch] /= sum;
	   }
	       for (int s = 1; s < cell.size(); s++){
                 cell[0].insert(cell[0].end(),cell[s].begin(),cell[s].end());
	       }
               for (int j = 0; j < cell[0].size(); j++){
 	         top_data[i*tempsize_*N_+record*N_+j] = cell[0][j];
               }
               record += tempBlock*tempBlock;  
           }
         }
           break;
          default:
            LOG(FATAL) << "Unknow HoG method.";
         }
         //TopForward<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(count,N_,K_,H,W,top[2]->gpu_data(),tempBlock,top_data);

}

template <typename Dtype>
__global__ void Backward(const int nthreads,const int N,const int K,const Dtype* top_diff,const Dtype* index_data,Dtype* bottom_diff){
    CUDA_KERNEL_LOOP(index,nthreads){
       for(int k = 0; k< K; k++){
         int in = index_data[index*K+k];
         bottom_diff[index*N*K+in*K+k] += top_diff[index*K+k];
       }
    }
}


template <typename Dtype>
void HoGLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[1]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* index_data = top[2]->gpu_data();
    const int count = bottom[0]->count();
    caffe_gpu_set(count,Dtype(0),bottom_diff);
    
    int nthreads = M_;
    
    Backward<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(nthreads,N_,K_,top_diff,index_data,bottom_diff);

  }
}   

INSTANTIATE_LAYER_GPU_FUNCS(HoGLayer);

}  // namespace caffe
