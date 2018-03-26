#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_cos_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Compute_center_norm_gpu(int nthreads, const int K, const Dtype* center_data,Dtype* center_norm ){
	CUDA_KERNEL_LOOP(index,nthreads){
	// index N_
    Dtype temp_norm = (Dtype)0.;
	for (int i = 0;i<K;i++){
		temp_norm += center_data[index*K+i]*center_data[index*K+i];
	}
	temp_norm = sqrt(temp_norm);
	for (int i = 0;i<K;i++){
		center_norm[index*K+i] = center_data[index*K+i]/temp_norm;
	}
	}               
}

template <typename Dtype>
__global__ void Compute_norm_x_gpu(int nthreads,const int K,const Dtype* bottom_data,Dtype* norm_x){
	CUDA_KERNEL_LOOP(index,nthreads){
		//index M_
		Dtype temp_norm_x = (Dtype)0.;
		for (int i=0;i<K;i++){
			temp_norm_x += bottom_data[index*K+i]*bottom_data[index*K+i];
		}
		temp_norm_x = sqrt(temp_norm_x);
		for(int i =0;i<K;i++){
			norm_x[index*K+i] = bottom_data[index*K+i]/temp_norm_x;
		}
	}
}
template <typename Dtype>
__global__ void Compute_loss_gpu(int nthreads,const int K,const Dtype* label,const Dtype* center_norm,const Dtype* norm_x,Dtype* temp_lossd){
	CUDA_KERNEL_LOOP(index,nthreads){
		//index M_
        Dtype temp_loss= (Dtype)0.;
		const int label_value = static_cast<int>(label[index]);
		for (int i=0;i<K;i++){
			temp_loss += center_norm[label_value*K+i]*norm_x[index*K+i];
		}
		temp_lossd[index] = temp_loss;
	}
}


template <typename Dtype>
__global__ void Compute_center_diff_gpu(int nthreads,const int M,const int K, const Dtype* label,
		 Dtype* variation_sum,const Dtype* center_data,const Dtype* bottom_data, Dtype* center_diff){
	CUDA_KERNEL_LOOP(index,nthreads){
		// index N_
		int count = 0;
		Dtype nc = (Dtype)0.; Dtype nx= (Dtype)0.; Dtype nxc= (Dtype)0.; Dtype dotc= (Dtype)0.; Dtype dotcx = (Dtype)0.;
		for (int m =0; m<M;m++){
			const int label_value = static_cast<int>(label[m]);
			if (label_value == index){
				count++;
				for (int k = 0; k< K; k++){
					nc += center_data[label_value*K+k]*center_data[label_value*K+k];
					nx += bottom_data[m*K+k]*bottom_data[m*K+k];
					nxc += center_data[label_value*K+k]*bottom_data[m*K+k];
				}
				nc = sqrt(nc);
				nx = sqrt(nx);
				dotc = nxc/nx/nc/nc/nc;
				dotcx = (Dtype)1./nx/nc;
				for (int k = 0; k<K;k++){
					variation_sum[index*K+k] += dotc*center_data[label_value*K+k]-dotcx*bottom_data[m*K+k];
				}
			}
		}
		for (int k =0;k<K;k++){
			center_diff[index*K+k] = variation_sum[index*K+k]/(count+(Dtype)1.);
		}
	}
}

template <typename Dtype>
__global__ void Compute_data_diff_gpu(int nthreads,const int M,const int K, const Dtype* label,
		const Dtype* center_data,const Dtype* bottom_data, const Dtype* topdiff,Dtype* data_diff){
	CUDA_KERNEL_LOOP(index,nthreads){
		// index M
		Dtype nc = (Dtype)0.; Dtype nx= (Dtype)0.; Dtype nxc= (Dtype)0.; Dtype dotc= (Dtype)0.; Dtype dotcx = (Dtype)0.;
		const int label_value = static_cast<int>(label[index]);

		for (int k = 0; k< K; k++){
			nc += center_data[label_value*K+k]*center_data[label_value*K+k];
			nx += bottom_data[index*K+k]*bottom_data[index*K+k];
			nxc += center_data[label_value*K+k]*bottom_data[index*K+k];
		}
		nc = sqrt(nc);
		nx = sqrt(nx);
		dotc = (Dtype)1./nx/nc;
		dotcx = nxc/nx/nx/nx/nc;
		for (int k = 0; k<K;k++){
			data_diff[index*K+k] = topdiff[0]*(dotcx*bottom_data[index*K+k]-dotc*center_data[label_value*K+k])/M;
		}

	}
}

template <typename Dtype>
void CenterCosLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  int nthreads = N_;
  Compute_center_norm_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),CAFFE_CUDA_NUM_THREADS>>>(nthreads,K_,this->blobs_[0]->gpu_data(),center_norm_.mutable_gpu_data());
  
  nthreads = M_;
  Compute_norm_x_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),CAFFE_CUDA_NUM_THREADS>>>(nthreads,K_,bottom[0]->gpu_data(),norm_x_.mutable_gpu_data());
  
  Dtype loss = (Dtype)0.;
  caffe_gpu_set(M_, (Dtype)0., temp_loss_.mutable_cpu_data());
  Compute_loss_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),CAFFE_CUDA_NUM_THREADS>>>(nthreads,K_,bottom[1]->gpu_data(),center_norm_.gpu_data(),norm_x_.gpu_data(),temp_loss_.mutable_cpu_data());
  //for(int i = 0; i<M_;i++){
	//loss -= temp_loss_.cpu_data()[i];
//} 
  caffe_gpu_asum(M_,temp_loss_.cpu_data(),&loss);
//loss = -loss/M_;
 top[0]->mutable_cpu_data()[0] = -loss;
//loss = loss/M_;
  //top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CenterCosLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int nthreads = N_;
  caffe_gpu_set(N_ * K_, (Dtype)0., variation_sum_.mutable_cpu_data());
  Compute_center_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, bottom[1]->gpu_data(),
                                variation_sum_.mutable_cpu_data(),this->blobs_[0]->gpu_data(),bottom[0]->gpu_data(),this->blobs_[0]->mutable_gpu_diff());

  if (propagate_down[0]) {
  nthreads = M_;
  Compute_data_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	  CAFFE_CUDA_NUM_THREADS>>>(nthreads,M_,K_,bottom[1]->gpu_data(),this->blobs_[0]->gpu_data(),bottom[0]->gpu_data(),top[0]->cpu_diff(),
			bottom[0]->mutable_gpu_diff());  //????????????
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CenterCosLossLayer);

}  // namespace caffe
