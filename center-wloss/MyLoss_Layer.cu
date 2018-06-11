#include <algorithm>  
#include <cfloat>  
#include <vector>  
  
#include "caffe/layers/my_loss_layer.hpp"  
#include "caffe/util/math_functions.hpp"  

namespace caffe{

//template <typename Dtype>
//__global__ voidCal

template <typename Dtype>
void MyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	 const Dtype* label = bottom[1]->cpu_data();
   const Dtype* wgallery = bottom[2]->cpu_data();
        for(int i=0;i<batch_size;++i){  
            //求最大值，并减最大值  
            Dtype mmax=-10000000;  
		for(int j=0;j<label_num;++j){
			 if(mmax<bottom[0]->data_at(i,j,0,0))
				mmax = bottom[0]->data_at(i,j,0,0);
		}  
            Dtype sum=0.0;   //求出分母
            for(int j=0;j<label_num;++j)  {
                prob_[i][j]=bottom[0]->data_at(i,j,0,0)-mmax;  
            

					  sum += exp(prob_[i][j]);

				}

            for(int j=0;j<label_num;++j){
            int curr_label = static_cast<int>(label[i]);
               if (j==curr_label)
					prob_[i][j]=(wgallery[curr_label]*exp(prob_[i][j]))/(sum+wgallery[curr_label]*exp(prob_[i][j]));
				else
				  prob_[i][j]=exp(prob_[i][j])/(sum);
			}
		} 
	 //CalProb<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(count,label_num,bottom_data,prob_);
	Dtype loss=0;  
	for(int i=0;i<batch_size;++i){  
		int realLabel=static_cast<int>(label[i]);  //图片i的真实标签
		  Dtype tmpProb=prob_[i][realLabel];         //属于真实标签的置信度  
		  loss -= log(tmpProb);   //防止数据溢出问题  
		  
	}
	top[0]->mutable_cpu_data()[0] = loss / batch_size;  
 }

//template <typename Dtype>
//__global__ void CalDiff(const int nthreads,const int label_num,)

template <typename Dtype>
void MyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if(propagate_down[0]){
	  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	  const Dtype* label = bottom[1]->cpu_data();
	  
			  for(int i=0;i<batch_size;++i){  
		int realLabel=static_cast<int>(label[i]);  //图片i的真实标签  
		for(int j=0;j<label_num;++j){  
			int offset=bottom[0]->offset(i,j);  
			if(j==realLabel)                       //按照公式，如果分量就是真实标签，直接在置信度上减去1，就得到该分量的梯度  
				bottom_diff[offset]=prob_[i][j]-1;  
			else                                  //否则，梯度等于置信度  
				bottom_diff[offset]=prob_[i][j];   
		   

		}  
	}  
	for(int i=0;i<bottom[0]->count();++i)   //梯度归一化，除以batch大小  
		bottom_diff[i]/=batch_size;  
  }  
	  //CaldDiff<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(count,label_num,prob,label,bottom_diff);
	  
	}

INSTANTIATE_LAYER_GPU_FUNCS(MyLossLayer);;
} //namaespace caffe