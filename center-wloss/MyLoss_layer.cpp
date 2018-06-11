    #include <algorithm>  
    #include <cfloat>  
    #include <vector>  
      
    #include "caffe/layers/my_loss_layer.hpp"  
    #include "caffe/util/math_functions.hpp"  
    using namespace std;  
    namespace caffe {  
      
    template <typename Dtype>  
    void MyLossLayer<Dtype>::LayerSetUp(  
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  
      LossLayer<Dtype>::LayerSetUp(bottom, top);  
      temp = 0.01;
      label_limit = 10572;
    }  
      
    template <typename Dtype>  
    void MyLossLayer<Dtype>::Reshape(  
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  
      LossLayer<Dtype>::Reshape(bottom, top);  
      this->label_num=bottom[0]->channels();   //标签数 ，比如mnist为10  
      this->batch_size=bottom[0]->num();       //batch大小，比如mnist 一次输入64个  
      this->prob_=vector<vector<Dtype> >(batch_size,vector<Dtype>(label_num,Dtype(0)));  //置信度数组 64*10  
    }  
      
    template <typename Dtype>  
    void MyLossLayer<Dtype>::Forward_cpu(  
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  

		const Dtype* label = bottom[1]->cpu_data();   //标签数组  64  
    const Dtype* wgallery = bottom[2]->cpu_data();
        //为了避免数值问题，计算prob_时，先减最大值，再按照softmax公式计算各置信度  
        for(int i=0;i<batch_size;++i){  
            //求最大值，并减最大值  
            Dtype mmax=-10000000;  
            for(int j=0;j<label_num;++j)  
                mmax=max<Dtype>(mmax,bottom[0]->data_at(i,j,0,0));  
            Dtype sum=0.0;   //求出分母
		      	Dtype sumG = 0.0; 
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
        //根据计算好的置信度，计算loss  
        Dtype loss=0.0;  
        for(int i=0;i<batch_size;++i){  
            int realLabel=static_cast<int>(label[i]);  //图片i的真实标签
              Dtype tmpProb=prob_[i][realLabel];         //属于真实标签的置信度  
              loss -= log(max<Dtype>(tmpProb,Dtype(FLT_MIN)));   //防止数据溢出问题  
              
		}
        top[0]->mutable_cpu_data()[0] = loss / batch_size;  
    }  
      
    //反向传播，计算梯度  
    template <typename Dtype>  
    void MyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,  
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {  
      if (propagate_down[0]) {  
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();  
        const Dtype* label = bottom[1]->cpu_data();   //标签   
      
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
    }  
      
      
    INSTANTIATE_CLASS(MyLossLayer);  
    REGISTER_LAYER_CLASS(MyLoss);  
      
} // namespace caffe  