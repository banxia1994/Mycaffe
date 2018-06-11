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
      this->label_num=bottom[0]->channels();   //��ǩ�� ������mnistΪ10  
      this->batch_size=bottom[0]->num();       //batch��С������mnist һ������64��  
      this->prob_=vector<vector<Dtype> >(batch_size,vector<Dtype>(label_num,Dtype(0)));  //���Ŷ����� 64*10  
    }  
      
    template <typename Dtype>  
    void MyLossLayer<Dtype>::Forward_cpu(  
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  

		const Dtype* label = bottom[1]->cpu_data();   //��ǩ����  64  
    const Dtype* wgallery = bottom[2]->cpu_data();
        //Ϊ�˱�����ֵ���⣬����prob_ʱ���ȼ����ֵ���ٰ���softmax��ʽ��������Ŷ�  
        for(int i=0;i<batch_size;++i){  
            //�����ֵ���������ֵ  
            Dtype mmax=-10000000;  
            for(int j=0;j<label_num;++j)  
                mmax=max<Dtype>(mmax,bottom[0]->data_at(i,j,0,0));  
            Dtype sum=0.0;   //�����ĸ
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
        //���ݼ���õ����Ŷȣ�����loss  
        Dtype loss=0.0;  
        for(int i=0;i<batch_size;++i){  
            int realLabel=static_cast<int>(label[i]);  //ͼƬi����ʵ��ǩ
              Dtype tmpProb=prob_[i][realLabel];         //������ʵ��ǩ�����Ŷ�  
              loss -= log(max<Dtype>(tmpProb,Dtype(FLT_MIN)));   //��ֹ�����������  
              
		}
        top[0]->mutable_cpu_data()[0] = loss / batch_size;  
    }  
      
    //���򴫲��������ݶ�  
    template <typename Dtype>  
    void MyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,  
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {  
      if (propagate_down[0]) {  
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();  
        const Dtype* label = bottom[1]->cpu_data();   //��ǩ   
      
        for(int i=0;i<batch_size;++i){  
            int realLabel=static_cast<int>(label[i]);  //ͼƬi����ʵ��ǩ  
            for(int j=0;j<label_num;++j){  
                int offset=bottom[0]->offset(i,j);  
                if(j==realLabel)                       //���չ�ʽ���������������ʵ��ǩ��ֱ�������Ŷ��ϼ�ȥ1���͵õ��÷������ݶ�  
                    bottom_diff[offset]=prob_[i][j]-1;  
                else                                  //�����ݶȵ������Ŷ�  
                    bottom_diff[offset]=prob_[i][j];   
			   

			}  
        }  
        for(int i=0;i<bottom[0]->count();++i)   //�ݶȹ�һ��������batch��С  
            bottom_diff[i]/=batch_size;  
      }  
    }  
      
      
    INSTANTIATE_CLASS(MyLossLayer);  
    REGISTER_LAYER_CLASS(MyLoss);  
      
} // namespace caffe  