 #ifndef CAFFE_MY_LOSS_LAYER_HPP_  
    #define CAFFE_MY_LOSS_LAYER_HPP_  
      
    #include <vector>  
      
    #include "caffe/blob.hpp"  
    #include "caffe/layer.hpp"  
    #include "caffe/proto/caffe.pb.h"  
      
    #include "caffe/layers/loss_layer.hpp"  
    #include "caffe/layers/softmax_layer.hpp"  
      
    namespace caffe {  
      
    template <typename Dtype>  
    class MyLossLayer : public LossLayer<Dtype> {  
     public:  
      explicit MyLossLayer(const LayerParameter& param)  
          : LossLayer<Dtype>(param) {}  
      virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,  
          const vector<Blob<Dtype>*>& top);  
      virtual void Reshape(const vector<Blob<Dtype>*>& bottom,  
          const vector<Blob<Dtype>*>& top);  

      
      virtual inline const char* type() const { return "MyLoss"; }  
      virtual inline int ExactNumTopBlobs() const { return 1; }  
      virtual inline int MinTopBlobs() const { return 1; }  
      virtual inline int MaxTopBlobs() const { return 2; }  
      
     protected:  
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
          const vector<Blob<Dtype>*>& top);  
      virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,  
          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);  
          
      virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

      vector<vector<Dtype> > prob_;   //保存置信度  
      int label_num;    //标签个数  
      int batch_size;   //批大小
      Dtype temp;
      int label_limit;
      
    };  
      
    }  // namespace caffe  
      
#endif // CAFFE_MY_LOSS_LAYER_HPP_  