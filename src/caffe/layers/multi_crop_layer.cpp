#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/layers/multi_crop_layer.hpp"
#include "caffe/net.hpp"


namespace caffe {

template <typename Dtype>
void MultiCropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
	// LayerSetup() handles the number of dimensions; Reshape() handles the sizes.
	// bottom[0] supplies the data
	const MultiCropParameter& param = this->layer_param_.multi_crop_param();
	
	const int num_crop = param.num_crop(); //16          （39x39）
	const int patch_size = param.patch_size(); //30 每个patch的大小
	const int stride_size = param.stride(); //3  步长

	const int num = bottom[0]->shape(0);
	const int channel = bottom[0]->shape(1);
	const int height = bottom[0]->shape(2);
	const int width = bottom[0]->shape(3);

	each_crop_shape.resize(4,0);
	each_crop_shape[0] = num;
	each_crop_shape[1] = channel;
	each_crop_shape[2] = patch_size;
	each_crop_shape[3] = patch_size;

	top_shape.resize(4,0);
	top_shape[0] = num;
	top_shape[1] = channel*num_crop;   // 把所有的local croped 合在一起
	top_shape[2] = patch_size;
	top_shape[3] = patch_size;

	const int num_w = (width-patch_size)/stride_size+1;
	const int num_h = (height-patch_size)/stride_size+1;
	CHECK_EQ(num_w*num_h, num_crop) << "patch_size, stride_size and num_crop don't match.";

	offsets_bottom_vector.clear();
	offsets_top_vector.clear();
	int k = 0;
	for (int hk = 0; hk < num_h; ++hk)
	{
		for (int wk = 0; wk < num_w; ++wk)
		{
			//the num and channel won't change
			vector<int>off_bottom(4, 0);
			//offset along height 
			off_bottom[2] = hk*stride_size;
			//offset along width
			off_bottom[3] = wk*stride_size;
			offsets_bottom_vector.push_back(off_bottom);

			vector<int>off_top(4, 0);
			//only the channel will change
			off_top[1] = k*channel;
			offsets_top_vector.push_back(off_top);
			
			++k;
		}
	}

	CHECK_EQ(k, num_crop) << "k must be equal to num_crop.";
}

template <typename Dtype>
void MultiCropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MultiCropLayer<Dtype>::crop_copy(const vector<Blob<Dtype>*>& bottom,
             const vector<Blob<Dtype>*>& top,
             const vector<int>& offsets_bottom,
			 const vector<int>& offsets_top,
             vector<int> indices,
             int cur_dim,
             const Dtype* src_data,
             Dtype* dest_data,
             bool is_forward) 
{
	if (cur_dim + 1 < each_crop_shape.size()) 
	{
		// We are not yet at the final dimension, call copy recursively 递归
		for (int i = 0; i < each_crop_shape[cur_dim]; ++i) 
		{
			indices[cur_dim] = i;
			crop_copy(bottom, top, offsets_bottom, offsets_top, indices, cur_dim+1,
				src_data, dest_data, is_forward);
		}
	} 
	else
	{
		// We are at the last dimensions, which is stored continously in memory
		// prepare index vector reduced(red) and with offsets_bottom(off)
		vector<int> ind_red(cur_dim+1, 0);//top
		vector<int> ind_off(cur_dim+1, 0);//bottom
		for (int j = 0; j < cur_dim; ++j) 
		{
			ind_red[j] = indices[j] + offsets_top[j];
			ind_off[j] = indices[j] + offsets_bottom[j];
		}
		ind_red[cur_dim] = offsets_top[cur_dim];
		ind_off[cur_dim] = offsets_bottom[cur_dim];
		// do the copy
		if (is_forward) 
		{
			caffe_copy(each_crop_shape[cur_dim], src_data + bottom[0]->offset(ind_off),
				dest_data + top[0]->offset(ind_red));
		} 
		else 
		{
			// in the backwards pass the src_data is top_diff
			// and the dest_data is bottom_diff
			//caffe_copy(top[0]->shape(cur_dim), src_data + top[0]->offset(ind_red),
			//	dest_data + bottom[0]->offset(ind_off));
			//because of the underlying overlap use caffe_axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);
			// Y=alpha*X+Y 
			caffe_axpy(each_crop_shape[cur_dim], static_cast<Dtype>(1.0), src_data + top[0]->offset(ind_red),
				dest_data + bottom[0]->offset(ind_off));
		}

	}
}

template <typename Dtype>
void MultiCropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	const int patch_num = offsets_bottom_vector.size();
	for (int k = 0; k < patch_num; ++k)
	{
		vector<int> indices(4, 0);
		crop_copy(bottom, top, offsets_bottom_vector[k], offsets_top_vector[k], indices, 0, bottom_data, top_data, true);
	}
}

template <typename Dtype>
void MultiCropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
	if (propagate_down[0]) 
	{
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		//very important! because patches may overlap so bottom_diff = bottom_diff + top_diff
		caffe_set(bottom[0]->count(), static_cast<Dtype>(0.0), bottom_diff);
		
		const int patch_num = offsets_bottom_vector.size();
		for (int k = 0; k < patch_num; ++k)
		{
			vector<int> indices(4, 0);
			crop_copy(bottom, top, offsets_bottom_vector[k], offsets_top_vector[k], indices, 0, top_diff, bottom_diff, false);
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(MultiCropLayer);
#endif

INSTANTIATE_CLASS(MultiCropLayer);
REGISTER_LAYER_CLASS(MultiCrop);

}  // namespace caffe
