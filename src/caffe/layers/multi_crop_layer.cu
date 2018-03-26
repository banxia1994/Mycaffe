#include <vector>

#include "caffe/layers/multi_crop_layer.hpp"

namespace caffe {

// Copy (one line per thread) from one array to another, with arbitrary
// strides in the last two dimensions.
//i think when this is invoked n == height so index / height is 0
	//so index / height * src_outer_stride can be removed
template <typename Dtype>
__global__ void copy_kernel_forward(const int n, const int height, const int width,
    const int src_outer_stride, const int src_inner_stride,
    const int dest_outer_stride, const int dest_inner_stride,
    const Dtype* src, Dtype* dest) {
  CUDA_KERNEL_LOOP(index, n) {
    int src_start = index / height * src_outer_stride
                  + index % height * src_inner_stride;
    int dest_start = index / height * dest_outer_stride
                   + index % height * dest_inner_stride;
    for (int i = 0; i < width; ++i) {
      dest[dest_start + i] = src[src_start + i];
    }
  }
}

template <typename Dtype>
__global__ void copy_kernel_backward(const int n, const int height, const int width,
    const int src_outer_stride, const int src_inner_stride,
    const int dest_outer_stride, const int dest_inner_stride,
    const Dtype* src, Dtype* dest) {
  CUDA_KERNEL_LOOP(index, n) {
    int src_start = index / height * src_outer_stride
                  + index % height * src_inner_stride;
    int dest_start = index / height * dest_outer_stride
                   + index % height * dest_inner_stride;
    for (int i = 0; i < width; ++i) {
		//because patches may overlap so bottom_diff = bottom_diff + top_diff
		//bottom_diff has been initialized as zero
      dest[dest_start + i] += src[src_start + i];
    }
  }
}

template <typename Dtype>
void MultiCropLayer<Dtype>::crop_copy_gpu(const vector<Blob<Dtype>*>& bottom,
             const vector<Blob<Dtype>*>& top,
             const vector<int>& offsets_bottom,
			 const vector<int>& offsets_top,
             vector<int> indices,
             int cur_dim,
             const Dtype* src_data,
             Dtype* dest_data,
             bool is_forward) 
{
	if (cur_dim + 2 < each_crop_shape.size()) 
	{
		// We are not yet at the final dimension, call copy recursivley
		for (int i = 0; i < each_crop_shape[cur_dim]; ++i) 
		{
			indices[cur_dim] = i;
			crop_copy_gpu(bottom, top, offsets_bottom, offsets_top, indices, cur_dim+1,
				src_data, dest_data, is_forward);
		}
	} 
	else 
	{
		// We are at the last two dimensions, which are stored continously in memory
		// With (N,C,H,W)
		//      (0,1,2,3) cur_dim   -> H
		//                cur_dim+1 -> W
		const int lines = each_crop_shape[cur_dim];
		const int height = each_crop_shape[cur_dim];
		const int width = each_crop_shape[cur_dim+1];
		vector<int> ind_off(cur_dim+2, 0);//bottom
		vector<int> ind_red(cur_dim+2, 0);//top
		for (int j = 0; j < cur_dim; ++j) 
		{
			ind_off[j] = indices[j] + offsets_bottom[j];
			ind_red[j] = indices[j] + offsets_top[j];
		}
		ind_off[cur_dim] = offsets_bottom[cur_dim];
		ind_off[cur_dim+1] = offsets_bottom[cur_dim+1];
		ind_red[cur_dim] = offsets_top[cur_dim];
		ind_red[cur_dim+1] = offsets_top[cur_dim+1];
		// Compute copy strides
		const int src_outer_stride = bottom[0]->shape(cur_dim)*bottom[0]->shape(cur_dim+1);
		const int src_inner_stride = bottom[0]->shape(cur_dim+1);
		const int dest_outer_stride = top[0]->shape(cur_dim)*top[0]->shape(cur_dim+1);
		const int dest_inner_stride = top[0]->shape(cur_dim+1);

		if (is_forward) 
		{
			const Dtype* bottom_data = bottom[0]->gpu_data() +
				bottom[0]->offset(ind_off);
			Dtype* top_data = top[0]->mutable_gpu_data() +
				top[0]->offset(ind_red);
			// NOLINT_NEXT_LINE(whitespace/operators)
			copy_kernel_forward<<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(
				lines, height, width,
				src_outer_stride, src_inner_stride,
				dest_outer_stride, dest_inner_stride,
				bottom_data, top_data);

		} 
		else 
		{
			const Dtype* top_diff = top[0]->gpu_diff() +
				top[0]->offset(ind_red);
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff() +
				bottom[0]->offset(ind_off);
			// NOLINT_NEXT_LINE(whitespace/operators)
			copy_kernel_backward<<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(
				lines, height, width,
				dest_outer_stride, dest_inner_stride,
				src_outer_stride, src_inner_stride,
				top_diff, bottom_diff);
		}
	}
}

template <typename Dtype>
void MultiCropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();

	const int patch_num = offsets_bottom_vector.size();
	for (int k = 0; k < patch_num; ++k)
	{
		vector<int> indices(4, 0);
		crop_copy_gpu(bottom, top, offsets_bottom_vector[k], offsets_top_vector[k], indices, 0, bottom_data, top_data, true);
	}
}

template <typename Dtype>
void MultiCropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
	if (propagate_down[0])
	{
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

		//very important! because patches may overlap so bottom_diff = bottom_diff + top_diff
		caffe_gpu_set(bottom[0]->count(), static_cast<Dtype>(0.0), bottom_diff);

		const int patch_num = offsets_bottom_vector.size();
		for (int k = 0; k < patch_num; ++k)
		{
			vector<int> indices(4, 0);
			crop_copy_gpu(bottom, top, offsets_bottom_vector[k], offsets_top_vector[k], indices, 0, top_diff, bottom_diff, false);
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(MultiCropLayer);

}  // namespace caffe
