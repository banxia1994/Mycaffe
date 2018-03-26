#ifndef CAFFE_MULTI_CROP_LAYER_HPP_
#define CAFFE_MULTI_CROP_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Takes a Blob and crop it to get multiple sub blobs
 * this layer is designed for the following local convolution use
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */

template <typename Dtype>
class MultiCropLayer : public Layer<Dtype> {
 public:
  explicit MultiCropLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiCrop"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  vector< vector<int> > offsets_bottom_vector;//each entry store the offset of each cropped local patch
  vector< vector<int> > offsets_top_vector;//each entry store the offset of each concatenated local patch
  vector<int> each_crop_shape;//the shape of one local cropped patch 
  vector<int> top_shape;//all local cropped patches are concatenated along the channel axis
 private:
  // Recursive copy function.
  void crop_copy(const vector<Blob<Dtype>*>& bottom,
               const vector<Blob<Dtype>*>& top,
               const vector<int>& offsets_bottom,
			   const vector<int>& offsets_top,
               vector<int> indices,
               int cur_dim,
               const Dtype* src_data,
               Dtype* dest_data,
               bool is_forward);

  // Recursive copy function: this is similar to crop_copy() but loops over all
  // but the last two dimensions to allow for ND cropping while still relying on
  // a CUDA kernel for the innermost two dimensions for performance reasons.  An
  // alterantive implementation could rely on the kernel more by passing
  // offsets_bottom, but this is problematic because of its variable length.
  // Since in the standard (N,C,W,H) case N,C are usually not cropped a speedup
  // could be achieved by not looping the application of the copy_kernel around
  // these dimensions.
  void crop_copy_gpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top,
                const vector<int>& offsets_bottom,
				const vector<int>& offsets_top,
                vector<int> indices,
                int cur_dim,
                const Dtype* src_data,
                Dtype* dest_data,
                bool is_forward);
};
}  // namespace caffe

#endif  // CAFFE_MULTI_CROP_LAYER_HPP_
