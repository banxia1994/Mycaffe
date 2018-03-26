# Mycaffe
add
HoG layer: combine deeplearning with HoG



MyLoss:  add gallery to training data


multicrop layer:  crop feature map for global and local learing
MultiCropParameter multi_crop_param
message MultiCropParameter {
  optional uint32 patch_size = 1; // The kernel size
  optional uint32 stride = 2; // The stride; defaults to 1
  optional uint32 num_crop = 3; // The number of outputs for the layer
}
layer {
  name: "multicrop"
  type: "MultiCrop"
  bottom: "pool1"
  top: "multicrop"
  multi_crop_param {
    patch_size: 20
    stride: 6
    num_crop: 16
  }
}



center_cos_loss: center-loss using cos method



cos_loss
  optional CenterCosLossParameter center_cos_loss_param = 150;
  optional CosLossParameter Cos_loss_param = 151;
  
  
  message CenterCosLossParameter {
  optional uint32 num_output = 1; // The number of outputs for the layer
  optional FillerParameter center_cos_filler = 2; // The filler for the centers
  // The first axis to be lumped into a single inner product computation;
  // all preceding axes are retained in the output.
  // May be negative to index from the end (e.g., -1 for the last axis).
  optional int32 axis = 3 [default = 1];
}
message CosLossParameter {
  optional uint32 num_output = 1; // The number of outputs for the layer
  optional FillerParameter Cos_filler = 2; // The filler for the centers
  // The first axis to be lumped into a single inner product computation;
  // all preceding axes are retained in the output.
  // May be negative to index from the end (e.g., -1 for the last axis).
  optional int32 axis = 3 [default = 1];
}


  layer {
  name: "center_cos_loss"
  type: "CenterCosLoss"
  bottom: "fc5"
  bottom: "label"
  top: "center_cos_loss"
  param {
    lr_mult: 1
    decay_mult: 0
  }
  center_cos_loss_param {
    num_output: 10572
    center_cos_filler {
      type: "xavier"
    }
  }
  loss_weight: 0.01
}


