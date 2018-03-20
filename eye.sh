#/usr/bin/env sh

/root/NewDisk/daxing2/caffe-face-caffe-face/.build_release/tools/caffe train --solver=./face_solvergpu.prototxt -gpu 3  2>&1 | tee center-cosMbn0.01.log

#/root/NewDisk/daxing2/sphereface/tools/caffe-sphereface/.build_release/tools/caffe train --solver=./face_solver.prototxt 


