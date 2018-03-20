#/usr/bin/env/ sh
/home/sensetime/caffe-test/build/tools/caffe train --solver=./mnist_solver.prototxt 2>&1 | tee lenetHOG.log
#/root/NewDisk/daxing2/sphereface/tools/caffe-sphereface/.build_release/tools/caffe train --solver=./mnist_solver.prototxt
