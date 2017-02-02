# Web-demo

Install Caffe
Uncommenting the line in the Makefile.config
WITH_PYTHON_LAYER := 1

make all
make pycaffe
make distribute

Add the LD_LIBRARY_PATH for the rc3 file.

Compile faster RCNN with option GPU

