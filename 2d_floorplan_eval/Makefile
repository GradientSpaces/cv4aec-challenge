# Copyright 2021 Allvision IO, Inc.
# author: ryan@allvision.io
all: build/linux/warping_error

build/linux/warping_error: warping_error/warping_error.cpp warping_error/main.cpp warping_error/warping_error.h
	mkdir -p build/linux
	g++ -o build/linux/warping_error -I 3P/jsonpp warping_error/warping_error.cpp warping_error/main.cpp warping_error/utility.cpp -l opencv_core -l opencv_imgproc -l opencv_imgcodecs -l opencv_highgui
