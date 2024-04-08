# Copyright 2021 Allvision IO, Inc.
# author: ryan@allvision.io
# usage: docker build -t <tag> -f <path/to/this/file>
FROM ubuntu:18.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libpng-dev python3 python3-pip libopencv-dev libboost-dev cmake
RUN pip3 install --upgrade awscli
RUN pip3 install scikit-build==0.11.1
RUN pip3 install scipy numpy==1.19.2 matplotlib
RUN pip3 install opencv-python==4.4.0.44
WORKDIR /allvision/dev
COPY ./ /allvision/dev
RUN mkdir /allvision/build
RUN mkdir /allvision/samples
RUN make && cp ./build/linux/warping_error /allvision/build/ && cp -R IOU_precision_recall /allvision/build/
RUN rm -Rf /allvision/build/IOU_precision_recall/__pycache__ && rm -Rf /allvision/build/IOU_precision_recall/ipynb
RUN cp -R /allvision/dev/samples/* /allvision/samples
WORKDIR /allvision
RUN rm -Rf /allvision/dev
