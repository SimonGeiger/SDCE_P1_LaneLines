# Project #01 - Detection of Lane Lines
This repository was forked as part of the [Udacity Self-Driving Car Engineer Nanodegree Program](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013). 

The goal of the first project is to design an algorithm to detect lane lines utilizing Python and OpenCV. Test images & videos are taken on the Californian Interstate 280. 

[//]: # (Image References)

[image1]: ./test_images_output/solidYellowCurve.jpg "Solid Yellow Curve"
[image2]: ./test_images_output/2021-06-06_test_img05.png "Overview of Processing Steps"

![alt text][image1]

---
## Project Description 

The lane detection is based mainly on color masking, canny edge detection and a hough transformation. This is an overview of the processing steps. Please find a brief description in the [writeup](./writeup.md)

![alt text][image2]

---
## Setup Information 

To simply see the results the [jupyter notebook](./P1.ipynb) is probably best. To run the lane detection locally please start by executing m<span>ain.p</span>y, an envrionment.yml is included to setup the utilized libraries. I followed the [original Udacity readme](./README_UdacityOriginal.md) to setup my envrionment.
