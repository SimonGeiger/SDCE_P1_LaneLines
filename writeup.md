# **Project 1 - Finding Lane Lines on the Road** 



**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./test_images_output/2021-06-06_test_img03.png "Overview of Processing Steps"

---

## Reflections

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

steps in lane finding pipeline:
- reading image
- creating color mask 
- applying ROI
- applying canny edge detection
- applying hough transformation to find long edges
- extracting lanes from found hough lines
- blend found lanes into original image

![alt text][image2]

Extracting the lane lines:
- First of all it seemed to be important to find the right parameters for the color mask, canny and hough transformation, as this was the base for good results. 
- The hough transformation is outputting a list of Line objects each with slope, y-intercept
- the slope is used to categorize each line into either potential candidates for the left and right lane or discarded because the slope is too horizontal
- the average slope and y-intercept of each category results in the detected lanes
- lane is then drawn from the lower edge (y=540) up to y = 320 (which is cose enough to the vanishing point of the lanes)

### 2. Identify potential shortcomings with your current pipeline

- unstable behavior, some stabilization utilizing previous frames would be favorable
- as lanes are categorized by slope, only straight segemetents can be detected, curves need a higher dimension (see challenge) 
- worse light/weather condtions could influece performance
- detection of adjacent lanes
- detection of road boundaries, could be also a more felxible ROI

### 3. Suggest possible improvements to your pipeline

- unflexible code, since parameters/ROI are hard-coded
- split algorithm and visualization/drawing