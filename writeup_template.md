# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.
---
**Given Advice & Goals**
* The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection. 
* Your goal is piece together a pipeline to ...
  * detect the line segments in the image
  * then average/extrapolate them and draw them onto the image for display
  * Once you have a working pipeline, try it out on the video stream below.
---
**Imagined Pipeline**

1) get image from video stream
2) find lanes
    * image preprocessing
      * color space
      * noise filtering
      * ROI masking
    * feature extraction
      * color selection
      * canny edge detection
      * hough transformation
    * lane finding algorithm
      * previous image
      * slope of detected edge
      

3) return image with lanes marked and stich back together to stream
---

**Finding Lane Lines on the Road**

The steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

* feature extraction
  * canny edge detection
  * hough transformation
* lane recognition
  * left & right

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

* hard-coded ROI