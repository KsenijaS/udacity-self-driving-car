# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then in order to reduce image noise I applied gaussian blur with kernel size of 5. Next step was using Canny edge detection function for displaying edges with paramters low_threshold = 50 and high_threshold = 150. After that I defined four sided polygon to mask everything outside the road and then applied Hough transform function in order to find lane lines.   

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by extrapolating lines given by hough function. Coeffictiens of line, that should represent lane line, are given by polyfit function from numpy package. To find start and end of the line first we need to decide of the lenght of the line, more specifically the height of the line (y coordinate) compared to image size. So the start should have y coordinate that is equal to image height and for the end y coordinate I decided around the same height as polygon intended to mask image. So having coefficient of tangent line and values of start and end of y coordinate it was easy to find x coordinate.  
 

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when Hough transformation function doesn't dectect well broken line on the road and as a result detected line goes way outside the lane line bounds


Another shortcoming could be that in certain frames there is no detection of the broken line by Hough transormation.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to improve Hough transoframtion to better detectbroken line

Another potential improvement could be to stabilze detected line because sometimes line moves a lot between frames. Maybe by keeping the results of previous lines and averaging the result line.
