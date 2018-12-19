## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.

Then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(8, 8)` can be found under output_images/hog_features.png


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and my final choice of paramenters were those who gave the best performace for SVM classifier. 
My parameter configuration is:
colorspace = 'YUV'
orient = 8
pix_per_cell = 16
cell_per_block = 2
hog_channel = "ALL"
Also changing block_norm to "L2" instead of "L2-Hys" improved the performance.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using only HOG features and achieved test accuracy of 98.51%. The code can be found under Train a classifier

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used find_car function to extract hog features for the entire image which can be subsampled for the given window size. Function returns  modified image with rectagles(if a car was found) and list of rectangles.

I experimeted with different size widows can came up with three groups of window: small for a car in distance, medium one for car in the midlle of the road an a large one for the car right in  front of us.


Example for one window size (400, 530) with scale 1.4 can be found under output_images/sliding_windows.png

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using only YUV 3-channel HOG features in the feature vector, which provided a nice result.  Some example images can be found under output_images/heat_map2.png


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
It's provided in a zip file  under project_video.mp4


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. Exept that in my case treshold is set to 0 because if it's >= 1 heatmap doesn't detect anything.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


### Here are six frames and their corresponding heatmaps:

See output_images.heat_map2.png

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from one of the frames:
See output_images/labels.png


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Problems that I have faced are mainly related to classifier and it was difficult to find the right combination of parameters that worked resonably well. I don't think that classifier is robust enogh. False positives cannot be filtered through heatmap using treshold and I think that's my main issue. For any value of treshold 1 or above heatmap doesn't detect very well, it doesn't have false positives bu also doesn't detect cars very well either.

