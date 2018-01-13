## Project: Search and Sample Return
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook).
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands.
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./write-up-img/original_img.png
[image2]: ./write-up-img/nav_terrain_obst.png
[image3]: ./write-up-img/first_threshold_img.png
[image4]: ./write-up-img/second_threshold_img.png
[image5]: ./write-up-img/rock_sample.png
[image6]: ./write-up-img/rock_img.png

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.
##### 1.1 Identifying the navigable terrain and obstacle
The test image taken by the rover camera is displayed below
<p align='center'>
![alt text][image1]
</p>
Apparently, the navigable terrain is the brightest part of the original image. Hence, the navigable terrain can be identified by setting a lower threshold for RGB value of each pixel as the following.
~~~
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select
~~~
On the other hand, obstacle is the left over part of the original image, so the obstacle image is the complement of the navigable terrain image.
~~~
obstacle = cv2.bitwise_not(nav_terrain)
~~~
The navigable terrain identified by `color_thresh()` and the obstacle image are
<p align='center'>
![alt text][image2]
</p>
##### 1.2 Identifying rock samples
Rock sample is a bit more tricky to be found because they are not as bright as the navigable terrain and not as dark as the other obstacle (mountain or big rocks in the middle of the map).
<p align="center">
![alt text] [image6]
</p>
To handle this issue, I use the adaptive threshold. The main idea of this threshold is that the binary value of each pixel is defined by its RGB value compared to its neighbor.Converting the rock image to gray image and applying the adaptive threshold with the parameters below yield the picture on the left.
```
gray_rock_img = cv2.cvtColor(rock_img, cv2.COLOR_RGB2GRAY) # convert color image
#to gray image
binary_img_1 = cv2.adaptiveThreshold(gray_rock_img, 255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 255, 5)
```
<p align="center">
![alt text] [image3]
</p>
As can be seen, the rock sample in the picture on the left is not in a good shape due to noise. Eliminate this noise using the `cv2.medianBlur()` to get the smoothed image on the right.
```
binary_img_smooth = cv2.medianBlur(binary_img, 11)
```
Change the threshold parameters and apply it to rock image (in the gray scale) again to get 2nd threshold image
```
mountain_img = cv2.adaptiveThreshold(gray_img, 255,
                         cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, 20)
mountain_img_smooth = cv2.medianBlur(mountain_img, 11)
```
<p align="center">
![alt text] [image4]
</p>
At this point, the only difference between the first and the second image is the rock sample, so the rock sample can be found by XOR these images.
 ```
 bin_rock_img = cv2.bitwise_xor(binary_img_smooth, mountain_img_smooth)
 bin_rock_img = cv2.medianBlur(rock_img, 3) # smooth the bin_rock_img
 ```
 <p align="center">
 ![alt text] [image5]
 </p>
#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result.
And another!

![alt text][image2]
### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.


#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  



![alt text][image3]
