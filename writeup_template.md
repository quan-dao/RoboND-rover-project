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

### 1. Notebook Analysis
#### 1.1 Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.
##### 1.1.1 Identifying the navigable terrain and obstacle
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
##### 1.1.2 Identifying rock samples
Rock sample is a bit more tricky to be found because they are not as bright as the navigable terrain and not as dark as the other obstacle (mountain or big rocks in the middle of the map).
<p align="center">
![alt text] [image6]
</p>
To handle this issue, I use the adaptive threshold. The main idea of this threshold is that the binary value of each pixel is defined by its RGB value compared to its neighbor.Converting the rock image to gray image and applying the adaptive threshold with the parameters below yield the picture on the left.
```
gray_rock_img = cv2.cvtColor(rock_img, cv2.COLOR_RGB2GRAY) # convert color image to gray image
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
#### 1.2 Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result.
##### 1.2.1 Define the destination for the perspective transform
The specification of the destination image is defined below
```
dst_size = 5 # half size of destination square
bottom_offset = 6
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]]) # coordinate of 4 corners of the source square
destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
              [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
              [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset],
              [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
              ]) # coordinate of 4 corners of the destination square
```
##### 1.2.2 Apply the perspective transform
With the perspective transform function being defined as
```
def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst) # get transformation matrix
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    return warped
```
The perspective transform is implemented in the `process_image()` by calling
```
warped = perspect_transform(img, source, destination)
```
##### 1.2.3 Apply color threshold to find navigable terrain, obstacle, and rock sample
The `process_image()` can call the `color_thresh()` defined in section 1.1.1 to get the navigable terrain out of the image resulted by warping the input image using perspective transform.
```
nav_terrain = color_thresh(warped)
```
As mentioned in section 1.1.1, the obstacle image is the complement of the navigable terrain image.  
```
obstacle = cv2.bitwise_not(nav_terrain)
```
The process of finding rock samples in a color image presented in section 1.1.2 is sum up in `rock_thresh()`
```
def rock_thresh(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # convert from color img to gray img
    # apply adaptive threhold to isolate navigale terrain from rocks & mountain
    binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 255, 5)
    binary_img = cv2.medianBlur(binary_img, 11) # smooth the binary image
    # apply adaptive threhold again to isolate mountain from rocks & navigable terrain
    mountain_img = cv2.adaptiveThreshold(gray_img, 255,
                         cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, 20)
    mountain_img = cv2.medianBlur(mountain_img, 11)
    # Find the rock img
    rock_img = cv2.bitwise_xor(binary_img, mountain_img)
    rock_img = cv2.medianBlur(rock_img, 3)
    return rock_img
```
Rock samples is found by calling this function inside `process_image()`.
```
rock_img_bin = rock_thresh(img)
```
##### 1.2.4 Create the world map
To create the world map, first convert the coordinates of navigable terrain, obstacle, and rock samples to the rover frame.
```
nav_xpix, nav_ypix = rover_coords(nav_terrain)
obs_xpix, obs_ypix = rover_coords(obstacle)
rock_xpix, rock_ypix = rover_coords(rock_img_bin)
```
Then, convert these rover-based coordinates to world coordinates.
 ```
nav_xworld, nav_yworld = pix_to_world(nav_xpix, nav_ypix, data.xpos[data.count], data.ypos[data.count],
                                           data.yaw[data.count], 200, 10)
obs_xworld, obs_yworld = pix_to_world(obs_xpix, obs_ypix, data.xpos[data.count], data.ypos[data.count],
                                           data.yaw[data.count], 200, 10)
rock_xworld, rock_yworld = pix_to_world(rock_xpix, rock_ypix, data.xpos[data.count], data.ypos[data.count],
                                           data.yaw[data.count], 200, 10)
 ```
 In the `pix_to_world()` last two parameters (200 and 10) are respectively the side of the world map and the number of pixels in the destination image needed to represent 1 meter in real world (i.e. the side of the destination square).  
### 2. Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.


#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
