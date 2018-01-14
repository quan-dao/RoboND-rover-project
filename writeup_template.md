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
[image3]: ./write-up-img/rock_img.png
[image4]: ./write-up-img/first_threshold_img.png
[image5]: ./write-up-img/second_threshold_img.png
[image6]: ./write-up-img/rock_sample.png
[image7]: ./write-up-img/fov_normal_2.png
[image8]: ./write-up-img/fov_obs_ahead.png
[image9]: ./write-up-img/angles_vs_dist.png
[image10]: ./write-up-img/extremum_ang_dist.png
[image11]: ./write-up-img/old_new_steer_2.png
[image12]: ./write-up-img/result_3.png
[image13]: ./write-up-img/result_2.png
[image14]: ./write-up-img/pipe_line_leak_1.png

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
# 1. Notebook Analysis
## 1.1 Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.
### 1.1.1 Identifying the navigable terrain and obstacle
The test image taken by the rover camera is displayed in Fig.1

![alt text][image1]

*Fig.1 Test image*

Apparently, the navigable terrain is the brightest part of the original image. Hence, the navigable terrain can be identified by setting a lower threshold for RGB value of each pixel as the following (this function is provided by the RoboND lecture).
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
The navigable terrain identified by `color_thresh()` and the obstacle image are shown in Fig.2

![alt text][image2]

*Fig.2 The navigable terrain image & obstacle image*

### 1.1.2 Identifying rock samples
Rock sample is a bit more tricky to be found because they are not as bright as the navigable terrain and not as dark as the other obstacle (mountain or big rocks in the middle of the map) (shown in Fig.3).

![alt text][image3]

*Fig.3 Color and gray scale image of rock sample*

To handle this issue, I use the adaptive threshold. The main idea of this threshold is that the binary value of each pixel is defined by its RGB value compared to its neighbor.Converting the rock image to gray image and applying the adaptive threshold with the parameters below yield the sub figure on the left of Fig.4.
```
gray_rock_img = cv2.cvtColor(rock_img, cv2.COLOR_RGB2GRAY) # convert color image to gray image
binary_img_1 = cv2.adaptiveThreshold(gray_rock_img, 255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 255, 5)
```

![alt text][image4]

*Fig.4 1st adaptive threshold result*

As can be seen, the rock sample in the picture on the left is not in a good shape due to noise. Eliminate this noise using the `cv2.medianBlur()` to get the smoothed image on the right.
```
binary_img_smooth = cv2.medianBlur(binary_img, 11)
```
Change the threshold parameters and apply it to rock image (in the gray scale) again to get 2nd threshold image (Fig.5)
```
mountain_img = cv2.adaptiveThreshold(gray_img, 255,
                         cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, 20)
mountain_img_smooth = cv2.medianBlur(mountain_img, 11)
```

![alt text][image5]

*Fig.5 2nd adaptive threshold result*

At this point, the only difference between Fig.4 and Fig.5 is the rock sample, so the rock sample (shown in Fig.6) can be found by XOR these two images.
 ```
 bin_rock_img = cv2.bitwise_xor(binary_img_smooth, mountain_img_smooth)
 bin_rock_img = cv2.medianBlur(rock_img, 3) # smooth the bin_rock_img
 ```

![alt text][image6]

*Fig.6 Binary image of rock sample*

## 1.2 Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result.
#### 1.2.1 Define the destination for the perspective transform
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
### 1.2.2 Apply the perspective transform
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
### 1.2.3 Apply color threshold to find navigable terrain, obstacle, and rock sample
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
### 1.2.4 Create the world map
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
# 2. Autonomous Navigation and Mapping

## 2.1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.
### 2.1.1 The `perception_step()`
The role of the `perception_step()` is to
* Create the world map
* Provide the coordinate of navigable terrain with respect to the rover in the polar form for the `decision_step()`

#### 2.1.1.1 Create the world map
The process of creating the wolrd map in `perception_step()` is the same with the one used by the notebook (presented in section 1.2.4). To  find the navigable terrain, I first perform perspective transform (using `perspect_transform()`) on the camera image (coming under the name of `Rover.img`). After that, the warped image is fed into `color_thresh()` to yield the desired result. The obstacle is easily identified by complementing the navigable terrain image, while the rock samples is found by `rock_thresh(Rover.img)`. Finally, the pixel coordinates of navigable terrain, obstacle, and rock sample are transform to the coordinates in rover centric frame with `rover_coords()`, and then to the world frame with `pix_to_world()`.

The perspective transform is accurate if the rover's roll and pitch angles are relatively near 0. Therefore, I check these two angles prior to updating the world map.
```
p_r_angles_threhold = 1 # deg, threshold of pitch & roll for perspect_transform to be considered accurate
if (np.abs(Rover.pitch) < p_r_angles_threhold) or (np.abs(Rover.pitch - 360) < p_r_angles_threhold):
    flag_pitch_ok = True
else:
    flag_pitch_ok = False
if (np.abs(Rover.roll) < p_r_angles_threhold) or (np.abs(Rover.roll - 360) < p_r_angles_threhold):
    flag_roll_ok = True
else:
    flag_roll_ok = False
if flag_pitch_ok & flag_roll_ok:
    # Update the world map
```
#### 2.1.1.2 Find the open part of rover camera field of view
The distance and angle coordinate relative to the rover frame of navigable terrain pixels is found by
```
dist, angles = rover_coord(nav_terrain)
```
When the rover is at the open field (like in the figure below), it is alright to determine it steering angle, which is denoted by the red arrow in Fig.7 by calculating the mean of angle coordinate of every pixel.
```
steering_angle = np.mean(angles)
```

![alt text][image7]

*Fig.7 Field of view with no obstacles*

However, if the rover is near an obstacle this strategy does not perform well.

![alt text][image8]

*Fig.8 Field of view with an obstacle in the middle*

Fig.8 shows that when there is an obstacle near the middle of the rover camera's field of the view, the steering direction calculated by the mean angles tend to point toward the obstacle.

My idea for handling this is that an obstacle(s) divides the rover 's camera field of into two (or more) parts, so I will choose the bigger (or the biggest) part and calculate mean angles of this part to get the good rover steering angle.

Plotting the pixels' angle versus pixels' distance coordinate results in Fig.9.

![alt text][image9]

*Fig.9 Graph of angles vs dist*

The sub figure on the left shows no clear pattern. But, when I sort the pixels angle coordinate (the `angles` array) and arrange the pixels distance coordinate  (the `dist` array) with the order of the `angle` array as following,
```
sort_index = np.argsort(angles) # extract the sort result in the form of the index of angles array
sorted_angles = angles[sort_index] # arrange the angles array with the index above
sorted_dist = dist[sort_index]
```
the pattern on the sub figure to the right emerges. The patter shows that the obstacle causes a suddenly fall in the distance coordinate of pixels in its neighbor. Hence, to help the rover sense the presence of the obstacle, I need to detect this sudden fall.

Make the rover be aware of this fall is hard because the field of view (formed by `dist, angles`) has too many elements, most of which contain no information of the fall. As long as the fall is concerned, this field is best described by the maximum value of distance coordinate of pixels having relatively the same angle coordinate. Therefore, I make a list of pixels angle coordinate where each value of angle appears only once (the list named `base_ang_array`). With each angle value of in this list, I find the maximum distance coordinate of pixels having relatively the same angle coordinate and store it in `max_dist_resp_ang` array. The code for implementing this process is displayed below
```
threshold = 3 * np.pi/180 # rad, threshold for considering two angles are equal
max_dist_resp_ang = []
base_ang_array = []
i_start = 0 # starting index of the relatively equal angles
i_end = 0 # ending index of the relatively equal angles
base_ang = sorted_angles[i_start] # angle for comparison
while i_end < len(sorted_angles):
  if np.abs(sorted_angles[i_end] - base_ang) < threshold: # two angles are relatively equal
      i_end += 1 # move the end index to next element
  else:
      # two angles are not equal, so the current angle is bigger than the base angle
      # this mean the angles having index in the range from i_start to (i_end - 1) are equal
      max_dist_resp_ang.append(max(sorted_dist[i_start:i_end])) # find the max of associated distance coordinate
      base_ang_array.append(base_ang) # store the base angle
      # update looping variable
      i_start = i_end
      i_end = i_start + 1
      base_ang = sorted_angles[i_start]

if i_start == len(sorted_angles) - 1:
    max_dist_resp_ang.append(sorted_dist[i_start])
    base_ang_array.append(sorted_angles[i_start])
```
Note that this process is conducted on the sorted version of pixels angle and distance coordinate array (i.e. `sorted_angles` & `sorted_dist`) to exploit the ordered structure of these two arrays. Plotting the `base_ang_array` versus `max_dist_resp_ang`, I get the profile of the rover camera's field of view (the red line on Fig.9).

Having the field profile, I characterize the fall by a large drop (beyond a threshold) in the distance coordinate of extremum of the profile, compared to other neighbor extremums. As a result, these extremums are needed to be found. I find them by keeping in mind that an extremum is a point that higher (or lower) than its adjacent point to the left and to the right. This idea is implemented below
```
n = len(max_dist_resp_ang)
extre_dist = [] # store the distance coordinate of extremums
extre_ang = [] # store the angle coordinate of extremums
for i in range(1, n-1):
    flag_max = (max_dist_resp_ang[i] > max_dist_resp_ang[i - 1]) & (max_dist_resp_ang[i] > max_dist_resp_ang[i + 1])
    flag_min = (max_dist_resp_ang[i] < max_dist_resp_ang[i - 1]) & (max_dist_resp_ang[i] < max_dist_resp_ang[i + 1])
    if flag_max or flag_min:
        extre_dist.append(max_dist_resp_ang[i])
        extre_ang.append(base_ang_array[i])
```
The field of view profile and its extremums is shown in Fig.10.

![alt text][image10]

*Fig.10 Field of view 's profile and its extremums*

At this point, the hard work is done. The left over is to identify there is a fall or not, then choose the appropriate portion of the field of view. As mentioned in the previous paragraph, the fall is a large drop in extremums' distance coordinate. I am going to use this definition in the following code to detect the fall.
```
extre_threhold = 30 # threshold for drop in extremums distance coordinate
cut_off_ang = [] # store the angle coordiante of the extremum where the drop happens
i = 1
while i < len(extre_dist):
    if np.abs(extre_dist[i] - extre_dist[i - 1]) > extre_threhold:
        cut_off_ang.append(extre_ang[i])
        i += 2 # skip one extremum to prevent a fall get counted twice because a fall is followed be a rise
    else:
        i += 1
```
Finally I choose the steering angle to be the mean of the bigger (or the biggest) part of the filed of view based on the places of the angle coordinate of the drop extremums in the sorted angles coordinate.  
```
n_cut_off = len(cut_off_ang) # number of drop extremums       
if n_cut_off == 1:
    ind_sorted_angles = 0
    while ind_sorted_angles < len(sorted_angles):
        if sorted_angles[ind_sorted_angles] < cut_off_ang[0]:
            ind_sorted_angles += 1
        else:
            break;
    if ind_sorted_angles > (len(sorted_angles) / 2):
        steering_angle = np.mean(sorted_angles[:ind_sorted_angles])
    else:
        steering_angle = np.mean(sorted_angles[ind_sorted_angles:])
elif n_cut_off > 1:
    left_part = sorted_angles[sorted_angles <= min(cut_off_ang)]
    right_part = sorted_angles[sorted_angles >= max(cut_off_ang)]
    mid_part = sorted_angles[(sorted_angles > min(cut_off_ang)) & (sorted_angles < max(cut_off_ang))]
    n_left = len(left_part)
    n_right = len(right_part)
    n_mid = len(mid_part)
    if n_left == max(n_left, n_right, n_mid):
        steering_angle = np.mean(left_part)
    elif n_right == max(n_left, n_right, n_mid):
        steering_angle = np.mean(right_part)
    else:
        steering_angle = np.mean(mid_part)
else:
    steering_angle = np.mean(sorted_angles)
```
The comparison between steering angle calculated based on the mean of whole and the appropriate part of the field of view is displayed in Fig.11

![alt text][image11]

*Fig.11 Comparison of two methods of calculating steering angle*

The whole process in this section in synthesized in the function `find_open_part()` (in the file `decision_step.py`) which takes the distance and angle coordinates of navigable terrain and return the those coordinate of the bigger (or open) part of the field of view, and an logic variable being true when there is at least an obstacle.
```
Rover.nav_dist, Rover.nav_angles, Rover.obst_in_view = find_open_part(dist, angles)
```   

### 2.1.2 The `decision_step()`
The open part of the navigable terrain found by the `find_open_part()` function established in section 2.1.1.2 pays the way for the decision step. The underlying policy of `decision_step()` is the decision tree provided in the lecture. However, I add a modification to how the rover steering angle (`Rover.steer`) is calculated to make the rover a wall crawler. This means the rover moves in the way that one of its side, which is the left in my case is kept close to the wall (made up by the mountain in the simulation environment) as soon as it is possible (no obstacle in the field of view). This policy is implemented by altering the `Rover.steer` from the mean of angle coordinate (stored in `Rover.nav_angles`) of the whole navigable terrain's open part to the mean of angles that exceed the mean.
```
if not Rover.obst_in_view: # there is no obstacle
    left_mean_ang = Rover.nav_angles[Rover.nav_angles > nav_angles_mean] # mean of all angle to the left of the navigable terrain
    Rover.steer = np.clip(np.mean(left_mean_ang * 180/np.pi), -15, 15)
    # Check if there are any obstacle ahead if move with Rover.steer angle
    dists_around_steer = np.abs(Rover.nav_angles - Rover.steer) < (2 * np.pi/180)
    mean_dist_ahead = np.mean(Rover.nav_dists[dists_around_steer])
    if mean_dist_ahead < Rover.min_dist_ahead_thres:
        Rover.mode = 'stop'
else: # there is obstacle
    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
```
If there is obstacle in the field of view, the rover can not strictly follow the wall crawling policy because the policy can force the rover to hit the obstacle. In this case, the `Rover.steer` is kept equal to the mean of whole navigable terrain, or, more precisely, the open part of the navigable terrain produced by the `find_open_part()` function.

When the rover reach the end of a small branch of the world map, the stop condition which is threshold of the numbered of pixels of the navigable terrain is triggered. Once completely stop, the rover needs do either a left or right four-wheel turn to spin around. Because of the wall crawler policy, the rover had turned left to reach the end, so it need to do a right four-wheel turn to get out without moving back to the same path it used to go in. This is the reason why I set
```
Rover.steer = -15
```   
to induce a four-wheel turn when the rover in the `stop` mode and `len(Rover.nav_angles) < Rover.go_forward`.

In addition, when `stop` mode is triggered, I also make the brake process less aggressive by increasing the `Rover.brake` incrementally to `Rover.brake_set`.
```
if Rover.brake < Rover.brake_set:
    Rover.brake += 0.3
```    
## 2.2 Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

### Note: The simulator setting is listed below
* The screen resolution is 1024 x 768
* Graphics quality is Good
* Windows is checked
* FPS output to terminal is in the range of 13 to 23

As shown in Fig.12, using the method of processing the image taken by the rover's camera and the wall crawler policy respectively presented in section 2.1.1 and section 2.1.2, the rover has successfully met the minimum requirement of the Search & Sample Return project after 60 seconds of operating.

![alt text][image12]

*Fig.12 The rover has mapped 40% of the environment with more than 70% fidelity*

Let the rover operate for more than 600 seconds, it has mapped 98% of the environment (Fig.13).

![alt text][image13]

*Fig.13 The rover has mapped nearly the entire environment*

As displayed in Fig.13, the rover also collected one rock sample. This has been done unintentionally because I have not integrated a strategy for approaching a rock sample (if there is one near the rover) into the wall crawler policy. What really happened is that the rover happened to reach the end of a map branch, so the `stop` mode is triggered and luckily there is a rock sample near by.

The drawback of wall crawler policy is the rover can get stuck by the wall's foot when it move too close to the wall (Fig.14). To get out of this situation, the rover needs to perform a four-wheel turn. However, the `stop` mode is not triggered because the field of view is quite large; therefore, the rover keeps moving forward. This situation does not resolve on it own because the stuck of the rover make the field of view stay the same, so the steering angle also stay the same.

![alt text][image14]

*Fig.14 The rover get stuck by the foot of the mountain*

I try to avoid this situation by adding a bias to wall crawler policy as following,
```
left_mean_ang = Rover.nav_angles[Rover.nav_angles > nav_angles_mean - bias_angle]
```
This bias can keep the rover a little bit further away from the wall, but it can not guarantee that the rover will not get stuck. Adding a mean to help the rover know if it is stuck is what I will do to upgrade the rover's program, in addition to the strategy of approaching the rock samples and how to bring them to the starting point.  
