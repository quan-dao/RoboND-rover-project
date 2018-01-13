
[//]: # (Image References)

[image1]: ./write-up-img/original_img.png
[image2]: ./write-up-img/nav_terrain_obst.png
[image3]: ./write-up-img/first_threshold_img.png
[image4]: ./write-up-img/second_threshold_img.png
[image5]: ./write-up-img/rock_sample.png
[image6]: ./write-up-img/rock_img.png
[image7]: ./write-up-img/fov_normal_2.png
[image8]: ./write-up-img/fov_obs_ahead.png
[image9]: ./write-up-img/angles_vs_dist.png


#### 2.1.1.2 Find the open part of rover camera field of view
The distance and angle coordinate relative to the rover frame of navigable terrain pixels is found by
```
dist, angles = rover_coord(nav_terrain)
```
When the rover is at the open field (like in the figure below), it is alright to determine it steering angle, which is denoted by the red arrow by calculating the mean of angle coordinate of every pixel.
```
steering_angle = np.mean(angles)
```
![alt text] [image7]
However, if the rover is near an obstacle this strategy does not perform well.
![alt text] [image8]
The figure above show that when there is an obstacle near the middle of the field of the view of rover camera, the steering direction calculated by the mean angles tend to point toward the obstacle.

My idea for handling this is that an obstacle(s) divides the rover 's camera field of into two (or more) parts, so I will choose the bigger (or the biggest) part and calculate mean angles of this part to get the good rover steering angle.

The presence of the obstacle causes a suddenly fall in the distance coordinate of pixel near the obstacle. Plotting the pixels' angle versus pixels' distance coordinate results in the figure below.
![alt text] [image9]
The sub figure on the left shows no clear pattern. But, when I sort the pixels angle coordinate (the `angles` array) and arrange the pixels distance coordinate  (the `dist` array) with the order of the `angle` array as following,
```
sort_index = np.argsort(angles) # extract the sort result in the form of the index of angles array
sorted_angles = angles[sort_index] # arrange the angles array with the index above
```
the pattern on the sub figure to the right emerges.  
