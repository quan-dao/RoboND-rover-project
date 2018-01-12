import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
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

# Threshold for rocks
def rock_thresh(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # convert from color img to gray img
    # apply adaptive threhold to the whole img to isolate navigale terrain from rocks & mountain
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

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    return warped

# Find the open part of the Rover.nav_angles for the decision step        
def find_nav_angles(dist, angles):
    # 1.st dectect the discontinuous in the field of view
    # 2.nd if there are any discont, choose the open side othe field of view, else the whole field is open
    # Return the open side of the field of view

    # sort the angles array in the increasing order (the first one the smallest)
    sort_index = np.argsort(angles) # extract the sort result in the form of the index of angles array
    sorted_angles = angles[sort_index] # arrange the angles array with the index above
    sorted_dist = dist[sort_index] # arrange the dist array using the order of the angles array
    # there are many dist have relatively equal angles
    # next to find out with a given value of angle, what is the maximum value of distances having this angle
    threhold = 3 * np.pi/180 # (rad), criterion to for 2 angles to be considered the same
    max_dist_resp_ang = [] # maximum distance with a given angles (there are dist have the same angles)
    base_ang_array = [] # array to store angle having different distance
    i_start = 0 # looping variable
    i_end = 0 # looping variable
    base_ang = sorted_angles[i_start]
    while i_end < len(sorted_angles):
        if np.abs(sorted_angles[i_end] - base_ang) < threhold: # check the relatively equal criterion
            i_end += 1
        else:
            max_dist_resp_ang.append(max(sorted_dist[i_start:i_end]))
            base_ang_array.append(base_ang)
            # update looping variable
            i_start = i_end
            i_end = i_start + 1
            base_ang = sorted_angles[i_start]

    if i_start == len(sorted_angles) - 1: # i_start reach the end of the angles array, so i_end is out of range
        max_dist_resp_ang.append(sorted_dist[i_start])
        base_ang_array.append(sorted_angles[i_start])
    # Detect the local maxima (maximum & minimum) of the graph (base angles, maximum dist respect to base angles)
    n = len(max_dist_resp_ang)
    maxima_dist_value = []
    maxima_dist_base_ang = []
    for i in range(1, n-1):
        flag_max = (max_dist_resp_ang[i] > max_dist_resp_ang[i - 1]) \
                    & (max_dist_resp_ang[i] > max_dist_resp_ang[i + 1]) # condition of maximum
        flag_min = (max_dist_resp_ang[i] < max_dist_resp_ang[i - 1]) \
                    & (max_dist_resp_ang[i] < max_dist_resp_ang[i + 1]) # condition of minimum
        if flag_max or flag_min:
            maxima_dist_value.append(max_dist_resp_ang[i])
            maxima_dist_base_ang.append(base_ang_array[i])
    # Detect the discontinuous points, i.e. the rapid change in the value of local maxima
    maxima_threhold = 30 # condition detect a discontinuous
    cut_off_ang = [] # store the value of cut off angle
    for i in range(1, len(maxima_dist_value)):
        if np.abs(maxima_dist_value[i] - maxima_dist_value[i - 1]) > maxima_threhold:
            cut_off_ang.append(maxima_dist_base_ang[i])
    # Check the number of discont points, then choose the approriate part of the field
    n_cut_off = len(cut_off_ang)
    if n_cut_off == 1: # one distcont in the field of view
        ind_sorted_angles = 0
        while ind_sorted_angles < len(sorted_angles):
            if sorted_angles[ind_sorted_angles] < 0.5:
                ind_sorted_angles += 1
            else:
                break;
        # Find the open side of the field of view, i.e. the bigger part of the field of view
        if ind_sorted_angles > (len(sorted_angles) / 2):
            open_angles = sorted_angles[:ind_sorted_angles]
            open_dist = sorted_dist[:ind_sorted_angles]
        else:
            open_angles = sorted_angles[ind_sorted_angles:]
            open_dist = sorted_dist[ind_sorted_angles:]
    elif n_cut_off > 1: # more than 1 discont, may be cannot be more than 2
        # Divide the field of view into 3 parts
        left_part = sorted_angles[sorted_angles <= min(cut_off_ang)]
        right_part = sorted_angles[sorted_angles >= max(cut_off_ang)]
        mid_part = sorted_angles[(sorted_angles > min(cut_off_ang)) \
                                 & (sorted_angles < max(cut_off_ang))]
        # Find the biggest part
        n_left = len(left_part)
        n_right = len(right_part)
        n_mid = len(mid_part)
        if n_left == max(n_left, n_right, n_mid):
            open_angles = left_part
            open_dist = sorted_dist[sorted_angles <= min(cut_off_ang)]
        elif n_right == max(n_left, n_right, n_mid):
            open_angles = right_part
            open_dist = sorted_dist[sorted_angles >= max(cut_off_ang)]
        else:
            open_angles = mid_part
            open_dist = sorted_dist[(sorted_angles > min(cut_off_ang))
                                 & (sorted_angles < max(cut_off_ang))]
    else: # there is no discont, so the whole field is open
        open_angles = sorted_angles
        open_dist = sorted_dist

    return open_dist, open_angles

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO:
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    dst_size = 5
    bottom_offset = 6
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                  [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    # 2) Apply perspective transform
    warped = perspect_transform(Rover.img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    nav_terrain = color_thresh(warped)
    obstacle = cv2.bitwise_not(nav_terrain)
    rock = rock_thresh(Rover.img)
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,0] = obstacle
    Rover.vision_image[:,:,1] = rock
    Rover.vision_image[:,:,2] = nav_terrain
    # 5) Convert map image pixel values to rover-centric coords
    nav_xpix_rov, nav_ypix_rov = rover_coords(nav_terrain)  # navigable terrain
    obs_xpix_rov, obs_ypix_rov = rover_coords(obstacle)  # obstacle
    # 6) Convert rover-centric pixel values to world coordinates
    nav_x_world, nav_y_world = pix_to_world(nav_xpix_rov, nav_ypix_rov, Rover.pos[0], Rover.pos[1],
                                            Rover.yaw, 200, 10)
    obs_x_world, obs_y_world = pix_to_world(obs_xpix_rov, obs_ypix_rov, Rover.pos[0], Rover.pos[1],
                                            Rover.yaw, 200, 10)
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    Rover.worldmap[nav_y_world, nav_x_world, 2] += 1
    Rover.worldmap[obs_y_world, obs_x_world, 0] += 1
    # Check for rock
    yrock, xrock = rock.nonzero()
    if yrock is not None:  # a rough guess of the number of white pixels if there is a rock
        rock_xpix_rov, rock_ypix_rov = rover_coords(rock)
        rock_x_world, rock_y_world = pix_to_world(rock_xpix_rov, rock_ypix_rov, Rover.pos[0], Rover.pos[1],
                                                Rover.yaw, 200, 10)
        Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        # check if rover is near rocks
        # rov_rock_dist, rov_rock_angle = to_polar_coords(rock_xpix_rov, rock_ypix_rov)
        # if rov_rock_dist is not None:
        #     if np.min(rov_rock_dist) < 10:
        #         Rover.mode = 'stop'
    # 8) Convert rover-centric pixel positions to polar coordinates
    rov_dist, rov_angle = to_polar_coords(nav_xpix_rov, nav_ypix_rov)
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    Rover.nav_dists = rov_dist  # this is an array
    Rover.nav_angles = rov_angle



    return Rover
