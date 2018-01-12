import numpy as np

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

# This is where you can build a decision tree for determining throttle, brake and steer
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        nav_angles_margin = len(Rover.nav_angles) # always positive
        # because the Rover.nav_angles array is sorted
        # Check for Rover.mode status
        if Rover.mode == 'forward':
            # check the stop criterion
            if (nav_angles_margin > Rover.stop_forward):
                # If the above condition is true, then the terrain looks good
                # If mode is forward, navigable terrain looks good
                # and velocity is below max, then throttle
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # use wall crawling policy to design steering angle
                nav_angles_mean = np.mean(Rover.nav_angles)
                if not Rover.obst_in_view:
                    if Rover.steer_favor_left:
                        left_mean_ang = Rover.nav_angles[Rover.nav_angles > nav_angles_mean]
                        Rover.steer = np.clip(np.mean(left_mean_ang * 180/np.pi), -15, 15)
                    else:
                        right_mean_ang = Rover.nav_angles[Rover.nav_angles < nav_angles_mean + 2 * np.pi/180]
                        Rover.steer = np.clip(np.mean(right_mean_ang * 180/np.pi), -15, 15)
                    # Check if there are any obstacle ahead if move with Rover.steer angle
                    dists_around_steer = np.abs(Rover.nav_angles - Rover.steer) < (2 * np.pi/180)
                    mean_dist_ahead = np.mean(Rover.nav_dists[dists_around_steer])
                    if mean_dist_ahead < Rover.min_dist_ahead_thres:
                        Rover.mode = 'stop'
                else:
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
            # If there's a lack of nav_angles_margin or nav_dists_mean, then go to 'stop' mode
            else:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    if Rover.brake < Rover.brake_set:
                        Rover.brake += 0.3
                    Rover.steer = 0
                    Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        else: # Rover.mode == 'stop'
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                if Rover.brake < Rover.brake_set:
                    Rover.brake += 0.3
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            else:
                # Now we're stopped and we have vision data to see if there's a path forward
                if nav_angles_margin < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                else:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'
                    # if Rover.steer_unchange_cnt < (16 * 3):
                    #     Rover.steer_favor_left = not Rover.steer_favor_left
    # Just to make the rover do something
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    # Update value of Rover.steer_prev for the next loop
    Rover.steer_prev = Rover.steer

    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True

    return Rover
