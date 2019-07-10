import math
import numpy as np
import random
import sys
from PIL import Image, ImageDraw, ImageOps

# LiDar Parameters
max_range = 200
interval = 5  # in degree
period = 1  # in seconds

# Movement Parameters
pix_per_m = 10
max_turn = 30  # Degrees


def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def generateLocalMap(map_img, map_size, local_map_size, scale, curr_x, curr_y):
    # Scale, center
    scaled_map_size = round(map_size * scale)
    scaled_map_img = map_img.resize((scaled_map_size, scaled_map_size), Image.ANTIALIAS)
    scaled_x = round(curr_x * scale)
    scaled_y = round(curr_y * scale)
    left_bound = max(0, round(scaled_x - local_map_size/2))
    up_bound = max(0, round(scaled_y - local_map_size/2))
    right_bound = min(scaled_map_size, round(scaled_x + local_map_size/2))
    down_bound = min(scaled_map_size, round(scaled_y + local_map_size/2))
    temp_img = scaled_map_img.crop((left_bound, up_bound, right_bound, down_bound))

    paste_x = round(local_map_size / 2 - (scaled_x - left_bound))
    paste_y = round(local_map_size / 2 - (scaled_y - up_bound))

    out_img = Image.new('L', (local_map_size, local_map_size), 255)
    out_img.paste(temp_img, (paste_x, paste_y))
    # out_img.show()
    return out_img


def getAngleGradient(curr_x, curr_y, near_x, near_y, depths, point_angles, sigma, curr_a):
    # Finds gradient of negative log likelihood of initial points wrt curr_a (Initial heading)
    # Used for gradient descent to find curr_a
    num_points = len(depths)
    gradient = 0
    for i in range(num_points):
        sweep_angle = point_angles[i]
        ref_a = curr_a + sweep_angle
        term_a = (near_x[i] - curr_x ) * math.sin(math.radians(ref_a))
        term_b = (-near_y[i] + curr_y) * math.cos(math.radians(ref_a))
        gradient = gradient + depths[i] * (term_a - term_b)
    gradient = gradient/(sigma**2)
    return gradient


def getRefDepthAngle(curr_x, curr_y, near_x, near_y, curr_scale):
    # Finds and returns a tuple of all reference point depth and a tuple of all reference point angle wrt x axis.
    # ref_pt_depth is c_i (Depth when scale = 1)
    ref_pt_angles = ()
    ref_pt_depth = ()
    num_points = len(near_x)
    for i in range(num_points):
        if near_x[i] == curr_x:
            angle = 180 + np.sign(curr_y - near_y[i]) * 90
        else:
            angle = math.degrees(math.atan((near_y[i] - curr_y)/(near_x[i] - curr_x)))
        ref_depth = (near_x[i] - curr_x)/(curr_scale * math.cos(math.radians(angle)))

        ref_pt_angles = ref_pt_angles + (angle, )
        ref_pt_depth = ref_pt_depth + (ref_depth,)
    return ref_pt_depth, ref_pt_angles


def getScaleGradient(curr_x, curr_y, near_x, near_y, depths, point_angles, sigma, curr_a, ref_pt_angles, ref_pt_depth):
    # Finds gradient of negative log likelihood of initial points wrt curr_scale
    # Used for gradient descent to find curr_scale
    num_points = len(depths)
    gradient = 0
    for i in range(num_points):
        sweep_angle = point_angles[i]
        obs_a = curr_a + sweep_angle
        ref_angle = ref_pt_angles[i]
        ref_depth = ref_pt_depth[i]  # c_i
        term_a = (curr_x + depths[i] * math.cos(math.radians(obs_a)) - near_x[i]) * ref_depth * math.cos(math.radians(ref_angle))
        term_b = (curr_y - depths[i] * math.sin(math.radians(obs_a)) - near_y[i]) * ref_depth * math.sin(math.radians(ref_angle))
        gradient = gradient + term_a + term_b
    gradient = - gradient/(sigma**2)
    return gradient


def getReadingCoordinates(curr_x, curr_y, curr_a, lidar_readings, point_angles):
    # Find coordinates of all points read by lidar
    points = ()
    for i in range(len(lidar_readings)):
        depth = lidar_readings[i]
        sweep_angle = point_angles[i]
        if depth < 0:
            depth = 0
        ref_a = (sweep_angle + curr_a) % 360
        sense_x = curr_x + depth * math.cos(math.radians(ref_a))
        sense_y = curr_y - depth * math.sin(math.radians(ref_a))
        points = points + ((sense_x, sense_y),)
    return points


def getPositionGradient(curr_points, near_x, near_y, sigma):
    num_points = len(curr_points)

    x_grad = 0
    y_grad = 0

    for i in range(num_points):
        x_grad = x_grad + (curr_points[i][0] - near_x[i])
        y_grad = y_grad + (curr_points[i][1] - near_y[i])

    x_grad = x_grad/sigma
    y_grad = y_grad/sigma

    return x_grad, y_grad


def getCost(curr_x, curr_y, near_x, near_y, depths, point_angles, sigma, curr_a):
    # Evaluate cost function (NLL) of initial points wrt curr_a
    num_points = len(depths)
    detected = getReadingCoordinates(curr_x, curr_y, curr_a, depths, point_angles)
    cost = 0
    for i in range(num_points):
        x_i = detected[i][0]
        y_i = detected[i][1]
        cost = cost + ((x_i - near_x[i])**2 + (y_i - near_y[i])**2)
    cost = cost * (1/(2*(sigma**2)))
    return cost

def findPose(curr_x, curr_y, map_arr, map_img, lidar_readings, point_angles, sigma):
    # Use gradient descent to find initial pose
    angle_lr = 0.0003  # Learn rate
    scale_lr = 0.000001
    pos_lr = 0.06

    next_a = 100
    curr_a = 9999
    next_scale = 1.2
    curr_scale = 9999
    excess_x = 0
    excess_y = 0
    next_x = 10
    next_y = -10

    # Termination Threshold
    angle_terminate = 0.1
    scale_terminate = 0.05
    pos_terminate = 0.1

    # Local Map Parameters
    map_size = map_arr.shape[0]  # Size of given image map
    local_size = 2 * max_range  # Size of local map to work with

    # Define curr x and y as center of local map
    center_x = round(local_size / 2)
    center_y = round(local_size / 2)

    iteration = 1
    while (abs(next_a - curr_a) > angle_terminate) | (abs(next_scale - curr_scale) > scale_terminate) | (abs(next_x - excess_x) > pos_terminate) | (abs(next_y - excess_y) > pos_terminate):
        # Set curr_a to the next_a
        curr_a = next_a
        curr_scale = next_scale
        excess_x = next_x
        excess_y = next_y
        local_x = center_x + excess_x
        local_y = center_y + excess_y

        # Create local map based on curr_scale
        local_img = generateLocalMap(map_img, map_size, local_size, curr_scale, curr_x, curr_y)
        local_arr = np.array(local_img.convert('L'))

        # Find points wrt to curr_a
        curr_points = getReadingCoordinates(local_x, local_y, curr_a, lidar_readings, point_angles)

        # Find nearest neighbour to each point
        min_dist = [99999, ] * len(lidar_readings)
        near_x = [0, ] * len(lidar_readings)
        near_y = [0, ] * len(lidar_readings)

        for i in range(local_arr.shape[0]):
            for j in range(local_arr.shape[1]):
                # Check if corner pixel
                if (i != 0) & (i != (local_arr.shape[0] - 1)) & (j != 0) & (j != (local_arr.shape[1] - 1)):
                    # Skip non-corner black pixels
                    if local_arr[i][j] == 0:
                        continue
                    # Skip non corner white pixels with that are not edge (no black in 8-neighbour)
                    elif (local_arr[i-1][j-1] != 0) & (local_arr[i-1][j] != 0) & (local_arr[i-1][j+1] != 0) & (local_arr[i][j-1] != 0) & (local_arr[i][j+1] != 0) & (local_arr[i+1][j-1] != 0) & (local_arr[i+1][j] != 0) & (local_arr[i+1][j+1] != 0):
                        continue
                for k in range(len(curr_points)):
                    point = curr_points[k]
                    # Compute distance
                    dist = distance(j, i, point[0], point[1])
                    # Update Minimum
                    if dist < min_dist[k]:
                        min_dist[k] = dist
                        near_x[k] = j
                        near_y[k] = i

        # Find Current Cost
        cost = getCost(local_x, local_y, near_x, near_y, lidar_readings, point_angles, sigma, curr_a)

        # Visualise
        vis_img = Image.fromarray(local_arr, 'L')
        vis_img = vis_img.convert("RGB")
        temp_map = vis_img.copy()
        vis_draw = ImageDraw.Draw(temp_map)
        vis_draw.ellipse((local_x - 5, local_y - 5, local_x + 5, local_y + 5), fill='red', outline='red')
        for dot in curr_points:
            vis_draw.ellipse((dot[0] - 2, dot[1] - 2, dot[0] + 2, dot[1] + 2), fill='green', outline='green')
        for i in range(len(near_x)):
            vis_draw.ellipse((near_x[i] - 2, near_y[i] - 2, near_x[i] + 2, near_y[i] + 2), fill='blue', outline='blue')
        temp_map.show()

        # Compute Gradient
        angle_gradient = getAngleGradient(local_x, local_y, near_x, near_y, lidar_readings, point_angles, sigma, curr_a)
        (ref_pt_depth, ref_pt_angles) = getRefDepthAngle(local_x, local_y, near_x, near_y, curr_scale)
        scale_gradient = getScaleGradient(local_x, local_y, near_x, near_y, lidar_readings, point_angles, sigma, curr_a, ref_pt_angles, ref_pt_depth)
        (x_gradient, y_gradient) = getPositionGradient(curr_points, near_x, near_y, sigma)

        # Display Iteration Information
        print("iteration {}: excess_x = {}, excess_y = {}, curr_a = {}, curr_scale = {}, cost = {}, a_grad = {}, s_grad = {}, x_grad = {}, y_grad = {}".format(iteration, excess_x, excess_y, next_a, next_scale, cost, angle_gradient, scale_gradient, x_gradient, y_gradient))

        # Find the next curr_a to use
        next_a = curr_a - angle_gradient * angle_lr
        next_scale = curr_scale - scale_gradient * scale_lr
        next_x = excess_x - x_gradient * pos_lr
        next_y = excess_y - y_gradient * pos_lr

        iteration += 1

    return next_a, next_scale, next_x, next_y


# START OF MAIN PROGRAM
def main():
    # Open Map Image
    map_img = Image.open("map.png")
    # Open Actions and LiDar Readings
    action = open("readings_stateaction.txt", "r")
    readings = open("readings_lidar.txt", "r")

    # Convert map image into an array
    map_arr = np.array(map_img.convert('L'))

    # Get the start coordinates
    start = action.readline()
    (start_x, start_y) = start.split()
    curr_x = float(start_x)
    curr_y = float(start_y)

    # Get the start LiDar readings
    start_readings = readings.readline()
    start_readings = start_readings.split()
    start_readings = list(map(lambda x: float(x), start_readings))

    # Remove points that did not find anything
    usable_readings = start_readings.copy()
    usable_headings = list(range(0, 360, interval))
    for i in range(len(usable_readings)-1, -1, -1):
        if usable_readings[i] > (max_range * 0.98):
            del usable_readings[i]
            del usable_headings[i]

    sigma = 0.95
    (curr_a, curr_scale, excess_x, excess_y) = findPose(curr_x, curr_y, map_arr, map_img, usable_readings, usable_headings, sigma)
    curr_x = curr_x + excess_x
    curr_y = curr_y + excess_y

    # TODO make use of curr_scale to project lidar readings back on global map
    # Interpret Initial LiDar Readings
    sweep_angles = list(range(0, 360, interval))
    global_depths = list(map(lambda x: x/curr_scale, start_readings))
    points = getReadingCoordinates(curr_x, curr_y, curr_a, global_depths, sweep_angles)

    # Visualise Observations
    map_img = map_img.convert("RGB")
    temp_map = map_img.copy()
    map_draw = ImageDraw.Draw(temp_map)
    map_draw.ellipse((curr_x - 5, curr_y - 5, curr_x + 5, curr_y + 5), fill='red', outline='red')
    for dot in points:
        map_draw.ellipse((dot[0] - 2, dot[1] - 2, dot[0] + 2, dot[1] + 2), fill='green', outline='green')
    temp_map.show()

    return
# TODO Gradient Descent for only works if NN association mostly correct. Set Initial heading and scale manually.
# TODO Nearest Neighbour is extremely slow when there are a lot of white pixels.


if __name__ == "__main__":
    main()
