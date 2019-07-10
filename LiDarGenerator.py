import math
import numpy as np
import random
from PIL import Image, ImageDraw


def addMoveNoise(speed, steer):
    sd_speed = 0.02
    sd_steer = 0.02
    return speed + random.gauss(0, sd_speed), steer + random.gauss(0, sd_steer)


def addObsNoise(obs):
    sd = 1
    for i in range(len(obs)):
        obs[i] = obs[i] + random.gauss(0, sd)
    return obs


def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2 )


def checkCollision(x, y, ground_truth):
    if ground_truth[y, x] != 0:
        return True
    else:
        return False


def getObservation(max_range, interval, curr_x, curr_y, curr_a, ground_truth, map_width, map_height):
    measurements = []
    positions = ()
    # Simulate 360degree sweep
    for sweep_angle in range(0, 360, interval):
        check_x = int(round(curr_x))
        check_y = int(round(curr_y))
        ref_a = (curr_a + sweep_angle) % 360
        obs_depth = 0
        while obs_depth < max_range:
            # Extend Checking
            if (ref_a <= 45) | (ref_a > 315):
                check_x = check_x + 1
                check_y = round(curr_y - math.tan(math.radians(ref_a)) * (check_x - curr_x))
            elif (ref_a > 45) & (ref_a <= 135):
                check_y = check_y - 1
                check_x = round(curr_x + ((curr_y - check_y) / (math.tan(math.radians(ref_a)))))
            elif (ref_a > 135) & (ref_a <= 225):
                check_x = check_x - 1
                check_y = round(curr_y - math.tan(math.radians(ref_a)) * (check_x - curr_x))
            elif (ref_a > 225) & (ref_a <= 315):
                check_y = check_y + 1
                check_x = round(curr_x + ((curr_y - check_y) / (math.tan(math.radians(ref_a)))))
            # Check if out of range
            if (check_x < 0) | (check_y < 0) | (check_x >= map_width) | (check_y >= map_height):
                break

            obs_depth = distance(check_x, check_y, curr_x, curr_y)

            # Determine if obstacle exist
            if checkCollision(check_x, check_y, ground_truth):
                break

        if obs_depth > max_range:
            obs_depth = max_range
            check_x = round(curr_x + (math.cos(math.radians(ref_a)) * max_range))
            check_y = round(curr_y - (math.sin(math.radians(ref_a)) * max_range))
        else:
            check_x = round(curr_x + (math.cos(math.radians(ref_a)) * obs_depth))
            check_y = round(curr_y - (math.sin(math.radians(ref_a)) * obs_depth))

        measurements = measurements + [obs_depth, ]
        positions = positions + ((check_x, check_y), )
    return measurements, positions


# LiDar Parameters
max_range = 200
interval = 5  # in degree
period = 1  # in seconds

# Movement Parameters
pix_per_m = 10
max_turn = 30  # Degrees

# Load Ground Truth Generated
ground_truth = np.load("ground_truth.npy")
map_width = ground_truth.shape[1]
map_height = ground_truth.shape[0]

# Start file to save LiDar readings
action = open("readings_stateaction.txt", "w")
readings = open("readings_lidar.txt", "w")

# Initial Position of LiDar
curr_x = map_width/2
curr_y = int(map_height * (3/4))
curr_a = 90  # In degree. 0 degree is in +ve x direction in cartesian axis
action.write(str(curr_x)+" "+str(curr_y)+"\n")

# Belief (Used for visualisation)
fake_x = curr_x
fake_y = curr_y
fake_a = curr_a

# Loop through a simulated series of actions and observations
for cycles in range(50):
    # Observe with noise
    (measurements, positions) = getObservation(max_range, interval, curr_x, curr_y, curr_a, ground_truth, map_width, map_height)

    # Visualise Observations without Noise
    # map_img = Image.fromarray(ground_truth, 'L')
    # map_img = map_img.convert("RGB")
    # temp_map = map_img.copy()
    # map_draw = ImageDraw.Draw(temp_map)
    # map_draw.ellipse((curr_x - 5, curr_y - 5, curr_x + 5, curr_y + 5), fill='red', outline='red')
    # for point in positions:
    #     map_draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill='green', outline='green')
    # temp_map.show()

    # Introduce Noise to Depth Measurement
    measurements = addObsNoise(measurements)

    # Visualise Observations with Noise
    # map_img = Image.fromarray(ground_truth, 'L')
    # map_img = map_img.convert("RGB")
    # temp_map = map_img.copy()
    # map_draw = ImageDraw.Draw(temp_map)
    # map_draw.ellipse((curr_x - 5, curr_y - 5, curr_x + 5, curr_y + 5), fill='red', outline='red')
    # sweep_angle = 0
    # for depth in measurements:
    #     ref_a = (curr_a + sweep_angle) % 360
    #     point_x = round(curr_x + (math.cos(math.radians(ref_a)) * depth))
    #     point_y = round(curr_y - (math.sin(math.radians(ref_a)) * depth))
    #     map_draw.ellipse((point_x - 2, point_y - 2, point_x + 2, point_y + 2), fill='green', outline='green')
    #     sweep_angle = sweep_angle + 5
    # temp_map.show()

    # Record Observations
    for depth in measurements:
        readings.write(str(depth) + " ")
    readings.write("\n")

    # Move with noise and record control action
    speed = 5  # in m/s
    steer = 0  # Range from -1(Hard Left) to 1(Hard Right), 0 Neutral
    action.write(str(speed) + " " + str(steer) + "\n")  # Format: Speed Steer \n
    (true_speed, true_steer) = addMoveNoise(speed, steer)
    # Update True Position
    curr_a = curr_a + true_steer * max_turn
    move_dist = true_speed * period
    curr_x = curr_x + (move_dist * math.cos(math.radians(curr_a)))
    curr_y = curr_y - move_dist * math.sin(math.radians(curr_a))

    # Visualise True and Perceived Positions
    # fake_a = fake_a + steer * max_turn
    # fake_dist = speed * period
    # fake_x = fake_x + (fake_dist * math.cos(math.radians(fake_a)))
    # fake_y = fake_y - fake_dist * math.sin(math.radians(fake_a))
    # map_img = Image.fromarray(ground_truth, 'L')
    # map_img = map_img.convert("RGB")
    # temp_map = map_img
    # map_draw = ImageDraw.Draw(temp_map)
    # map_draw.ellipse((new_x - 5, new_y - 5, new_x + 5, new_y + 5), fill='red', outline='red')
    # map_draw.ellipse((fake_x - 5, fake_y - 5, fake_x + 5, fake_y + 5), fill='yellow', outline='yellow')
    # temp_map.show()

