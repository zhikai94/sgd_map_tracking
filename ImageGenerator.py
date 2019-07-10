from PIL import Image, ImageDraw
import numpy as np
import random


def rand_object(map_draw, cen_x, cen_y):
    size = random.randint(10, 20)
    shape = random.randint(1, 2)
    if shape == 1: #Square
        map_draw.rectangle((cen_x-size, cen_y-size, cen_x+size, cen_y+size), fill='white', outline='white')
    elif shape == 2: #Circle
        map_draw.ellipse((cen_x-size, cen_y-size, cen_x+size, cen_y+size), fill='white', outline='white')

    return map_draw


# Create Map Image and populate it with num_obj number of obstacles
# Note: The obstacle file lists cen_x then cen_y of obstacles
map_size = 512
num_obj = 3

obstacle_file = open("obstacle.txt", "w")
map_arr = np.full((map_size, map_size, 3), 0, dtype=np.uint8)
map_img = Image.fromarray(map_arr, 'RGB')
map_draw = ImageDraw.Draw(map_img)
# Obstacle Create so that they are by the side of a pavement
for i in range(num_obj):
    # cen_x = random.randint(20, map_arr.shape[1] - 20)
    cen_x = random.randint(70, 120)
    x_side = random.randint(1, 2)
    if x_side == 1:
        cen_x = int(np.floor(map_arr.shape[1]/2) - cen_x)
    elif x_side == 2:
        cen_x = int(np.floor(map_arr.shape[1] / 2) + cen_x)
    cen_y = int( random.randint(20, np.floor((map_arr.shape[0])/num_obj)-20) + np.floor((map_arr.shape[0])/num_obj)*i )
    obstacle_file.write(str(cen_x) + " " + str(cen_x) + "\n")
    rand_object(map_draw, cen_x, cen_y)

map_img.show()
map_img.save('map.png')

# Make map of ground truth
map_arr = np.array(map_img.convert('L'))
np.save("ground_truth", map_arr)
