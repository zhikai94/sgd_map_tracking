import numpy as np
from PIL import Image
from skimage import measure

threshold = 60
stray_size = 3

def thresholding(point):
    if point < threshold:
        return 0
    else:
        return 255

def invert(point):
    return 255 - point


def main():
    map_img = Image.open("PossibleMaps/TownPlaza.bmp")
    map_img = map_img.convert('L')
    map_img = map_img.point(thresholding)
    # Shows original Greyscaled map
    map_img.show()

    # Convert map image into an array
    map_arr = np.array(map_img.point(invert))
    # Run Connected Component Analysis
    map_labels, num_labels = measure.label(map_arr, return_num=True)
    label_count = [0, ]*(num_labels+1)
    (im_height, im_width) = map_arr.shape
    # Count the Number of Each Label
    for i in range(im_height):
        for j in range(im_width):
            label_count[map_labels[i][j]] = label_count[map_labels[i][j]] + 1
    # Remove Components with too few label counts (Small Regions)
    for i in range(im_height):
        for j in range(im_width):
            if label_count[map_labels[i][j]] < stray_size:
                map_arr[i][j] = 0

    cleaned = Image.fromarray(np.uint8(map_arr))
    cleaned = cleaned.point(invert)
    cleaned.show()


if __name__ == '__main__':
    main()
