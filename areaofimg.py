import cv2

# Load the image
image = cv2.imread('/home/nakshtra/Desktop/GroundingDINO/rich/last_captured_image.jpg_8260_s00_00013.jpg')

# Get the dimensions of the image
height, width, channels = image.shape

# Calculate the area (total number of pixels)
area = height * width

print(f"Area of the entire image: {area} pixels")
