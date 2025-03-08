import cv2
import numpy as np

# Load the image
image = cv2.imread('/home/nakshtra/Desktop/GroundingDINO/rich/last_captured_image.jpg_8260_s00_00013.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a threshold to segment the object from the background
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate the area of the first contour
if contours:
    area = cv2.contourArea(contours[0])
    print(f"Area of the object: {area} pixels")

    # Optionally, draw the contour on the image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    # Display the image
    cv2.imshow('Image with Contour', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No contours found")
