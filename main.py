import cv2
import numpy as np
from image_processing import *

# Parameters
Image_Size = 540
Upper_Boundary = 0.8
Low_Boundary = 5000
Color_Range_1 = (151, 135, 16)
Color_Range_2 = (255, 255, 255)


# Macros
def show_image(image_space, image_name):
    return cv2.imread(f'{image_space}/{image_name}')


def start_processing_image(image):
    # Scale image
    image = scale(image, Image_Size)
    # Add border
    image = addborder(image, 100)
    # Denoise image
    image = denoise(image, 15)
    return image


def image_masking(image, image_mask):
    # Gray image
    image_gray = gray(image_mask)

    # Threshold
    image_threshold = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contour
    contours = cv2.findContours(image_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    higher_boundary = (image.shape[0] * image.shape[1]) * Upper_Boundary
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        cv2.fillPoly(image_threshold, [cnt], color=(0, 0, 0))
        if Low_Boundary < area < higher_boundary:
            filtered_contours.append(cnt)

    # Fill contour
    for cnt in filtered_contours:
        cv2.fillPoly(image_threshold, [cnt], color=(255, 255, 255))

    # Mask original image with processed threshold mask
    final_image = cv2.bitwise_and(image, image, mask=image_threshold)

    # Highlight object
    object_number = 0
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(final_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(final_image, f'Rose {object_number + 1}', (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        object_number += 1
    return final_image

def run_modify(image_file_name, image):
    cv2.namedWindow("HSV Lower Boundary Slider")
    cv2.namedWindow("HSV Upper Boundary Slider")
    cv2.createTrackbar("Lower Hue", "HSV Lower Boundary Slider", Color_Range_1[0], 255, lambda x: x)
    cv2.createTrackbar("Lower Saturation", "HSV Lower Boundary Slider", Color_Range_1[1], 255, lambda x: x)
    cv2.createTrackbar("Lower Value", "HSV Lower Boundary Slider", Color_Range_1[2], 255, lambda x: x)
    cv2.createTrackbar("Upper Hue", "HSV Upper Boundary Slider", Color_Range_2[0], 255, lambda x: x)
    cv2.createTrackbar("Upper Saturation", "HSV Upper Boundary Slider", Color_Range_2[1], 255, lambda x: x)
    cv2.createTrackbar("Upper Value", "HSV Upper Boundary Slider", Color_Range_2[2], 255, lambda x: x)

    while True:
        lower_hue = cv2.getTrackbarPos("Lower Hue", "HSV Lower Boundary Slider")
        lower_saturation = cv2.getTrackbarPos("Lower Saturation", "HSV Lower Boundary Slider")
        lower_value = cv2.getTrackbarPos("Lower Value", "HSV Lower Boundary Slider")
        upper_hue = cv2.getTrackbarPos("Upper Hue", "HSV Upper Boundary Slider")
        upper_saturation = cv2.getTrackbarPos("Upper Saturation", "HSV Upper Boundary Slider")
        upper_value = cv2.getTrackbarPos("Upper Value", "HSV Upper Boundary Slider")
        lower = np.array([lower_hue, lower_saturation, lower_value])
        upper = np.array([upper_hue, upper_saturation, upper_value])

        # Create and apply mask
        mask_image = create_apply_mask(image, lower, upper)
        # Process the image with mask
        final_image = image_masking(image, mask_image)

        cv2.putText(final_image, f'Debug mode', (0, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv2.putText(final_image, f'Press Esc Twice to Exit...', (0, 35), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv2.imshow(image_file_name, final_image)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break


def create_apply_mask(image, first, last):
    # HSV functions
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, first, last)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Masking
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)
    mask_image = cv2.bitwise_and(image, image, mask=mask_close)
    return mask_image


def main(Image_space, image_file_name, Is_Modify):
    # Load image
    image = show_image(Image_space, image_file_name)
    # Preprocess image
    image = start_processing_image(image)

    if Is_Modify:
        run_modify(image_file_name, image)
        return

    mask = create_apply_mask(image, Color_Range_1, Color_Range_2)
    final_image = image_masking(image, mask)

    cv2.imshow(image_file_name, final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
