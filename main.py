# Author: Michael
# Date: October 5, 2023,
# Description: This script perform object detection or hole detection inside the seed tray.


import cv2
import numpy as np
import threading
import queue

circle_count = 0


# Function to process frames
def process_frame(frame, result_queue):
    global circle_count
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ----------------brown--------------------
    # Hmin = 0, h_max=179, s_min=33, smax=255, v_min=0, v_max=255, erode=40
    # lower brown
    h_min = 0
    s_min = 15
    v_min = 0

    # upper brown
    h_max = 60
    s_max = 255
    v_max = 255

    lower_brown = np.array([h_min, s_min, v_min])
    upper_brown = np.array([h_max, s_max, v_max])

    # Create a mask from lower and upper bound of brown
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    blurred = cv2.GaussianBlur(mask_brown, (3, 3), 2)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    # Apply the erosion operation to the image based on the trackbar value 23, 23
    erosion_min = 25
    erosion_max = 25
    kernel = np.ones((erosion_min, erosion_max), np.uint8)
    eroded_thresh = cv2.erode(thresh, kernel)

    # Apply a dilation to the image 20, 20
    dilation_kernel = np.ones((20, 20), np.uint8)
    dilated_thresh = cv2.dilate(eroded_thresh, dilation_kernel, iterations=1)

    # Apply canny edge detection to the dilate image
    dilated_edges = cv2.Canny(dilated_thresh, 100, 255)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(dilated_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set the rectangle width and height
    rectangle_width = 35  # Adjust as needed
    rectangle_height = 35  # Adjust as needed

    # set the diameter of the circle
    circle_diameter = 10  # adjust as needed

    # Loop over the contours and draw bounding boxes
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Adjust this threshold as needed
            (x, y, w, h) = cv2.boundingRect(contour)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate the center coordinates
            centerX = x + w // 2
            centerY = y + h // 2

            # Calculate the new rectangle coordinates
            new_x = centerX - rectangle_width // 2
            new_y = centerY - rectangle_height // 2

            # Draw a circle at the center
            cv2.circle(frame, (centerX, centerY), circle_diameter, (255, 255, 0), 1)

            # Draw a bounding box with the adjusted size
            cv2.rectangle(frame, (new_x, new_y), (new_x + rectangle_width, new_y + rectangle_height), (255, 255, 0), 1)

            # # Calculate the radius for the circle
            # radius = max(w, h) // 2 + circle_diameter // 2

            # Calculate the radius for the circle
            radius = max(rectangle_width, rectangle_height) // 2 + circle_diameter // 2

            # Add a label to the top-left corner of the bounding box
            label = f'Circle'  # {circle_count}'
            cv2.putText(frame, label, (new_x, new_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

            # Color
            # Red: (255, 0, 0)
            # Orange: (255, 165, 0)
            # Yellow: (255, 255, 0)
            # Green: (0, 128, 0)
            # Blue: (0, 0, 255)
            # Indigo: (75, 0, 130)
            # Violet: (148, 0, 211)

            # Draw a circle around the bounding box
            cv2.circle(frame, (centerX, centerY), radius, (0, 255, 0), 2)

    # Increment the bounding box count
    circle_count += 1

    result_queue.put((frame, thresh, dilated_edges, dilated_thresh, eroded_thresh))


# Open the video file
video_capture = cv2.VideoCapture('sample.mp4')

# Create a queue to store processing results
result_queue = queue.Queue()

# Set the desired frame rate (frames per second)
desired_fps = 30  # Adjust this value as needed

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    # Create a thread to process the frame
    frame_thread = threading.Thread(target=process_frame, args=(frame, result_queue))
    frame_thread.start()

    # Extract the results from the queue
    frame, thresh, dilated_edges, dilated_thresh, eroded_thresh = result_queue.get()

    # Display the images
    cv2.imshow("circle_detection", frame)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Dilation Image", dilated_edges)
    cv2.imshow("Edges Image", dilated_thresh)
    cv2.imshow("Eroded Image", eroded_thresh)

    # Print the total number of circle detected
    print(f'Total circle count detected: {circle_count}')

    # delay
    delay = int(1000 / desired_fps)

    print(f"{desired_fps} fps")

    # Break the loop if the user presses 'q' or if the video ends
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
