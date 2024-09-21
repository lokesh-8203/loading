# import cv2
# import numpy as np
# import os
# from ultralytics import YOLO

# model = YOLO('export_box.pt')  # Upload the pt file here

# video_path = r'D:\vChanel\export_box\IMG_1983.MOV'  # Path for the video
# cap = cv2.VideoCapture(video_path)

# # Get video properties
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Define the codec and create VideoWriter object
# output_path = 'Truck_loading_area.mp4'  # Name for the video to save loading_area.mp4
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# # Define the zone coordinates
# zone1 = [(1431,645), (1474,649), (1472,863), (1429,868)]  # Zone 1 coordinates
# zone2 = [(1368,321), (1417,319), (1421,524), (1374,524)]  # Zone 2 coordinates (adjust based on the video)

# # Initialize counters for objects entering the zones
# object_count1 = 0
# object_count2 = 0
# previous_count1 = 0
# previous_count2 = 0

# # Track frames where objects have been counted
# recent_frames1 = []
# recent_frames2 = []

# # Create an output directory if it doesn't exist
# output_folder = 'Export_boxes'
# os.makedirs(output_folder, exist_ok=True)

# # Function to check if a frame is recent
# def is_frame_recent(frame_index, recent_frames):
#     for i in recent_frames:
#         if frame_index - i < 20:
#             return True
#     return False

# frame_index = 0  # Initialize the frame index

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform object detection on the frame
#     results = model(frame)

#     # Draw bounding boxes and centroids on detected objects with confidence >= 0.70
#     for result in results:
#         for box in result.boxes:
#             # Filter by confidence
#             if box.conf >= 0.70:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 class_id = int(box.cls)

#                 # Calculate the width and height of the bounding box
#                 width = x2 - x1
#                 height = y2 - y1

#                 # Draw the bounding box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), -1)

#                 # Calculate and draw the centroid
#                 centroid_x = int((x1 + x2) / 2)
#                 centroid_y = int((y1 + y2) / 2)
#                 cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 255, 0), -1)

#                 # Check if the centroid is inside zone 1
#                 if cv2.pointPolygonTest(np.array(zone1, np.int32), (centroid_x, centroid_y), False) >= 0:
#                     if not is_frame_recent(frame_index, recent_frames1):
#                         object_count1 += 1
#                         recent_frames1.append(frame_index)  # Mark this frame as counted for zone 1
#                         # Mark the object as counted by drawing a different color
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 5)

#                 # Check if the centroid is inside zone 2
#                 if cv2.pointPolygonTest(np.array(zone2, np.int32), (centroid_x, centroid_y), False) >= 0:
#                     if not is_frame_recent(frame_index, recent_frames2):
#                         object_count1 += 1
#                         recent_frames2.append(frame_index)  # Mark this frame as counted for zone 2
#                         # Mark the object as counted by drawing a different color
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 5)

#     # Draw the zone bounding boxes
#     cv2.polylines(frame, [np.array(zone1, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
#     cv2.polylines(frame, [np.array(zone2, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

#     # Display the counts for each zone
#     cv2.putText(frame, f"Zone 1 Count: {object_count1}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     # cv2.putText(frame, f"Zone 2 Count: {object_count2}", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     # Check if the object count has increased in zone 1
#     if object_count1 > previous_count1:
#         frame_filename1 = os.path.join(output_folder, f"frame_zone1_{frame_index}.jpg")
#         cv2.imwrite(frame_filename1, frame)
#         previous_count1 = object_count1  # Update the previous count for zone 1

#     # Check if the object count has increased in zone 2
#     if object_count2 > previous_count2:
#         frame_filename2 = os.path.join(output_folder, f"frame_zone2_{frame_index}.jpg")
#         cv2.imwrite(frame_filename2, frame)
#         previous_count2 = object_count2  # Update the previous count for zone 2

#     # Write the frame to the output video
#     out.write(frame)

#     disp = cv2.resize(frame, (800, 800))
#     # Display the frame
#     cv2.imshow('Object Detection', disp)

#     # Increment the frame index
#     frame_index += 1

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()




















import cv2
import numpy as np
import os
from ultralytics import YOLO

model = YOLO('box.pt')#upload the pt file here

video_path = r'ch12_20120226014540.mp4'#path for the video
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
output_path = 'box_output.mp4'#name for the video to save  loading_area.mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Define the zone coordinates
# zone = [(78,203), (1805,205), (1805,250), (81,257)]
# zone = [(1040,21), (1139,19), (1141,1026), (1038,1028)]#adjust the coordinates based on the video
# zone=[(593,203), (962,206), (963,244), (594,239)]
zone=[(668,125), (704,125), (704,523), (668,523)]



# Initialize a counter for objects entering the zone
object_count = 0
previous_count = 0

# Track frames where objects have been counted
recent_frames = []

# Create an output directory if it doesn't exist
# output_folder = 'ECT_bag'
# os.makedirs(output_folder, exist_ok=True)

# Function to check if a frame is recent
def is_frame_recent(frame_index, recent_frames):
    for i in recent_frames:
        if frame_index - i < 50:
            return True
    return False

frame_index = 0  # Initialize the frame index

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame)

    # Draw bounding boxes and centroids on detected persons with confidence >= 0.75
    for result in results:
        for box in result.boxes:
            # Filter by confidence
            if box.conf >= 0.70:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)

                # Calculate the width and height of the bounding box
                width = x2 - x1
                height = y2 - y1

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), -1)

                # Calculate and draw the centroid
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 255, 0), -1)

                # Display the confidence score
                confidence_text = f"{box.conf.item():.2f}"  # Convert tensor to float
                # cv2.putText(frame, confidence_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Display the size of the bounding box
                size_text = f"W: {width} H: {height}"
                # cv2.putText(frame, size_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Check if the size of the bounding box is greater than 200x200
                if width > 20 and height > 20:
                    # Check if the centroid is inside the zone
                    if cv2.pointPolygonTest(np.array(zone, np.int32), (centroid_x, centroid_y), False) >= 0:
                        if not is_frame_recent(frame_index, recent_frames):
                            object_count += 1


                            recent_frames.append(frame_index)  # Mark this frame as counted
                            # Mark the object as counted by drawing a different color
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 5)

    # Draw the zone bounding box
    # cv2.polylines(frame, [np.array(zone, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the count
    cv2.putText(frame, f"Count: {object_count}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 8)

    # Check if the object count has increased
    if object_count > previous_count:
        # frame_filename = os.path.join(output_folder, f"frame_{frame_index}.jpg")
        # cv2.imwrite(frame_filename, frame)
        previous_count = object_count  # Update the previous count

    # Write the frame to the output video
    out.write(frame)

    disp = cv2.resize(frame, (800, 800))
    # Display the frame
    cv2.imshow('Object Detection', disp)

    # Increment the frame index
    frame_index += 1

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
