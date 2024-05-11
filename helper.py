from ultralytics import YOLO
import time
import streamlit as st
import cv2
import supervision as sv
import settings
import numpy as np




def load_model(model_path, classes=None):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model




# Function to calculate the coord of a bounding box

def get_bbox_coord(box):
    x2, y2 = box[2], box[3]
    return int(x2), int(y2)


# Function to check if a point is inside a polygon
def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0


def track_and_count_people(
    model, frame, st_frame,
    area1, area2,
    people_entering, entering,
    people_exiting, exiting,
    video_writer=None,
    
   
):
    results = model.track(frame, conf=.7,
                          persist=True, classes=[0],
                          tracker="bytetrack.yaml")

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()
        

        for box, track_id in zip(boxes, track_ids):

            # Calculate bounding box x y
            track_bbox = get_bbox_coord(box)

            # Check if person is entering or exiting
            entering_area1_result = point_in_polygon(track_bbox, area1)
            if entering_area1_result:
                people_entering[track_id] = track_bbox

            if track_id in people_entering:
                entering_area2_result = point_in_polygon(track_bbox, area2)
                if entering_area2_result:
                    entering.add(track_id)
                    del people_entering[track_id]

            exiting_area2_result = point_in_polygon(track_bbox, area2)
            if exiting_area2_result:
                people_exiting[track_id] = track_bbox

            if track_id in people_exiting:
                exiting_area1_result = point_in_polygon(track_bbox, area1)
                if exiting_area1_result:
                    exiting.add(track_id)
                    del people_exiting[track_id]

        
            # Visualize areas of interest and statistics
            cv2.putText(frame, 'Number of entering people= ' + str(len(entering)),
                        (20, 44), cv2.FONT_HERSHEY_COMPLEX, (1), (0, 255, 0), 2)
            cv2.putText(frame, 'Number of Exiting people= ' + str(len(exiting)),
                        (20, 70), cv2.FONT_HERSHEY_COMPLEX, (1), (0, 255, 0), 2)

            st_frame.image(frame, caption='Detected Video',
                           channels="BGR", use_column_width=True)

            if video_writer is not None:
                video_writer.write(frame)
            
            


def run_tracking_video(model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    if source_vid == "video_2":
        # Define areas of interest (replace with your desired coordinates)
        area1 = [(494, 289), (505, 499), (578, 496), (530, 292)]
        area2 = [(548, 290), (600, 496), (637, 493), (574, 288)]
    else:
        area2 = [(312, 388), (289, 390), (474, 469), (497, 462)]
        area1 = [(279, 392), (250, 397), (423, 477), (454, 469)]

    entering = set()
    exiting = set()
    people_entering = {}
    people_exiting = {}

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)
    else:
        st.error("Error: Empty video file!")

    if st.sidebar.button('Count People'):
        try:
            # Video setup
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))

            new_width = 1020
            new_height = 500

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, frame = vid_cap.read()

                if not success:
                    vid_cap.release()
                    cv2.destroyAllWindows()
                    break
                # writer = create_video_writer(vid_cap, "Output.mp4")
                frame = cv2.resize(frame, (new_width, new_height))
                track_and_count_people(
                    model=model, frame=frame,
                    st_frame=st_frame,
                    area1=area1, area2=area2,
                    people_entering=people_entering, entering=entering,
                    people_exiting=people_exiting, exiting=exiting,
                  
                )
               
            # Success message (optional)
            st.success(f"Video processing complete")
            vid_cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
