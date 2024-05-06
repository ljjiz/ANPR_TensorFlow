#!/usr/bin/env python
# coding: utf-8

# # 1. Setup Paths
import os

# CUSTOM_MODEL_NAME = 'my_ssd_mobnet2' 
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map_3.pbtxt'



paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
}

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

for path in paths.values():
    if not os.path.exists(path):
        os.makedirs(path)




# # 2. Create Label Map


#check tensorflow
#import object_detection
# labels = [{'name':'number_plate', 'id':1}]

# with open(files['LABELMAP'], 'w') as f:
#     for label in labels:
#         f.write('item { \n')
#         f.write('\tname:\'{}\'\n'.format(label['name']))
#         f.write('\tid:{}\n'.format(label['id']))
#         f.write('}\n')


# # 3. Load Train Model From Checkpoint

import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

gpus = tf.config.list_physical_devices('GPU')

# Prevent GPU complete consumption
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    except RuntimeError as e:
        print(e)
        
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint

checkpoint_path = 'S:\\Indonesia\\skripsi\\anpr\\Tensorflow\\workspace\\models\\my_ssd_mobnet\\model_dir\\ckpt-76' 


ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(checkpoint_path).expect_partial()




@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# # 4. Detect from an Image from tested images

import cv2 
import numpy as np
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'new', 'Cars0.png')

img = cv2.imread(IMAGE_PATH)
image_np = np.array(img)

input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)

plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))


# # 5. Apply OCR to Detection 

import easyocr

detection_threshold = 0.7

image = image_np_with_detections
scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
boxes = detections ['detection_boxes'][:len(scores)]
classes = detections['detection_classes'][:len(scores)]

width = image.shape[1]
height = image.shape[0]

#Apply ROI filtering and OCR
for idx, box in enumerate(boxes):
    print(box)
    roi = box*[height, width, height, width]
    print(roi)
    region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
    
    reader = easyocr.Reader(['en'])
    ocr_result = reader.readtext(region)
    print(ocr_result)
    
    plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))


# # 6. OCR Filtering
region_threshold = 0.06

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
             
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate


filter_text(region, ocr_result, region_threshold)



# 7. Bring 5&6 together
region_threshold = 0.05

def ocr_it(image, detections, detection_threshold, region_threshold):
    
    #scores boxes and classes above threshold
    scores = list(filter(lambda x: x> detection_threshold, detections["detection_scores"]))
    boxes = detections ["detection_boxes"][:len(scores)]
    classes = detections["detection_classes"][:len(scores)]
    
    #full image dimensions
    width = image.shape[1]
    height = image.shape[0]
    
    # Apply ROI filtering and OCR
    for idx, box in enumerate (boxes) :
        roi = box*[height, width, height, width]
        region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]

        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(region)
        #print(ocr_result) 
        
        text = filter_text(region, ocr_result, region_threshold)
        
        plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        print(text)
        return text, region
    
    # Return default values if no text is recognized
    return "No text"
    
text, region = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)


# # 8. Save Result

import csv
import uuid

"{}.jpg".format(uuid.uuid1())

def save_results(text, region, csv_filename, folder_path):
    img_name= "{}.jpg".format(uuid.uuid1())
    
    cv2.imwrite(os.path.join(folder_path, img_name), region)
    
    with open(csv_filename, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name, text])

save_results(text, region, "detection_results.csv", "Detection_Images")








# # 9. Real Time Detections from Webcam 

import cv2
import tensorflow as tf
import numpy as np
from object_detection.utils import visualization_utils as viz_utils
import tkinter as tk
from tkinter import END
from PIL import Image, ImageDraw, ImageTk

def capIsOpen():
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

    while cap.isOpened(): 
        ret, frame = cap.read()
        image_np = np.array(frame)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.8,
                    agnostic_mode=False)
        

        try:
            text, region = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)
            save_results(text, region, "realtimeresults.csv", "Detection_Images")
            
            # insert the output text into the first Entry widget
            entry_1.delete(0, END)  # clear any previous text
            entry_1.insert(END, text)
            
            # create a PIL image object from the Numpy array
            region_img = Image.fromarray(region)
            
            # create a PIL image object from the Numpy array and resize it
            #region_img = Image.fromarray(region).resize((2 * region.shape[1], 2 * region.shape[0]))

            # create a Draw object to draw text on the image
            draw = ImageDraw.Draw(region_img)

            # draw the text on the image
            draw.text((10, 10), text)

            # convert the PIL image to a Tkinter-compatible image object
            region_img_tk = ImageTk.PhotoImage(region_img)

            # update the label with the new image
            label.config(image=region_img_tk)

            # keep a reference to the image to prevent it from being garbage collected
            label.image = region_img_tk

        except:
            pass

        cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

            

# # 10. Create GUI-TKINTER

# In[14]:

import os
from pathlib import Path
from tkinter import *
from PIL import Image, ImageDraw, ImageTk
from tkinter import filedialog
from tkinter.font import Font
from tkinter import ttk
from tkinter import Tk, ttk, END
from datetime import datetime
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


OUTPUT_PATH = os.path.dirname(os.path.realpath('__file__'))
ASSETS_PATH = OUTPUT_PATH / Path(r"S:\Indonesia\semester 5\CV\skripsi\GUI\build\assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def detect_and_display(frame):
    
    # Placeholder for your actual number plate detection logic
    
    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the RGB frame to a Tkinter-compatible PhotoImage
    img_tk = ImageTk.PhotoImage(Image.fromarray(rgb_frame))

    # Update the label with the new image
    label.config(image=img_tk)
    label.image = img_tk

    # Perform object detection on the frame (replace this with your actual detection logic)
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Visualize bounding boxes and labels on the image
    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        detections['detection_boxes'],
        detections['detection_classes'] + 1,  # Add 1 to avoid the background class
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=0.8,
        agnostic_mode=False
    )

    try:
        # Perform OCR on the detected regions
        text, region = ocr_it(frame, detections, detection_threshold, region_threshold)

        # Save results (optional)
        save_results(text, region, "realtimeresults.csv", "Detection_Images")

        # Insert the detected number plate text into entry_1 (if needed)
        entry_1.delete(0, END)
        entry_1.insert(END, text)

        # Convert the region to RGB format for display
        rgb_region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

        # Convert the RGB region to a Tkinter-compatible PhotoImage
        region_img_tk = ImageTk.PhotoImage(Image.fromarray(rgb_region))

        # Update the label with the new region image
        label.config(image=region_img_tk)
        label.image = region_img_tk


        display_result(text, region_img_tk)

    except Exception as e:
        print(f"Error processing frame: {e}")

    # Check for the 'q' key to quit the camera feed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        # Optionally, you can release the video capture here if it's not needed continuously
        # cap.release()


def detect(frame):
    # Placeholder for your actual number plate detection logic
    
    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection on the frame (replace this with your actual detection logic)
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Visualize bounding boxes and labels on the image
    viz_utils.visualize_boxes_and_labels_on_image_array(
        rgb_frame,
        detections['detection_boxes'],
        detections['detection_classes'] + 1,  # Add 1 to avoid the background class
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=0.8,
        agnostic_mode=False
    )

    # Convert the RGB frame to a Tkinter-compatible PhotoImage
    img_tk = ImageTk.PhotoImage(Image.fromarray(rgb_frame))

    # Update the label with the new image
    label.config(image=img_tk)
    label.image = img_tk

    try:
        # Perform OCR on the detected regions
        text, region = ocr_it(frame, detections, detection_threshold, region_threshold)
        resized_license_plate_crop = cv2.resize(region, (220, 60),
                                                    interpolation=cv2.INTER_LINEAR)
        # Save results (optional)
        save_results(text, resized_license_plate_crop, "realtimeresults.csv", "Detection_Images")

        # Insert the detected number plate text into entry_1 (if needed)
        entry_1.delete(0, END)
        entry_1.insert(END, text)

        # Convert the region to RGB format for display
        rgb_region = cv2.cvtColor(resized_license_plate_crop, cv2.COLOR_BGR2RGB)

        # Convert the RGB region to a Tkinter-compatible PhotoImage
        region_img_tk = ImageTk.PhotoImage(Image.fromarray(rgb_region))

        # # Update the label with the new region image
        # label.config(image=region_img_tk)
        # label.image = region_img_tk

        display_result(text, region_img_tk)

    except Exception as e:
        print(f"Error processing frame: {e}")

    # Check for the 'q' key to quit the camera feed
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()




# Function to update the label with a new image
def update_label_with_image(image_np):
    # Resize the frame to a smaller size (e.g., 640x480)
    #   resized_frame = cv2.resize(image_np, (640, 480))

    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # Convert the RGB frame to a Tkinter-compatible PhotoImage
    img_tk = ImageTk.PhotoImage(Image.fromarray(rgb_frame))

    # Update the label with the new image
    label.config(image=img_tk)
    label.image = img_tk

    
def resize_image(image, max_size):
    # Get the dimensions of the image
    height, width, _ = image.shape
    
    # Determine the maximum dimension
    max_dimension = max(height, width)
    
    # Calculate the scaling factor
    scale = max_size / max_dimension
    
    # Resize the image
    resized_image = cv2.resize(image, (int(width * scale), int(height * scale)))
    
    return resized_image


def upload_photo():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        image = cv2.imread(file_path)
        # Perform object detection on the resized image
        detect(image)
        # Resize the image for display with a maximum size of 640 pixels
        resized_image = resize_image(image, 640)
        # Update the label with the resized image
        update_label_with_image(resized_image)


cap = None

def stop_current_process():
    global cap, stop_camera
    if cap is not None and cap.isOpened():
        cap.release()
    stop_camera = True


def upload_video():
    global cap, stop_camera  # Access the global cap and stop_camera variables
    stop_camera = False  # Reset the stop_camera flag
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    print("Selected file:", file_path)  # Debugging: Print selected file path
    if file_path:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("Error: Unable to open video file")
            return

        # Define the desired display size
        display_width = 700
        display_height = 450

        while cap.isOpened() and not stop_camera:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize the frame to the desired display size
            resized_frame = cv2.resize(frame, (display_width, display_height))

            # Perform detection on the resized frame
            detect(resized_frame)
            
            # Update the window to display the new image
            window.update()  

            # Move the waitKey outside the loop
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()







# Declare a global variable for stopping the camera
stop_camera = False

def open_camera():
    stop_current_process()
    global stop_camera, cap

    def process_frame():
        global stop_camera, cap
        ret, frame = cap.read()
        
        if ret and window.winfo_exists():
            detect(frame)
            window.update()
            if not stop_camera:
                window.after(10, process_frame)  # Schedule the next frame processing

    cap = cv2.VideoCapture(0)
    # Reset the stop_camera flag
    stop_camera = False
    # Schedule the first frame processing
    window.after(10, process_frame)


def extract_number_plate():
    global stop_camera
    if label.image:
        # Stop the camera
        stop_camera = True

        image_np = np.array(label.image)
        text, region = ocr_it(image_np, detections, detection_threshold, region_threshold)
        
        # Display the extracted number plate image on the label
        region_img = Image.fromarray(region)
        region_img_tk = ImageTk.PhotoImage(region_img)
        label.config(image=region_img_tk)
        label.image = region_img_tk
        
        # Insert the detected number plate text into entry_1
        entry_1.delete(0, END)
        entry_1.insert(END, text)





def quit_window():
    for child in window.winfo_children():
        child.destroy()
    window.quit()

from tkinter import Tk, ttk, Label, PhotoImage
from datetime import datetime
from PIL import Image, ImageTk
import cv2
import numpy as np
from tkinter import PhotoImage
from PIL import Image, ImageTk




# Create the main GUI window
window = tk.Tk()
window.geometry("1700x823")  # Adjust the width as needed
window.configure(bg="#FFFFFF")
window.title("Result Display")

# Create a frame for the application on the left side
app_frame = ttk.Frame(window, width=650, height=820)  # Adjust the width and height as needed
app_frame.grid(row=0, column=0, padx=200, pady=10, sticky="nsew")

style = ttk.Style()

# Set the background color for the Treeview
style.configure('Treeview', rowheight=70, background='#F0F0F0')  # Adjust the background color as needed


result_tree = ttk.Treeview(window, columns=("Image", "Text", "Time"))

result_tree.heading("#0", text="License Plate Image", anchor=tk.CENTER)
result_tree.heading("#1", text="License Plate Text", anchor=tk.CENTER)
result_tree.heading("#2", text="Time of Detection", anchor=tk.CENTER)
result_tree.heading("#3", text="", anchor=tk.CENTER)  # Hide the default column


result_tree.column("#0", width=300, anchor=tk.CENTER)
result_tree.column("#1", width=180, anchor=tk.CENTER)
result_tree.column("#2", width=150, anchor=tk.CENTER)
result_tree.column("#3", width=0, stretch=tk.NO)  # Hide the default column


# Place the Treeview in the window
result_tree.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# Create a vertical scrollbar
scrollbar = Scrollbar(window, orient=VERTICAL, command=result_tree.yview)
scrollbar.grid(row=0, column=2, padx=10, pady=10, sticky="ns")

# Configure the Treeview to use the scrollbar
result_tree.configure(yscrollcommand=scrollbar.set)

# Counter for result IDs
result_id_counter = 1

# Global dictionary to store references to PhotoImage objects
photoimage_references = {}

class CustomPhotoImage:
    def __init__(self, image=None):
        if image:
            self._pil_image = Image.open(image)
            self._resize_image()
        else:
            self._pil_image = None
            self._photo_image = PhotoImage()

    def _resize_image(self):
        if self._pil_image:
            # Resize the image while maintaining aspect ratio
            width, height = self._pil_image.size
            new_width = 250
            new_height = int((new_width / width) * height)
            resized_img_pil = self._pil_image.resize((new_width, new_height), Image.ANTIALIAS)
            self._photo_image = ImageTk.PhotoImage(resized_img_pil)

    def get_photo_image(self):
        return self._photo_image

    def resize(self, new_size):
        if self._pil_image:
            # Resize the image to the specified new size
            resized_img_pil = self._pil_image.resize(new_size, Image.ANTIALIAS)
            self._photo_image = ImageTk.PhotoImage(resized_img_pil)
            return self._photo_image
        else:
            return self._photo_image  # No resizing for empty PhotoImage
def display_result(text, region_img_tk):
    global result_id_counter

    # Get the current time for the time of detection
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create a CustomPhotoImage instance
    original_img = CustomPhotoImage()

    # Set the existing PhotoImage to the CustomPhotoImage instance
    original_img._photo_image = region_img_tk

    # Resize the image within the CustomPhotoImage instance
    resized_img_tk = original_img.resize((250, 50))  # Fixed size (width, height)

    # Ensure that the resized image is retained as a reference
    photoimage_references[result_id_counter] = resized_img_tk

    # Create a new row with fixed height at the beginning of the Treeview
    result_tree.insert(parent="",
                       index=0,  # Insert at the beginning
                       iid=str(result_id_counter),
                       values=(text, current_time),
                       image=resized_img_tk,
                       tags=str(result_id_counter))  # Set dynamic row height

    # Increment the result ID counter
    result_id_counter += 1


# Create a label to display the image
label = Label(window, bg="#F1F5FF")
label.place(x=300, y=150)

# Add a button to trigger extraction
button_extract = Button(
    text="Extract",
    borderwidth=0,
    highlightthickness=0,
    command=extract_number_plate,
    relief="flat"
)
button_extract.place(x=594.0, y=640.0, width=126.0, height=61.0)


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 823,
    width = 1047,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    13.0,
    0.0,
    277.0,
    823.0,
    fill="#FFFFFF",
    outline="")

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_upload_image = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=upload_photo,
    relief="flat"
)
button_upload_image.place(
    x=38.0,
    y=117.0,
    width=180.0,
    height=61.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_upload_vid = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=upload_video,
    relief="flat"
)
button_upload_vid.place(
    x=38.0,
    y=214.0,
    width=180.0,
    height=61.0
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_from_camera = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=open_camera,
    relief="flat"
)
button_from_camera.place(
    x=38.0,
    y=311.0,
    width=180.0,
    height=61.0
)

button_image_web_cam = PhotoImage(
    file=relative_to_assets("web1.png")
)

button_web_cam = Button(
    image=button_image_web_cam,
    borderwidth=0,
    highlightthickness=0,
    command=capIsOpen,  
    relief="flat"
)

button_web_cam.place(
    x=38.0,
    y=415.0
   
)

button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_extract = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=extract_number_plate,
    relief="flat"
)
#button_4.bind("<Button-1>", handle_q_press) # bind button press event to handle_q_press function
button_extract.place(
    x=594.0,
    y=640.0,
    width=126.0,
    height=61.0
)

button_image_5 = PhotoImage(
    file=relative_to_assets("button_5.png"))
button_quit = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=quit_window,
    relief="flat"
)
button_quit.place(
    x=920.0,
    y=728.0,
    width=82.0,
    height=61.0
)

button_image_6 = PhotoImage(
    file=relative_to_assets("button_6.png"))
button_refresh = Button(
    image=button_image_6,
    borderwidth=0,
    highlightthickness=0,
    command=capIsOpen,
    relief="flat"
)
button_refresh.place(
    x=62.0,
    y=736.0,
    width=127.0,
    height=46.0
)



entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    657.0,
    760.5,
    image=entry_image_1
)

# create a placeholder image with a solid color
img = Image.new('RGB', (670, 400), color='#F1F5FF')
img_tk = ImageTk.PhotoImage(img)

# create a label to display the image
label = Label(window, image=img_tk)
label.place(x=300, y=150)

#output1 = text
entry_1 = Entry(
    bd=0,
    bg="#F1F5FF",
    fg="#000716",
    highlightthickness=0,
    
)
# create a custom font with a larger size
font = Font(size=25)

# set the custom font as the font for the Entry widget
entry_1.config(font=font)
entry_1.place(
    x=528.0,
    y=729.0,
    width=258.0,
    height=61.0
)
#entry_1.insert(END, output1)

canvas.create_text(
    21.0,
    23.0,
    anchor="nw",
    text="Automatic",
    fill="#000000",
    font=("Arial BoldMT", 45 * -1)
)

canvas.create_text(
    297.0,
    23.0,
    anchor="nw",
    text="Number Plate Recognition ANPR",
    fill="#000000",
    font=("Arial BoldMT", 45 * -1)
)

canvas.create_text(
    291.0,
    741.0,
    anchor="nw",
    text="Number Plate :",
    fill="#000000",
    font=("Arial BoldMT", 30 * -1)
)

window.resizable(False, False)
window.mainloop()





