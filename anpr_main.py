# # 1. Setup Paths
import os

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


# Restore checkpoint with the desired path

checkpoint_path = 'S:\\Indonesia\\skripsi\\anpr\\Tensorflow\\workspace\\models\\my_ssd_mobnet\\model_dir\\ckpt-76' 

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(checkpoint_path).expect_partial()




@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# 8. Save Result
import os
import csv
import cv2
import uuid
from datetime import datetime


def save_results11(license_plate_text, resized_license_plate_crop, csv_filename, folder_path,real_time):
    # Generate a unique image name
    img_name = "{}.jpg".format(uuid.uuid1())
    
    # Save the license plate crop as an image
    cv2.imwrite(os.path.join(folder_path, img_name), resized_license_plate_crop)
    
    # Get the current time in the desired format
    real_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Append the results to the CSV file
    with open(csv_filename, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name, license_plate_text, real_time])

            
# 10. Setup for  GUI-TKINTER

import os
from pathlib import Path
from tkinter import *
from PIL import Image, ImageDraw, ImageTk
from tkinter import filedialog
from tkinter.font import Font
from tkinter import ttk
import tkinter as tk
from tkinter import Tk, ttk, END
from datetime import datetime
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


OUTPUT_PATH = os.path.dirname(os.path.realpath('__file__'))
ASSETS_PATH = OUTPUT_PATH / Path(r"S:\Indonesia\skripsi\ANPR_TensorFlow\GUI")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


from sort.sort import *   
import util_11  
import cv2
from util_11 import get_car,write_csv
from ultralytics import YOLO

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
vehicles = [2, 3, 5, 7]

def get_car(license_plate, vehicle_track_ids):
    
    x1, y1, x2, y2= license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1

import easyocr

reader = easyocr.Reader(['en'], gpu=True)
def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.readtext(license_plate_crop)

    width = img.shape[1]
    height = img.shape[0]
    
    if detections == []:
        return None, None

    rectangle_size = license_plate_crop.shape[0] * license_plate_crop.shape[1]

    plate = [] 

    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length * height / rectangle_size > 0.17:
            bbox, text, score = result
            text = result[1]
            text = text.upper()
            # text = format_license(text)  
            scores += score
            plate.append(text)
    
    if len(plate) != 0: 
        return " ".join(plate), scores / len(plate)
    else:
        return " ".join(plate), 0
    

results = {}


previous_results = {}  # Initialize previous results
best_scores = {}  # Initialize best scores for each vehicle
best_texts = {}  # Initialize best texts for each vehicle
best_license_plate_crops = {}  # Initialize best license plate crops for each vehicle

def detect_license_plate(frame, reader, coco_model, detect_fn, vehicles):
    global previous_results, best_scores, best_texts, best_license_plate_crops

    results = {}

    # Detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # Track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))

    # Detect license plates
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    for box_idx in range(len(detections['detection_boxes'])):
        box = detections['detection_boxes'][box_idx]
        class_id = detections['detection_classes'][box_idx]
        score = detections['detection_scores'][box_idx]

        # Extract box coordinates
        y1, x1, y2, x2 = box
        y1 = int(y1 * frame.shape[0])
        x1 = int(x1 * frame.shape[1])
        y2= int(y2 * frame.shape[0])
        x2 = int(x2 * frame.shape[1])

        # Assign license plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car([y1, x1, y2, x2], track_ids)
        cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), thickness=2)

        if car_id != -1:
            # Crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
            
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_thresh = cv2.threshold(license_plate_crop_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            
            # Update detections dictionary with license plate bounding box information
            detections['detection_boxes'] = np.concatenate((detections['detection_boxes'], [[y1/frame.shape[0], x1/frame.shape[1], y2/frame.shape[0], x2/frame.shape[1]]]), axis=0)
            detections['detection_classes'] = np.concatenate((detections['detection_classes'], [class_id]), axis=0)
            detections['detection_scores'] = np.concatenate((detections['detection_scores'], [score]), axis=0)

            # Read license plate
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_thresh, frame)

            if license_plate_text is not None:
                weighted_score = 0.6 * score + 0.4 * license_plate_text_score
                if car_id not in previous_results or weighted_score > previous_results[car_id]['weighted_score']:
                    previous_results[car_id] = {
                        'license_plate_text': license_plate_text,
                        'license_plate_text_score': license_plate_text_score,
                        'weighted_score': weighted_score,
                        'car_bbox': [xcar1, ycar1, xcar2, ycar2]
                    }
                    best_scores[car_id] = weighted_score
                    best_texts[car_id] = license_plate_text
                    best_license_plate_crops[car_id] = license_plate_crop

                    results[car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {'bbox': [x1, y1, x2, y2],
                                        'text': license_plate_text,
                                        'bbox_score': score,
                                        'text_score': license_plate_text_score}
                }
                    

    # Display the best result if available
    best_car_id = None
    if results:
        best_car_id = max(results, key=lambda x: results[x]['license_plate']['text_score'])
        if best_car_id:
            resized_license_plate_crop = cv2.resize(best_license_plate_crops[best_car_id], (220, 60),
                                                    interpolation=cv2.INTER_LINEAR)
            rgb_region = cv2.cvtColor(resized_license_plate_crop, cv2.COLOR_BGR2RGB)
            img_tk_region = ImageTk.PhotoImage(Image.fromarray(rgb_region))
            display_result(best_texts[best_car_id], img_tk_region)
            cv2.rectangle(frame, (results[best_car_id]['license_plate']['bbox'][0],
                                  results[best_car_id]['license_plate']['bbox'][1]),
                          (results[best_car_id]['license_plate']['bbox'][2],
                           results[best_car_id]['license_plate']['bbox'][3]), (0, 255, 0), 2)

            best_license_plate_crop = best_license_plate_crops[best_car_id]
            best_license_plate_crops_list = [best_license_plate_crop]

            save_results11([best_texts[best_car_id]], resized_license_plate_crop, "detection_results.csv",
                        "Detection_Images", real_time="YYYY-MM-DD HH:MM:SS")
            write_csv({best_car_id: results[best_car_id]}, "./detection.csv")
    else:
        best_license_plate_crops_list = []

    img_wth_box = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return [img_wth_box, [best_texts.get(best_car_id, "")], best_license_plate_crops_list]





# Function to update the label with a new image
def update_label_with_image(image_np):
    # Resize the frame to a smaller size (e.g., 640x480)
    resized_frame = cv2.resize(image_np, (640, 480))

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
        display_height = 394

        # display_width = 1300
        # display_height = 731
        


        while cap.isOpened() and not stop_camera:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize the frame to the desired display size
            resized_frame = cv2.resize(frame, (display_width, display_height))

            # Perform detection on the resized frame
            try:
                processed_image, licenses_texts, license_plate_crops_total = detect_license_plate(resized_frame, reader, coco_model, detect_fn, vehicles)
            except ValueError as e:
                # Handle the case where not enough values are unpacked
                processed_image = resized_frame  # Assign resized_frame directly
                licenses_texts = None
                license_plate_crops_total = None
            frame = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            # Update the label with the resized image
            update_label_with_image(frame)
            print(licenses_texts, license_plate_crops_total)
            # Update the window to display the new image
            window.update() 

        cap.release()
        cv2.destroyAllWindows()

# Declare a global variable for stopping the camera
stop_camera = False


def quit_window():
    for child in window.winfo_children():
        child.destroy()
    window.quit()




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
def display_result(license_plate_text, region_img_tk):
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
                       values=(license_plate_text, current_time),
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
    command='',
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
    command='',
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
    command='',
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
    command='',  
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
    command='',
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
    command='',
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





