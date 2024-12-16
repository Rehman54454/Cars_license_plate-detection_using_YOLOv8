**Cars_license_plate_detection_using_YOLOv8**
**Demo:**
![3](https://github.com/user-attachments/assets/6c2bb7bd-129a-46c7-b8fe-013fbb3a2498)

**Description:**
This project demonstrates real-time license plate detection using a YOLOv8n model trained on a custom dataset. The detected license plates are then processed using Tesseract OCR to extract the license plate text. The system is capable of handling images and marking detected license plates with bounding boxes while displaying the extracted text.

**YOLOv8n Model:**
The YOLOv8n model is a lightweight and efficient object detection model, ideal for tasks with resource constraints. In this project, it was fine-tuned on a custom dataset to accurately detect license plates in various conditions.

**OpenCV:**
OpenCV is an open-source computer vision library that allows easy image processing and manipulation. It was used in this project for reading, resizing, and displaying images, as well as for drawing bounding boxes and text annotations.

**Tesseract OCR:**
Tesseract OCR is an optical character recognition engine used to extract text from images. In this project, it reads and processes cropped regions of license plates to extract the alphanumeric text.
You can download it from [here](https://github.com/tesseract-ocr/tesseract).

**Install YOLOv8 and dependencies:**
pip install ultralytics opencv-python pytesseract
Ensure Tesseract-OCR is installed:
Download and install Tesseract-OCR from here.
Update the pytesseract.pytesseract.tesseract_cmd path in the code to the location of the Tesseract executable on your machine.

**Model:**
Download trained model trainable weights to load for real time testing from [here](https://github.com/tesseract-ocr/tesseract).

**Datasets**
You can download and check dataset from [here](https://github.com/tesseract-ocr/tesseract).

**How to Run:**
First download the Requirement.txt from [here](https://github.com/Rehman54454/Cars_license_plate-detection_using_YOLOv8/blob/main/Requirements.txt).
Replace the img path with your input image file in main.py file code.
Replace the YOLO model path (license_plate_detector.pt) with your trained model file in min.py code.
Execute the script. Detected license plates and their extracted text will appear on the displayed image.

**Example Output:**
Detected License Plate: The system draws bounding boxes around license plates.
Extracted Text: Text is displayed on the top-left of the image.
