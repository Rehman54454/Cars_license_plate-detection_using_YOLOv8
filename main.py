import cv2
from ultralytics import YOLO
import pytesseract

# Ensure Tesseract-OCR is installed and set up correctly
# Change the path below to the installed tesseract executable if necessary
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
img = cv2.imread(r"C:\Users\3s\Downloads\15.jfif")
#print(img)

# Resize the image
img = cv2.resize(img, (640, 640))

# Load the YOLO model
model = YOLO(r"C:\Users\3s\Downloads\license_plate_detector.pt")

# Run inference
results = model(img)

# Extract bounding boxes
for box in results[0].boxes.xyxy:  # Iterate over detected bounding boxes
    x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
    plate_roi = img[y1:y2, x1:x2]  # Crop the license plate area

    # Perform OCR to read the text
    plate_text = pytesseract.image_to_string(plate_roi, config='--psm 7')  # '--psm 7' is for single line text
    plate_text = plate_text.strip()  # Clean up the text

    # Print the extracted text to the console
    print(f"Detected License Plate Text: {plate_text}")

    # Draw a white rectangle at the top-left corner
    cv2.rectangle(img, (10, 10), (300, 60), (255, 255, 255), -1)  # White filled rectangle
    # Write the text in black on the rectangle
    cv2.putText(img, plate_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Draw the bounding boxes on the image
detected_image = results[0].plot()

# Display the resulting image
cv2.imshow("Detected License Plate", detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
