import cv2
import serial
import time
from ultralytics import YOLO

# Arduino serial communication setup
ser_arduino = serial.Serial('COM3', 115200)

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, adjust if you have multiple cameras

model_path = "C:\\Users\\y4sef\\Downloads\\pt28\\train28\\weights\\best.pt"

# Egitilen modelin kullanilmasi
model = YOLO(model_path)


hue_lower_apple = 10  
hue_upper_apple = 40  

hue_lower_shirt = 90  
hue_upper_shirt = 130  

hue_lower_green = 45   
hue_upper_green = 65   

hue_lower_yellow = 20  
hue_upper_yellow = 30  

threshold = 0.01  

# Define size thresholds
min_size_threshold = 5 
max_size_threshold = 16000  

max_confidence = 0  
coordinates_to_send = None  

# Safety line parameters
safety_line_y_min = 140
safety_line_y_max = 400

while True:
    # Capture frame-by-frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define masks for apple, shirt, green, and yellow colors
    mask_apple = cv2.inRange(hsv_frame, (hue_lower_apple, 50, 50), (hue_upper_apple, 255, 255))
    mask_shirt = cv2.inRange(hsv_frame, (hue_lower_shirt, 50, 50), (hue_upper_shirt, 255, 255))
    mask_green = cv2.inRange(hsv_frame, (hue_lower_green, 50, 50), (hue_upper_green, 255, 255))
    mask_yellow = cv2.inRange(hsv_frame, (hue_lower_yellow, 50, 50), (hue_upper_yellow, 255, 255))

    # Combined mask for apple, shirt, green, and yellow colors
    combined_mask = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(mask_apple, mask_green), mask_shirt), mask_yellow)

    # Apply the combined mask to the frame
    filtered_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # Perform inference using the loaded model on the filtered frame
    results = model(filtered_frame)[0]

    # Reset values for each iteration
    max_confidence = 0
    coordinates_to_send = None

    # Detect only one object at a time (highest confidence)
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        box_size = (x2 - x1) * (y2 - y1)

        if score > threshold and min_size_threshold < box_size < max_size_threshold:
            # Calculate the center coordinates
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
        
            # Boost confidence for green color
            if hue_lower_green <= (hue_lower_apple + hue_upper_apple) / 2 <= hue_upper_green:
                score += 0.2
            
            # Boost confidence for yellow color
            if hue_lower_yellow <= (hue_lower_apple + hue_upper_apple) / 2 <= hue_upper_yellow:
                score += 0.5

            # Update the maximum confidence and coordinates
            if score > max_confidence:
                max_confidence = score
                coordinates_to_send = (center_x, center_y)
    
        
    # This function converts 1920x1080 pixel to 0 - 180 degrees
    def map_value(value, from_low, from_high, to_low, to_high):
        return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low
    
    # If there are coordinates to send, send them to the Arduino
    if coordinates_to_send is not None:
        data_to_send = f"{int(coordinates_to_send[0])},{int(coordinates_to_send[1])}\n"
        # Show the coordinates (x,y) that you will send to arduino
        # X --> from 1920 to 0 - 180 range "mapped_center_x"
        # Y --> from 1080 to 0 - 180 range "mapped_center_y"
        mapped_center_x = map_value(center_x, 0, 640, 0, 180)
        mapped_center_y = map_value(center_y, 0, 480, 0, 180)
        print("New Coordinate Values")
        #print("X= ", mapped_center_x, " angle", " Y= ", mapped_center_y, " angle") #Orijinal X ve Y koordinatları (Float cinsinden)
        print(f"X= {int(mapped_center_x):} angle, Y= {int(mapped_center_y):} angle") # X ve Y koordinatları (.2f cinsinden)
        #print(f"X= {int(mapped_center_x)} angle, Y= {int(mapped_center_y)} angle") # X ve Y koordinatları (Integer cinsinden)
        #Terminalden X ve Y degerlerini 0.2 saniye araliklar ile gormek icin time.sleep(0.2) kullandik
        ser_arduino.write(data_to_send.encode())
    """   
     # Draw safety line (red rectangle)
    cv2.rectangle(frame, (0, safety_line_y_min), (frame.shape[1], safety_line_y_max), (0, 0, 255), 2)

    # Add text next to the safety line
    safety_text_position = (10, (safety_line_y_min + safety_line_y_max) // 2)
    cv2.putText(frame, "Safety Area", safety_text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    """
    
    # Draw square around the detected object and display coordinates and confidence
    if coordinates_to_send is not None:
        x, y = coordinates_to_send
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        label = f"Coordinates: ({int(x)}, {int(y)}) | Confidence: {max_confidence:.2f}"
        #X ve Y piksel degerlerinin yazdirilmasi
        print(f"X coordinate: {int(x)}, Y coordinate: {int(y)}")
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
ser_arduino.close()
cv2.destroyAllWindows()