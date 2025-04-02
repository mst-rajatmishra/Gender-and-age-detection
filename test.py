import cv2
import argparse

# Global constants for the model paths and mean values
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
PADDING = 20

# Function to load the models
def load_models():
    try:
        face_net = cv2.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
        age_net = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
        gender_net = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")
        return face_net, age_net, gender_net
    except Exception as e:
        print(f"Error loading models: {e}")
        exit()

# Function to highlight the faces in the frame
def highlight_face(net, frame, conf_threshold=0.7):
    frame_height, frame_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * [frame_width, frame_height, frame_width, frame_height]).astype(int)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)
    return frame, face_boxes

# Function to predict the gender and age for each face
def predict_age_gender(face, age_net, gender_net):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    
    # Predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    
    # Predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]
    
    return gender, age

# Function to display results and annotate them on the frame
def display_results(result_img, gender, age, face_box):
    x1, y1, x2, y2 = face_box
    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f'{gender}, {age} years'
    
    # Adding a background for text for better readability
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(result_img, (x1, y1 - 30), (x1 + w, y1), (0, 0, 0), -1)
    cv2.putText(result_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

# Function to handle mouse click events
def click_and_save(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button clicked
        print("Image saved!")
        cv2.imwrite('detected_image.jpg', param)  # Save the frame with the annotations

# Main function to run the age and gender detection
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help="Path to input image or video file")
    args = parser.parse_args()

    # Load the models
    face_net, age_net, gender_net = load_models()

    # Initialize video capture
    video = cv2.VideoCapture(args.image if args.image else 0)
    if not video.isOpened():
        print("Error: Could not open video source.")
        exit()

    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to capture image")
            break
        
        # Detect faces in the frame
        result_img, face_boxes = highlight_face(face_net, frame)

        if not face_boxes:
            print("No face detected")
            continue

        for face_box in face_boxes:
            # Extract the face region
            x1, y1, x2, y2 = face_box
            face = frame[max(0, y1 - PADDING):min(y2 + PADDING, frame.shape[0]), 
                         max(0, x1 - PADDING):min(x2 + PADDING, frame.shape[1])]
            
            # Predict age and gender
            gender, age = predict_age_gender(face, age_net, gender_net)

            # Annotate the results
            display_results(result_img, gender, age, face_box)

        # Display the resulting image
        cv2.imshow("Age and Gender Detection", result_img)

        # Set the callback for mouse click event to save the image
        cv2.setMouseCallback("Age and Gender Detection", click_and_save, result_img)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
