import cv2
import numpy as np
import os

background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350


def cal_accum_avg(frame, accumulated_weight):

    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment_hand(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Grab the external contours for the image
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        return (thresholded, hand_segment_max_cont)


cam = cv2.VideoCapture(0)

# Set windows to stay on top
cv2.namedWindow('Sign Detection', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Sign Detection', cv2.WND_PROP_TOPMOST, 1)
cv2.namedWindow('Thresholded Hand Image', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Thresholded Hand Image', cv2.WND_PROP_TOPMOST, 1)

num_frames = 0
num_imgs_taken = 0

# Create dynamic paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set dataset paths dynamically
train_path = os.path.join(BASE_DIR, 'gesture', 'train')
test_path = os.path.join(BASE_DIR, 'gesture', 'test')

# Ensure directories exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Get input for what to capture
print("Enter what to capture:")
print("1. Numbers (0-9)")
print("2. Alphabets (A-Z)")
choice = input("Enter choice (1 or 2): ")

if choice == '1':
    elements = range(10)  # 0 to 9
else:
    elements = [chr(i) for i in range(65, 91)]  # A to Z

for element in elements:
    num_frames = 0
    num_imgs_taken = 0
    
    # Create directory for current element
    os.makedirs(os.path.join(train_path, str(element)), exist_ok=True)
    
    print(f"Capturing gestures for {element}")
    
    while True:
        ret, frame = cam.read()

        # filpping the frame to prevent inverted image of captured frame...
        frame = cv2.flip(frame, 1)

        frame_copy = frame.copy()

        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

        if num_frames < 60:
            cal_accum_avg(gray_frame, accumulated_weight)
            if num_frames <= 59:
                cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                
        elif num_frames <= 300: 
            hand = segment_hand(gray_frame)
            
            cv2.putText(frame_copy, "Adjust hand...Gesture for " + str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            if hand is not None:
                thresholded, hand_segment = hand
                cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0),1)
                cv2.putText(frame_copy, str(num_frames)+"For " + str(element), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("Thresholded Hand Image", thresholded)
        
        else: 
            hand = segment_hand(gray_frame)
            
            if hand is not None:
                thresholded, hand_segment = hand
                cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0),1)
                
                cv2.putText(frame_copy, str(num_frames), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(frame_copy, str(num_imgs_taken) + ' images ' +"For " + str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                cv2.imshow("Thresholded Hand Image", thresholded)
                if num_imgs_taken <= 300:
                    save_path = os.path.join(train_path, str(element), f"{num_imgs_taken+300}.jpg")
                    cv2.imwrite(save_path, thresholded)
                else:
                    break
                num_imgs_taken +=1
            else:
                cv2.putText(frame_copy, 'No hand detected...', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)
        
        cv2.putText(frame_copy, "DataFlair hand sign recognition_ _ _", (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
        
        num_frames += 1

        cv2.imshow("Sign Detection", frame_copy)

        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break
    
    # Ask if user wants to continue to next element
    if k != 27:  # If not escaped
        input(f"Press Enter to continue to next gesture or Ctrl+C to exit...")

cv2.destroyAllWindows()
cam.release()