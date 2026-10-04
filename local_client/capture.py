import cv2
import requests
import time
from dispatcher import ActionDispatcher

# Cloud API endpoint (Set to localhost for now, will update to AWS ELB URL later)
API_URL = "http://localhost:8000/predict"

def main():
    # Initialize the webcam (0 is the default built-in or USB camera)
    cap = cv2.VideoCapture(0)
    
    # Initialize the dispatcher mapping gestures to OS commands (cooldown prevents spamming)
    dispatcher = ActionDispatcher(cooldown=1.2)
    
    print("Initializing Heads Up Display. Press 'q' to terminate.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from hardware.")
            break
            
        # Flip the frame horizontally for a natural mirror-like view
        frame = cv2.flip(frame, 1)
        
        # Compress the frame to JPEG to drastically reduce upload payload size
        # Quality 80 balances inference accuracy with network speed
        _, encoded_image = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        
        # Prepare the multipart/form-data payload for FastAPI
        files = {'file': ('frame.jpg', encoded_image.tobytes(), 'image/jpeg')}
        
        try:
            # Send frame to the cloud backend.
            # strict 0.25s timeout ensures the UI doesn't freeze if the network stutters.
            response = requests.post(API_URL, files=files, timeout=0.25)
            
            if response.status_code == 200:
                data = response.json()
                gesture = data.get("gesture")
                confidence = data.get("confidence", 0.0)
                
                # If the neural network is highly confident, execute the shortcut
                if confidence > 0.85:
                    dispatcher.execute(gesture)
                    
                # Overlay the active prediction on the video feed
                cv2.putText(frame, f"Detected: {gesture} ({confidence:.2f})", 
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                            
        except requests.exceptions.RequestException:
            # Display a warning without interrupting the video loop if the API drops
            cv2.putText(frame, "API Offline / Reconnecting...", 
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Display the live feed window
        cv2.imshow("Local Gesture Client", frame)
        
        # Listen for the 'q' key to cleanly exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release hardware resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()