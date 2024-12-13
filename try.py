import cv2
from tkinter import *
from PIL import Image, ImageTk
import ElbowUpDown  # # # Replace with your actual module name




# Function to update the camera feed
def update_camera_feed():
    global cap, camera_label
    ret, frame = cap.read()
    if ret:
        # Process the frame with the pose detection logic
        frame = ElbowUpDown.run_exercise()  # Annotate the frame with pose detection

        # Convert the frame to an image format suitable for Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the label with the new image
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

    # Schedule the next frame update
    camera_label.after(10, update_camera_feed)


# Function to quit the application
def quit_app():
    global cap, root
    cap.release()
    root.quit()


# Initialize the camera
cap = cv2.VideoCapture(0)

# Create the Tkinter GUI
root = Tk()
root.title("Exercise Pose Detection")
root.geometry("800x400")  # Adjust window size as needed

# Create a frame for the camera
camera_frame = Frame(root, width=400, height=400)
camera_frame.pack(side=LEFT, fill=BOTH, expand=True)

camera_label = Label(camera_frame)
camera_label.pack()

# Create a frame for the buttons
button_frame = Frame(root, width=400, height=400, bg="lightgrey")
button_frame.pack(side=RIGHT, fill=BOTH, expand=True)

# Add buttons to the right-hand frame
button1 = Button(button_frame, text="Button 1", command=lambda: print("Button 1 Clicked"))
button1.pack(pady=10)

button2 = Button(button_frame, text="Button 2", command=lambda: print("Button 2 Clicked"))
button2.pack(pady=10)

exit_button = Button(button_frame, text="Exit", command=quit_app)
exit_button.pack(pady=10)

# Start the camera feed
update_camera_feed()

# Run the Tkinter main loop
root.mainloop()
