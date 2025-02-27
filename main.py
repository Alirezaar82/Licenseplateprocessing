import cv2
import numpy as np
from keras.models import load_model
from tkinter import *
from tkinter import filedialog

model = load_model('model.h5')

class_name = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
]

root = Tk()
root.title("Detect Persian number on the image")

root.geometry("600x400")

def select_image():
    file_path = filedialog.askopenfilename()  
    if file_path:
        process_image(file_path)


def process_image(image_path):
    numbers = []

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    detected_numbers = []
    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 30:  
            char_img = thresh[y:y+h, x:x+w+2]
            _, char_img = cv2.threshold(char_img, 200, 255, cv2.THRESH_BINARY)
            char_img = cv2.resize(char_img, (32, 32))
            char_img = cv2.blur(char_img, (1, 1))
            image_num = np.expand_dims(char_img, axis=2)
            image_num = np.array([image_num])
            
            # Predict the number
            out = model.predict(image_num)
            number = class_name[(np.argmax(out[0], axis=0))]
            detected_numbers.append(number)

    result_label.config(text=f"Detected numbers: {' '.join(detected_numbers)}")

select_button = Button(root, text="Select Image", command=select_image)
select_button.pack(pady=20)

result_label = Label(root, text="Detected numbers: ", font=("Helvetica", 14))
result_label.pack(pady=20)

root.mainloop()