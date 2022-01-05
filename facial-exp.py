"""
project : Facial Expression Recognition
author : Akash Sharma
connection : https://www.linkedin.com/in/akash-sharma-01775b14a/
portfolio : https://cosmix-6.github.io/
"""
from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras import models

# loading model
model = models.load_model("facial-exp.h5")
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

win = Tk()

win.title('Facial Expression Recognition')
win.geometry("300x350")
win.minsize(300, 350)
win.maxsize(300, 350)

frame1 = Frame(win)
frame1.pack(side=TOP, fill=X)

frame2 = Frame(win)
frame2.pack(side=TOP, fill=X)

# camera view
label = Label(frame1)
label.grid(row=0, column=0)

# output text
labelfont = ('times', 30, 'bold')
indicator = Label(frame2)
indicator.grid(row=15, column=0)
indicator.config(bg='#086E7D', fg='#FFFFFF')
indicator.config(font=labelfont)
indicator.pack(expand=YES, fill=BOTH)
cap = cv2.VideoCapture(0)


# capture video camera sub region
upper_left = (200, 0)
bottom_right = (500, 300)


def predict_expression(img_array):
    """
    function that predicts the expression by processing the array data of an image
    :param img_array:
    :return:
    """
    # converting the color & size of image as per used in model training
    conv_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    # here we're having the image of size 300x300
    conv_img = cv2.resize(conv_img, (48, 48))

    # here we've image image of 48x48
    # Let's convert image to numpy array and reshape the image to shape of 1x48x48x1
    img_array = np.array(conv_img)
    img_array = np.stack((img_array,) * 1, axis=-1)
    img_array_ex = np.expand_dims(img_array, axis=0) / 255

    # predict the image array using the model
    prediction = model.predict(img_array_ex)

    # returning the result
    return emotions[np.argmax(prediction)]


def show_frames():
    """
    fucntion to predict the display the video capture and predicted result
    :return: Image, Predicted String Value
    """
    # Get the latest frame and convert into Image
    cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
    # selecting subregion from captured image
    img_array = cv2image[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]
    img = Image.fromarray(img_array)

    # predict the image expression
    indicator.config(text=predict_expression(img_array))
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    # Repeat after an interval to capture continuously
    label.after(20, show_frames)


show_frames()
win.mainloop()
