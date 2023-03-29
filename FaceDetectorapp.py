#face detector ai project. 3 stpes: Load a TON of face images (already done with haarcascade file). Change color scale to black and white. Train the algorithm to detect faces.

import cv2
from random import randrange 

#we want to load pre-trained data on face frontal from opencv (open computer vision) (use the haarcascade frontal face default)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

#now we want to SELECt a file of an image to detect a face
img = cv2.imread('frontfacetest.jpeg')

#convert color to gray scale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#Draw rectangles to appear aroound face
# cv2.rectangle(img, ( 643, 181), (643 + 684, 181 + 684), (0, 255, 0), 2) # This is just hardrawing the box. #x,y and x+w, y+h are the cooridantes that we got: (x,y) = (643, 181) which is the top left corner of the rectangle. (x+w, y+h) = (684, 684), you add these values to x and y, . the color is BGR. the number 2 represents thickness.

#(x,y,w,h) = face_coordinates[0] #This only detects ONE face. To detect multiple, we create a loop.
#cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2) #this is what draws the rectangle by itself. 

for (x,y,w,h) in face_coordinates:  #this is to detect multiple faces in one picture.
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)  





#now we need to print of the coordinates to be able to draw the square to display around the face.
#print(face_coordinates)

#now we want to bring up the image
cv2.imshow('Gray image', img)
#cv2.imshow('Test Image', img) ######this prints out the image in color. changing 'img' to grayscaled_img is what brings up the gray image.
cv2.waitKey() #this tells the program to wait here, until a key is pressed.



print("Code Check")
