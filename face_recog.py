# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:20:39 2017

@author: Visharg Shah
"""

##############################Dataset_Creator##############################

#####Importing dependencies#####
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sqlite3

#####User Input#####
ID = input("Enter the ID:  ")
Name = input("Enter the name : ")
Age = input("Enter your Age : ")
Gender = input("Enter your Gender : ")
Criminal = input("Any Criminal Records? (Y/N):")

#####For no. of Images#####
samplenum = 0

#####Frontface Cascade#####
cam = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('C:/Users/Visharg Shah/Desktop/Face Detection/Face Recognition/haarcascade_frontalface_default.xml')

#####All values in the database#####
def update(ID, Name, Age, Gender, Criminal) :
    connection = sqlite3.connect("V:/Software/SQLiteStudio/FaceDatabase.db")
    cmd = "SELECT * FROM People WHERE ID="+str(ID)
    details = connection.execute(cmd)
    recordexist = 0
    for row in details:
        recordexist = 1
    if(recordexist==1):
        cmd = "UPDATE People SET Name="+str(Name)+", Age="+str(Age)+", Gender="+str(Gender)+", CriminalRecords="+str(Criminal)+" WHERE ID=" + str(ID)
    else:
        cmd = "INSERT into People(ID,Name,Age,Gender,CriminalRecords) values (?,?,?,?,?)"
        par = (ID,Name,Age,Gender,Criminal)
        connection.execute(cmd,par)
        connection.commit()
        connection.close()
            
update(ID,Name,Age,Gender,Criminal)

#####Capturing images from web cam#####
while(cam.isOpened()):  # check !
    # capture frame-by-frame
    ret,img = cam.read()

    if ret: # check ! (some webcam's need a "warmup")
        # our operation on frame come here
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray,1.2,5)
        for(x,y,w,h) in faces :
            samplenum = samplenum+1
            cv2.rectangle(img,(x,y), (x+w, y+h),(0,255,0),2)
            cv2.imwrite("Face_Dataset/User." + str(ID)+ "." +str(samplenum)+".jpg",gray[y:y+h,x:x+w])
        cv2.imshow('Face',img) # Display the resulting frame
        cv2.waitKey(1)
        if(samplenum>20):
            break;
           
#####When everything is done release the capture#####
cam.release()
cv2.destroyAllWindows()

##############################Model_Training##############################

#####Importing Dependencies#####
import os
import cv2
import numpy as np
from PIL import Image

#####Creating Face Recognizer#####
recognizer = cv2.face.createLBPHFaceRecognizer()
path = 'C:/Users/Visharg Shah/Desktop/Face Detection/Face Recognition/Face_Dataset'

def getImageId(path):
    imgpaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    ids = []
    for imgpath in imgpaths:
        faceimg = Image.open(imgpath).convert('L')
        facenp = np.array(faceimg,'uint8')
        ID = int(os.path.split(imgpath)[-1].split(".")[1])
        faces.append(facenp)
        ids.append(ID)
        cv2.imshow("training",facenp)
        cv2.waitKey(10)
    return np.array(ids) , faces

ID , faces = getImageId(path)

#####Training and saving the model#####
recognizer.train(faces,ID)
recognizer.save("Recognizer/traindata.yml")
cv2.destroyAllWindows()

##############################Detector##############################

#####Importing Dependencies#####
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sqlite3

#####Face Samples#####
samplenum = 0

#####Face Recognizer and Classifier#####
cam = cv2.VideoCapture(0)
recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('C:/Users/Visharg Shah/Desktop/Face Detection/Face Recognition/Recognizer/traindata.yml')
facedetect = cv2.CascadeClassifier('C:/Users/Visharg Shah/Desktop/Face Detection/Face Recognition/haarcascade_frontalface_default.xml')

#####Function to get ID for detected user#####
def getID(ID):
    connection = sqlite3.connect("V:/Software/SQLiteStudio/FaceDatabase.db")
    cmd = "SELECT * FROM People where ID="+str(ID)
    details = connection.execute(cmd)
    profile = None
    for det in details:
        profile = det
    connection.close()
    return profile
    
#####Setting Font style to display#####
font = cv2.FONT_HERSHEY_PLAIN

#####Displaying details of detected face#####
while(cam.isOpened()):  # check !
    # capture frame-by-frame
    ret, img = cam.read()

    if ret: # check ! (some webcam's need a "warmup")
        # our operation on frame come here
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces :       
            cv2.rectangle(img,(x,y), (x+w,y+h),(0,0,255),3)
            ID, conf = recognizer.predict(gray[y:y+h,x:x+w])
            profile = getID(ID)
            if(profile!=None):
                cv2.putText(img,str(profile[0]),(x,y+h+50),font,2,(0,0,255),2,cv2.LINE_AA)
                cv2.putText(img,str(profile[1]),(x,y+h+75),font,2,(0,0,255),2,cv2.LINE_AA)
                cv2.putText(img,str(profile[2]),(x,y+h+100),font,2,(0,0,255),2,cv2.LINE_AA)
                cv2.putText(img,str(profile[3]),(x,y+h+125),font,2,(0,0,255),2,cv2.LINE_AA)
                cv2.putText(img,str(profile[4]),(x,y+h+150),font,2,(0,0,255),2,cv2.LINE_AA)
            
        cv2.imshow('Face', img) # Display the resulting frame
        if(cv2.waitKey(1)==ord('q')):
            break;
    
######Release the cam#####
cam.release()
cv2.destroyAllWindows()