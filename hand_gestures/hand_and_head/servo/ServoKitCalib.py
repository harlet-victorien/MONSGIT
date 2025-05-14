# encoding: UTF-8
'''
    Copyright (c) 2020-8 Arducam <http://www.arducam.com>.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
    OR OTHER DEALINGS IN THE SOFTWARE.
'''

import time
import os
import sys
import time
import adafruit_servokit
import curses
import Jetson.GPIO as GPIO

from pynput import keyboard
import sys
#from ultralytics import YOLO
import numpy as np
#from jtop import jtop
import cv2
#import torch
import math
import time
np.bool = np.bool_
default_angleBottom1 = 70
default_angleTop1 = 50

default_angleBottom2 = 30
default_angleTop2 = 150

default_angleBottom3 = 60
default_angleTop3 = 130




class ServoKit(object):


    def __init__(self, num_ports, setRotation = 45, MiddleAngleServoBottom = 90, MiddleAngleServoTop = 90, NumCamera = 0):
        print("Initializing the servo...")
        self.kit = adafruit_servokit.ServoKit(channels=16)
        self.num_ports = num_ports
        self.setRotation = setRotation
        self.MiddleAngleServoBottom = MiddleAngleServoBottom
        self.MiddleAngleServoTop = MiddleAngleServoTop
        self.NumCamera = NumCamera

        # renseignement du tracking kit
        choixTK = input("Renseignez le numéro du tracking kit (1, 2 ou 3) : ")
        if (choixTK == "1"):
            print("Vous avez choisi le tracking kit 1")
            self.calibTrackingKit(1)

        elif (choixTK == "2"):
            print("Vous avez choisi le tracking kit 2")
            self.calibTrackingKit(2)
        elif (choixTK == "3"):
            print("Vous avez choisi le tracking kit 3")
            self.calibTrackingKit(3)
        else:
            print("Choix invalide")

        self.NumCamera = int(input("Renseignez le numéro (0,2, 4, 6 ...) assigné à la camera (Pour trouver le numéro, insérez dans un terminal : 'v4l2-ctl --list-devices') : "))

        self.resetAll()
        print("Initializing complete.")


    def calibTrackingKit(self, numéroKit):
        print("Tracking Kit Calibration")
        if(numéroKit == 1):
            self.MiddleAngleServoBottom = 70
            self.MiddleAngleServoTop = 50
            self.setMaxRotation(45)
        if (numéroKit == 2):
            self.MiddleAngleServoBottom = 40
            self.MiddleAngleServoTop = 160
            self.setMaxRotation(30)
        if (numéroKit == 3):
            self.MiddleAngleServoBottom = 60
            self.MiddleAngleServoTop = 130
            self.setMaxRotation(45)


    def setMaxRotation(self, angleMax):

        if angleMax > 90:
            angleMax = 90
        elif angleMax < -90:
            angleMax = -90
        self.setRotation = angleMax

    def getMaxRotation(self):
        return self.setRotation

    def setAngle(self, port, angle):
        if angle < 0:
            self.kit.servo[port].angle = 0
        elif angle > 180:
            self.kit.servo[port].angle = 180
        else:
            self.kit.servo[port].angle = angle

    def getNumCamera(self):
        return self.NumCamera

    def setAngleDeg(self, port, angle):

        if angle < -self.setRotation:
            angle = -self.setRotation
        elif angle > self.setRotation:
            angle = self.setRotation

        angle = angle / 135 * 180

        if(port == 0):
            angle = angle + self.MiddleAngleServoBottom
        elif (port == 1):
            angle = angle + self.MiddleAngleServoTop

        if angle < 0:
            self.kit.servo[port].angle = 0
        elif angle > 180:
            self.kit.servo[port].angle = 180
        else:
            self.kit.servo[port].angle = angle




    def getAngle(self, port):
        return self.kit.servo[port].angle

    def getAngleDeg(self, port):

        angle = self.kit.servo[port].angle

        if(port == 0):
            angle = angle - self.MiddleAngleServoBottom
        if (port == 1):
            angle = angle - self.MiddleAngleServoTop

        angle = (angle /180) * 135

        return angle

    def reset(self, port):
        self.kit.servo[port].angle = self.MiddleAngleServoBottom

    def resetAll(self):

        self.kit.servo[0].angle = self.MiddleAngleServoBottom
        self.kit.servo[1].angle = self.MiddleAngleServoTop





    def testCalibPlage(self):
        for i in range(-self.setRotation, self.setRotation, 5):
            servoKit.setAngleDeg(0, i)
            servoKit.setAngleDeg(2, i)
            time.sleep(.05)

        for i in range(-self.setRotation, self.setRotation, 5):
            servoKit.setAngleDeg(0, 180 - i)
            servoKit.setAngleDeg(2, 180 - i)
            time.sleep(.05)

        for i in range(-self.setRotation, self.setRotation, 5):
            servoKit.setAngleDeg(0, i)
            servoKit.setAngleDeg(2, i)
            time.sleep(.05)


        for i in range(-self.setRotation, self.setRotation, 5):
            servoKit.setAngleDeg(1, i)
            servoKit.setAngleDeg(3, i)
            time.sleep(.05)

        for i in range(-self.setRotation, self.setRotation, 5):
            servoKit.setAngleDeg(1, 180 - i)
            servoKit.setAngleDeg(3, 180 - i)
            time.sleep(.05)

        for i in range(-self.setRotation, self.setRotation, 5):
            servoKit.setAngleDeg(1, i)
            servoKit.setAngleDeg(3, i)
            time.sleep(.05)





# parse input key
def parseKey(servoKit, motor_step):
    global image_count
    if keyboard.is_pressed("s"):
        servoKit.setAngle(1, servoKit.getAngle(1) - motor_step)
        print("Moteur haut", servoKit.getAngle(1))
        print("Moteur bas ", servoKit.getAngle(0))
    if keyboard.is_pressed("z"):
        servoKit.setAngle(1, servoKit.getAngle(1) + motor_step)
        print("Moteur haut", servoKit.getAngle(1))
        print("Moteur bas ", servoKit.getAngle(0))
    if keyboard.is_pressed("q"):
        servoKit.setAngle(0, servoKit.getAngle(0) - motor_step)
        print("Moteur haut", servoKit.getAngle(1))

        print("Moteur bas ", servoKit.getAngle(0))
    if keyboard.is_pressed("d"):
        servoKit.setAngle(0, servoKit.getAngle(0) + motor_step)
        print("Moteur haut", servoKit.getAngle(1))
        print("Moteur bas ", servoKit.getAngle(0))



def testCalib():
    for i in range(servoKit.MiddleAngleServoBottom, 180, 5):
        servoKit.setAngle(0, i)
        servoKit.setAngle(3, i)
        time.sleep(.05)

    for i in range(0, 180, 5):
        servoKit.setAngle(0, 180 - i)
        servoKit.setAngle(3, 180 - i)
        time.sleep(.05)

    for i in range(0, servoKit.MiddleAngleServoBottom, 5):
        servoKit.setAngle(0, i)
        servoKit.setAngle(3, i)
        time.sleep(.05)


    for i in range(servoKit.MiddleAngleServoTop, 180, 5):
        servoKit.setAngle(1, i)
        servoKit.setAngle(3, i)
        time.sleep(.05)
        time.sleep(.05)

    for i in range(0, 180, 5):
        servoKit.setAngle(1, 180 - i)
        servoKit.setAngle(3, 180 - i)
        time.sleep(.05)

    for i in range(0, servoKit.MiddleAngleServoTop, 5):
        servoKit.setAngle(1, i)
        servoKit.setAngle(3, i)
        time.sleep(.05)









