import numpy as np 
import matplotlib.pyplot as plt
from ff_racing.PlannerUtils.VehicleStateHistory import VehicleStateHistory
from ff_racing.PlannerUtils.TrackLine import TrackLine

import cv2 as cv
from PIL import Image
import os

name = "MyFrenetPlanner"
path = f"Data/{name}/Scans/"


def process_img(counter=1):
    img_name = path + f"{name}_{counter}.png"
    print(img_name)
    img = cv.imread(img_name)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    edges = cv.Canny(gray, 50, 150, apertureSize=3, L2gradient=False)

    c, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(len(c))
    for i in range(len(c)):
        if len(c[i]) < 10:
            continue
        # print(i, len(c[i]))
        print(c[i].shape)
        img2 = cv.drawContours(img, c, i, (0,255,0), 1)
        cv.imshow('cont', img2)
        cv.waitKey(1000)
    # img2 = cv.drawContours(img, c, -1, (0,255,0), 1)
    

    cv.imshow('original', gray)
    cv.imshow('canny', edges)
    cv.waitKey(5000)
    cv.destroyAllWindows()
    


process_img()