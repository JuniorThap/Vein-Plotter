from src.vein_selection import model_session, pipeline
import cv2
import numpy as np
import torch

img = cv2.imread("experiment1_1\high_contrast\person_001_L1_IR.png", cv2.IMREAD_GRAYSCALE)
out, lines = pipeline(model_session, img)