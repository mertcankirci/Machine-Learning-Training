from pyexpat import model
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student_-mat.csv", sep=";") #sep for readin pandas dataframe with this it gets the correct values

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]] # Choosung arttirabutes we want to use

predict = "G3"







 
 
 