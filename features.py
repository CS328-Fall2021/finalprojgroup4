"""
This file is used for extracting features.
"""

import numpy as np
import pandas as pd
import math
from scipy import signal
from scipy.signal import butter, lfilter, freqz, iirnotch, filtfilt, firwin, argrelextrema

def compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    return np.mean(window, axis=0)

# TODO: define functions to compute more features

def get_accel_mag(window):
    accelMag = []
    for currPoint in window:
        x = (currPoint[0])
        y = (currPoint[1])
        z = (currPoint[2])
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        accelMag.append(r)
        
    order = 2
    fs = 150 #Sampling Frequency
    cutoff = 2
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=True)
    buttered_mag = filtfilt(b, a, np.array(accelMag))
    return np.array(accelMag)

#HELPER function for other functions
#Separates the x, y, z coordinates into separate lists
def _separate_coordinates(window):
    x_coords = []
    y_coords = []
    z_coords = []
    coordinate_list = [x_coords, y_coords, z_coords]
    for coords in window:
        for i in range(0, 3):
            coordinate_list[i].append(coords[i])

    return coordinate_list

def compute_total_peaks(window):
    coordinate_list = _separate_coordinates(window)

    peak_arrays = []
    for i in coordinate_list:
        temp_array = np.array(i)
        peak_arrays.append(temp_array[argrelextrema(temp_array, np.greater)])

    peak_arrays_lengths = []
    for i in peak_arrays:
        peak_arrays_lengths.append(len(i))

    return peak_arrays_lengths

def compute_total_troughs(window):
    coordinate_list = _separate_coordinates(window)

    peak_arrays = []
    for i in coordinate_list:
        temp_array = np.array(i)
        peak_arrays.append(temp_array[argrelextrema(temp_array, np.less)])

    peak_arrays_lengths = []
    for i in peak_arrays:
        peak_arrays_lengths.append(len(i))

    return peak_arrays_lengths

def compute_total_zero_crossings(window):
    #TODO
   
    def is_negative(number):
        if (number < 0):
            return "yup"
        else:
            return "naw dawg"
 
    x_crossings = 0
    y_crossings = 0
    z_crossings = 0
    output = [x_crossings, y_crossings, z_crossings]
 
    x_negative_status = is_negative(window[0][0])
    y_negative_status = is_negative(window[0][1])
    z_negative_status = is_negative(window[0][2])
 
    negative_statuses = [x_negative_status, y_negative_status, z_negative_status]
 
    for coordinates in window:
        for i in range(0, 3):
            if (negative_statuses[i] != is_negative(coordinates[i])):
                output[i] += 1
                negative_statuses[i] = is_negative(coordinates[i])
 
    return output

def compute_entropy(window):
    coordinate_list = _separate_coordinates(window)
    result= [0.0,0.0,0.0]
    
    for i in range(0,3):
        sum = 0.0
        for arr in np.histogram(coordinate_list[i], bins = 5):
            for val in arr:
                sum += val
        result[i] = sum
        
    return result

def compute_dfreq_range(window):
    #TODO
    x_coords = []
    y_coords = []
    z_coords = []
 
    coords = [x_coords, y_coords, z_coords]
 
    for coordinate in window:
        for i in range(0, 3):
            coords[i].append(coordinate[i])
 
    ranges = [0, 0, 0]
    for i in range(0, 3):
        ranges[i] = max(coords[i]) - min(coords[i])
 
    return ranges
 

def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """
    
    x = []
    feature_names = []

    x.append(compute_mean_features(window))
    feature_names.append("x_mean")
    feature_names.append("y_mean")
    feature_names.append("z_mean")
    
    x.append(compute_total_peaks(window))
    feature_names.append("x_peaks")
    feature_names.append("y_peaks")
    feature_names.append("z_peaks")
    
    x.append(compute_total_troughs(window))
    feature_names.append("x_troughs")
    feature_names.append("y_troughs")
    feature_names.append("z_troughs")
    
    x.append(compute_total_zero_crossings(window))
    feature_names.append("x_zero_crossings")
    feature_names.append("y_zero_crossings")
    feature_names.append("z_zero_crossings")
    
    x.append(compute_dfreq_range(window))
    feature_names.append("x_dfreq")
    feature_names.append("y_dfreq")
    feature_names.append("z_dfreq")
    
    x.append(compute_entropy(window))
    feature_names.append("x_entropy")
    feature_names.append("y_entropy")
    feature_names.append("z_entropy")

    # TODO: call functions to compute other features. Append the features to x and the names of these features to feature_names

    feature_vector = np.concatenate(x, axis=0) # convert the list of features to a single 1-dimensional vector
    return feature_names, feature_vector
