import sys
import os
import glob
import numpy as np
import pandas as pd
import pathlib
import cv2
import imutils
import numpy as np
from itertools import combinations
#from skimage.exposure import match_histograms


class Comparison(object):

    def __init__(self, hist_match=False, bit_depth=None, min_contour_area=None):

        """
        provides function loaders for pre-processing, optional scaling,
        image comparison and results score values
        """
        self.hist_match = hist_match
        self.bit_depth = bit_depth
        self.min_contour_area = min_contour_area

    def get_datapoint_list(self, inp_list):
        """
        if lst file is given /dont have to use generate_list.py
        generates a list of paths
        :param inp: csv file where column path is used OR a list of paths
        :return: list of paths
        """
        datapoint_list = []
        if isinstance(inp, str) and \
                pathlib.Path(inp).suffix in ['.lst', '.csv']:
            datapoint_list += pd.read_csv(inp)['path'].values.tolist()
        elif isinstance(inp, list):
            for i in inp:
                if isinstance(i, str):
                    if pathlib.Path(i).suffix in ['.lst', '.csv']:
                        datapoint_list += pd.read_csv(i)['path'].values.tolist()
                    else:
                        datapoint_list += i
                else:
                    raise ValueError('bad file inserted')
        else:
            raise ValueError('corrupt input to get_datapoint_list data inserted')
        return datapoint_list

    @staticmethod
    def draw_color_mask(img, borders, color=(0, 0, 0)):
        h = img.shape[0]
        w = img.shape[1]
        x_min = int(borders[0] * w / 100)
        x_max = w - int(borders[2] * w / 100)
        y_min = int(borders[1] * h / 100)
        y_max = h - int(borders[3] * h / 100)
        img = cv2.rectangle(img, (0, 0), (x_min, h), color, -1)
        img = cv2.rectangle(img, (0, 0), (w, y_min), color, -1)
        img = cv2.rectangle(img, (x_max, 0), (w, h), color, -1)
        img = cv2.rectangle(img, (0, y_max), (w, h), color, -1)
        return img

    def preprocess_image_change_detection(self, img, gaussian_blur_radius_list=None, black_mask=(5, 10, 5, 0)):
        """
        preprocess image for comparison
        """
        gray = img.copy()
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        if gaussian_blur_radius_list is not None:
            for radius in gaussian_blur_radius_list:
                gray = cv2.GaussianBlur(gray, (radius, radius), 0)
                # gray = gray1 - gray2 #DoG
                # cv2.imwrite('/home/ram/Downloads/norm.png',normalized_image)
        # median = cv2.medianBlur(gray, 5)
        gray = Comparison.draw_color_mask(gray, black_mask)

        return gray

    def compare_frames_change_detection(self, prev_frame, next_frame, min_contour_area):
        """
        image comparison based on absolute difference and thresholding
        """
        frame_delta = cv2.absdiff(prev_frame, next_frame)
        # thresh = cv2.adaptiveThreshold(frame_delta, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 13)
        thresh = cv2.threshold(frame_delta, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        # from IPython import embed;embed()
        thresh = thresh.astype('uint8')
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        score = 0
        res_cnts = []
        for c in cnts:
            if cv2.contourArea(c) < min_contour_area:
                continue
            res_cnts.append(c)
            score += cv2.contourArea(c)

        return score, thresh

    def standardization(self, img, bit_depth=12):
        # mean = np.mean(img)
        # variance = np.var(img)
        # img = (img - mean)/variance
        img = img / float((2 ** bit_depth) - 1)
        # from IPython import embed;embed()
        # img2 = img2 / float((2**bit_depth)-1)

        return img

    def histogram_matching(self, img1_pre, img2_pre):
        """
        image comparison based on histogram matching
        """
        # time consuming-varied in increasing accuracy than normal comparison
        matched = match_histograms(img1_pre, img2_pre)
        matched = matched.astype('uint8')

        return matched


