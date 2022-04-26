# !/usr/bin/python3
import os
import cv2
import pandas as pd
import numpy as np
import tarfile
import argparse
import glob
from itertools import combinations
import generate_list
from new_comparison import Comparison

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tar-files', dest='tar_file', type=str, help='path to tar files containing input data.',
                        default=None)
    parser.add_argument('-i', '--input-path', dest='input_path', type=str, help='path to image input files.',
                        required=True)
    parser.add_argument('-o', '--output-path', dest='output_lst', type=str, help='path to output lst file.',
                        required=True)
    parser.add_argument('-g', '--gauss-kernel', dest='gaussian_kernel', type=int, help='gaussian kernel size.', nargs=2,
                        default=[7, 7])
    parser.add_argument('-s', '--score-thresh', dest='score_thresh', type=int, help='score threshold.', nargs=1,
                        default=1000)
    parser.add_argument('-m', '--hist-match', dest='hist_match', type=bool, help='use histogram match to compare.',
                        default=False)
    # parser.add_argument('-c','--camera-names', dest='camera_number',   type=str, help='compare images from the given cameras numbers', nargs=4, required=True)

    args = parser.parse_args()
    # camera_number_lst = []
    # dp_list_c1 = []
    # dp_list_c2 = []
    # dp_list_c3 = []
    # dp_list_c4 = []
    filepath = args.tar_file
    inpath = args.input_path
    outpath = args.output_lst
    hist_match = args.hist_match
    threshold_score_value = args.score_thresh
    gaussian_blur_radius_list = args.gaussian_kernel
    # from IPython import embed;embed()
    # assert len(camera_number_lst)>1
    dp_list = generate_list.image_gen(filepath, inpath, outpath)
    img_comparison = Comparison(hist_match, bit_depth=12, min_contour_area=100)
    # dp_list = img_comparison.get_datapoint_list("path to input lst")
    n = 2
    removed_img_lst = []
    loaded_img_list = []
    for read_img in dp_list:
        load_img = cv2.imread(read_img)
        load_img = img_comparison.preprocess_image_change_detection(load_img, gaussian_blur_radius_list)
        load_img = cv2.resize(load_img, (800,600), interpolation=cv2.INTER_LANCZOS4)
        # load_img = img_comparison.standardization(load_img)
        loaded_img_list.append(load_img)
    # from IPython import embed;embed()
    for (img1, img2), (path_img1, path_img2) in zip(combinations(loaded_img_list.copy(), n),
                                                    combinations(dp_list.copy(), n)):

        if hist_match:
            matched = img_comparison.histogram_matching(img1, img2)
            score1, thresh1 = img_comparison.compare_frames_change_detection(img1, matched, 100)
            score2, thresh2 = img_comparison.compare_frames_change_detection(img2, matched, 100)
            if score1 < score2:
                score = score1
            else:
                score = score2
        else:
            score, thresh = img_comparison.compare_frames_change_detection(img1, img2, 100)
        # print(score)
        if score <= 1000:
            if path_img2 not in removed_img_lst:
                removed_img_lst.append(path_img2)
                if os.path.exists(path_img2):
                    if path_img2 in dp_list:
                        dp_list.remove(path_img2)
                # else:
                # print("The file does not exist")
    # images in the folder are removed
    for rem in removed_img_lst:
        if os.path.exists(rem):
            os.remove(rem)
    # saving list of images deleted and remaining images
    datapoint_df = pd.DataFrame({'path': removed_img_lst})
    datapoint_df1 = pd.DataFrame({'path': dp_list})
    datapoint_df.to_csv(outpath + '/' + 'removed_imgs_path.lst', index=False)
    datapoint_df1.to_csv(outpath + '/' + 'remaining_imgs_path.lst', index=False)