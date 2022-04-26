import os
import sys
import glob
import numpy as np
import pandas as pd


def image_gen(tar_file_path, inpath, outpath):
    """Extracts image from tar file to inpath and
    creates a list of inpath images and stores in as lst
    """

    if tar_file_path is not None:
        file = tarfile.open(tar_file_path)
        file.extractall(inpath)
        file.close()
    else:
        print("File already extracted")

    if not os.path.isdir(inpath) or not os.access(inpath, os.R_OK):
            print("input dir does not exist or is not readable: {}. Creating".format(inpath))
            os.makedirs(inpath)

    all_datapoint_paths = glob.glob(inpath+'/**/*.png', recursive=True)
    all_datapoint_paths = [os.path.abspath(p) for p in all_datapoint_paths]
    #generate random
    all_datapoint_paths = np.random.permutation(all_datapoint_paths)
    #for filePath in all_datapoint_paths:
    #    print("{}".format(filePath))
    datapoint_df = pd.DataFrame({'path': all_datapoint_paths})
    datapoint_df.to_csv(outpath+'/'+'total_path.lst',index=False)
    datapoint_list = list(datapoint_df['path'])

    return datapoint_list