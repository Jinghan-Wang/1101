import os

import cv2
import numpy as np
import nibabel as nib
from skimage import morphology


thres_val = 0.5
k_size = 10
dataNum = 3


input_foremask_path =  r'/data/ZZJ/code/1101/dataset/process/extract_mask/'
input_initmask_path =  r'/data/ZZJ/code/1101/dataset/process/fsl_result/'

save_processedmask_path =  r'/data/ZZJ/code/1101/dataset/process/processed_mask/'
save_finalmask_path =  r'/data/ZZJ/code/1101/dataset/process/Training/final_skull/'

files=os.listdir(input_foremask_path)

for file in files:

    print('processing MR: id {:0>2d}: \n',file)
    input_foremask_name=input_foremask_path+file
    input_initmask_name=input_initmask_path+file+'_mask.nii.gz'

    input_foremask = nib.load(input_foremask_name)
    input_foremask_vol = input_foremask.get_fdata().astype('single')

    input_initmask = nib.load(input_initmask_name)
    input_initmask_vol = input_initmask.get_fdata().astype('single')

    sz = np.shape(input_foremask_vol)
    print(sz)
    init_mask = np.zeros(sz)
    init_mask[input_initmask_vol > thres_val] = 1

    processed_mask = np.zeros(sz)

    idSlice_all = np.arange(sz[2])
    for idS in idSlice_all:

        tmp = init_mask[:,:,idS]

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        tmp = cv2.dilate(tmp, kernel_dilate, iterations=1)

        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size+10, k_size+10))
        tmp = cv2.erode(tmp, kernel_erode, iterations=2)

        processed_mask[:,:,idS] = tmp

    final_mask = np.zeros(sz)
    final_mask[input_foremask_vol > thres_val] = 1
    final_mask[processed_mask > thres_val] = 0

    # save mask.
    save_processed_mask = nib.Nifti1Image(processed_mask, input_initmask.affine, input_initmask.header)
    nib.save(save_processed_mask, save_processedmask_path+file)

    save_final_mask = nib.Nifti1Image(final_mask, input_initmask.affine, input_initmask.header)
    nib.save(save_final_mask, save_finalmask_path+file)