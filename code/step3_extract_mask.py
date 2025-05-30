import os

import cv2
import numpy as np
import nibabel as nib
from skimage import morphology


thres_val = -900
k_size = 10


# ct_path='/media/yyz/11685CA812B59B21/MR_CT/data/dayi/dayi_nii/0_CT/'
mr_path='/data/ZZJ/code/1101/dataset/process/Training/MRI/'
files=os.listdir(mr_path)


# save_mr_path = '/media/yyz/11685CA812B59B21/MR_CT/data/dayi/dayi_nii/3_masked_mrct/MR/'
# save_ct_path = '/media/yyz/11685CA812B59B21/MR_CT/data/dayi/dayi_nii/3_masked_mrct/CT/'
save_mask_path = '/data/ZZJ/code/1101/dataset/process/extract_mask/'

for file in files:

    print('processing MR: ',file)

    input_mr = nib.load(mr_path+file)
    input_mr_vol = input_mr.get_fdata().astype('single')

    # input_ct = nib.load(ct_path+file)
    # input_ct_vol = input_ct.get_fdata().astype('single')

    sz = np.shape(input_mr_vol)

    init_mask = np.zeros(sz)
    init_mask[input_mr_vol > thres_val] = 1

    kernel = np.ones((k_size, k_size))
    mask = np.zeros(sz)

    idSlice_all = np.arange(sz[2])
    for idS in idSlice_all:

        tmp = init_mask[:,:,idS]

        tmp = cv2.erode(tmp, kernel, iterations=1)
        tmp = cv2.dilate(tmp, kernel, iterations=1)
        tmp = morphology.convex_hull_image(tmp)

        mask[:,:,idS] = tmp

    # save mask.
    save_mask = nib.Nifti1Image(mask, input_mr.affine, input_mr.header)
    nib.save(save_mask, save_mask_path+file)

    # # save masked ct.
    # masked_ct = np.ones(sz) * (-1000)
    # masked_ct[mask == 1] = input_ct_vol[mask == 1]
    # save_masked_ct = nib.Nifti1Image(masked_ct, input_ct.affine, input_ct.header)
    # nib.save(save_masked_ct, save_ct_path+file)
    #
    # # save masked mr.
    # masked_mr = np.ones(sz) * (-1000)
    # masked_mr[mask == 1] = input_mr_vol[mask == 1]
    # save_masked_mr = nib.Nifti1Image(masked_mr, input_mr.affine, input_mr.header)
    # nib.save(save_masked_mr, save_mr_path+file)