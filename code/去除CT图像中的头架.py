import numpy as np
import cv2
from skimage import morphology
import SimpleITK as sitk
from tqdm import tqdm
import os
def background_noise(input,jiaoda):
    # thres_val = -900
    thres_val = -800
    k_size = 10
    input_mr_vol = input.astype('float32')
    jiaoda_ct = jiaoda.astype('float32')
    sz = np.shape(input_mr_vol)
    init_mask = np.zeros(sz)

    init_mask[input_mr_vol > thres_val] = 1

    kernel = np.ones((k_size, k_size))  # 核
    mask = np.zeros(sz)  # 掩码(1, 512, 512)

    idSlice_all = np.arange(sz[0])
    for idS in idSlice_all:
        tmp = init_mask[idS, :, :]

        tmp = cv2.erode(tmp, kernel, iterations=1)
        tmp = cv2.dilate(tmp, kernel, iterations=1)
        tmp = morphology.convex_hull_image(tmp)

        mask[idS, :, :] = tmp
    # save masked mr.
    masked_mr = np.ones(sz) * (-1000)
    masked_ct = np.ones(sz) * (-1000)

    masked_mr[mask == 1] = input_mr_vol[mask == 1]
    masked_ct[mask == 1] = jiaoda_ct[mask == 1]

    return masked_ct, masked_mr

# jiaoda_dir = r'C:\code\DeepLearning\Dayi\TensorRT Deployment Process\02 MRI2sCT\3channelModel\resister_20241024\06 calculate\01 mr2ct\gt\id02_mr2ct_gt.nii.gz'
# predict_dir = r'C:\code\DeepLearning\Dayi\TensorRT Deployment Process\02 MRI2sCT\3channelModel\resister_20241024\06 calculate\01 mr2ct\pre_changedirection\id02_mr2ct_result.nii.gz'

jiaoda_dir = r'/data/ZZJ/code/1101/dataset/process/nii_taihe/CT'
predict_dir =r'/data/ZZJ/code/1101/dataset/process/Registration/MR'
ct_save_root_path = r'/data/ZZJ/code/1101/dataset/process/Training/CT'
mr_save_root_path = r'/data/ZZJ/code/1101/dataset/process/Training/MRI'
jiaoda_dir_list = os.listdir(jiaoda_dir)
for indx in range(len(jiaoda_dir_list)):
    jiaoda_dir1 = os.path.join(jiaoda_dir, jiaoda_dir_list[indx])
    # patient_name = jiaoda_dir.split("\\")[-1]
    predict_dir1 = os.path.join(predict_dir, jiaoda_dir_list[indx])
    # patient_name = predict_dir.split("\\")[-1]
    print('jiaoda_dir_list[i]:',jiaoda_dir_list[indx])
    print('CT:',jiaoda_dir1)
    print('MRI',predict_dir1 )

    jiaoda_img = sitk.ReadImage(jiaoda_dir1)
    jiaoda_npy = sitk.GetArrayFromImage(jiaoda_img)

    spacing = jiaoda_img.GetSpacing()
    direction = jiaoda_img.GetDirection()
    origin = jiaoda_img.GetOrigin()




    predict_img = sitk.ReadImage(predict_dir1)
    predict_npy = sitk.GetArrayFromImage(predict_img)

    spacing1 = predict_img.GetSpacing()
    direction1 = predict_img.GetDirection()
    origin1 = predict_img.GetOrigin()

    save_ = np.zeros(jiaoda_npy.shape)
    save1_ = np.zeros(predict_npy.shape)
    ct_save_path = os.path.join(ct_save_root_path, jiaoda_dir_list[indx])
    mr_save_path = os.path.join(mr_save_root_path, jiaoda_dir_list[indx])

    for i in tqdm(range(predict_npy.shape[0]), desc="Processing", ncols=100):
        predict_slice = predict_npy[i,:,:]
        jiaoda_slice = jiaoda_npy[i,:,:]

        #去除维度为1的
        predict_slice = np.expand_dims(predict_slice, axis=0)
        jiaoda_slice = np.expand_dims(jiaoda_slice, axis=0)

        jiaoda_slice_pro, predict_slice_pro = background_noise(predict_slice,jiaoda_slice)

        jiaoda_slice_pro = np.squeeze(jiaoda_slice_pro)
        predict_slice_pro = np.squeeze(predict_slice_pro)
        save_[i,:,:] = jiaoda_slice_pro
        save1_[i,:,:] = predict_slice_pro

    save_img = sitk.GetImageFromArray(save_)
    save_img.SetSpacing(spacing)
    save_img.SetDirection(direction)
    save_img.SetOrigin(origin)
    sitk.WriteImage(save_img, ct_save_path)

    save_img1 = sitk.GetImageFromArray(save1_)
    save_img1.SetSpacing(spacing1)
    save_img1.SetDirection(direction1)
    save_img1.SetOrigin(origin1)
    sitk.WriteImage(save_img1, mr_save_path)






