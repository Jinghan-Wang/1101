import numpy as np
from hausdorff import hausdorff_distance
import os
import SimpleITK as sitk

"""
分割任务的一些指标计算
DICE、Jaccard、hausdroff distance
"""

def dice_coefficient(prediction, ground_truth):
    prediction = np.round(prediction).astype(int)
    ground_truth = np.round(ground_truth).astype(int)
    return np.sum(prediction[ground_truth == 1]) * 2.0 / (np.sum(prediction) + np.sum(ground_truth))

def CalSimJaccard(dataA, dataB):
    '''【目的】计算Jaccard相似度，度量集合之间的差异，共有的元素越多则越相似
       【输入】np.array
       【输出】取值[0,1]，数值越大表示相似性越高，数值为1代表完全相似'''
    # 展平数组
    dataA_flat = dataA.flatten()
    dataB_flat = dataB.flatten()
    # 计算交集
    #intersection = np.sum((dataA_flat == dataB_flat) & (dataA_flat == 1))
    intersection = np.sum((dataA_flat == dataB_flat) & (dataA_flat == 1)) + np.sum((dataA_flat == dataB_flat) & (dataA_flat == 0))
    # 计算并集
    #union = np.sum(dataA_flat == 1) + np.sum(dataB_flat == 1) - intersection
    union = np.sum((dataA_flat == 1) & (dataB_flat == 1)) + np.sum((dataA_flat == 0) & (dataB_flat == 0)) + np.sum((dataA_flat == 1) & (dataB_flat == 0)) + np.sum((dataA_flat == 0) & (dataB_flat == 1))
    if union == 0:
        return np.nan
        #return 1
    return intersection / union


# Test computation of Hausdorff distance with different base distances
# print(f"Hausdorff distance test: {hausdorff_distance(X, Y, distance='manhattan')}")
# print(f"Hausdorff distance test: {hausdorff_distance(X, Y, distance='euclidean')}")
# print(f"Hausdorff distance test: {hausdorff_distance(X, Y, distance='chebyshev')}")
# print(f"Hausdorff distance test: {hausdorff_distance(X, Y, distance='cosine')}")


# ct_dir = r'/data/CycleGan/WJF/5.Project/SkullStripping/TrainNp/test/51-62/1.2 ct_sckull_delete40'
ct_dir = r'D:\code\1101\0000070252_nii'
sct_dir = r'D:\code\1101\0000070252_nii'


ct_list = sorted(os.listdir(ct_dir))
sct_list = sorted(os.listdir(sct_dir))

psnr_result_total = []
ssim_result_total = []
mae_result_total_bone = []
mae_result_total_tissue = []

for i in range(len(ct_list)):
    print("----------------------------------")
    loss = 0
    name = ct_list[i].split('.')[0]
    ct_path = os.path.join(ct_dir, ct_list[i])
    sct_path = os.path.join(sct_dir, sct_list[i])

    sub_ctimg_path = os.path.join(ct_path)
    sub_ctimg_path = 'D:/code/1101/0000070252_nii/CT_Tougu.nii.gz'
    sub_ctimg_ = sitk.ReadImage(sub_ctimg_path)
    sub_ctimg_npy = sitk.GetArrayFromImage(sub_ctimg_)
    # print(sct_path)
    sub_sctimg_path = os.path.join(sct_path)
    sub_sctimg_path = 'D:/code/1101/0000070252_nii/SCT_Tougu.nii.gz'
    sub_sctimg_ = sitk.ReadImage(sub_sctimg_path)
    sub_sctimg_npy = sitk.GetArrayFromImage(sub_sctimg_)
    hausdorff_list = []
    dice_list = []
    jaccard_List = []
    for j in range(sub_ctimg_npy.shape[0]):
        ct_Label = sub_ctimg_npy[j,:,:]
        sct_Label = sub_sctimg_npy[j,:,:]
        hausdorff_list.append(hausdorff_distance(ct_Label, sct_Label, distance='euclidean'))
        if not (np.sum(sct_Label) + np.sum(ct_Label))==0:
            exp = dice_coefficient(sct_Label,ct_Label)
        else:
            exp = 1
        dice_list.append(exp)
        if not np.isnan(CalSimJaccard(ct_Label,sct_Label)):
            jaccard_List.append(CalSimJaccard(ct_Label,sct_Label))
        else:
            jaccard_List.append(1)
    print("当前患者{} 的dice为 {}, jaccard为 {}, Hausdorff distance 为 {} ".format(name,np.sum(dice_list)/len(dice_list), np.sum(jaccard_List)/len(jaccard_List) ,np.sum(hausdorff_list)/len(hausdorff_list)))
    # print(f"Hausdorff distance test: {np.sum(hausdorff_list)/len(hausdorff_list)}")
    # break