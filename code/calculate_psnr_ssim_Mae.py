from skimage.metrics import structural_similarity,peak_signal_noise_ratio
from sklearn.metrics import mean_absolute_error
import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt

"""
生成任务的一些指标计算
PSNR、SSIM、MAE
"""



def psnr_3d(gt, pr):
    mask=np.zeros(gt.shape)
    mask[gt!=-1000]=1
    # psnr_fore_3d = peak_signal_noise_ratio(gt, pr, data_range=4000)
    psnr_fore_3d=peak_signal_noise_ratio(gt[mask > 0.5],pr[mask > 0.5],data_range=4000)

    return psnr_fore_3d

def ssim_3d(gt, pr):
    ssim_result=structural_similarity(gt,pr,data_range=4000)
    return ssim_result

def mae_3d(gt, pr):
    mask=np.zeros(gt.shape)
    mask_bone=np.zeros(gt.shape)
    mask_tissue=np.zeros(gt.shape)

    mask[gt!=-1000]=1
    mask_bone = gt.copy()
    mask_tissue = gt.copy()

    mask_bone[mask_bone < 150] = 0
    mask_bone[mask_bone >= 150] = 1

    mask_tissue[mask_tissue <= 150] = 1
    mask_tissue[mask_tissue > 150] = 0

    mae_result=np.average(np.average(np.average(np.abs(pr - gt))))
    mae_fore=np.average(np.average(np.average(np.abs(pr[mask > 0.5] - gt[mask > 0.5]))))
    mae_bone=np.average(np.average(np.average(np.abs(pr[mask_bone > 0.5] - gt[mask_bone > 0.5]))))
    mae_tissue=np.average(np.average(np.average(np.abs(pr[mask_tissue > 0.5] - gt[mask_tissue > 0.5]))))
    return mae_result,mae_fore,mae_bone,mae_tissue

# ct_dir = r'D:\code\1101\1101\code\test_taihe_finetune_96epoch_result22\result\D\nii\gt'
ct_dir = r'D:\code\1101\1101\code\test_taihe_finetune_96epoch_result22\result\D\nii\gt'


sct_dir = r'D:\code\1101\1101\code\test_taihe_finetune_96epoch_result22\result\D\nii\syn'



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
    #sub_ctimg_path = r'D:\code\1101\0000070252_nii\CT_NoToujia.nii.gz'
    sub_ctimg_ = sitk.ReadImage(sub_ctimg_path)
    sub_ctimg_npy = sitk.GetArrayFromImage(sub_ctimg_)
    # print(sct_path)
    sub_sctimg_path = os.path.join(sct_path)
    #sub_sctimg_path = r'D:\code\1101\0000070252_nii\SCT.nii.gz'
    sub_sctimg_ = sitk.ReadImage(sub_sctimg_path)
    sub_sctimg_npy = sitk.GetArrayFromImage(sub_sctimg_)

    # psnr_3d_result = psnr_3d(sub_ctimg_npy, sub_sctimg_npy)
    # ssim_3d_result = ssim_3d(sub_ctimg_npy, sub_sctimg_npy)
    # mae_3d_result, mae_3d_result_fore, mae_bone, mae_tissue = mae_3d(sub_ctimg_npy, sub_sctimg_npy)

    # new_sub_ctimg_npy = np.ones((126,512,512))*-999
    # new_sub_sctimg_npy = np.ones((126,512,512))*-999

    # new_sub_ctimg_npy = np.ones((107, 512, 512)) * -999
    # new_sub_sctimg_npy = np.ones((107, 512, 512)) * -999

    # new_sub_ctimg_npy = np.ones((107, 512, 512)) * -999
    # new_sub_sctimg_npy = np.ones((107, 512, 512)) * -999

    new_list = []
    j = 0

    # for i in range(sub_ctimg_npy.shape[0]):
    #     if sub_ctimg_npy[i,:,:].max() > 0:
    #         new_sub_ctimg_npy[j,:,:] = sub_ctimg_npy[i,:,:]
    #         new_sub_sctimg_npy[j,:,:] = sub_sctimg_npy[i,:,:]
    #         j = j+1
    # print('aaa', new_sub_sctimg_npy.shape)
    # new_sub_ctimg_img = sitk.GetImageFromArray(new_sub_ctimg_npy)
    # new_sub_sctimg_img = sitk.GetImageFromArray(new_sub_sctimg_npy)

    psnr_3d_result = psnr_3d(sub_ctimg_npy, sub_sctimg_npy)
    ssim_3d_result = ssim_3d(sub_ctimg_npy, sub_sctimg_npy)
    mae_3d_result, mae_3d_result_fore, mae_bone, mae_tissue = mae_3d(sub_ctimg_npy, sub_sctimg_npy)

    # psnr_3d_result = psnr_3d(new_sub_ctimg_npy, new_sub_sctimg_npy)
    # ssim_3d_result = ssim_3d(new_sub_ctimg_npy, new_sub_sctimg_npy)
    # mae_3d_result,mae_3d_result_fore,mae_bone,mae_tissue = mae_3d(new_sub_ctimg_npy, new_sub_sctimg_npy)
    print("{} patient psnr:{}   ssim:{}   mae_result:{}  mae_fore:{} mae_bone:{} mae_tissue:{} ".format(name,psnr_3d_result,ssim_3d_result,mae_3d_result,mae_3d_result_fore,mae_bone,mae_tissue))
#
    break