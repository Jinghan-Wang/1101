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


# ct_dir = r'D:\code\1101\1101\code\data0527\train_data_result\result_200epoch\D\nii\gt'
ct_dir = r'D:\code\1101\1101\code\data0527\test_data_result\result\D\nii\gt'
# sct_dir = r'D:\code\1101\1101\code\data0527\train_data_result\result_200epoch\D\nii\syn'
sct_dir = r'D:\code\1101\1101\code\data0527\test_data_result\result\D\nii\syn'

subs = os.listdir(ct_dir)

# 初始化指标累加器
nums = 0
psnrs_fore_3d = 0.0
ssims_3d = 0.0
maes_3d = 0.0
maes_fore_3d = 0
maes_bone_3d = 0
maes_tissue_3d = 0

for sub in subs:
    # print(f"Calculating metrics for subject: {sub}")

    # 读取GT和合成图像
    ct_path = os.path.join(ct_dir, sub)
    sct_path = os.path.join(sct_dir, sub)


    sub_ctimg_ = sitk.ReadImage(ct_path)
    sub_ctimg_npy = sitk.GetArrayFromImage(sub_ctimg_)
    sub_sctimg_ = sitk.ReadImage(sct_path)
    sub_sctimg_npy = sitk.GetArrayFromImage(sub_sctimg_)

    # # 显示第十张切片
    # slice_index = 30  # 第十张切片索引
    # if sub_ctimg_npy.shape[0] > slice_index:  # 确保有足够的切片
    #     plt.figure(figsize=(10, 8))
    #     plt.imshow(sub_ctimg_npy[slice_index], cmap='gray')
    #     plt.colorbar()
    #     plt.title(f'Subject: {sub} - GT Slice {slice_index}')
    #     plt.show()
    #
    #     # 显示合成图像的第十张切片
    # if sub_sctimg_npy.shape[0] > slice_index:  # 确保有足够的切片
    #     plt.figure(figsize=(10, 8))
    #     plt.imshow(sub_sctimg_npy[slice_index], cmap='gray')
    #     plt.colorbar()
    #     plt.title(f'Subject: {sub} - Synthetic Slice {slice_index}')
    #     plt.show()

    # [修改] 检查是否有足够的切片
    if sub_ctimg_npy.shape[0] <= 20:
        print(f"警告: {sub} 的切片数量不足 ({sub_ctimg_npy.shape[0]}层), 跳过该患者")
        continue

    # psnr_3d_result = psnr_3d(sub_ctimg_npy, sub_sctimg_npy)
    # ssim_3d_result = ssim_3d(sub_ctimg_npy, sub_sctimg_npy)
    # mae_3d_result, mae_3d_result_fore, mae_bone, mae_tissue = mae_3d(sub_ctimg_npy, sub_sctimg_npy)
    # [修改] 跳过前20层切片
    sub_ctimg_npy_skip = sub_ctimg_npy[30:, :, :]
    print('sub_ctimg_npy_skip',sub_ctimg_npy_skip.shape)
    sub_sctimg_npy_skip = sub_sctimg_npy[30:, :, :]

    # [修改] 使用跳过前20层的数据计算指标
    psnr_3d_result = psnr_3d(sub_ctimg_npy_skip, sub_sctimg_npy_skip)
    ssim_3d_result = ssim_3d(sub_ctimg_npy_skip, sub_sctimg_npy_skip)
    mae_3d_result, mae_3d_result_fore, mae_bone, mae_tissue = mae_3d(sub_ctimg_npy_skip, sub_sctimg_npy_skip)

    ssims_3d += ssim_3d_result
    psnrs_fore_3d += psnr_3d_result
    maes_3d += mae_3d_result
    maes_fore_3d += mae_3d_result_fore
    maes_bone_3d += mae_bone
    maes_tissue_3d += mae_tissue

    nums += 1

    print("{} patient ssim:{} psnr:{}  mae_result:{}  mae_fore:{} mae_bone:{} mae_tissue:{} ".format(sub,ssim_3d_result,psnr_3d_result,mae_3d_result,mae_3d_result_fore,mae_bone,mae_tissue))
#

# 计算平均指标
if nums > 0:
    ssims_3d /= nums
    psnrs_fore_3d /= nums
    maes_3d /= nums
    maes_fore_3d /= nums
    maes_bone_3d /= nums
    maes_tissue_3d /= nums


    print(f"Number of subjects: {nums}")
    print(
        'ssim_3d: {:.4f} | psnr_3d_fore: {:.4f} | mae_3d: {:.4f} | mae_3d_fore: {:.4f} | mae_bone: {:.4f} | mae_tissu: {:.4f}'.format(
            ssims_3d, psnrs_fore_3d, maes_3d, maes_fore_3d, maes_bone_3d, maes_tissue_3d
        )
    )
