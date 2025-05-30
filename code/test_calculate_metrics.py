import os
import argparse
import numpy as np
import SimpleITK as sitk
# from metrics import *
from skimage.metrics import structural_similarity,peak_signal_noise_ratio

def psnr_3d(gt, pr):
    mask=np.zeros(gt.shape)
    mask[gt!=-1000]=1
    # psnr_fore_3d=peak_signal_noise_ratio(gt,pr,data_range=4000)
    psnr_fore_3d = peak_signal_noise_ratio(gt[mask > 0.5], pr[mask > 0.5], data_range=4000)
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

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_dir', dest='dataset_dir',
    #                     default='D:/code/1101/1101/dataset/',
    #                     help='path of the dataset')

    parser.add_argument('--result_path', dest='result_path',
                        default='D:/code/1101/1101/code/data0527/train_data_result/',
                        help='results are saved here')

    parser.add_argument('--load_size0', dest='load_size0', type=int, default=520, help='scale images to this size')
    parser.add_argument('--load_size1', dest='load_size1', type=int, default=520, help='scale images to this size')
    parser.add_argument('--fine_size0', dest='fine_size0', type=int, default=512, help='then crop to this size')
    parser.add_argument('--fine_size1', dest='fine_size1', type=int, default=512, help='then crop to this size')

    args = parser.parse_args()

    print('Start Metrics Calculation')

    result_path = os.path.join(args.result_path)
    center = 'D'
    center_result_path = os.path.join(result_path, 'result/D')
    nii_path = os.path.join(center_result_path, 'nii/')
    gt_nii_path = os.path.join(result_path, 'result/D/nii/gt/')
    syn_nii_path = os.path.join(result_path, 'result/D/nii/syn/')
    print(f"GT path: {gt_nii_path}")
    # 检查路径是否存在
    if not os.path.exists(gt_nii_path) or not os.path.exists(syn_nii_path):
        print("Error: GT or synthetic image paths do not exist!")
        print(f"GT path: {gt_nii_path}")
        print(f"Syn path: {syn_nii_path}")
        return

    # 获取skull数据路径
    # test_data_path = os.path.join(args.dataset_dir, 'test22')
    # CT_test = os.path.join(test_data_path, 'CT')
    # skull_test = os.path.join(test_data_path, 'final_skull')

    # 获取所有subject
    subs = os.listdir(gt_nii_path)

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
        gt_file = os.path.join(gt_nii_path, sub )
        syn_file = os.path.join(syn_nii_path, sub)

        if not os.path.exists(gt_file) or not os.path.exists(syn_file):
            print(f"Warning: Files not found for subject {syn_file }, skipping...")
            continue

        ct_gt_nii = sitk.ReadImage(gt_file)
        ct_gt = sitk.GetArrayFromImage(ct_gt_nii)

        ct_syn_nii = sitk.ReadImage(syn_file)
        ct_syn_aug = sitk.GetArrayFromImage(ct_syn_nii)

        # 读取skull数据用于计算分区指标
        # skull_sub_path = os.path.join(skull_test, sub)
        # skullimg = sitk.ReadImage(skull_sub_path)
        # skullimg = sitk.GetArrayFromImage(skullimg)
        # skullimg = np.flip(skullimg, axis=1)

        # 处理skull数据以匹配预测图像的尺寸
        skull_all = []
        tissu_all = []

        # for i in range(min(skullimg.shape[0] - 40, ct_gt.shape[0])):
        #     img_skull_slice = skullimg[i + 30]
        #
        #     # 进行相同的padding操作 (与生成预测时保持一致)
        #     padB_size0 = args.load_size0 - img_skull_slice.shape[0]
        #     padB_size1 = args.load_size1 - img_skull_slice.shape[1]
        #
        #     img_skull_slice = np.pad(img_skull_slice,
        #                              ((int(padB_size0 // 2), int(padB_size0) - int(padB_size0 // 2)),
        #                               (int(padB_size1 // 2), int(padB_size1) - int(padB_size1 // 2))),
        #                              mode='constant', constant_values=0)
        #
        #     # 进行相同的cropping操作 (从520x520裁剪到512x512)
        #     crop_start0 = (args.load_size0 - args.fine_size0) // 2
        #     crop_end0 = crop_start0 + args.fine_size0
        #     crop_start1 = (args.load_size1 - args.fine_size1) // 2
        #     crop_end1 = crop_start1 + args.fine_size1
        #
        #     img_skull_slice = img_skull_slice[crop_start0:crop_end0, crop_start1:crop_end1]
        #
        #     # 创建tissue mask
        #     current_gt_slice = ct_gt[i]
        #     mask = np.zeros(current_gt_slice.shape)
        #     mask[current_gt_slice > -990] = 1
        #     tissu_mask = mask - img_skull_slice
        #
        #     skull_all.append(img_skull_slice.astype('int16'))
        #     tissu_all.append(tissu_mask.astype('int16'))
        #
        # skull_all = np.array(skull_all)
        # tissu_all = np.array(tissu_all)

        # 确保数组长度匹配
        # min_len = min(len(ct_gt), len(skull_all))
        # ct_gt = ct_gt[:min_len]
        # ct_syn_aug = ct_syn_aug[:min_len]
        # skull_all = skull_all[:min_len]
        # tissu_all = tissu_all[:min_len]

        # print(f"Processing {min_len} slices for subject {sub[:2]}")
        # print(f"CT GT shape: {ct_gt.shape}")
        # print(f"CT Syn shape: {ct_syn_aug.shape}")
        # print(f"Skull shape: {skull_all.shape}")
        # print(f"Tissue shape: {tissu_all.shape}")

        # 计算3D指标
        ssim_3d_result = ssim_3d(ct_gt, ct_syn_aug)
        psnr_fore_3d_result = psnr_3d(ct_gt, ct_syn_aug)
        # mae_result, mae_fore = mae_3d(ct_gt, ct_syn_aug)
        mae_3d_result, mae_3d_result_fore, mae_bone, mae_tissue = mae_3d(ct_gt, ct_syn_aug)
        # # 计算骨骼和组织的MAE
        # bone_mask = skull_all > 0.5
        # tissue_mask = tissu_all > 0.5
        #
        # if np.sum(bone_mask) > 0:
        #     mae_bone = np.average(np.abs(ct_syn_aug[bone_mask] - ct_gt[bone_mask]))
        # else:
        #     mae_bone = 0.0
        #     print(f"Warning: No bone pixels found for subject {sub[:2]}")
        #
        # if np.sum(tissue_mask) > 0:
        #     mae_tissu = np.average(np.abs(ct_syn_aug[tissue_mask] - ct_gt[tissue_mask]))
        # else:
        #     mae_tissu = 0.0
        #     print(f"Warning: No tissue pixels found for subject {sub[:2]}")

        # 累加指标
        ssims_3d += ssim_3d_result
        psnrs_fore_3d += psnr_fore_3d_result
        maes_3d += mae_3d_result
        maes_fore_3d += mae_3d_result_fore
        maes_bone_3d += mae_bone
        maes_tissue_3d += mae_tissue

        nums += 1

        print(f"Subject {sub} - SSIM: {ssim_3d_result:.4f}, PSNR: {psnr_fore_3d_result:.4f}, MAE: {mae_3d_result:.4f}, MAE_FORE: {mae_3d_result_fore:.4f},MAE_Bone: {mae_bone:.4f}, MAE_Tissue: {mae_tissue:.4f}")
        # print(f"Subject {sub} - MAE_Bone: {mae_bone:.4f}, MAE_Tissue: {mae_tissue:.4f}")

    # 计算平均指标
    if nums > 0:
        ssims_3d /= nums
        psnrs_fore_3d /= nums
        maes_3d /= nums
        maes_fore_3d /= nums
        maes_bone_3d /= nums
        maes_tissue_3d /= nums

        print(f"\n=== Final Results for {center} ===")
        print(f"Number of subjects: {nums}")
        print(
            'ssim_3d: {:.4f} | psnr_3d_fore: {:.4f} | mae_3d: {:.4f} | mae_3d_fore: {:.4f} | mae_bone: {:.4f} | mae_tissu: {:.4f}'.format(
                ssims_3d, psnrs_fore_3d, maes_3d, maes_fore_3d, maes_bone_3d, maes_tissue_3d
            )
        )

        # 保存结果到文件
        results_file = os.path.join(center_result_path, 'metrics_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"Center: {center}\n")
            f.write(f"Number of subjects: {nums}\n")
            f.write(f"SSIM_3D: {ssims_3d:.4f}\n")
            f.write(f"PSNR_3D_FORE: {psnrs_fore_3d:.4f}\n")
            f.write(f"MAE_3D: {maes_3d:.4f}\n")
            f.write(f"MAE_3D_FORE: {maes_fore_3d:.4f}\n")
            f.write(f"MAE_BONE: {maes_bone_3d:.4f}\n")
            f.write(f"MAE_TISSUE: {maes_tissue_3d:.4f}\n")

        print(f"Results saved to: {results_file}")
    else:
        print("No valid subjects found for metric calculation!")


if __name__ == "__main__":
    main()
