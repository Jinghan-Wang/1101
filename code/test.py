import imageio
import scipy.io as sio
import os
import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import imageio
import scipy.io as sio
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 
import torch.optim as optim
from tqdm import tqdm
import SimpleITK as sitk
from model import Uformer
from dataloaders.BrainDataset_ori import load_data_3d
from metrics import *
import time
from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", default='0', help="specify the GPU(s)", type=str)

parser.add_argument('--dataset_dir', dest='dataset_dir',
                default='D:/code/1101/1101/dataset/',
                help='path of the dataset')

parser.add_argument("--time_str", default='2024_10_26_18_15_02', type=str)

parser.add_argument('--epoch', dest='epoch', type=int, default=30, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--threads', dest='threads', type=int, default=0, help='# num_workers')

parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='initial gen learning rate for adam')
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-4, help='weight_decay')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--w', type=float, default=1.0, help='skull w')

parser.add_argument('--load_size0', dest='load_size0', type=int, default=520, help='scale images to this size')
parser.add_argument('--load_size1', dest='load_size1', type=int, default=520, help='scale images to this size')
parser.add_argument('--fine_size0', dest='fine_size0', type=int, default=512, help='then crop to this size')
parser.add_argument('--fine_size1', dest='fine_size1', type=int, default=512, help='then crop to this size')

parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='sample', help='sample are saved here')

parser.add_argument('--result_path', dest='result_path', default='test_taihe_finetune_96epoch_result22/',
                    help='results are saved here')

parser.add_argument('--test_dir', dest='test_dir', default='test', help='test sample are saved here')

args = parser.parse_args()

print('Start Test')
time_str=args.time_str
print(time_str)
result_path = os.path.join(args.result_path)
test_path = result_path
# test_checkpoint = os.path.join(test_path, 'checkpoint/last_epoch_weights.pth')
test_checkpoint = 'D:/code/1101/1101/epoch_96_weights.pth'
center = 'D'
# result_dir = os.path.join(test_path, 'test12/' + center)
result_path = os.path.join(test_path, 'result')
if not os.path.exists(result_path):
    os.makedirs(result_path)
center_result_path = os.path.join(result_path, center)

if not os.path.exists(center_result_path):
    os.makedirs(center_result_path)

# if not os.path.exists(result_dir):
#     os.makedirs(result_dir)

nii_path=os.path.join(center_result_path, 'nii/')
if not os.path.exists(nii_path):
    os.makedirs(nii_path)

gt_nii_path = os.path.join(nii_path, 'gt/')
syn_nii_path = os.path.join(nii_path, 'syn/')
if not os.path.exists(gt_nii_path):
    os.makedirs(gt_nii_path)
if not os.path.exists(syn_nii_path):
    os.makedirs(syn_nii_path)

test_data_path = os.path.join(args.dataset_dir, 'test22')
CT_test = os.path.join(test_data_path,  'CT')
print('CT_test',CT_test)
MR_test = os.path.join(test_data_path,  'MR')
skull_test = os.path.join(test_data_path, 'final_skull')
os.environ["CUDA_VISIBLE_DEVICES"] = args.device


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = Uformer(img_size=args.fine_size0, dd_in=3,embed_dim=16,depths=[3,3, 3, 3, 3,3, 3, 3, 3], win_size=8, token_projection='linear',
                                token_mlp='leff', modulator=True)
model = model.to(device)
model.load_state_dict(torch.load(test_checkpoint, map_location=device))
subs = os.listdir(CT_test)
nums = 0
psnrs_fore_3d = 0.0
ssims_3d = 0.0
maes_3d = 0.0
maes_fore_3d = 0
maes_bone_3d = 0
maes_tissue_3d = 0
model.eval()
for sub in subs:
    ct_sub_path = os.path.join(CT_test, sub)
    mr_sub_path = os.path.join(MR_test, sub)
    skull_sub_path = os.path.join(skull_test, sub)
    ctimg = sitk.ReadImage(ct_sub_path)
    ctimg = sitk.GetArrayFromImage(ctimg)
    ctimg=np.flip(ctimg,axis=1)
    mrimg = sitk.ReadImage(mr_sub_path)
    mrimg = sitk.GetArrayFromImage(mrimg)
    mrimg=np.flip(mrimg,axis=1)
    skullimg = sitk.ReadImage(skull_sub_path)
    skullimg = sitk.GetArrayFromImage(skullimg)
    skullimg=np.flip(skullimg,axis=1)

    ct_3d = []
    mr_3d = []
    skull_3d = []

    for i in range(ctimg.shape[0]-40):

        img_mr_slice = mrimg[i+30]
        img_ct_slice = ctimg[i+30]
        img_skull_slice = skullimg[i+30]

        img_mr_slice_before = mrimg[i+30-1]

        img_mr_slice_after = mrimg[i+30+1]

        img_ct_slice[img_ct_slice < -1000] = -1000
        img_ct_slice[img_ct_slice > 3000] = 3000

        img_mr_slice[img_mr_slice < -1000] = -1000
        img_mr_slice[img_mr_slice > 3000] = 3000

        img_mr_slice_before[img_mr_slice_before < -1000] = -1000
        img_mr_slice_before[img_mr_slice_before > 3000] = 3000

        img_mr_slice_after[img_mr_slice_after < -1000] = -1000
        img_mr_slice_after[img_mr_slice_after > 3000] = 3000

        img_A = img_ct_slice.copy() 
        img_B = img_mr_slice.copy()

        padA_size0 = args.load_size0 - img_A.shape[0]
        padA_size1 = args.load_size1 - img_A.shape[1]
        padB_size0 = args.load_size0 - img_B.shape[0]
        padB_size1 = args.load_size1 - img_B.shape[1]

        img_A = np.pad(img_A, ((int(padA_size0 // 2), int(padA_size0) - int(padA_size0 // 2)),
                                (int(padA_size1 // 2), int(padA_size1) - int(padA_size1 // 2))), mode='constant',
                        constant_values=-1000)
        img_B = np.pad(img_B, ((int(padB_size0 // 2), int(padB_size0) - int(padB_size0 // 2)),
                                (int(padB_size1 // 2), int(padB_size1) - int(padB_size1 // 2))), mode='constant',
                        constant_values=-1000 )
        
        img_mr_slice_before = np.pad(img_mr_slice_before, ((int(padB_size0 // 2), int(padB_size0) - int(padB_size0 // 2)),
                                (int(padB_size1 // 2), int(padB_size1) - int(padB_size1 // 2))), mode='constant',
                        constant_values=-1000)
        
        img_mr_slice_after = np.pad(img_mr_slice_after, ((int(padB_size0 // 2), int(padB_size0) - int(padB_size0 // 2)),
                                (int(padB_size1 // 2), int(padB_size1) - int(padB_size1 // 2))), mode='constant',
                        constant_values=-1000)
        
        img_skull_slice = np.pad(img_skull_slice, ((int(padB_size0 // 2), int(padB_size0) - int(padB_size0 // 2)),
                                (int(padB_size1 // 2), int(padB_size1) - int(padB_size1 // 2))), mode='constant',
                        constant_values=0)
        
        if (img_B.max()-img_B.min())>0 and (img_A.max()-img_A.min())>0:
            img_A = (img_A + 1000) / (3000 + 1000)
            img_B = (img_B + 1000) / (3000 + 1000)

            if (img_mr_slice_before.max()-img_mr_slice_before.min())>0:
                img_mr_slice_before = (img_mr_slice_before + 1000) / (3000 + 1000)
            else:
                img_mr_slice_before = img_B

            if (img_mr_slice_after.max()-img_mr_slice_after.min())>0:
                img_mr_slice_after = (img_mr_slice_after + 1000) / (3000 + 1000)
            else:
                img_mr_slice_after = img_B

            img_mr_slice_before=np.expand_dims(img_mr_slice_before,axis=0)
            img_B=np.expand_dims(img_B,axis=0)
            img_mr_slice_after=np.expand_dims(img_mr_slice_after,axis=0)

            img_MR=np.concatenate((img_mr_slice_before,img_B,img_mr_slice_after),axis=0)

            ct_3d.append(img_A) 
            mr_3d.append(img_MR)       
            skull_3d.append(img_skull_slice)
        
    test_loader = load_data_3d(args.batch_size, ct_3d, mr_3d,skull_3d, train=False, load_size0=args.load_size0,
                                load_size1=args.load_size1,
                                fine_size0=args.fine_size0, fine_size1=args.fine_size1,
                                num_workers=args.threads)
    ct_gt = []
    ct_syn_aug = []
    skull_all = []
    tissu_all = []

    for idx, batch in enumerate(test_loader):
        with torch.no_grad():
            CT_image, MR_image, skull_images= batch
            CT_image = CT_image.to(device)  
            MR_image = MR_image.to(device)  
            skull_images = skull_images.to(device) 
            

            if MR_image.max()>0:
                skull_image=skull_images.cpu().numpy()

                output = model(MR_image)
                fake_img = output.cpu().numpy()
                fake_img[fake_img < 0] = 0
                CT_images = CT_image.cpu().numpy()
                MR_images = MR_image.cpu().numpy()

                fake_img_255 = fake_img * 255
                
                temp = fake_img_255 / 255. * (3000. + 1000.) - 1000.
                gen_ct = temp
                ct_syn_aug.append(np.array(gen_ct).astype('int16').reshape(
                    [args.fine_size0, args.fine_size1]))

                gt_img_255 = CT_images * 255
             
                temp = gt_img_255 / 255. * (3000. + 1000.) - 1000.
                gt_ct = temp
                ct_gt.append(np.array(gt_ct).astype('int16').reshape([args.fine_size0, args.fine_size1])) 

                mask=np.zeros(gt_ct.shape)
                mask[gt_ct>-990]=1
                tissu_mask = mask-skull_image

                skull_all.append(np.array(skull_image).astype('int16').reshape([args.fine_size0, args.fine_size1])) 
                tissu_all.append(tissu_mask.astype('int16').reshape([args.fine_size0, args.fine_size1])) 

                # psnr_plt=peak_signal_noise_ratio(gt_ct[mask > 0.5],gen_ct[mask > 0.5],data_range=4000)
                # ssim_plt=structural_similarity(gt_ct[0,0],gen_ct[0,0],data_range=4000)
                # mae_fore_plt=np.average(np.average(np.average(np.abs(gen_ct[mask > 0.5] - gt_ct[mask > 0.5]))))
                # mae_plt=np.average(np.average(np.average(np.abs(gen_ct - gt_ct))))
                # mae_bone_plt = np.average(np.average(np.average(np.abs(gen_ct[skull_image > 0.5] - gt_ct[skull_image > 0.5]))))
                # mae_tissue_plt = np.average(np.average(np.average(np.abs(gen_ct[tissu_mask > 0.5] - gt_ct[tissu_mask > 0.5]))))

                fake_img=output.cpu().numpy()
                fake_img[fake_img < 0] = 0
                ctimgs=CT_image.cpu().numpy()
                mrimgs = MR_image[:,1,:,:].cpu().numpy()
                bone_imgs=(fake_img*skull_image)
                tissu_imgs=(fake_img*tissu_mask)
                res=fake_img-ctimgs

                nums += 1

                # aug_mr = np.array(mrimgs).reshape([args.fine_size0, args.fine_size1])
                # aug_mr = aug_mr  * 255
                #
                # ori_ct=np.array(ctimgs).reshape([args.fine_size0, args.fine_size1])
                # ori_ct = ori_ct  * 255
                #
                # gen_ct=np.array(fake_img).reshape([args.fine_size0, args.fine_size1])
                # gen_ct = gen_ct  * 255
                #
                # bone_ct=np.array(bone_imgs).reshape([args.fine_size0, args.fine_size1])
                # bone_ct = bone_ct  * 255
                #
                # tissu_ct=np.array(tissu_imgs).reshape([args.fine_size0, args.fine_size1])
                # tissu_ct = tissu_ct  * 255
                #
                # res_ct=np.array(res).reshape([args.fine_size0, args.fine_size1])
                # res_ct = res_ct  * 255
                #
                # fig = plt.figure(figsize=(12, 8))
                # gs = GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
                #
                # ax1 = plt.subplot(gs[0, 0])
                # ax1.imshow(aug_mr, cmap='gray')
                #
                # ax1.axis('off')
                # ax1.set_frame_on(False)
                #
                # ax2 = plt.subplot(gs[0, 1])
                # ax2.imshow(ori_ct, cmap='gray')
                #
                # ax2.axis('off')
                # ax2.set_frame_on(False)
                #
                # ax3 = plt.subplot(gs[0, 2])
                # ax3.imshow(gen_ct, cmap='gray')
                #
                # ax3.axis('off')
                # ax3.set_frame_on(False)
                #
                # ax3.text(15, 20, f'PSNR: {psnr_plt:.4f}', color='white', fontsize=10, ha='left')
                # ax3.text(15, 50, f'SSIM: {ssim_plt:.4f}', color='white', fontsize=10, ha='left')
                # ax3.text(15, 80, f'MAE: {mae_plt:.4f}', color='white', fontsize=10, ha='left')
                # ax3.text(15, 110, f'MAE_FORE: {mae_fore_plt:.4f}', color='white', fontsize=10, ha='left')
                #
                # ax4 = plt.subplot(gs[1, 0])
                # ax4.imshow(bone_ct, cmap='gray')
                #
                # ax4.axis('off')
                # ax4.set_frame_on(False)
                #
                # ax4.text(15, 20, f'bone_MAE: {mae_bone_plt:.4f}', color='white', fontsize=10, ha='left')
                #
                # ax5 = plt.subplot(gs[1, 1])
                # ax5.imshow(tissu_ct, cmap='gray')
                #
                # ax5.axis('off')
                # ax5.set_frame_on(False)
                #
                # ax5.text(15, 20, f'tissu_MAE: {mae_tissue_plt:.4f}', color='white', fontsize=10, ha='left')
                #
                # ax6 = plt.subplot(gs[1, 2])
                # ax6.imshow(res_ct, cmap='seismic', vmin=-1, vmax=1)
                #
                # ax6.axis('off')
                # ax6.set_frame_on(False)
                #
                # plt.tight_layout()
                #
                # plt.savefig('{}/{}_{}_aug.jpg'.format(result_dir, sub[:2], idx), dpi=300)


    ct_gt= np.array(ct_gt)
    ct_syn_aug= np.array(ct_syn_aug)

    skull_all= np.array(skull_all)
    tissu_all= np.array(tissu_all)

    ct_gt_nii=sitk.GetImageFromArray(ct_gt)
    sitk.WriteImage(ct_gt_nii,gt_nii_path+ sub[:2]+'.nii')
    ct_syn_nii=sitk.GetImageFromArray(ct_syn_aug)
    sitk.WriteImage(ct_syn_nii,syn_nii_path+ sub[:2]+'.nii')

    ssim_3d_result=ssim_3d(ct_gt,ct_syn_aug)
    psnr_fore_3d=psnr_3d(ct_gt,ct_syn_aug)
    mae_result,mae_fore=mae_3d(ct_gt,ct_syn_aug)

    mae_bone=np.average(np.average(np.average(np.abs(ct_syn_aug[skull_all > 0.5] - ct_gt[skull_all > 0.5]))))
    mae_tissu=np.average(np.average(np.average(np.abs(ct_syn_aug[tissu_all > 0.5] - ct_gt[tissu_all > 0.5]))))

    ssims_3d+=ssim_3d_result
    psnrs_fore_3d+=psnr_fore_3d
    maes_3d+=mae_result
    maes_fore_3d+=mae_fore

    maes_bone_3d +=mae_bone
    maes_tissue_3d +=mae_tissu


ssims_3d/=len(subs)
psnrs_fore_3d/=len(subs)
maes_3d/=len(subs)
maes_fore_3d/=len(subs)
maes_bone_3d/=len(subs)
maes_tissue_3d/=len(subs)

print(center)


print(
    'ssim_3d: {:.4f} | psnr_3d_fore: {:.4f} |mae_3d:{:.4f} |mae_3d_fore:{:.4f} |mae_bone:{:.4f} |mae_tissu:{:.4f}'.format(
        ssims_3d, psnrs_fore_3d, maes_3d, maes_fore_3d,maes_bone_3d,maes_tissue_3d
    )
)