from skimage.metrics import structural_similarity,peak_signal_noise_ratio
from sklearn.metrics import mean_absolute_error
import numpy as np


def ssim_score(y_true, y_pred):
    ssim_sum=0
    y_true=y_true.cpu().detach().numpy()
    y_pred=y_pred.cpu().detach().numpy()
    for i in range(y_true.shape[0]):
        score=structural_similarity(y_true[i][0], y_pred[i][0],data_range=y_true[i][0].max())
        ssim_sum+=score
    ssim_mean=ssim_sum/y_true.shape[0]
    return ssim_mean

def ssim_loss(y_true, y_pred):
	loss_ssim = 1.0 - ssim_score(y_true, y_pred)
	return loss_ssim

def psnr(reference_image, img2):
    reference_image=reference_image.cpu().numpy()
    img2=img2.cpu().detach().numpy()
    psnr_sum=0
    for i in range(reference_image.shape[0]):
        psnr=peak_signal_noise_ratio(reference_image[i][0], img2[i][0])
        psnr_sum=psnr_sum+psnr
    psnr_mean=psnr_sum/reference_image.shape[0]
    return psnr_mean

def mae(reference_image, img2):
    reference_image=reference_image.cpu().numpy()
    img2=img2.cpu().detach().numpy()
    mae_sum=0
    for i in range(reference_image.shape[0]):
        mae=mean_absolute_error(reference_image[i][0], img2[i][0])
        mae_sum=mae_sum+mae
    mae_mean=mae_sum/reference_image.shape[0]
    return mae_mean

def psnr_3d(gt, pr):
    mask=np.zeros(gt.shape)
    mask[gt!=-1000]=1
    # psnr_fore_3d=peak_signal_noise_ratio(gt,pr,data_range=4000)
    psnr_fore_3d = peak_signal_noise_ratio(gt[mask > 0.5], pr[mask > 0.5], data_range=4000)
    return psnr_fore_3d

def ssim_3d(gt, pr):
    ssim_result=structural_similarity(gt,pr,data_range=4000)
    return ssim_result

def mae_3d0(gt, pr):
    mask=np.zeros(gt.shape)
    mask[gt!=-1000]=1
    mae_result=np.average(np.average(np.average(np.abs(pr - gt))))
    mae_fore=np.average(np.average(np.average(np.abs(pr[mask > 0.5] - gt[mask > 0.5]))))
    return mae_result,mae_fore

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
#