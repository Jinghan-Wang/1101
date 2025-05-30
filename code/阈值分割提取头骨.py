import SimpleITK as sitk
import os
import pandas as pd
import numpy as np
import cv2
import pydicom
from skimage import morphology
from FuncCutBG import save_nifti,keep_largest_connected_component,find_largest_contour


"""从CT数据中提取头骨轮廓标签"""

def background_noise(input):
    # thres_val = -900
    thres_val = -800
    k_size = 10
    input_mr_vol = input.astype('float32')
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
    masked_mr[mask == 1] = input_mr_vol[mask == 1]
    return masked_mr

def _get_instance_number(dicom_path):
    img_reader = sitk.ImageFileReader()
    img_reader.SetFileName(dicom_path)
    img_reader.LoadPrivateTagsOn()
    img_reader.ReadImageInformation()
    number_str = img_reader.GetMetaData('0020|0013')  # 获取Instance Number
    return int(number_str)

def get_slice(dicom_path):
    # 构建DICOM序列文件阅读器，并获取文件名列表
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
    # 按Instance Number排序
    r = []
    for name in dicom_names:
        r.append({"instance_number": _get_instance_number(name), "dcm_name": name})
    r = pd.DataFrame(r)
    # r = r.sort_values("instance_number")
    r = r.sort_values("instance_number", ascending=False)   # 反向排序

    # 获取排序后的切片路径
    return tuple(r["dcm_name"])
def get_slice_thickness(dicom_file):
    """从DICOM文件中获取切片厚度和切片间距"""
    ds = pydicom.dcmread(dicom_file)
    slice_thickness = None
    spacing_between_slices = None
    if 'SliceThickness' in ds:
        slice_thickness = float(ds.SliceThickness)
    if 'SpacingBetweenSlices' in ds:
        spacing_between_slices = float(ds.SpacingBetweenSlices)
    return slice_thickness, spacing_between_slices
def calculate_slice_distance(dicom_dir):
    """计算DICOM序列中相邻切片的距离"""
    # dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)]
    dicom_files.sort()  # 确保文件按照序列顺序排序
    if len(dicom_files) < 2:
        raise ValueError("DICOM目录中至少需要两个文件来计算切片距离")
    # 读取第一个文件的间距信息
    slice_thickness, spacing_between_slices = get_slice_thickness(dicom_files[0])
    # 如果存在`SpacingBetweenSlices`，优先使用
    if spacing_between_slices is not None:
        return spacing_between_slices
    elif slice_thickness is not None:
        return slice_thickness
    else:
        raise ValueError("无法从DICOM文件中获取切片厚度或切片间距信息")

def skullStripping(CT_dir,patient_name):
    # print(CT_dir)
    # # 给传进来的CT序列按照标签信息排序
    # file_list = get_slice(CT_dir)
    #
    # # 获取相邻切片之间的距离作为新的spacing
    # try:
    #     slice_distance = calculate_slice_distance(CT_dir)
    #     # print(f"相邻切片之间的距离: {slice_distance} mm")
    # except ValueError as e:
    #     print(e)
    #
    # new_file_list = []
    # # 将排序之后的CT dcm序列存为nii
    # save_skullstripping = np.zeros((len(file_list),512,512))
    # num = 0
    # for file in file_list:
    #     new_file_list.append(file)
    #     image = sitk.ReadImage(file)
    #     old_spacing = image.GetSpacing()
    #     image_array = sitk.GetArrayFromImage(image)
    #     image_array = background_noise(image_array)
    #     save_skullstripping[num, :, :] = image_array
    #     num += 1
    # # print(old_spacing)
    # new_spacing = (old_spacing[0],old_spacing[1],slice_distance)
    # image_ = sitk.GetImageFromArray(save_skullstripping)
    # image_.SetSpacing(new_spacing)
    #
    # save_image_path = './image/'+patient_name+".nii.gz"
    # sitk.WriteImage(image_, save_image_path)


    img_IMG = sitk.ReadImage(CT_dir)
    old_spacing = img_IMG.GetSpacing()
    # npy_ = sitk.GetArrayFromImage(img_IMG)
    new_spacing = (old_spacing)

    itkimage = img_IMG
    Spacing = itkimage.GetSpacing()
    Origin = itkimage.GetOrigin()
    Direction = itkimage.GetDirection()
    Data = sitk.GetArrayFromImage(itkimage)
    DataC = Data.copy()
    Result = np.zeros_like(Data)
    for i in range(Data.shape[0]):
        Slice = Data[i]
        SliceOrg = Slice.copy()
        Slice[Slice < -400] = 0
        Slice[Slice != 0] = 1
        Slice = keep_largest_connected_component(Slice)
        Slice = cv2.convertScaleAbs(Slice)
        SliceMask = find_largest_contour(Slice)
        SliceResult = SliceMask * SliceOrg  # +(SliceMask-1)*(-100)
        otherPart = (SliceMask - 1) * (-100)
        otherPart[otherPart < -999] = -999
        Result[i, :, :] = otherPart + SliceResult
    nii_ct = save_nifti(Result, './Cut.nii.gz', Spacing, Origin, Direction)
    nii_ct = sitk.GetArrayFromImage(nii_ct)
    save_bias_skull = np.zeros_like(nii_ct)
    new_kernel = np.ones((3, 3))  # 核
    for i in range(nii_ct.shape[0]):
        temp = nii_ct[i, :, :]
        # temp[temp < 650] = 0
        # temp[temp > 0] = 1

        temp[temp < 150] = 0
        temp[temp >= 150] = 1

        # temp = cv2.dilate(temp, new_kernel, iterations=1)
        # temp = cv2.erode(temp, new_kernel, iterations=1)
        save_bias_skull[i, :, :] = temp
    save_bias_skull_Image = sitk.GetImageFromArray(save_bias_skull)
    save_bias_skull_Image.SetSpacing(new_spacing)
    save_bias_skull_Image.SetDirection(itkimage.GetDirection())
    # save_path = './label/'+patient_name+".gz"
    # save_path = 'C:/Users/OUR/Desktop/SkullStrippingTestPatient/SkullStrippingTest_data/final_test/sct_getskullstripping/'+patient_name+".gz"
    # save_path = 'C:/Users/OUR/Desktop/SkullStrippingTestPatient/SkullStrippingTest_data/final_test/real_ct_getskullstripping/'+patient_name+".gz"
    save_path = 'C:/code/DeepLearning/Dayi/TensorRT Deployment Process/02 MRI2sCT/3channelModel/resister_20241024/06 calculate/01 mr2ct/4 pre_skull/'+patient_name
    print("save_path :",save_path)
    sitk.WriteImage(save_bias_skull_Image, save_path)


if __name__=='__main__':

    # flor_dir = r'C:\code\DeepLearning\Dayi\1 segment\SkullStripping\image_CT'
    # flor_dir = r'C:\Users\OUR\Desktop\SkullStrippingTestPatient\SkullStrippingTest_data\final_test\real_ct'
    # flor_dir = r'C:\Users\OUR\Desktop\SkullStrippingTestPatient\SkullStrippingTest_data\final_test\sct'
    # flor_dir = r'C:\code\DeepLearning\Dayi\1 segment\SkullStripping\skulltest\SCT'
    flor_dir = r'C:\code\DeepLearning\Dayi\TensorRT Deployment Process\02 MRI2sCT\3channelModel\resister_20241024\06 calculate\01 mr2ct\3 pre_delete40'
    """测试数据"""
    # flor_dir = r'E:\CT _dataset\3 MRsCT\02 taihe data'
    flo_list = os.listdir(flor_dir)
    for i in range(len(flo_list)):
        sub_dir = os.path.join(flor_dir,flo_list[i])
        patient_name = sub_dir.split("\\")[-1]
        skullStripping(sub_dir,patient_name)
