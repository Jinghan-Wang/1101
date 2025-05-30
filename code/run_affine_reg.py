import ants
import os
"""
对核磁和CT进行配准，并且保存纠正过位置之后的nii
"""

# root_path = r'/data/CycleGan/Data/Register_data'
# data_num = 1
ct_root_path = r'/data/ZZJ/code/1101/dataset/train_taihe/CT'
mri_root_path = r'/data/ZZJ/code/1101/dataset/train_taihe/MR'

ct_save_path = r'/data/ZZJ/code/1101/dataset/Registration/CT'
mri_save_path = r'/data/ZZJ/code/1101/dataset/Registration/MR'
# 确保保存目录存在
os.makedirs(ct_save_path, exist_ok=True)
os.makedirs(mri_save_path, exist_ok=True)

ct_list = os.listdir(ct_root_path)
for folder in ct_list:

    ct_path = os.path.join(ct_root_path, folder)
    mri_path = os.path.join(mri_root_path, folder)

    # 检查文件路径是否存在
    if not os.path.isfile(ct_path) or not os.path.isfile(mri_path):
        print(f"Warning: File {folder} does not exist in both CT and MRI directories.")
        continue

    fixed_img = ants.image_read(mri_path)
    moving_img = ants.image_read(ct_path)
    print('ct_path',ct_path)
    print('mri_path', mri_path)
    # 执行图像配准，将 CT 图像（moving_img）与 MR 图像（fixed_img）进行配准，使用刚体变换
    ct2mr = ants.registration(fixed_img,moving_img,type_of_transform='Rigid')
    # print("ct2mr  ",ct2mr)
    #应用配准变换，将变换应用于移动图像（CT），生成变换后的图像。
    warped_moving_img = ants.apply_transforms(fixed_img,moving_img,transformlist=ct2mr['fwdtransforms'],defaultvalue=-1000)
    #执行反向图像配准，将 MR 图像与 CT 图像进行配准。
    mr2ct = ants.registration(moving_img,fixed_img,type_of_transform='Rigid')
    print("--------------------------------")
    # print("mr2ct  ",mr2ct)
    #应用反向配准变换，将变换应用于固定图像（MR）
    warped_fixed_img = ants.apply_transforms(moving_img,fixed_img,transformlist=mr2ct['fwdtransforms'],defaultvalue=-1000)
    #应用反向配准变换，将变换应用于固定图像（MR）
    # 保存配准后的图像
    ct_save_file = os.path.join(ct_save_path, folder)
    mri_save_file = os.path.join(mri_save_path, folder)
    ants.image_write(warped_moving_img, ct_save_file)
    ants.image_write(warped_fixed_img, mri_save_file)
    print(f"Saved registered CT to: {ct_save_file}")
    print(f"Saved registered MR to: {mri_save_file}")

    # # 获取正向变换的参数
    # forward_transform_parameters = ct2mr['fwdtransforms'][0]['parameters']
    # print("forward_transform_parameters  ",forward_transform_parameters)
    # 获取反向变换的参数
    # inverse_transform_parameters = mr2ct['fwdtransforms'][0]['parameters']
    # print("inverse_transform_parameters  ",inverse_transform_parameters)