import ants
import os
"""
step 2 配准  获取配准之后的核磁进行预测
执行
"""


# ct_dir = r'/data/CycleGan/Data/Register_data/chongqing_20250416/ct_nii'
# mr_dir = r'/data/CycleGan/Data/Register_data/chongqing_20250416/mr_nii'
# war_ct_dir = r'/data/CycleGan/Data/Register_data/chongqing_20250416/war_ct_nii'
# war_mr_dir = r'/data/CycleGan/Data/Register_data/chongqing_20250416/war_mr_nii'

ct_dir = r'D:/python_code/MRIGensCT/data/Zhangguohua/CTnii'
mr_dir = r'D:/python_code/MRIGensCT/data/Zhangguohua/MRnii'
war_ct_dir = r'D:/python_code/MRIGensCT/data/Zhangguohua/war_ct_nii'
war_mr_dir = r'D:/python_code/MRIGensCT/data/Zhangguohua/war_mr_nii'


ct_list = sorted(os.listdir(ct_dir))
sct_list = sorted(os.listdir(mr_dir))

for i in range(len(sct_list)):
    sub_ct_dir = os.path.join(ct_dir, ct_list[i])
    sub_sct_dir = os.path.join(mr_dir, sct_list[i])
    print(i)

    moving_img = ants.image_read(sub_ct_dir)    #读取ct
    fixed_img = ants.image_read(sub_sct_dir)    #读取sct
    # 执行图像配准，将 CT 图像（moving_img）与 MR 图像（fixed_img）进行配准，使用刚体变换
    # ct2mr = ants.registration(fixed_img,moving_img,type_of_transform='Rigid')
    #应用配准变换，将变换应用于移动图像（CT），生成变换后的图像。
    # warped_moving_img = ants.apply_transforms(fixed_img,moving_img,transformlist=ct2mr['fwdtransforms'],defaultvalue=-1000)
    #执行反向图像配准，将 MR 图像与 CT 图像进行配准。
    mr2ct = ants.registration(moving_img,fixed_img,type_of_transform='Rigid')

    # 获取变换文件路径
    transform_path = mr2ct['fwdtransforms'][0]  # 刚体变换通常是第一个变换文件
    # 读取变换文件内容
    transform_data = ants.read_transform(transform_path)
    # 获取旋转矩阵和平移向量
    rotation_matrix = transform_data.parameters.reshape(4, 3)[:3, :3]  # 取前 9 个元素组成 3x3 矩阵
    translation = transform_data.parameters[9:12]  # 最后 3 个是平移
    # 使用 scipy 将旋转矩阵转换为欧拉角（即旋转角度）
    from scipy.spatial.transform import Rotation as R
    rot = R.from_matrix(rotation_matrix)
    euler_angles = rot.as_euler('xyz', degrees=True)  # 单位是“度”
    print("Euler Angles (degrees):", euler_angles)  # 3个角度值
    print("Translation (x, y, z):", translation)  # 3个平移值


    print("--------------------------------")
    #应用反向配准变换，将变换应用于固定图像（MR）
    warped_fixed_img = ants.apply_transforms(moving_img,fixed_img,transformlist=mr2ct['fwdtransforms'],defaultvalue=-1000)

    save_war_ct_dir = os.path.join(war_ct_dir, ct_list[i])
    save_war_sct_dir = os.path.join(war_mr_dir, sct_list[i])

    save_war_sct_dir = save_war_sct_dir.replace(' ','')
    save_war_ct_dir  = save_war_ct_dir.replace(' ','')

    #ants.image_write(warped_moving_img, save_war_ct_dir)
    ants.image_write(warped_fixed_img, save_war_sct_dir)
    # break
