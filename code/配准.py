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

ct_dir = r'/data/CycleGan/Data/Register_data/chongqing_20250429_21/ct_nii'
mr_dir = r'/data/CycleGan/Data/Register_data/chongqing_20250429_21/mr_nii'
war_ct_dir = r'/data/CycleGan/Data/Register_data/chongqing_20250429_21/war_ct_nii'
war_mr_dir = r'/data/CycleGan/Data/Register_data/chongqing_20250429_21/war_mr_nii'


ct_list = sorted(os.listdir(ct_dir))
sct_list = sorted(os.listdir(mr_dir))

for i in range(len(sct_list)):
    sub_ct_dir = os.path.join(ct_dir, ct_list[i])
    sub_sct_dir = os.path.join(mr_dir, sct_list[i])
    print(i)

    moving_img = ants.image_read(sub_ct_dir)    #读取ct
    fixed_img = ants.image_read(sub_sct_dir)    #读取sct
    # 执行图像配准，将 CT 图像（moving_img）与 MR 图像（fixed_img）进行配准，使用刚体变换
    ct2mr = ants.registration(fixed_img,moving_img,type_of_transform='Rigid')
    #应用配准变换，将变换应用于移动图像（CT），生成变换后的图像。
    warped_moving_img = ants.apply_transforms(fixed_img,moving_img,transformlist=ct2mr['fwdtransforms'],defaultvalue=-1000)
    #执行反向图像配准，将 MR 图像与 CT 图像进行配准。
    mr2ct = ants.registration(moving_img,fixed_img,type_of_transform='Rigid')
    print("--------------------------------")
    #应用反向配准变换，将变换应用于固定图像（MR）
    warped_fixed_img = ants.apply_transforms(moving_img,fixed_img,transformlist=mr2ct['fwdtransforms'],defaultvalue=-1000)

    save_war_ct_dir = os.path.join(war_ct_dir, ct_list[i])
    save_war_sct_dir = os.path.join(war_mr_dir, sct_list[i])

    save_war_sct_dir = save_war_sct_dir.replace(' ','')
    save_war_ct_dir  = save_war_ct_dir.replace(' ','')

    ants.image_write(warped_moving_img, save_war_ct_dir)
    ants.image_write(warped_fixed_img, save_war_sct_dir)
    # break
