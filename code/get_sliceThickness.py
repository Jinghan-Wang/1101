import os
import SimpleITK as sitk

# 指定目录
directory = '/data/ZZJ/code/1101/dataset/test/CT'
print(directory)
# 遍历目录中的所有文件
for filename in os.listdir(directory):
    # if filename.endswith('.nii.gz'):
        # 构建完整的文件路径
        filepath = os.path.join(directory, filename)

        # 读取 NIfTI 文件
        img = sitk.ReadImage(filepath)

        # 获取图像的物理尺寸（spacing）
        spacing = img.GetSpacing()

        # 假设切片厚度在 Z 轴方向
        slice_thickness = spacing[2]  # Z 轴的间距
        print(f'File: {filename}, Slice Thickness: {slice_thickness} mm')  # 假设单位为毫米