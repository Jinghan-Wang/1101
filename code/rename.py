import os

# 设置目标目录
directory = "/data/ZZJ/code/1101/dataset/process/Training/CT"

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    # 仅处理 .nii.gz 文件
    if filename.endswith(".nii.gz"):
        old_file_path = os.path.join(directory, filename)

        # 替换空格为下划线
        new_filename = filename.replace(" ", "_")

        # 获取新文件的完整路径
        new_file_path = os.path.join(directory, new_filename)

        # 重命名文件
        if old_file_path != new_file_path:  # 避免重复重命名
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")
