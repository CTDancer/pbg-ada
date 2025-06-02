import os
import pandas as pd


def merge_csv_files(directory, output_file):
    # 获取文件夹中所有以"processed"开头的csv文件
    csv_files = [f for f in os.listdir(directory) if f.startswith("processed") and f.endswith(".csv")]

    # 按文件名排序
    csv_files.sort()

    # 初始化一个空的DataFrame列表
    dfs = []

    # 读取每个csv文件并追加到DataFrame列表中
    for file in csv_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    # 将所有DataFrame合并成一个
    merged_df = pd.concat(dfs, ignore_index=True)

    # 将合并后的DataFrame保存为一个新的csv文件
    merged_df.to_csv(output_file, index=False)
    print(f"所有文件已合并到 {output_file}")


# 使用示例
directory = "/home/tc415/PPI_datasets"
output_file = '/home/tc415/dataset.csv'
merge_csv_files(directory, output_file)
