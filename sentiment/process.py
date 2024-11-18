import pandas as pd

# 读取两个CSV文件
df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')

# 合并两个DataFrame
# 可以选择按行（axis=0）或按列（axis=1）合并
# 这里按行合并（默认）
df_merged = pd.concat([df1, df2], axis=0, ignore_index=True)

# 保存合并后的结果到新CSV文件
df_merged.to_csv('sentiment.csv', index=False)
