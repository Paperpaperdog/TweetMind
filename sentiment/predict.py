import pandas as pd
from transformers import pipeline

# 加载微调后的模型
sentiment_analyzer = pipeline("text-classification", model="./sentiment/fine_tuned_roberta")

# 设置每次处理的块大小，假设每次处理 1000 行数据
chunksize = 1000


diseases = ['adhd', 'anxiety', 'bipolar', 'depression', 'neg', 'ocd', 'ptsd']
for disease in diseases:
    # 输入文件路径
    input_file = f"data\\{disease}\\test.csv"  # 替换为你的文件路径
    output_file = f"./sentiment/predicted_output_{disease}.csv"  # 输出文件路径

    # 初始化一个空的 DataFrame 用于存储预测结果
    predicted_df = pd.DataFrame()

    # 分块读取 CSV 文件并处理
    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        # 确保 CSV 文件中有 'sentence' 列
        if "sentence" not in chunk.columns:
            raise ValueError("CSV 文件必须包含 'sentence' 列")

        # 获取文本列
        texts = chunk["sentence"].tolist()

        # 批量预测
        predictions = sentiment_analyzer(texts)
        predicted_labels = [prediction["label"] for prediction in predictions]

        # 将预测结果添加到 DataFrame 中
        chunk["predicted_sentiment"] = predicted_labels

        # 将处理过的块追加到预测结果 DataFrame
        predicted_df = pd.concat([predicted_df, chunk], ignore_index=True)

        print(f"已处理 {len(predicted_df)} 行数据")

        # 保存最终的预测结果到 CSV 文件
        predicted_df.to_csv(output_file, index=False)

        print(f"预测结果已保存至 {output_file}")
