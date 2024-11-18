import math
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# 自定义数据集
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all", problem_type="multi_label_classification")
model.eval()
class_mapping = model.config.id2label


diseases = ['adhd', 'anxiety', 'bipolar', 'depression', 'neg', 'ocd', 'ptsd']
for disease in diseases:
    print(disease)
    # 假设 CSV 文件的路径是 'data.csv'
    df = pd.read_csv(f'{disease}\\test.csv')

    # 获取文本列
    texts = df['sentence'].tolist()
    print(texts[:50])

    # 分批次处理
    batch_size = 8  # 根据显存大小调整批次大小
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    print(dataloader[0])

    # 输出文件路径
    output_file = f'predicted_results_{disease}.csv'

    # 创建一个空的列表来存储每个样本的预测结果
    predictions = []

    # 获取所有类的名称（从模型的 class_mapping 中提取）
    class_labels = list(class_mapping.values())

    # 打开文件以便写入
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入CSV头部：每一列为一个类别
        f.write('sentence,' + ','.join(class_labels) + '\n')

        # 逐批次预测并写入结果
        for batch in dataloader:
            input_ids = batch['input_ids'].squeeze(1)
            attention_mask = batch['attention_mask'].squeeze(1)

            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=attention_mask)

            # 处理每个样本
            for idx in range(len(input_ids)):
                # 根据 sigmoid 输出进行二分类（0 或 1）
                flags = [sigmoid(s) > 0.5 for s in output[0][idx].detach().tolist()]
                
                # 构建每个样本的预测结果行，标签为 0 或 1
                result_row = [texts[idx]] + [str(int(flag)) for flag in flags]
                f.write(','.join(result_row) + '\n')

    print("Predictions written to", output_file)


