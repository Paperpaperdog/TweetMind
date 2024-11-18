from datasets import load_dataset

# 假设数据是 CSV 格式
dataset = load_dataset("csv", data_files="train.csv")

# 查看数据格式
print(dataset["train"][0])

train_test_split = dataset["train"].train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "roberta-base"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 数据清理函数
def clean_data(dataset):
    return dataset.filter(lambda example: example.get("text") is not None and isinstance(example["text"], str))

train_dataset = clean_data(train_dataset)
test_dataset = clean_data(test_dataset)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 三分类

def preprocess_function(examples):
    sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
    examples["labels"] = [sentiment_map[sentiment] for sentiment in examples["sentiment"]]
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)


# 应用预处理
try:
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)
except Exception as e:
    print("Preprocessing error:", e)
    for example in train_dataset:
        print("Inspecting sample:", example)
        break

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./sentiment/results",         # 输出目录
    evaluation_strategy="epoch",   # 每轮评估一次
    learning_rate=2e-5,            # 学习率
    per_device_train_batch_size=16,  # 训练批量大小
    per_device_eval_batch_size=64,   # 验证批量大小
    num_train_epochs=3,            # 训练轮数
    weight_decay=0.01,             # 权重衰减
    logging_dir="./sentiment/logs",          # 日志目录
    save_total_limit=2,            # 最多保存的模型数
)

from transformers import Trainer

trainer = Trainer(
    model=model,                         # 微调的模型
    args=training_args,                  # 训练参数
    train_dataset=tokenized_train,       # 训练数据集
    eval_dataset=tokenized_test,         # 验证数据集
    tokenizer=tokenizer,                 # 分词器
)
trainer.train()

model.save_pretrained("./fine_tuned_roberta")
tokenizer.save_pretrained("./fine_tuned_roberta")

from transformers import pipeline

# 加载微调模型
sentiment_analyzer = pipeline("text-classification", model="./sentiment/fine_tuned_roberta")

# 测试推理
result = sentiment_analyzer("I love this movie! It's amazing.")
print(result)

from sklearn.metrics import classification_report

# 获取模型预测
predictions = trainer.predict(tokenized_test)
y_pred = predictions.predictions.argmax(-1)
y_true = tokenized_test["label"]

# 打印分类报告
print(classification_report(y_true, y_pred, target_names=["negative", "neutral", "positive"]))
