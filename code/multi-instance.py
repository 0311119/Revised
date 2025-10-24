import math
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, SwinModel
from tqdm import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler
import random
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import json
from pathlib import Path

'''
    -----------------------------数据初始化--------------------------------
'''

config_path = Path(__file__).resolve().parent.parent / "config" / "multi_instance.json"
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

# Load pretrained model from config
# Available models:
# - swin-tiny-patch4-window7-224: https://huggingface.co/microsoft/swin-tiny-patch4-window7-224
# - swin-tiny-patch4-window7-224-finetuned-face-emotion-v12: https://huggingface.co/MahmoudWSegni/swin-tiny-patch4-window7-224-finetuned-face-emotion-v12
# - swinv2-tiny-patch4-window8-256: https://huggingface.co/microsoft/swinv2-tiny-patch4-window8-256
pretrained_model_name = cfg.get("pretrained_model", "swin-tiny-patch4-window7-224-finetuned-face-emotion-v12")
pretrained_model_path = f"./weights/{pretrained_model_name}"
swin_processor = AutoImageProcessor.from_pretrained(pretrained_model_path)
swin_model = SwinModel.from_pretrained(pretrained_model_path)

subject_id = os.getenv("ID", "s08")
tb_dir = os.getenv("TB_DIR", "runs/run2")

device_str = cfg.get("device", "cuda:1")
if torch.cuda.is_available() and str(device_str).startswith("cuda"):
    device = torch.device(device_str)
else:
    device = torch.device("cpu")

epochs = cfg.get("epochs", 100)
lr = cfg.get("lr", 0.0001)
lrf = cfg.get("lrf", 0.1)
max_lr = cfg.get("max_lr", 0.00001)
batch_size = cfg.get("batch_size", 14)
test_batch_size = cfg.get("test_batch_size", batch_size)
seed = cfg.get("seed", 0)
num_instances_cfg = cfg.get("num_instances", 10)
instance_selection_method = cfg.get("instance_selection_method", "attention_weighted_topk")
fusion_type = cfg.get("fusion_type", "none")
num_select = cfg.get("num_select", 2)


channels = 32
samples = 384
eeg_data = np.load(f"./DEAP/EEGData/{subject_id}_eeg.npy")
labels = np.load(f"./DEAP/EEGData/{subject_id}_labels.npy")
label_counts = np.bincount(labels)

'''
    -----------------------------组织图像数据,与eeg对齐--------------------------------
    1. 得到图像数据的numpy数组 (800, 3, 224, 224)
'''
# 创建一个列表，用于存储受试者的图像文件的路径
image_paths = []
for i in range(1, 801):
    img_path = f"./DEAP/faces_multi_instance/{subject_id}/{i}.jpg"
    image_paths.append(img_path)
print("image_paths:", len(image_paths)) # 800

# eeg、图片组合在一起
combined_data = list(zip(eeg_data, image_paths)) # (800, 32, 384) + (800)
# 现在，combined_data是一个长度为800的数组，每个元素都是一个元组，元组中包含一个32x384的EEG数组和一个图像文件路径
paired_data = combined_data

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
set_seed(seed)



class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels, num_instances=10):
        self.samples = samples
        self.labels = labels
        self.num_instances = num_instances  # 要使用的实例数量
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        eeg_sample = self.samples[index][0].astype(np.float32)
        base_image_path = self.samples[index][1]
        
        # 获取主编号(1-800)
        trial_index = int(base_image_path.split('/')[-1].split('.')[0])
        
        # 计算要选择的副编 号(从30张图片中均匀选择10张)
        if subject_id=='s01' or subject_id=='s02':
            step = 30 // self.num_instances  # 间隔 = 30/10 = 3
            sub_indices = list(range(1, 31, step))[:self.num_instances]  # [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
        else:
            sub_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # 加载选定的图片
        images_stack = []
        for sub_idx in sub_indices:
            img_file = f"./DEAP/faces_multi_instance/{subject_id}/{trial_index}_{sub_idx}.jpg"
            img = transforms.Resize((224, 224))(Image.open(img_file))
            img = np.array(img).astype(np.float32)
            images_stack.append(img)
        
        # 堆叠所有图片
        images_stack = np.stack(images_stack)  # [num_instances, 224, 224, 3]
        
        label = self.labels[index]
        return eeg_sample, images_stack, label

# 分类模型
class MultiModalClassifier(nn.Module):
    def __init__(self, input_size=768, num_classes=4, 
                 num_heads=12, dim_feedforward=2048, num_encoder_layers=2, device=device, 
                 eeg_size=384, transformer_dropout_rate=0.2, cls_dropout_rate=0.1,
                 fusion_type='cross_attention',  # options: 'none', 'cross_attention', 'mlp'
                 instance_selection_method='attention_weighted_topk',  # options: 'none', 'softmax', 'amil', 'attention_topk', 'attention_weighted_topk'
                 num_select = 3, num_instances=10
                 ):
        super().__init__()
        # Core hyperparameters and options
        self.transformer_dropout_rate = transformer_dropout_rate  # Dropout used in Transformer layers
        self.cls_dropout_rate = cls_dropout_rate  # Dropout applied to the CLS token output
        self.fusion_type = fusion_type  # Fusion strategy for image features
        self.instance_selection_method = instance_selection_method  # MIL instance selection method

        # Swin image processor and backbone (fine-tuned)
        self.img_processor = swin_processor
        self.swin_model = swin_model
        for param in self.swin_model.parameters():
            param.requires_grad = True  # Enable fine-tuning

        # Token type embeddings: 0 for image tokens, 1 for EEG tokens
        self.token_type_embeddings = nn.Embedding(2, input_size)
        
        # Transformer encoder over concatenated image and EEG tokens
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=transformer_dropout_rate, batch_first=True),
            num_layers=num_encoder_layers
        )

        # EEG projection and normalization
        self.eeg_proj = nn.Linear(eeg_size, input_size)  # Project EEG from eeg_size to hidden size
        self.activation = nn.ReLU()
        self.layernorm = nn.LayerNorm(eeg_size)  # Normalize along the last EEG dimension
        # Classification head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_size)).to(device)  # Learnable CLS token
        self.dropout = nn.Dropout(cls_dropout_rate)  # Dropout before classifier
        self.classifier = nn.Linear(input_size,num_classes)  # Final classifier

        # Initialize cross-attention components only when fusion_type == 'cross_attention'
        if fusion_type == 'cross_attention':
            self.num_queries = 147
            self.query_tokens = nn.Parameter(torch.zeros(1, self.num_queries, input_size))
            nn.init.normal_(self.query_tokens, std=0.02)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=input_size,
                num_heads=num_heads,
                dropout=transformer_dropout_rate,
                batch_first=True
            )
        # MLP up/down projection components
        elif fusion_type == 'mlp':
            self.mlp_up = nn.Linear(input_size, 4*input_size)  # 768 -> 3072
            self.mlp_act = nn.GELU()
            self.mlp_down = nn.Linear(4*input_size, input_size)  # 3072 -> 768


        self.num_instances = num_instances
        self.num_select = num_select
        # Instance weights parameter for the softmax method
        self.instance_weights = nn.Parameter(torch.ones(1, num_instances))  # Global learnable instance weights (used by 'softmax' method)

        # AMIL attention projection layers
        self.amil_value_proj = nn.Linear(input_size, input_size)
        self.amil_weight_proj = nn.Linear(input_size, 1)

        # Top-K attention projection layers
        self.topk_value_proj = nn.Linear(input_size * 49, input_size * 49)
        self.topk_weight_proj = nn.Linear(input_size * 49, 1)
        self.topk_dimension_recover = nn.Linear(input_size * 49, self.num_instances)  # recover per-instance dimension

    def select_instances(self, images_embedding):
        # images_embedding shape: [batch_size, num_instances, 49, 768]
        if self.instance_selection_method == 'none':
            # Return the embedding of the first image
            return images_embedding[:, 0, :, :].unsqueeze(1)
        
        elif self.instance_selection_method == 'softmax':
            # Compute weight for each instance
            weights = F.softmax(self.instance_weights, dim=1)
            # Select top-k instances
            _, indices = torch.topk(weights, self.num_select, dim=1)
            selected_embeddings = []
            batch_size = images_embedding.size(0)
            
            for i in range(batch_size):
                selected = images_embedding[i, indices[0], :]  # Select the k instances with the highest weights
                selected_embeddings.append(selected)
                
            return torch.stack(selected_embeddings)
        
        elif self.instance_selection_method == 'amil':
            # [batch_size, num_instances, 49, 768]
            batch_size = images_embedding.size(0)
            
            # Compute per-instance representation
            instance_features = images_embedding.mean(dim=2)  # [batch_size, num_instances, 768]
            
            # AMIL attention scores
            hidden = torch.tanh(self.amil_value_proj(instance_features))  # [batch_size, num_instances, 768]
            weights = self.amil_weight_proj(hidden)  # [batch_size, num_instances, 1]
            weights = F.softmax(weights, dim=1)  # [batch_size, num_instances, 1]
            
            # Weighted sum
            weighted_features = (instance_features * weights).sum(dim=1)  # [batch_size, 768]
            
            return weighted_features
        
        elif self.instance_selection_method == 'attention_topk':
            # [batch_size, num_instances, 49, 768]
            batch_size = images_embedding.size(0)
            
            # Compute per-instance representation
            instance_features = images_embedding.view(batch_size, self.num_instances, -1)  # [batch_size, num_instances, 49*768]
            
            # Attention scores
            hidden = torch.tanh(self.topk_value_proj(instance_features))  # [batch_size, num_instances, 49*768]
            weights = self.topk_weight_proj(hidden)  # [batch_size, num_instances, 1]
            weights = F.softmax(weights, dim=1)  # [batch_size, num_instances, 1]

            # Weighted sum
            weighted_features = (instance_features * weights).sum(dim=1)  # [batch_size, 49*768] ([10, 37632])   
            # Recover per-instance weights
            recovered_weights = self.topk_dimension_recover(weighted_features).view(batch_size, self.num_instances, 1)  # [batch_size, num_instances, 1] ([10, 10, 1])
            
            _, indices = torch.topk(recovered_weights, self.num_select, dim=1)
            selected_embeddings = []
            
            for i in range(batch_size):
                selected = images_embedding[i, indices[i], :, :]  # Select top-k instances by recovered weights
                selected_embeddings.append(selected)
            
            return torch.stack(selected_embeddings)  # [batch_size, num_select, 49, 768]
        
        elif self.instance_selection_method == 'attention_weighted_topk':
            # [batch_size, num_instances, 49, 768]
            batch_size = images_embedding.size(0)
            
            # Compute per-instance representation
            instance_features = images_embedding.view(batch_size, self.num_instances, -1)  # [batch_size, num_instances, 49*768]
            
            # Attention scores
            hidden = torch.tanh(self.topk_value_proj(instance_features))  # [batch_size, num_instances, 49*768]
            weights = self.topk_weight_proj(hidden)  # [batch_size, num_instances, 1]
            weights = F.softmax(weights, dim=1)  # [batch_size, num_instances, 1]

            # Select top-k instances
            _, indices = torch.topk(weights, self.num_select, dim=1)
            selected_embeddings = []
            
            for i in range(batch_size):
                selected = images_embedding[i, indices[i].squeeze(), :, :]  # Select the k instances with the highest weights
                selected_embeddings.append(selected)
            
            return torch.stack(selected_embeddings)  # [batch_size, num_select, 49, 768]

    def forward(self, eeg_data, images_data):
        '''
            eeg_data            # [batch_size, 32, 384]
            eeg_embedding       # [batch_size, 32, 768]
            images_data         # [batch_size, num_instances, 224, 224, 3]
            images_embedding    # [batch_size, L_img, 768] (L_img = num_select*49 or num_queries)
            multi_embedding     # [batch_size, (L_img + 32 + 1), 768]
        '''
        batch_size = images_data.size(0)
        
        # Process multiple images
        images_embedding = []
        for i in range(self.num_instances):
            image = images_data[:, i]  # [batch_size, 224, 224, 3]
            vision_inputs = self.img_processor(image, return_tensors="pt").to(device)
            embedding = self.swin_model(**vision_inputs).last_hidden_state
            images_embedding.append(embedding)
        
        images_embedding = torch.stack(images_embedding, dim=1)  # [batch_size, num_instances, 49, 768] (49 = 7x7 patches)
        # print("After stack:", images_embedding.shape)  # torch.Size([16, 10, 49, 768])

        # Select instances
        selected_embeddings = self.select_instances(images_embedding)  # [batch_size, num_select, 49, 768]
        # print("After select:", selected_embeddings.shape)  # torch.Size([16, 3, 49, 768])

        selected_embeddings = selected_embeddings.view(batch_size, -1, 768)  # [batch_size, num_select*49, 768]
        # print("After view:", selected_embeddings.shape)  # torch.Size([16, 147, 768])
        
        # Process according to fusion_type
        if self.fusion_type == 'cross_attention':
            query_tokens = self.query_tokens.expand(batch_size, -1, -1)
            image_features, _ = self.cross_attention(
                query=query_tokens,
                key=selected_embeddings,
                value=selected_embeddings
            )
            images_embedding = image_features  # [batch_size, self.num_queries, 768]
        
        elif self.fusion_type == 'mlp':
            # MLP up/down projection
            x = self.mlp_up(selected_embeddings)  # [batch_size, num_select*49, 3072]
            x = self.mlp_act(x)
            images_embedding = self.mlp_down(x)  # [batch_size, num_select*49, 768]
        
        else:
            images_embedding = selected_embeddings  # [batch_size, num_select*49, 768]

        eeg_data = self.layernorm(eeg_data)
        eeg_embedding = self.eeg_proj(eeg_data)
        eeg_embedding = self.activation(eeg_embedding)

        # Add token-type embeddings to distinguish modalities
        images_embedding, eeg_embedding = (
            images_embedding + self.token_type_embeddings(torch.zeros(images_embedding.shape[0], 1, dtype=torch.long, device=device)),
            eeg_embedding + self.token_type_embeddings(torch.ones(eeg_embedding.shape[0], 1, dtype=torch.long, device=device))
        )

        # Concatenate image and EEG tokens, prepend CLS, and encode
        multi_embedding = torch.cat((images_embedding, eeg_embedding), dim=1)
        multi_embedding = torch.cat((self.cls_token.expand(multi_embedding.size(0), -1, -1), multi_embedding), dim=1)
        multi_embedding = self.transformer_encoder(multi_embedding)

        # Take the output of the CLS token
        cls_token_output = multi_embedding[:, 0, :]
        cls_token_output = self.dropout(cls_token_output)
        x = self.classifier(cls_token_output)

        return x

# 一次划分
# train_index, test_index = train_test_split(range(len(paired_data)), test_size=0.2, random_state=random_state)
train_index, test_index = train_test_split(range(len(paired_data)), test_size=0.2, stratify=labels)

model = MultiModalClassifier(
    fusion_type=fusion_type,
    instance_selection_method=instance_selection_method,
    num_select=num_select,
    num_instances=num_instances_cfg,
    device=device
).to(device) 

epochs = int(epochs)
lr = float(lr)
lrf = float(lrf)
max_lr = float(max_lr)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# ViT学习率
cosine_lr_lambda = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lr_lambda)

# 分割数据
train_data = [paired_data[i] for i in train_index]
test_data = [paired_data[i] for i in test_index]
train_labels = labels[train_index]
test_labels = labels[test_index]

# 计算并打印每个类别的数量
unique_train, counts_train = np.unique(train_labels, return_counts=True)
unique_test, counts_test = np.unique(test_labels, return_counts=True)
print("训练集中每个类别的数量：", dict(zip(unique_train, counts_train)))
print("测试集中每个类别的数量：", dict(zip(unique_test, counts_test)))

# 创建数据集、数据加载器
train_dataset = MultiModalDataset(train_data, train_labels, num_instances=num_instances_cfg)
test_dataset = MultiModalDataset(test_data, test_labels, num_instances=num_instances_cfg)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader.dataset) // train_loader.batch_size, epochs=epochs, pct_start=0.20)
# print("steps_per_epoch:", len(train_loader.dataset) // train_loader.batch_size)

start_time = datetime.now()
writer = SummaryWriter(f'{tb_dir}')
writer.add_text('ID', subject_id, global_step=0)
writer.add_text('transformer dropout', str(model.transformer_dropout_rate), global_step=0)
writer.add_text('cls dropout', str(model.cls_dropout_rate), global_step=0)
test_accuracies = []
test_precisions = []
test_recalls = []
test_f1s = []
for epoch in range(epochs):
    train_bar = tqdm(enumerate(train_loader),total=len(train_loader),desc="Training", leave=False)
    model.train()

    for i, (eeg_data, image_data, label) in enumerate(train_loader):
        eeg_data = eeg_data.to(device)
        image_data = image_data.to(device)
        label = label.to(device)

        output = model(eeg_data, image_data)
        loss = nn.CrossEntropyLoss()(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新warm-up学习率 
        # scheduler.step()

        if i % 10 == 0:  # 每10个批次，记录损失和准确率
            _, predicted = torch.max(output, 1)
            correct = (predicted == label).sum().item()
            acc = correct / label.size(0)
            writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)  # loss 800
            writer.add_scalar('training accuracy', acc, epoch * len(train_loader) + i)  # acc 800

        train_bar.update(1)
        train_bar.write(f"Epoch: {epoch + 1}, Training Loss: {loss.item()}")

    # 在测试集上评估模型
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for i, (eeg_data, image_data, label) in enumerate(test_loader):
            eeg_data, image_data, label = eeg_data.to(device), image_data.to(device), label.to(device)
            outputs = model(eeg_data, image_data)
            _, predicted = torch.max(outputs.data, 1)
            test_total += label.size(0)
            test_correct += (predicted == label).sum().item()

            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # 计算测试集的损失
            loss = nn.CrossEntropyLoss()(outputs, label)
            test_loss += loss.item()

        # print("All labels:", all_labels)
        # print("All predictions:", all_predictions)
        acc = test_correct / test_total
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        test_accuracies.append(acc)
        test_precisions.append(precision)
        test_recalls.append(recall)
        test_f1s.append(f1)
        writer.add_scalar('test accuracy', acc, epoch + 1) # 从0-9变成1-10
        writer.add_scalar('test loss', test_loss / len(test_loader), epoch + 1)
        writer.add_scalar('test precision', precision, epoch + 1)
        writer.add_scalar('test recall', recall, epoch + 1)
        writer.add_scalar('test f1 score', f1, epoch + 1)
        print(f"Accuracy: {acc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
    # 更新ViT的学习率 
    writer.add_scalar('learning rate', optimizer.param_groups[0]["lr"], epoch)
    scheduler.step()

writer.close()

print("\n--------------------------训练结束--------------------------")
max_acc_index = np.argmax(test_accuracies)
print(f"Max accuracy: {test_accuracies[max_acc_index]}")
print(f"Precision at max accuracy: {test_precisions[max_acc_index]}")
print(f"Recall at max accuracy: {test_recalls[max_acc_index]}")
print(f"F1 Score at max accuracy: {test_f1s[max_acc_index]}")

end_time = datetime.now()
run_time = end_time - start_time
run_time_seconds = run_time.total_seconds()  # 获取总秒数
run_time_min = round(run_time_seconds / 60)  # 转换为分钟
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total runtime: {run_time_min} minutes")
print(f"ID: {subject_id}")

with open('./第23轮/第23轮记录.md', 'a') as f:
    f.write('{}  {} precision:{} recall:{} f1:{}\n'.format(subject_id, max(test_accuracies), max(test_precisions), max(test_recalls), max(test_f1s)))