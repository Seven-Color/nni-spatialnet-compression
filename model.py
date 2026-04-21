"""模型定义模块"""
import torch
import torch.nn as nn


class CNN(nn.Module):
    """可配置层数的 CNN 模型"""
    
    def __init__(self, in_channels=1, num_classes=10, channels=[16, 32], dropout=0.25):
        super().__init__()
        
        self.channels = channels
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # 卷积层
        prev_ch = in_channels
        for ch in channels:
            self.convs.append(nn.Conv2d(prev_ch, ch, kernel_size=3, padding=1))
            self.bns.append(nn.BatchNorm2d(ch))
            prev_ch = ch
        
        # 全连接层
        self.fc = nn.Linear(channels[-1] * 7 * 7, num_classes)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = self.pool(self.relu(bn(conv(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)
    
    @property
    def name(self):
        return f"CNN_{len(self.channels)}L_{'-'.join(map(str, self.channels))}"


def build_model(config):
    """根据配置构建模型"""
    model = CNN(
        in_channels=config['dataset']['channels'],
        num_classes=config['dataset']['num_classes'],
        channels=config['model']['channels'],
        dropout=config['model'].get('dropout', 0.25)
    )
    return model


def count_params(model):
    """计算参数量"""
    return sum(p.numel() for p in model.parameters())


def get_model_size(state_dict, path="temp.pt"):
    """获取模型文件大小 (KB)"""
    import os
    torch.save(state_dict, path)
    size_kb = os.path.getsize(path) / 1024
    os.remove(path)
    return size_kb
