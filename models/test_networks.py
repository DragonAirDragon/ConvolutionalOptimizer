"""
Тестовые нейросети для сравнения оптимизаторов

Модели:
- SimpleNet: простая 3-слойная сеть
- DeepNet: глубокая сеть (6+ слоёв)
- IllConditionedNet: сеть с плохой обусловленностью (разные масштабы весов)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """
    Простая 3-слойная полносвязная сеть для классификации.
    
    Архитектура: Input -> FC -> ReLU -> FC -> ReLU -> FC -> Output
    
    Args:
        input_dim: размерность входа (по умолчанию 20)
        hidden_dim: размерность скрытых слоёв (по умолчанию 64)
        output_dim: количество классов (по умолчанию 10)
    """
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, output_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DeepNet(nn.Module):
    """
    Глубокая полносвязная сеть (6+ слоёв).
    
    Глубокие сети часто имеют проблемы с vanishing/exploding gradients.
    Сверточный оптимизатор может помочь сгладить эти проблемы.
    
    Args:
        input_dim: размерность входа (по умолчанию 20)
        hidden_dim: размерность скрытых слоёв (по умолчанию 32)
        output_dim: количество классов (по умолчанию 10)
        n_layers: количество скрытых слоёв (по умолчанию 6)
    """
    
    def __init__(
        self, 
        input_dim: int = 20, 
        hidden_dim: int = 32, 
        output_dim: int = 10, 
        n_layers: int = 6
    ):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IllConditionedNet(nn.Module):
    """
    Сеть с плохой обусловленностью — разные масштабы весов.
    
    Эта сеть имеет искусственно созданную плохую обусловленность:
    - Первый слой имеет большие веса (×10)
    - Второй слой имеет маленькие веса (×0.1)
    - Третий слой имеет средние веса (×5)
    
    Такая конфигурация создаёт сложный ландшафт loss'а.
    Здесь сверточный оптимизатор должен показать преимущество!
    
    Args:
        input_dim: размерность входа (по умолчанию 20)
        output_dim: количество классов (по умолчанию 10)
    """
    
    def __init__(self, input_dim: int = 20, output_dim: int = 10):
        super().__init__()
        # Адаптивные скрытые размеры на основе входа
        hidden1 = min(256, max(64, input_dim // 3))
        hidden2 = min(128, max(32, input_dim // 6))
        
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)
        
        # Искусственно создаём плохую обусловленность
        with torch.no_grad():
            self.fc1.weight.mul_(10.0)   # Большие веса
            self.fc2.weight.mul_(0.1)    # Маленькие веса
            self.fc3.weight.mul_(5.0)    # Средние веса
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ConvNet(nn.Module):
    """
    Простая свёрточная сеть для классификации изображений.
    
    Архитектура: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC -> Output
    
    Args:
        input_channels: количество входных каналов (по умолчанию 1)
        num_classes: количество классов (по умолчанию 10)
    """
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)             # (batch, 32*7*7)
        return self.fc(x)


class ResidualBlock(nn.Module):
    """Остаточный блок для ResNet-подобной архитектуры"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return F.relu(x + residual)


class ResNet(nn.Module):
    """
    Простая ResNet-подобная архитектура.
    
    Residual connections помогают с gradient flow.
    Интересно сравнить, как сверточный оптимизатор работает с residual сетями.
    
    Args:
        input_dim: размерность входа (по умолчанию 20)
        hidden_dim: размерность скрытых слоёв (по умолчанию 64)
        output_dim: количество классов (по умолчанию 10)
        n_blocks: количество residual блоков (по умолчанию 3)
    """
    
    def __init__(
        self, 
        input_dim: int = 20, 
        hidden_dim: int = 64, 
        output_dim: int = 10, 
        n_blocks: int = 3
    ):
        super().__init__()
        
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_fc(x))
        x = self.blocks(x)
        return self.output_fc(x)
