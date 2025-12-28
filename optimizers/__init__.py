"""
Сверточные оптимизаторы для нейросетей

Модули:
- ConvolutionalSGD: базовый сверточный SGD с адаптивным ядром
- LocalLossConvSGD: оптимизатор с локальными функциями потерь
- ConvolutionalSGDv2: продвинутый оптимизатор с 2D свёрткой и обучаемым ядром
"""

from .conv_sgd import ConvolutionalSGD
from .local_loss_sgd import LocalLossConvSGD
from .conv_sgd_v2 import ConvolutionalSGDv2

__all__ = ['ConvolutionalSGD', 'LocalLossConvSGD', 'ConvolutionalSGDv2']
