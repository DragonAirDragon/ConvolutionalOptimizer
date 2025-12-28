"""
Сверточный SGD — сглаживает градиенты перед обновлением весов

Формула: θ_{k+1} = θ_k - lr * (W ⊗ ∇L)

где W — обучаемое ядро свёртки, которое сглаживает градиент
"""

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable


class ConvolutionalSGD(Optimizer):
    """
    Сверточный SGD с адаптивным ядром сглаживания.
    
    Ключевая идея: градиенты сглаживаются свёрткой перед обновлением весов.
    Ядро свёртки W обучается вместе с сетью на основе глобального loss.
    
    Args:
        params: параметры модели для оптимизации
        lr: learning rate (по умолчанию 0.01)
        momentum: коэффициент моментума (по умолчанию 0.9)
        kernel_size: размер ядра свёртки (по умолчанию 3)
        adaptive_kernel: включить адаптацию ядра (по умолчанию True)
        kernel_lr: learning rate для обновления ядра (по умолчанию 0.001)
    
    Пример использования:
        >>> model = nn.Linear(10, 2)
        >>> optimizer = ConvolutionalSGD(model.parameters(), lr=0.01)
        >>> 
        >>> for data, target in dataloader:
        >>>     optimizer.zero_grad()
        >>>     loss = criterion(model(data), target)
        >>>     loss.backward()
        >>>     
        >>>     # Для адаптации ядра передаём closure
        >>>     def closure():
        >>>         return criterion(model(data), target)
        >>>     optimizer.step(closure)
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        kernel_size: int = 3,
        adaptive_kernel: bool = True,
        kernel_lr: float = 0.001
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be positive odd number, got: {kernel_size}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            kernel_size=kernel_size,
            adaptive_kernel=adaptive_kernel,
            kernel_lr=kernel_lr
        )
        super().__init__(params, defaults)
        
        # Инициализация ядра свёртки (равномерное распределение)
        # Ядро нормализовано: сумма элементов = 1
        self.kernel = torch.ones(kernel_size) / kernel_size
        
        # Для отслеживания изменения loss
        self.prev_loss: Optional[float] = None
        
        # Статистика для анализа
        self.kernel_history = [self.kernel.clone()]
        
    def _smooth_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Ключевая операция: свёртка градиента с ядром W
        
        Это "сглаживает" градиент, убирая шум и резкие переходы.
        Математически: smoothed_grad = W ⊗ grad
        
        Args:
            grad: тензор градиентов
            
        Returns:
            Сглаженный градиент той же формы
        """
        original_shape = grad.shape
        grad_flat = grad.view(-1)
        
        # Если градиент слишком маленький — свёртка невозможна
        if len(grad_flat) < len(self.kernel):
            return grad
        
        # Переносим ядро на то же устройство, что и градиент
        kernel = self.kernel.to(grad.device)
        pad_size = len(kernel) // 2
        
        # Reflect padding сохраняет граничные значения
        grad_padded = F.pad(
            grad_flat.unsqueeze(0).unsqueeze(0),
            (pad_size, pad_size),
            mode='reflect'
        )
        
        # 1D свёртка
        kernel_3d = kernel.view(1, 1, -1)
        smoothed = F.conv1d(grad_padded, kernel_3d)
        
        return smoothed.view(original_shape)
    
    def _update_kernel(self, current_loss: float):
        """
        Метаоптимизация: адаптируем ядро W на основе изменения loss
        
        Это реализует идею "функция потерь сети оптимизирует ядро":
        - Если loss уменьшился → ядро работает хорошо, усиливаем центр
        - Если loss увеличился → делаем ядро более сглаживающим
        
        Args:
            current_loss: текущее значение loss
        """
        if self.prev_loss is None:
            self.prev_loss = current_loss
            return
            
        kernel_lr = self.defaults['kernel_lr']
        loss_diff = current_loss - self.prev_loss
        
        if loss_diff < 0:
            # Loss уменьшился → ядро работает хорошо
            # Усиливаем центральный элемент (более точный градиент)
            center = len(self.kernel) // 2
            self.kernel[center] += kernel_lr
        else:
            # Loss увеличился → нужно больше сглаживания
            # Двигаемся к равномерному ядру
            uniform = torch.ones_like(self.kernel) / len(self.kernel)
            self.kernel = 0.9 * self.kernel + 0.1 * uniform
        
        # Нормализация (сумма = 1)
        self.kernel = self.kernel / self.kernel.sum()
        self.prev_loss = current_loss
        
        # Сохраняем историю ядра
        self.kernel_history.append(self.kernel.clone())
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Выполняет один шаг оптимизации.
        
        Алгоритм:
        1. Для каждого параметра сглаживаем градиент свёрткой
        2. Применяем моментум (как в обычном SGD)
        3. Обновляем веса
        4. Если передан closure — адаптируем ядро
        
        Args:
            closure: функция, вычисляющая loss (нужна для адаптации ядра)
            
        Returns:
            Значение loss (если передан closure) или None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # === КЛЮЧЕВОЙ ШАГ: сглаживание градиента ===
                smoothed_grad = self._smooth_gradient(grad)
                
                # Моментум (как в обычном SGD)
                if momentum != 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(smoothed_grad)
                    smoothed_grad = buf
                
                # Обновление весов: θ = θ - lr * smoothed_grad
                p.add_(smoothed_grad, alpha=-lr)
        
        # Адаптация ядра (если включено и есть loss)
        if self.defaults['adaptive_kernel'] and loss is not None:
            self._update_kernel(loss.item())
        
        return loss
    
    def get_kernel(self) -> torch.Tensor:
        """Возвращает текущее ядро свёртки"""
        return self.kernel.clone()
    
    def get_kernel_history(self) -> list:
        """Возвращает историю изменений ядра"""
        return [k.clone() for k in self.kernel_history]
