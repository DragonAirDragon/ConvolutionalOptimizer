"""
Оптимизатор с локальными функциями потерь

Идея: "У нейрона есть своя функция потерь"
Каждый слой/параметр имеет своё ядро W, которое оптимизируется локально.

Локальный loss для ядра:
    L_local = -cos_similarity(grad, smoothed_grad) + λ * variance(smoothed_grad)

Цели локального loss:
1. Сохранить направление градиента (косинусное сходство)
2. Уменьшить вариацию (сглаживание шума)
"""

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable, Dict, Any


class LocalLossConvSGD(Optimizer):
    """
    Сверточный SGD с локальными функциями потерь для каждого параметра.
    
    Каждый параметр имеет:
    - Своё ядро сглаживания W_i
    - Локальный loss: насколько хорошо W_i сглаживает градиент
    
    Глобальный loss координирует локальные оптимизации.
    
    Args:
        params: параметры модели для оптимизации
        lr: learning rate для весов (по умолчанию 0.01)
        momentum: коэффициент моментума (по умолчанию 0.9)
        kernel_size: размер ядра свёртки (по умолчанию 3)
        local_lr: learning rate для обновления локальных ядер (по умолчанию 0.001)
        smoothness_weight: вес штрафа за вариацию в локальном loss (по умолчанию 0.1)
    
    Пример использования:
        >>> model = nn.Linear(10, 2)
        >>> optimizer = LocalLossConvSGD(model.parameters(), lr=0.01)
        >>> 
        >>> for data, target in dataloader:
        >>>     optimizer.zero_grad()
        >>>     loss = criterion(model(data), target)
        >>>     loss.backward()
        >>>     optimizer.step()
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        kernel_size: int = 3,
        local_lr: float = 0.001,
        smoothness_weight: float = 0.1
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
            local_lr=local_lr,
            smoothness_weight=smoothness_weight
        )
        super().__init__(params, defaults)
        
        # Статистика локальных loss'ов
        self.local_loss_history: Dict[int, list] = {}
    
    def _init_state(self, p: torch.Tensor, kernel_size: int) -> Dict[str, Any]:
        """
        Инициализация состояния для параметра.
        
        Каждый параметр получает:
        - Локальное ядро свёртки
        - Буфер моментума
        
        Args:
            p: тензор параметра
            kernel_size: размер ядра
            
        Returns:
            Словарь состояния параметра
        """
        state = self.state[p]
        if 'kernel' not in state:
            # Локальное ядро для этого параметра (равномерное)
            state['kernel'] = torch.ones(kernel_size, device=p.device) / kernel_size
            state['momentum_buffer'] = torch.zeros_like(p)
            state['local_loss_history'] = []
        return state
    
    def _local_loss(self, grad: torch.Tensor, smoothed_grad: torch.Tensor) -> torch.Tensor:
        """
        Локальная функция потерь для ядра.
        
        Формула: L = -cos_sim(grad, smoothed_grad) + λ * var(smoothed_grad)
        
        Цели:
        1. Максимизировать косинусное сходство (сохранить направление)
        2. Минимизировать вариацию (сглаживание)
        
        Args:
            grad: оригинальный градиент
            smoothed_grad: сглаженный градиент
            
        Returns:
            Скалярное значение локального loss
        """
        # Косинусное сходство между оригинальным и сглаженным градиентом
        grad_flat = grad.view(1, -1)
        smoothed_flat = smoothed_grad.view(1, -1)
        
        # Защита от нулевых градиентов
        grad_norm = torch.norm(grad_flat)
        smoothed_norm = torch.norm(smoothed_flat)
        
        if grad_norm < 1e-8 or smoothed_norm < 1e-8:
            cos_sim = torch.tensor(1.0, device=grad.device)
        else:
            cos_sim = F.cosine_similarity(grad_flat, smoothed_flat)
        
        # Штраф за высокую вариацию (шум)
        variance = torch.var(smoothed_grad)
        
        # Локальный loss: максимизируем сходство, минимизируем вариацию
        smoothness_weight = self.defaults['smoothness_weight']
        return -cos_sim + smoothness_weight * variance
    
    def _smooth_gradient(self, grad: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        Свёртка градиента с ядром.
        
        Args:
            grad: тензор градиентов
            kernel: ядро свёртки
            
        Returns:
            Сглаженный градиент той же формы
        """
        original_shape = grad.shape
        grad_flat = grad.view(-1)
        
        if len(grad_flat) < len(kernel):
            return grad
        
        pad_size = len(kernel) // 2
        grad_padded = F.pad(
            grad_flat.unsqueeze(0).unsqueeze(0),
            (pad_size, pad_size),
            mode='reflect'
        )
        kernel_3d = kernel.view(1, 1, -1)
        smoothed = F.conv1d(grad_padded, kernel_3d)
        
        return smoothed.view(original_shape)
    
    def _update_local_kernel(
        self, 
        state: Dict[str, Any], 
        grad: torch.Tensor, 
        local_lr: float
    ):
        """
        Обновление локального ядра на основе локального loss.
        
        Используем численное дифференцирование для обновления ядра.
        
        Args:
            state: состояние параметра
            grad: градиент параметра
            local_lr: learning rate для ядра
        """
        kernel = state['kernel']
        eps = 1e-4
        
        # Численный градиент для каждого элемента ядра
        kernel_grad = torch.zeros_like(kernel)
        
        for i in range(len(kernel)):
            # Ядро с возмущением +eps
            kernel_plus = kernel.clone()
            kernel_plus[i] += eps
            kernel_plus = kernel_plus / kernel_plus.sum()
            
            # Ядро с возмущением -eps
            kernel_minus = kernel.clone()
            kernel_minus[i] -= eps
            kernel_minus = kernel_minus / kernel_minus.sum()
            
            # Вычисляем loss для обоих возмущений
            smoothed_plus = self._smooth_gradient(grad, kernel_plus)
            smoothed_minus = self._smooth_gradient(grad, kernel_minus)
            
            loss_plus = self._local_loss(grad, smoothed_plus)
            loss_minus = self._local_loss(grad, smoothed_minus)
            
            # Численный градиент
            kernel_grad[i] = (loss_plus - loss_minus) / (2 * eps)
        
        # Обновляем ядро
        new_kernel = kernel - local_lr * kernel_grad
        
        # Проекция на симплекс (все элементы >= 0, сумма = 1)
        new_kernel = F.relu(new_kernel)  # Неотрицательность
        if new_kernel.sum() > 0:
            new_kernel = new_kernel / new_kernel.sum()  # Нормализация
        else:
            # Если все элементы стали <= 0, возвращаем равномерное ядро
            new_kernel = torch.ones_like(kernel) / len(kernel)
        
        state['kernel'] = new_kernel
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Выполняет один шаг оптимизации с локальной адаптацией ядер.
        
        Алгоритм для каждого параметра:
        1. Сглаживаем градиент локальным ядром
        2. Вычисляем локальный loss
        3. Обновляем локальное ядро
        4. Применяем моментум
        5. Обновляем веса
        
        Args:
            closure: функция, вычисляющая loss (опционально)
            
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
            kernel_size = group['kernel_size']
            local_lr = group['local_lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self._init_state(p, kernel_size)
                grad = p.grad.clone()  # Копируем градиент
                kernel = state['kernel']
                
                # === Сглаживаем градиент ===
                smoothed_grad = self._smooth_gradient(grad, kernel)
                
                # === Вычисляем локальный loss ===
                with torch.enable_grad():
                    local_l = self._local_loss(grad, smoothed_grad)
                    state['local_loss_history'].append(local_l.item())
                
                # === Обновляем локальное ядро ===
                self._update_local_kernel(state, grad, local_lr)
                
                # === Применяем моментум ===
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(smoothed_grad)
                
                # === Обновляем веса ===
                p.add_(buf, alpha=-lr)
        
        return loss
    
    def get_kernels(self) -> Dict[int, torch.Tensor]:
        """
        Возвращает все локальные ядра.
        
        Returns:
            Словарь {id параметра: ядро}
        """
        kernels = {}
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state and 'kernel' in self.state[p]:
                    kernels[id(p)] = self.state[p]['kernel'].clone()
        return kernels
    
    def get_local_losses(self) -> Dict[int, list]:
        """
        Возвращает историю локальных loss'ов для каждого параметра.
        
        Returns:
            Словарь {id параметра: список loss'ов}
        """
        losses = {}
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state and 'local_loss_history' in self.state[p]:
                    losses[id(p)] = self.state[p]['local_loss_history'].copy()
        return losses
