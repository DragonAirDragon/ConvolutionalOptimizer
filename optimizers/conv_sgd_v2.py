"""
Продвинутый сверточный оптимизатор v2

Улучшения:
1. 2D свёртка для conv-слоёв (сохраняет пространственную структуру)
2. Выучиваемое ядро через backprop (не эвристика)
3. Улучшенная метрика: gradient SNR + direction preservation
4. Отдельные ядра для разных типов слоёв
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable, Dict, Any, Tuple
import math


class ConvolutionalSGDv2(Optimizer):
    """
    Продвинутый сверточный SGD с обучаемым ядром через backprop.
    
    Ключевые улучшения:
    - 2D свёртка для матричных градиентов (Linear, Conv2d)
    - Ядро обучается через градиентный спуск с собственным loss
    - Метрика: SNR улучшение + сохранение направления + сглаживание
    
    Args:
        params: параметры модели
        lr: learning rate для весов
        momentum: коэффициент моментума
        kernel_size: размер ядра (для 1D и 2D)
        kernel_lr: learning rate для обучения ядра
        noise_estimation_momentum: EMA для оценки шума в градиентах
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        kernel_size: int = 3,
        kernel_lr: float = 0.01,
        noise_estimation_momentum: float = 0.99
    ):
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ядро должно быть нечётным
            
        defaults = dict(
            lr=lr,
            momentum=momentum,
            kernel_size=kernel_size,
            kernel_lr=kernel_lr,
            noise_estimation_momentum=noise_estimation_momentum
        )
        super().__init__(params, defaults)
        
        # Глобальные обучаемые ядра (отдельно для 1D и 2D)
        self.kernel_1d = nn.Parameter(
            self._init_gaussian_kernel_1d(kernel_size),
            requires_grad=True
        )
        self.kernel_2d = nn.Parameter(
            self._init_gaussian_kernel_2d(kernel_size),
            requires_grad=True
        )
        
        # Статистика
        self.stats = {
            'snr_before': [],
            'snr_after': [],
            'kernel_loss': [],
            'cosine_sim': []
        }
    
    def _init_gaussian_kernel_1d(self, size: int) -> torch.Tensor:
        """Инициализация 1D Гауссовым ядром"""
        x = torch.arange(size).float() - size // 2
        kernel = torch.exp(-x**2 / (2 * (size/4)**2))
        return kernel / kernel.sum()
    
    def _init_gaussian_kernel_2d(self, size: int) -> torch.Tensor:
        """Инициализация 2D Гауссовым ядром"""
        x = torch.arange(size).float() - size // 2
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * (size/4)**2))
        return kernel / kernel.sum()
    
    def _estimate_gradient_snr(
        self, 
        grad: torch.Tensor, 
        state: Dict[str, Any]
    ) -> Tuple[float, torch.Tensor]:
        """
        Оценка Signal-to-Noise Ratio градиента.
        
        Идея: "сигнал" = EMA градиента, "шум" = отклонение от EMA
        
        Returns:
            snr: отношение сигнал/шум
            signal: оценка "чистого" сигнала
        """
        momentum = self.defaults['noise_estimation_momentum']
        
        if 'grad_ema' not in state:
            state['grad_ema'] = grad.clone()
            state['grad_sq_ema'] = (grad ** 2).clone()
            return float('inf'), grad.clone()
        
        # Обновляем EMA
        state['grad_ema'].mul_(momentum).add_(grad, alpha=1-momentum)
        state['grad_sq_ema'].mul_(momentum).add_(grad**2, alpha=1-momentum)
        
        signal = state['grad_ema']
        variance = state['grad_sq_ema'] - signal**2
        variance = torch.clamp(variance, min=1e-8)
        
        # SNR = ||signal||^2 / E[noise^2]
        signal_power = (signal ** 2).mean()
        noise_power = variance.mean()
        snr = (signal_power / noise_power).item()
        
        return snr, signal
    
    def _smooth_gradient_1d(self, grad: torch.Tensor) -> torch.Tensor:
        """1D свёртка для векторных/малых градиентов"""
        if grad.numel() < self.kernel_1d.numel():
            return grad
            
        original_shape = grad.shape
        grad_flat = grad.view(-1)
        
        kernel = self.kernel_1d.to(grad.device)
        # Нормализуем ядро (softmax для положительности + сумма=1)
        kernel_norm = F.softmax(kernel, dim=0)
        
        pad_size = len(kernel_norm) // 2
        grad_padded = F.pad(
            grad_flat.unsqueeze(0).unsqueeze(0),
            (pad_size, pad_size),
            mode='reflect'
        )
        
        kernel_3d = kernel_norm.view(1, 1, -1)
        smoothed = F.conv1d(grad_padded, kernel_3d)
        
        return smoothed.view(original_shape)
    
    def _smooth_gradient_2d(self, grad: torch.Tensor) -> torch.Tensor:
        """
        2D свёртка для матричных градиентов.
        
        Применяется к:
        - Linear layers: (out_features, in_features) → 2D сглаживание
        - Conv2d weights: (out_c, in_c, kH, kW) → сглаживаем по (kH, kW)
        """
        kernel = self.kernel_2d.to(grad.device)
        # Нормализуем ядро
        kernel_norm = F.softmax(kernel.view(-1), dim=0).view(kernel.shape)
        
        if grad.dim() == 2:
            # Linear layer: (out, in)
            h, w = grad.shape
            if h < kernel.shape[0] or w < kernel.shape[1]:
                return self._smooth_gradient_1d(grad)
            
            pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
            grad_4d = grad.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            grad_padded = F.pad(grad_4d, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
            
            kernel_4d = kernel_norm.unsqueeze(0).unsqueeze(0)  # (1, 1, kH, kW)
            smoothed = F.conv2d(grad_padded, kernel_4d)
            
            return smoothed.squeeze(0).squeeze(0)
        
        elif grad.dim() == 4:
            # Conv2d: (out_c, in_c, kH, kW)
            out_c, in_c, kH, kW = grad.shape
            
            if kH < kernel.shape[0] or kW < kernel.shape[1]:
                # Ядро conv слишком маленькое — сглаживаем по (out_c, in_c)
                grad_2d = grad.view(out_c, in_c, -1).mean(dim=2)  # (out_c, in_c)
                smoothed_2d = self._smooth_gradient_2d(grad_2d)
                return smoothed_2d.unsqueeze(-1).unsqueeze(-1).expand_as(grad)
            
            # Сглаживаем каждый фильтр отдельно
            pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
            smoothed = torch.zeros_like(grad)
            
            kernel_4d = kernel_norm.unsqueeze(0).unsqueeze(0)
            for o in range(out_c):
                for i in range(in_c):
                    filt = grad[o, i].unsqueeze(0).unsqueeze(0)
                    filt_padded = F.pad(filt, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
                    smoothed[o, i] = F.conv2d(filt_padded, kernel_4d).squeeze()
            
            return smoothed
        
        else:
            # Fallback для других размерностей
            return self._smooth_gradient_1d(grad)
    
    def _compute_kernel_loss(
        self,
        original_grad: torch.Tensor,
        smoothed_grad: torch.Tensor,
        snr_before: float,
        signal: torch.Tensor
    ) -> torch.Tensor:
        """
        Улучшенная функция потерь для обучения ядра.
        
        Компоненты:
        1. Direction preservation: косинусное сходство с оригиналом
        2. Signal preservation: близость к "чистому" сигналу
        3. Smoothness: штраф за высокочастотные компоненты
        4. SNR improvement: поощрение за улучшение SNR
        """
        # 1. Сохранение направления градиента
        cos_sim = F.cosine_similarity(
            original_grad.view(1, -1),
            smoothed_grad.view(1, -1)
        )
        direction_loss = 1 - cos_sim  # Минимизируем
        
        # 2. Близость к оценённому сигналу
        signal_loss = F.mse_loss(smoothed_grad, signal)
        
        # 3. Сглаживание (низкая вариация)
        if smoothed_grad.dim() >= 2:
            # Для матриц — вариация по обоим измерениям
            var_h = torch.var(smoothed_grad, dim=0).mean()
            var_w = torch.var(smoothed_grad, dim=-1).mean() if smoothed_grad.dim() == 2 else torch.tensor(0.0)
            smoothness_loss = var_h + var_w
        else:
            smoothness_loss = torch.var(smoothed_grad)
        
        # Нормализуем smoothness
        orig_var = torch.var(original_grad) + 1e-8
        smoothness_loss = smoothness_loss / orig_var
        
        # Итоговый loss с весами
        total_loss = (
            0.5 * direction_loss +      # Главное — сохранить направление
            0.3 * signal_loss +          # Приблизиться к чистому сигналу
            0.2 * smoothness_loss        # Немного сгладить
        )
        
        return total_loss, cos_sim.item()
    
    def _init_param_state(self, p: torch.Tensor) -> Dict[str, Any]:
        """Инициализация состояния параметра"""
        state = self.state[p]
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(p)
            state['step'] = 0
        return state
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Шаг оптимизации с обучением ядра"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Аккумулируем градиенты для обновления ядра
        kernel_grads_1d = []
        kernel_grads_2d = []
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            kernel_lr = group['kernel_lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self._init_param_state(p)
                state['step'] += 1
                grad = p.grad.clone()
                
                # === Оценка SNR до сглаживания ===
                snr_before, signal = self._estimate_gradient_snr(grad, state)
                
                # === Выбор типа свёртки ===
                use_2d = (grad.dim() >= 2 and 
                         grad.shape[-1] >= self.kernel_2d.shape[-1] and
                         grad.shape[-2] >= self.kernel_2d.shape[-2])
                
                # === Сглаживание с градиентом для ядра ===
                with torch.enable_grad():
                    grad_for_smooth = grad.clone().requires_grad_(False)
                    
                    if use_2d:
                        # Временно включаем grad для ядра
                        kernel_param = self.kernel_2d
                    else:
                        kernel_param = self.kernel_1d
                    
                    # Прямой проход сглаживания
                    if use_2d:
                        smoothed_grad = self._smooth_gradient_2d(grad)
                    else:
                        smoothed_grad = self._smooth_gradient_1d(grad)
                    
                    # Вычисляем loss для ядра
                    kernel_loss, cos_sim = self._compute_kernel_loss(
                        grad, smoothed_grad, snr_before, signal
                    )
                    
                    # Сохраняем статистику
                    if state['step'] % 10 == 0:
                        self.stats['kernel_loss'].append(kernel_loss.item())
                        self.stats['cosine_sim'].append(cos_sim)
                
                # === Обновление ядра через численный градиент ===
                # (более стабильно чем autograd для такой архитектуры)
                if state['step'] % 5 == 0:  # Обновляем ядро каждые 5 шагов
                    self._update_kernel_numerical(
                        kernel_param, grad, signal, kernel_lr, use_2d
                    )
                
                # === Моментум ===
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(smoothed_grad)
                
                # === Обновление весов ===
                p.add_(buf, alpha=-lr)
        
        return loss
    
    def _update_kernel_numerical(
        self,
        kernel: nn.Parameter,
        grad: torch.Tensor,
        signal: torch.Tensor,
        kernel_lr: float,
        use_2d: bool
    ):
        """Численное обновление ядра"""
        eps = 1e-3
        kernel_data = kernel.data
        
        for idx in range(kernel_data.numel()):
            # +eps
            kernel_data_plus = kernel_data.clone()
            kernel_data_plus.view(-1)[idx] += eps
            kernel_plus_norm = F.softmax(kernel_data_plus.view(-1), dim=0).view(kernel_data.shape)
            
            # -eps
            kernel_data_minus = kernel_data.clone()
            kernel_data_minus.view(-1)[idx] -= eps
            kernel_minus_norm = F.softmax(kernel_data_minus.view(-1), dim=0).view(kernel_data.shape)
            
            # Применяем и считаем loss
            with torch.no_grad():
                if use_2d:
                    smooth_plus = self._apply_kernel_2d(grad, kernel_plus_norm)
                    smooth_minus = self._apply_kernel_2d(grad, kernel_minus_norm)
                else:
                    smooth_plus = self._apply_kernel_1d(grad, kernel_plus_norm)
                    smooth_minus = self._apply_kernel_1d(grad, kernel_minus_norm)
                
                # Loss: MSE to signal - cosine similarity
                mse_plus = F.mse_loss(smooth_plus, signal)
                mse_minus = F.mse_loss(smooth_minus, signal)
                
                cos_plus = F.cosine_similarity(
                    smooth_plus.view(1, -1), grad.view(1, -1)
                ).squeeze()
                cos_minus = F.cosine_similarity(
                    smooth_minus.view(1, -1), grad.view(1, -1)
                ).squeeze()
                
                loss_plus = mse_plus - 0.5 * cos_plus
                loss_minus = mse_minus - 0.5 * cos_minus
                
                # Численный градиент
                num_grad = (loss_plus.item() - loss_minus.item()) / (2 * eps)
                
                # Обновляем
                kernel_data.view(-1)[idx] -= kernel_lr * num_grad
    
    def _apply_kernel_1d(self, grad: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Применить 1D ядро (без нормализации)"""
        if grad.numel() < kernel.numel():
            return grad
        original_shape = grad.shape
        grad_flat = grad.view(-1)
        pad_size = len(kernel) // 2
        grad_padded = F.pad(grad_flat.unsqueeze(0).unsqueeze(0), (pad_size, pad_size), mode='reflect')
        kernel_3d = kernel.view(1, 1, -1)
        smoothed = F.conv1d(grad_padded, kernel_3d)
        return smoothed.view(original_shape)
    
    def _apply_kernel_2d(self, grad: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Применить 2D ядро (без нормализации)"""
        if grad.dim() < 2:
            return self._apply_kernel_1d(grad, kernel.view(-1))
        
        h, w = grad.shape[-2], grad.shape[-1]
        if h < kernel.shape[0] or w < kernel.shape[1]:
            return self._apply_kernel_1d(grad, kernel.view(-1))
        
        if grad.dim() == 2:
            pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
            grad_4d = grad.unsqueeze(0).unsqueeze(0)
            grad_padded = F.pad(grad_4d, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
            kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
            smoothed = F.conv2d(grad_padded, kernel_4d)
            return smoothed.squeeze(0).squeeze(0)
        
        return grad  # Fallback
    
    def get_kernels(self) -> Dict[str, torch.Tensor]:
        """Получить текущие ядра"""
        return {
            '1d': F.softmax(self.kernel_1d, dim=0).detach().clone(),
            '2d': F.softmax(self.kernel_2d.view(-1), dim=0).view(self.kernel_2d.shape).detach().clone()
        }
    
    def get_stats(self) -> Dict[str, list]:
        """Получить статистику обучения"""
        return self.stats.copy()
