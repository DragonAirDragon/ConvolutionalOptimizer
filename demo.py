"""
Демонстрация: сравнение сверточных оптимизаторов со стандартными

Этот скрипт:
1. Загружает датасет MNIST
2. Обучает одинаковые модели разными оптимизаторами
3. Строит графики сравнения (loss, accuracy)
4. Выводит итоговую таблицу

Запуск:
    python demo.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import matplotlib.pyplot as plt
from typing import Dict, Callable, List, Tuple, Optional
import time
import os

# Для загрузки датасета
from torchvision import datasets, transforms

# Импорты наших модулей
from optimizers.conv_sgd import ConvolutionalSGD
from optimizers.local_loss_sgd import LocalLossConvSGD
from models.test_networks import SimpleNet, DeepNet, IllConditionedNet


def load_mnist(
    data_dir: str = "./data",
    train_samples: Optional[int] = 5000,
    test_samples: Optional[int] = 1000,
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Загрузка датасета MNIST.
    
    Args:
        data_dir: директория для сохранения датасета
        train_samples: количество примеров для обучения (None = все)
        test_samples: количество примеров для теста (None = все)
        batch_size: размер батча
        
    Returns:
        train_loader: DataLoader для обучения
        test_loader: DataLoader для тестирования
        input_dim: размерность входа (784 для MNIST)
        num_classes: количество классов (10)
    """
    # Трансформации: в тензор + нормализация
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("📥 Загрузка MNIST...")
    
    # Скачиваем датасет
    train_dataset = datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Ограничиваем количество примеров для быстрого эксперимента
    if train_samples is not None and train_samples < len(train_dataset):
        train_dataset = Subset(train_dataset, range(train_samples))
    if test_samples is not None and test_samples < len(test_dataset):
        test_dataset = Subset(test_dataset, range(test_samples))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ MNIST загружен: {len(train_dataset)} train, {len(test_dataset)} test")
    
    return train_loader, test_loader, 784, 10


def generate_data(
    n_samples: int = 1000, 
    n_features: int = 20, 
    n_classes: int = 10,
    noise: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Генерация синтетических данных для классификации (fallback).
    
    Создаём линейно разделимые данные с добавлением шума.
    
    Args:
        n_samples: количество примеров
        n_features: размерность признаков
        n_classes: количество классов
        noise: уровень шума
        
    Returns:
        X: тензор признаков (n_samples, n_features)
        y: тензор меток (n_samples,)
    """
    X = torch.randn(n_samples, n_features)
    W = torch.randn(n_features, n_classes)
    logits = X @ W + noise * torch.randn(n_samples, n_classes)
    y = logits.argmax(dim=1)
    return X, y


def train_and_evaluate(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    epochs: int = 30,
    verbose: bool = True,
    flatten_input: bool = False
) -> Dict[str, List[float]]:
    """
    Обучение и оценка модели.
    
    Args:
        model: нейросеть для обучения
        optimizer: оптимизатор
        train_loader: DataLoader для обучения
        test_loader: DataLoader для тестирования
        epochs: количество эпох
        verbose: выводить прогресс
        flatten_input: выравнивать входные данные (для MNIST)
        
    Returns:
        Словарь с историей: train_loss, test_loss, accuracy, time_per_epoch
    """
    criterion = nn.CrossEntropyLoss()
    history = {
        'train_loss': [], 
        'test_loss': [], 
        'accuracy': [],
        'time_per_epoch': []
    }
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # === TRAIN ===
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            if len(batch) == 2:
                X, y = batch
            else:
                X, y = batch[0], batch[1]
            
            # Выравниваем для полносвязных сетей
            if flatten_input and X.dim() > 2:
                X = X.view(X.size(0), -1)
            
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            
            # Для наших оптимизаторов передаём closure для адаптации ядра
            if isinstance(optimizer, (ConvolutionalSGD, LocalLossConvSGD)):
                def closure():
                    out = model(X)
                    return criterion(out, y)
                optimizer.step(closure)
            else:
                optimizer.step()
            
            epoch_loss += loss.item()
        
        history['train_loss'].append(epoch_loss / len(train_loader))
        
        # === EVALUATE ===
        model.eval()
        correct = 0
        total = 0
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 2:
                    X, y = batch
                else:
                    X, y = batch[0], batch[1]
                
                if flatten_input and X.dim() > 2:
                    X = X.view(X.size(0), -1)
                
                output = model(X)
                test_loss += criterion(output, y).item()
                predictions = output.argmax(dim=1)
                correct += (predictions == y).sum().item()
                total += len(y)
        
        history['test_loss'].append(test_loss / len(test_loader))
        history['accuracy'].append(100 * correct / total)
        history['time_per_epoch'].append(time.time() - start_time)
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}: "
                  f"Loss={history['train_loss'][-1]:.4f}, "
                  f"Acc={history['accuracy'][-1]:.1f}%")
    
    return history


def run_comparison(
    model_class: type = SimpleNet,
    model_name: str = "SimpleNet",
    epochs: int = 30,
    seed: int = 42,
    use_mnist: bool = True,
    input_dim: int = 784,
    output_dim: int = 10
):
    """
    Сравнение оптимизаторов на одной модели.
    
    Args:
        model_class: класс модели для тестирования
        model_name: название модели для отображения
        epochs: количество эпох обучения
        seed: random seed для воспроизводимости
        use_mnist: использовать MNIST или синтетические данные
        input_dim: размерность входа
        output_dim: количество классов
    """
    print("="*60)
    print(f"  СРАВНЕНИЕ ОПТИМИЗАТОРОВ: {model_name}")
    print("="*60)
    
    torch.manual_seed(seed)
    
    if use_mnist:
        train_loader, test_loader, input_dim, output_dim = load_mnist(
            train_samples=5000, 
            test_samples=1000
        )
        flatten_input = True
    else:
        # Синтетические данные (fallback)
        X_train, y_train = generate_data(2000, n_features=input_dim, n_classes=output_dim)
        X_test, y_test = generate_data(500, n_features=input_dim, n_classes=output_dim)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)
        flatten_input = False
    
    # Конфигурация оптимизаторов
    optimizers_config: Dict[str, Callable] = {
        'SGD': lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
        'Adam': lambda p: torch.optim.Adam(p, lr=0.001),
        'ConvSGD': lambda p: ConvolutionalSGD(
            p, lr=0.01, momentum=0.9, kernel_size=3, adaptive_kernel=True
        ),
        'LocalLossConvSGD': lambda p: LocalLossConvSGD(
            p, lr=0.01, momentum=0.9, kernel_size=3
        ),
    }
    
    results: Dict[str, Dict] = {}
    
    for name, opt_fn in optimizers_config.items():
        print(f"\n--- {name} ---")
        torch.manual_seed(seed)  # Одинаковая инициализация весов
        
        # Создаём модель с правильными размерностями
        if model_class == SimpleNet:
            model = SimpleNet(input_dim=input_dim, hidden_dim=128, output_dim=output_dim)
        elif model_class == DeepNet:
            model = DeepNet(input_dim=input_dim, hidden_dim=64, output_dim=output_dim)
        elif model_class == IllConditionedNet:
            model = IllConditionedNet(input_dim=input_dim, output_dim=output_dim)
        else:
            model = model_class()
        
        optimizer = opt_fn(model.parameters())
        results[name] = train_and_evaluate(
            model, optimizer, train_loader, test_loader, epochs, 
            flatten_input=flatten_input
        )
    
    return results


def visualize_results(
    results: Dict[str, Dict], 
    title: str = "Optimizer Comparison",
    save_path: str = "optimizer_comparison.png"
):
    """
    Визуализация результатов сравнения.
    
    Args:
        results: словарь с результатами {optimizer_name: history}
        title: заголовок графика
        save_path: путь для сохранения графика
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for (name, hist), color in zip(results.items(), colors):
        axes[0].plot(hist['train_loss'], label=name, color=color, linewidth=2)
        axes[1].plot(hist['test_loss'], label=name, color=color, linewidth=2)
        axes[2].plot(hist['accuracy'], label=name, color=color, linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Test Loss')
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Test Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ График сохранён: {save_path}")


def print_summary(results: Dict[str, Dict]):
    """Вывод итоговой таблицы результатов"""
    print("\n" + "="*70)
    print("  ИТОГИ")
    print("="*70)
    print(f"{'Оптимизатор':<20} {'Final Loss':>12} {'Final Acc':>12} {'Avg Time/Ep':>15}")
    print("-"*70)
    
    for name, hist in results.items():
        final_loss = hist['train_loss'][-1]
        final_acc = hist['accuracy'][-1]
        avg_time = sum(hist['time_per_epoch']) / len(hist['time_per_epoch'])
        print(f"{name:<20} {final_loss:>12.4f} {final_acc:>11.1f}% {avg_time:>14.4f}s")
    
    print("-"*70)
    
    # Определяем лучший оптимизатор
    best_by_loss = min(results.items(), key=lambda x: x[1]['train_loss'][-1])
    best_by_acc = max(results.items(), key=lambda x: x[1]['accuracy'][-1])
    
    print(f"\n🏆 Лучший по Loss: {best_by_loss[0]} ({best_by_loss[1]['train_loss'][-1]:.4f})")
    print(f"🏆 Лучший по Accuracy: {best_by_acc[0]} ({best_by_acc[1]['accuracy'][-1]:.1f}%)")


def run_all_experiments():
    """Запуск всех экспериментов на MNIST"""
    
    # Эксперимент 1: SimpleNet на MNIST
    print("\n" + "="*70)
    print("  ЭКСПЕРИМЕНТ 1: SimpleNet на MNIST")
    print("="*70)
    results_simple = run_comparison(SimpleNet, "SimpleNet (MNIST)", epochs=20, use_mnist=True)
    visualize_results(results_simple, "SimpleNet on MNIST", "simple_comparison.png")
    print_summary(results_simple)
    
    # Эксперимент 2: DeepNet на MNIST
    print("\n\n" + "="*70)
    print("  ЭКСПЕРИМЕНТ 2: DeepNet (6 слоёв) на MNIST")
    print("="*70)
    results_deep = run_comparison(DeepNet, "DeepNet (MNIST)", epochs=20, use_mnist=True)
    visualize_results(results_deep, "DeepNet on MNIST", "deep_comparison.png")
    print_summary(results_deep)
    
    # Эксперимент 3: IllConditionedNet на MNIST
    print("\n\n" + "="*70)
    print("  ЭКСПЕРИМЕНТ 3: IllConditionedNet на MNIST")
    print("="*70)
    results_ill = run_comparison(IllConditionedNet, "IllConditionedNet (MNIST)", epochs=20, use_mnist=True)
    visualize_results(results_ill, "IllConditionedNet on MNIST", "ill_conditioned_comparison.png")
    print_summary(results_ill)
    
    return {
        'SimpleNet': results_simple,
        'DeepNet': results_deep,
        'IllConditionedNet': results_ill
    }


def demo_kernel_evolution():
    """Демонстрация эволюции ядра во время обучения"""
    print("\n" + "="*70)
    print("  ДЕМО: Эволюция ядра ConvolutionalSGD")
    print("="*70)
    
    torch.manual_seed(42)
    
    # Используем MNIST
    train_loader, _, input_dim, output_dim = load_mnist(train_samples=2000, test_samples=500)
    
    model = SimpleNet(input_dim=input_dim, hidden_dim=128, output_dim=output_dim)
    optimizer = ConvolutionalSGD(model.parameters(), lr=0.01, kernel_size=5, adaptive_kernel=True)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nНачальное ядро: {optimizer.get_kernel().numpy()}")
    
    for epoch in range(10):
        for batch in train_loader:
            X, y = batch
            X = X.view(X.size(0), -1)  # Flatten для MNIST
            
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            
            def closure():
                return criterion(model(X), y)
            optimizer.step(closure)
    
    print(f"Финальное ядро: {optimizer.get_kernel().numpy()}")
    
    # Визуализация истории ядра
    kernel_history = optimizer.get_kernel_history()
    if len(kernel_history) > 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        kernel_matrix = torch.stack(kernel_history).numpy()
        
        for i in range(kernel_matrix.shape[1]):
            ax.plot(kernel_matrix[:, i], label=f'K[{i}]', linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Kernel Value')
        ax.set_title('Kernel Evolution During Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('kernel_evolution.png', dpi=150)
        print(f"\n✓ График эволюции ядра сохранён: kernel_evolution.png")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║          СВЕРТОЧНЫЙ ОПТИМИЗАТОР ДЛЯ НЕЙРОСЕТЕЙ               ║
    ║                                                               ║
    ║   Датасет: MNIST (рукописные цифры)                          ║
    ║   Сравнение: SGD, Adam, ConvSGD, LocalLossConvSGD            ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Запуск всех экспериментов
    all_results = run_all_experiments()
    
    # Демо эволюции ядра
    demo_kernel_evolution()
    
    print("\n\n" + "="*70)
    print("  ГОТОВО! Созданы графики:")
    print("  - simple_comparison.png")
    print("  - deep_comparison.png")
    print("  - ill_conditioned_comparison.png")
    print("  - kernel_evolution.png")
    print("="*70)
