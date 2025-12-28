"""
Стресс-тест: сравнение оптимизаторов при высоком шуме в градиентах

Условия:
- Маленький batch_size (8) → зашумлённые градиенты
- Noisy labels (10% label noise)
- Это сценарий где сглаживание должно помочь!

Запуск:
    python demo_noisy.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from typing import Dict, Callable, List, Tuple
import time
import random

from torchvision import datasets, transforms

from optimizers.conv_sgd import ConvolutionalSGD
from optimizers.conv_sgd_v2 import ConvolutionalSGDv2
from models.test_networks import SimpleNet, DeepNet


class NoisyLabelDataset(torch.utils.data.Dataset):
    """Датасет с искусственным шумом в метках"""
    
    def __init__(self, dataset, noise_rate: float = 0.1, num_classes: int = 10):
        self.dataset = dataset
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        
        # Генерируем зашумлённые метки один раз
        self.noisy_labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            if random.random() < noise_rate:
                # Случайная неправильная метка
                wrong_labels = [l for l in range(num_classes) if l != label]
                label = random.choice(wrong_labels)
            self.noisy_labels.append(label)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img, self.noisy_labels[idx]


def load_noisy_mnist(
    data_dir: str = "./data",
    train_samples: int = 5000,
    test_samples: int = 1000,
    batch_size: int = 8,  # Маленький батч!
    noise_rate: float = 0.1  # 10% шума в метках
) -> Tuple[DataLoader, DataLoader]:
    """Загрузка MNIST с шумом"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print(f"📥 Загрузка MNIST с шумом...")
    print(f"   batch_size={batch_size}, noise_rate={noise_rate*100:.0f}%")
    
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    # Ограничиваем размер
    train_dataset = Subset(train_dataset, range(train_samples))
    test_dataset = Subset(test_dataset, range(test_samples))
    
    # Добавляем шум к train
    train_noisy = NoisyLabelDataset(train_dataset, noise_rate=noise_rate)
    
    train_loader = DataLoader(train_noisy, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"✓ Загружено: {len(train_noisy)} train (noisy), {len(test_dataset)} test (clean)")
    
    return train_loader, test_loader


def train_with_tracking(
    model: nn.Module,
    optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 30,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """Обучение с детальным трекингом"""
    
    criterion = nn.CrossEntropyLoss()
    history = {
        'train_loss': [],
        'test_loss': [],
        'accuracy': [],
        'grad_norm': [],
        'time': []
    }
    
    for epoch in range(epochs):
        start = time.time()
        model.train()
        epoch_loss = 0
        epoch_grad_norm = 0
        n_batches = 0
        
        for X, y in train_loader:
            X = X.view(X.size(0), -1)  # Flatten
            
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            
            # Трекинг нормы градиента
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            epoch_grad_norm += total_norm ** 0.5
            
            # Для наших оптимизаторов с closure
            if hasattr(optimizer, 'kernel_1d') or hasattr(optimizer, 'kernel'):
                def closure():
                    return criterion(model(X), y)
                optimizer.step(closure)
            else:
                optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        history['train_loss'].append(epoch_loss / n_batches)
        history['grad_norm'].append(epoch_grad_norm / n_batches)
        
        # Eval
        model.eval()
        correct = total = 0
        test_loss = 0
        with torch.no_grad():
            for X, y in test_loader:
                X = X.view(X.size(0), -1)
                out = model(X)
                test_loss += criterion(out, y).item()
                correct += (out.argmax(1) == y).sum().item()
                total += len(y)
        
        history['test_loss'].append(test_loss / len(test_loader))
        history['accuracy'].append(100 * correct / total)
        history['time'].append(time.time() - start)
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}: Loss={history['train_loss'][-1]:.4f}, "
                  f"Acc={history['accuracy'][-1]:.1f}%, "
                  f"GradNorm={history['grad_norm'][-1]:.2f}")
    
    return history


def run_noisy_comparison():
    """Главное сравнение при высоком шуме"""
    
    print("="*70)
    print("  СТРЕСС-ТЕСТ: Маленький батч + Шумные метки")
    print("  Здесь сглаживание градиентов должно помочь!")
    print("="*70)
    
    torch.manual_seed(42)
    random.seed(42)
    
    train_loader, test_loader = load_noisy_mnist(
        train_samples=5000,
        test_samples=1000,
        batch_size=8,      # Очень маленький → много шума
        noise_rate=0.15    # 15% неправильных меток
    )
    
    optimizers_config = {
        'SGD': lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
        'Adam': lambda p: torch.optim.Adam(p, lr=0.001),
        'ConvSGD v1': lambda p: ConvolutionalSGD(p, lr=0.01, momentum=0.9, kernel_size=5),
        'ConvSGD v2': lambda p: ConvolutionalSGDv2(p, lr=0.01, momentum=0.9, kernel_size=5, kernel_lr=0.005),
    }
    
    results = {}
    
    for name, opt_fn in optimizers_config.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print('='*50)
        
        torch.manual_seed(42)
        model = SimpleNet(input_dim=784, hidden_dim=128, output_dim=10)
        optimizer = opt_fn(model.parameters())
        
        results[name] = train_with_tracking(
            model, optimizer, train_loader, test_loader, epochs=30
        )
        
        # Для v2 выводим ядро
        if hasattr(optimizer, 'get_kernels'):
            kernels = optimizer.get_kernels()
            print(f"\n  Финальное 1D ядро: {kernels['1d'].numpy().round(3)}")
    
    return results


def visualize_noisy_results(results: Dict[str, Dict], filename: str = "noisy_comparison.png"):
    """Визуализация с дополнительными метриками"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Train Loss
    ax = axes[0, 0]
    for (name, hist), color in zip(results.items(), colors):
        ax.plot(hist['train_loss'], label=name, color=color, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss (noisy data)')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 2. Test Accuracy
    ax = axes[0, 1]
    for (name, hist), color in zip(results.items(), colors):
        ax.plot(hist['accuracy'], label=name, color=color, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Test Accuracy (clean labels)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Gradient Norm
    ax = axes[1, 0]
    for (name, hist), color in zip(results.items(), colors):
        ax.plot(hist['grad_norm'], label=name, color=color, linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Avg Gradient Norm')
    ax.set_title('Gradient Norm (lower = more stable)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Train vs Test Loss Gap (overfitting indicator)
    ax = axes[1, 1]
    for (name, hist), color in zip(results.items(), colors):
        gap = [tr - te for tr, te in zip(hist['train_loss'], hist['test_loss'])]
        ax.plot(gap, label=name, color=color, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train - Test Loss')
    ax.set_title('Generalization Gap (closer to 0 = better)')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Stress Test: batch_size=8, 15% label noise', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ График сохранён: {filename}")


def print_final_summary(results: Dict[str, Dict]):
    """Финальная таблица"""
    
    print("\n" + "="*75)
    print("  ИТОГИ СТРЕСС-ТЕСТА")
    print("="*75)
    print(f"{'Оптимизатор':<20} {'Final Loss':>12} {'Best Acc':>12} {'Final Acc':>12} {'Stability':>12}")
    print("-"*75)
    
    for name, hist in results.items():
        final_loss = hist['train_loss'][-1]
        best_acc = max(hist['accuracy'])
        final_acc = hist['accuracy'][-1]
        # Stability = std of gradient norms (lower = more stable)
        stability = torch.tensor(hist['grad_norm']).std().item()
        
        print(f"{name:<20} {final_loss:>12.4f} {best_acc:>11.1f}% {final_acc:>11.1f}% {stability:>12.2f}")
    
    print("-"*75)
    
    # Winners
    best_acc_opt = max(results.items(), key=lambda x: max(x[1]['accuracy']))
    best_stable = min(results.items(), key=lambda x: torch.tensor(x[1]['grad_norm']).std().item())
    
    print(f"\n🏆 Лучшая точность: {best_acc_opt[0]} ({max(best_acc_opt[1]['accuracy']):.1f}%)")
    print(f"🏆 Самый стабильный: {best_stable[0]} (std grad norm = {torch.tensor(best_stable[1]['grad_norm']).std():.2f})")


def run_batch_size_ablation():
    """Исследование влияния batch_size"""
    
    print("\n\n" + "="*70)
    print("  ABLATION: Влияние размера батча")
    print("="*70)
    
    batch_sizes = [8, 16, 32, 64]
    results_by_batch = {}
    
    for bs in batch_sizes:
        print(f"\n--- batch_size = {bs} ---")
        
        torch.manual_seed(42)
        random.seed(42)
        
        train_loader, test_loader = load_noisy_mnist(
            train_samples=3000,
            test_samples=500,
            batch_size=bs,
            noise_rate=0.1
        )
        
        # Сравниваем только SGD vs ConvSGD v2
        for opt_name, opt_fn in [
            ('SGD', lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9)),
            ('ConvSGD v2', lambda p: ConvolutionalSGDv2(p, lr=0.01, momentum=0.9, kernel_size=5))
        ]:
            torch.manual_seed(42)
            model = SimpleNet(input_dim=784, hidden_dim=64, output_dim=10)
            optimizer = opt_fn(model.parameters())
            
            hist = train_with_tracking(model, optimizer, train_loader, test_loader, epochs=15, verbose=False)
            
            key = f"{opt_name} (bs={bs})"
            results_by_batch[key] = hist
            print(f"  {opt_name}: Acc={max(hist['accuracy']):.1f}%")
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sgd_accs = []
    conv_accs = []
    
    for bs in batch_sizes:
        sgd_accs.append(max(results_by_batch[f'SGD (bs={bs})']['accuracy']))
        conv_accs.append(max(results_by_batch[f'ConvSGD v2 (bs={bs})']['accuracy']))
    
    x = range(len(batch_sizes))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], sgd_accs, width, label='SGD', color='#1f77b4')
    ax.bar([i + width/2 for i in x], conv_accs, width, label='ConvSGD v2', color='#2ca02c')
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_title('Batch Size Ablation: SGD vs ConvSGD v2')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Добавляем разницу
    for i, (sgd, conv) in enumerate(zip(sgd_accs, conv_accs)):
        diff = conv - sgd
        color = 'green' if diff > 0 else 'red'
        ax.annotate(f'{diff:+.1f}%', xy=(i, max(sgd, conv) + 1), ha='center', color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('batch_size_ablation.png', dpi=150)
    print(f"\n✓ График сохранён: batch_size_ablation.png")
    
    return results_by_batch


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║          СТРЕСС-ТЕСТ СВЕРТОЧНОГО ОПТИМИЗАТОРА                ║
    ║                                                               ║
    ║   Условия: batch_size=8, 15% label noise                     ║
    ║   Цель: показать преимущество сглаживания градиентов         ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Главное сравнение
    results = run_noisy_comparison()
    visualize_noisy_results(results)
    print_final_summary(results)
    
    # Ablation study
    run_batch_size_ablation()
    
    print("\n\n" + "="*70)
    print("  ГОТОВО!")
    print("  - noisy_comparison.png — сравнение при высоком шуме")
    print("  - batch_size_ablation.png — влияние размера батча")
    print("="*70)
