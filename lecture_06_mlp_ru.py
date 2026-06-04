import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")

class MLP(nn.Module):
    """Simple MLP: linear -> GeLU -> linear -> GeLU -> ... -> linear -> GeLU"""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        # Отметьте весь передний проход
        # EN: Mark the entire forward pass
        for i, layer in enumerate(self.layers):
            # Отметьте отдельно расчеты каждого слоя
            # EN: Mark each layer's computation separately
            with nvtx.range(f"layer_{i}"):
                x = layer(x)
                x = torch.nn.functional.gelu(x)
        
        return x

def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int, use_optimizer: bool = False):
    """Run forward and backward passes through an MLP.
    
    Args:
        dim: Dimension of each layer
        num_layers: Number of linear+GeLU layers
        batch_size: Number of samples to process at once
        num_steps: Number of forward/backward iterations
        use_optimizer: Whether to use Adam optimizer for weight updates
    """
    # Определение модели (с случайными весами)
    # EN: Define a model (with random weights)
    with nvtx.range("define_model"):
        model = MLP(dim, num_layers).to(get_device())
    
    # Инициировать оптимизацию при поступлении соответствующей просьбы
    # EN: Initialize optimizer if requested
    optimizer = torch.optim.Adam(model.parameters()) if use_optimizer else None

    # Определить вход (случайность)
    # EN: Define an input (random)
    with nvtx.range("define_input"):
        x = torch.randn(batch_size, dim, device=get_device())

    # Запустить модель "num_steps" раз
    # EN: Run the model `num_steps` times
    for step in range(num_steps):
        if step > 10:
            # начать профилирование после 10 итераций разогрева
            # EN: start profiling after 10 warmup iterations
            torch.cuda.cudart().cudaProfilerStart()

        nvtx.range_push(f"step_{step}")
        
        # Ноль градиентов
        # EN: Zero gradients
        if use_optimizer:
            optimizer.zero_grad()
        else:
            model.zero_grad(set_to_none=True)

        # Вперед
        # EN: Forward
        with nvtx.range("forward"):
            y = model(x).mean()

        # Назад
        # EN: Backward
        with nvtx.range("backward"):
            y.backward()

        # Оптимизатор при включении
        # EN: Optimizer step if enabled
        if use_optimizer:
            with nvtx.range("optimizer_step"):
                # отпечаток (f) "Штап {step}, потеря: {y.item(:6f})]
                # EN: print(f"Step {step}, loss: {y.item():.6f}")
                optimizer.step()
        
        nvtx.range_pop()

def main():
    # Запустить большую модель, если GPU доступен
    # EN: Run a larger model if GPU is available
    if torch.cuda.is_available():
        print("Running on GPU")
        run_mlp(dim=4096, num_layers=64, batch_size=1024, num_steps=15, use_optimizer=True)
    else:
        print("Running on CPU")
        run_mlp(dim=128, num_layers=16, batch_size=128, num_steps=15, use_optimizer=True)

if __name__ == "__main__":
    main()