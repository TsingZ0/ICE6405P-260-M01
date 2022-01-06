import torch


def filter_input(self, x: torch.Tensor) -> torch.Tensor:
    # Convert dtype
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    # Add extra dimension
    if x.dim() == 3:
        x = x.unsqueeze(0)

    _, _, height, width = x.shape

    # Interpolate large image
    if height != self.INPUT_HEIGHT or width != self.INPUT_WIDTH:
        x = torch.nn.functional.interpolate(x, (28, 28))

    # Normalize
    if x.max() > 1:
        x = x - x.min() / (x.max() - x.min())

    return x