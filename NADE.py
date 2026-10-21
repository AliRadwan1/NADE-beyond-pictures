import torch
import torch.nn as nn
import torch.nn.functional as F

class NADE(nn.Module):
    """Neural Autoregressive Distribution Estimator (NADE).
    
    Models binary vector distributions by predicting each pixel/feature 
    conditioned on all previous pixels.
    
    Attributes:
        D (int): Input feature dimension (e.g., 100 for text, 512 for speech).
        H (int): Hidden layer dimension (e.g., 50).
        W (nn.Parameter): Weight matrix for updating memory state, shape (D, H).
        V (nn.Parameter): Read weight matrix for predicting logits, shape (D, H).
        b (nn.Parameter): Output bias vector, shape (D,).
        c (nn.Parameter): Initial hidden memory state, shape (H,).
    """

    def __init__(self, D: int, H: int):
        """Initializes NADE layer parameters.

        Args:
            D (int): Input dimension (number of features/pixels).
            H (int): Hidden state dimension.
        """
        super().__init__()
        self.D = D
        self.H = H
        
        # Trainable parameters
        self.W = nn.Parameter(torch.randn(D, H))
        self.V = nn.Parameter(torch.randn(D, H))
        self.b = nn.Parameter(torch.zeros(D))
        self.c = nn.Parameter(torch.zeros(H))
          
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes unnormalized prediction logits for each feature.

        Args:
            x (torch.Tensor): Binary input tensor of shape (batch_size, D).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, D).
        """
        batch_size = x.size(0)
        # Expand initial memory c across batch dimension: shape (batch_size, H)
        memory = self.c.repeat(batch_size, 1)
        logits = []
        
        for i in range(self.D):
            # 1. Compute hidden representation for step i: shape (batch_size, H)
            h = torch.sigmoid(memory)
            
            # 2. Compute logit prediction for pixel i: shape (batch_size,)
            z = self.b[i] + (h * self.V[i]).sum(dim=1)
            logits.append(z)
            
            # 3. Update memory using true pixel i value: shape (batch_size, H)
            memory = memory + x[:, i].unsqueeze(1) * self.W[i]
        
        # Stack logits along feature dimension: shape (batch_size, D)
        return torch.stack(logits, dim=1)
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """Generates new synthetic binary samples pixel-by-pixel.

        Args:
            num_samples (int): Number of binary vectors to generate.

        Returns:
            torch.Tensor: Generated binary samples of shape (num_samples, D).
        """
        # Detect which device the model weights are currently on
        device = self.W.device
        
        samples = torch.zeros((num_samples, self.D), device=device)
        memory = self.c.unsqueeze(0).expand(num_samples, -1) # ensure memory is on same device
        
        for i in range(self.D):
            # Compute hidden state and prediction probability for pixel i
            h = torch.sigmoid(memory)
            z = self.b[i] + (h * self.V[i]).sum(dim=1)
            p_i = torch.sigmoid(z)
            
            # Draw binary choice (0 or 1) based on computed probability
            x_i = torch.bernoulli(p_i)
            samples[:, i] = x_i
            
            # Update memory state using generated binary sample
            memory = memory + samples[:, i].unsqueeze(1) * self.W[i]
        
        return samples
    
    def compute_surprise(self, x: torch.Tensor) -> torch.Tensor:
        """Computes average surprise (binary cross-entropy loss) per pixel.

        Args:
            x (torch.Tensor): Input binary tensor of shape (batch_size, D).

        Returns:
            torch.Tensor: Scalar tensor containing average surprise per pixel.
        """
        logits = self.forward(x)
        return F.binary_cross_entropy_with_logits(logits, x)