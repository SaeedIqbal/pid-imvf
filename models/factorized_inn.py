import torch
import torch.nn as nn
import math

class AffineCouplingLayer(nn.Module):
    """
    Affine Coupling Layer for the Invertible Neural Network core.
    Splits the input tensor into two halves based on a binary mask.
    Mathematical formulation: 
        y_{~mask} = x_{~mask} * exp(s(x_{mask})) + t(x_{mask})
        log |det J| = sum(s(x_{mask}))
    """
    def __init__(self, dim: int, mask: torch.Tensor):
        super().__init__()
        self.dim = dim
        self.register_buffer('mask', mask)
        
        # Neural network to predict scale (s) and translation (t) parameters
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim * 2)
        )
        
    def forward(self, x: torch.Tensor):
        """
        Forward transformation: x -> y
        """
        masked_x = x * self.mask
        out = self.net(masked_x)
        s, t = out.chunk(2, dim=-1)
        
        y = x.clone()
        # Apply affine transformation only to the unmasked dimensions
        y[:, ~self.mask] = x[:, ~self.mask] * torch.exp(s[:, ~self.mask]) + t[:, ~self.mask]
        
        # Analytical log-determinant of the Jacobian for this layer
        log_det = torch.sum(s[:, ~self.mask], dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor):
        """
        Inverse transformation: y -> x
        """
        masked_y = y * self.mask
        out = self.net(masked_y)
        s, t = out.chunk(2, dim=-1)
        
        x = y.clone()
        x[:, ~self.mask] = (y[:, ~self.mask] - t[:, ~self.mask]) * torch.exp(-s[:, ~self.mask])
        return x


class SubmanifoldProjector(nn.Module):
    """
    Projects high-dimensional views X_v \in R^{d_v} into a low-dimensional 
    data submanifold M_v \in R^{d'}, where d' = d_S + d_U.
    Mathematical formulation: Z' = X_v P, where P is an orthogonal projector.
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Learnable projection matrix P (without bias to maintain linearity)
        self.projection = nn.Linear(input_dim, latent_dim, bias=False)
        self._initialize_orthogonal()
        
    def _initialize_orthogonal(self):
        """
        Initialize weights orthogonally to approximate an isometric projection 
        onto the tangent space of the submanifold.
        """
        nn.init.orthogonal_(self.projection.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
        
    def get_projection_matrix(self) -> torch.Tensor:
        """Returns the projection matrix P."""
        return self.projection.weight


class FactorizedINN(nn.Module):
    """
    Factorized Invertible Neural Network (g_v).
    Inverts the multi-view generative process X_v = f_v(S, U_v, \epsilon_v) 
    on a low-dimensional data submanifold M_v.
    
    Maps X_v -> (\hat{S}_v, \hat{U}_v) while preserving the Riemannian 
    information geometry via the exact change-of-variables formula.
    """
    def __init__(self, input_dim: int, d_S: int, d_U: int, num_coupling_layers: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.d_S = d_S
        self.d_U = d_U
        self.latent_dim = d_S + d_U
        
        # 1. Submanifold Projector (P_x)
        self.projector = SubmanifoldProjector(input_dim, self.latent_dim)
        
        # 2. Invertible Neural Network Core (f_INN)
        self.coupling_layers = nn.ModuleList()
        for i in range(num_coupling_layers):
            # Alternating binary masks to ensure all dimensions are transformed
            mask = torch.arange(self.latent_dim) % 2 == (i % 2)
            self.coupling_layers.append(AffineCouplingLayer(self.latent_dim, mask))
            
    def forward(self, x: torch.Tensor):
        """
        Forward pass: X_v -> (\hat{S}_v, \hat{U}_v, log_det_f_INN)
        """
        batch_size = x.size(0)
        
        # Project high-dimensional observation onto the submanifold
        z_prime = self.projector(x)
        
        # Pass through INN coupling layers
        log_det_total = torch.zeros(batch_size, device=x.device)
        z = z_prime
        for layer in self.coupling_layers:
            z, log_det = layer(z)
            log_det_total += log_det
            
        # Split the latent representation into shared and view-specific factors
        s_hat = z[:, :self.d_S]
        u_hat = z[:, self.d_S:]
        
        return s_hat, u_hat, log_det_total

    def compute_riemannian_logdet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the exact Riemannian change-of-variables log-determinant:
        log det ( J_gv(x) P_x J_gv(x)^T )
        
        Mathematical Derivation:
        Since g_v(x) = f_INN(P x), the Jacobian J_gv = J_f_INN * P.
        The term becomes log det ( J_f_INN * P * P^T * J_f_INN^T ).
        Using determinant properties: = 2 * log |det J_f_INN| + log det(P * P^T).
        
        This avoids computing the intractable d_v x d_v full Jacobian, 
        relying instead on the analytical log-det of the INN and the 
        small d' x d' projection matrix.
        """
        # 1. Get the analytical log |det J_f_INN| from the forward pass
        _, _, log_det_f_inn = self.forward(x)
        
        # 2. Compute log det(P * P^T)
        P = self.projector.get_projection_matrix()  # Shape: (latent_dim, input_dim)
        P_PT = torch.matmul(P, P.T)                 # Shape: (latent_dim, latent_dim)
        
        # Use slogdet for numerical stability
        sign, log_det_P_PT = torch.linalg.slogdet(P_PT)
        
        # 3. Combine to get the final Riemannian log-determinant
        riemannian_log_det = 2.0 * log_det_f_inn + log_det_P_PT
        
        return riemannian_log_det

    def inverse(self, s_hat: torch.Tensor, u_hat: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass: (\hat{S}_v, \hat{U}_v) -> X_v
        Reconstructs the observation from the partitioned latent factors.
        """
        # Concatenate shared and unique factors
        z = torch.cat([s_hat, u_hat], dim=-1)
        
        # Inverse through coupling layers in reverse order
        for layer in reversed(self.coupling_layers):
            z = layer.inverse(z)
            
        # Inverse projection using the Moore-Penrose pseudo-inverse
        P = self.projector.get_projection_matrix()
        P_pinv = torch.linalg.pinv(P)  # Shape: (input_dim, latent_dim)
        
        # Map back to the ambient space R^{d_v}
        x = torch.matmul(z, P_pinv.T)
        
        return x