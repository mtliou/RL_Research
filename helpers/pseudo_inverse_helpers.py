import torch


class HippocampalSensoryLayer:
    def __init__(
        self,
        input_size: int,
        N_h: int,
        device=None,
    ):
        self.input_size = input_size
        self.N_h = N_h
        self.device = device

    def hippocampal_from_sensory(self, S: torch.Tensor) -> torch.Tensor:
        """
        Input shape: `(B, input_size)`

        Output shape: `(B, N_h)`

        Args:
            S (torch.Tensor): Sensory input tensor.

        """
        pass

    def sensory_from_hippocampal(self, H: torch.Tensor) -> torch.Tensor:
        """
        Input shape `(B, N_h)`

        Output shape `(B, input_size)`

        Args:
            H (torch.Tensor): Hippocampal state tensor.
        """
        pass

    def learn(self, h: torch.Tensor, s: torch.Tensor):
        """
        Associate a sensory input to a hippocampal fixed point.

        `h`: The hippocampal fixed point
        `s`: The sensory input
        """
        pass


class IterativeBidirectionalPseudoInverseHippocampalSensoryLayer(
    HippocampalSensoryLayer
):
    def __init__(
        self,
        input_size: int,
        N_h: int,
        hidden_layer_factor: int,
        stationary=True,
        epsilon_sh=None,
        epsilon_hs=None,
        device=None,
    ):
        super().__init__(input_size, N_h, device)

        self.hidden_layer_factor = hidden_layer_factor
        self.stationary = stationary
        self.epsilon_hs = epsilon_hs
        self.epsilon_sh = epsilon_sh

        hidden_size_sh = self.N_h * self.hidden_layer_factor
        if hidden_size_sh == 0:
            hidden_size_sh = self.N_h
        else:
            self.hidden_sh = torch.rand((hidden_size_sh, self.N_h), device=device) - 0.5
        self.W_sh = torch.zeros((self.input_size, hidden_size_sh), device=device)

        if epsilon_sh == None:
            self.epsilon_sh = hidden_size_sh
        else:
            self.epsilon_sh = epsilon_sh
        self.inhibition_matrix_sh = torch.eye(hidden_size_sh, device=device) / (
            self.epsilon_sh**2
        )

        hidden_size_hs = self.input_size * self.hidden_layer_factor
        if hidden_size_hs == 0:
            hidden_size_hs = self.input_size
        else:
            self.hidden_hs = (
                torch.rand((hidden_size_hs, self.input_size), device=device) - 0.5
            )
        self.W_hs = torch.zeros((self.N_h, hidden_size_hs), device=device)
        if epsilon_hs == None:
            self.epsilon_hs = hidden_size_hs
        else:
            self.epsilon_hs = epsilon_hs
        self.inhibition_matrix_hs = torch.eye(hidden_size_hs, device=device) / (
            self.epsilon_hs**2
        )

    @torch.no_grad()
    def learned_pseudo_inverse_hs(self, input, output):
        if self.stationary:
            b_k_hs = (self.inhibition_matrix_hs @ input) / (
                1 + input.T @ self.inhibition_matrix_hs @ input
            )

            self.inhibition_matrix_hs = (
                self.inhibition_matrix_hs
                - self.inhibition_matrix_hs @ torch.outer(input, b_k_hs.T)
            )

            self.W_hs += torch.outer((output - self.W_hs @ input), b_k_hs.T)
        else:
            b_k_hs = (self.inhibition_matrix_hs @ input) / (
                1 + input.T @ self.inhibition_matrix_hs @ input
            )
            # ERROR VECTOR EK
            e_k = output - self.W_hs @ input

            # NORMALIZATION FACTOR
            E = ((e_k.T @ e_k) / self.inhibition_matrix_hs.shape[0]) / (
                1 + input.T @ self.inhibition_matrix_hs @ input
            )
            # E = torch.abs(E)

            # GAMMA CALCULATION
            gamma = 1 / (1 + ((1 - torch.exp(-E)) / self.epsilon_hs))

            self.inhibition_matrix_hs = gamma * (
                self.inhibition_matrix_hs
                - self.inhibition_matrix_hs @ torch.outer(input, b_k_hs.T)
                + ((1 - torch.exp(-E)) / self.epsilon_hs)
                * torch.eye(self.inhibition_matrix_hs.shape[0], device=self.device)
            )
            self.W_hs += torch.outer((output - self.W_hs @ input), b_k_hs.T)

    @torch.no_grad()
    def learned_pseudo_inverse_sh(self, input, output):
        if self.stationary:
            b_k_sh = (self.inhibition_matrix_sh @ input) / (
                1 + input.T @ self.inhibition_matrix_sh @ input
            )

            self.inhibition_matrix_sh = (
                self.inhibition_matrix_sh
                - self.inhibition_matrix_sh @ torch.outer(input, b_k_sh.T)
            )

            self.W_sh += torch.outer((output - self.W_sh @ input), b_k_sh.T)
        else:
            # (N_h, N_h) x (N_h, 1) / (1 + (1, N_h) x (N_h, N_h) x (N_h, 1)) = (N_h, 1)
            b_k_sh = (self.inhibition_matrix_sh @ input) / (
                1 + input.T @ self.inhibition_matrix_sh @ input
            )

            # (784, 1) - (784, N_h) x (N_h, 1) = (784, 1)
            e_k = output - self.W_sh @ input

            # ((1, 784) x (784, 1) / (1)) / ((1, N_h) x (N_h, N_h) x (N_h x 1))
            E = ((e_k.T @ e_k) / self.inhibition_matrix_sh.shape[0]) / (
                1 + input.T @ self.inhibition_matrix_sh @ input
            )
            # E = torch.abs(E)

            # scalar
            gamma = 1 / (1 + ((1 - torch.exp(-E)) / self.epsilon_sh))

            # (N_h, N_h) - (N_h, N_h) x (N_h, 1) x (1, N_h) + scalar * (N_h, N_h) = (N_h, N_h)
            self.inhibition_matrix_sh = gamma * (
                self.inhibition_matrix_sh
                - self.inhibition_matrix_sh @ torch.outer(input, b_k_sh.T)
                + ((1 - torch.exp(-E)) / self.epsilon_sh)
                * torch.eye(self.inhibition_matrix_sh.shape[0], device=self.device)
            )
            self.W_sh += torch.outer((output - self.W_sh @ input), b_k_sh.T)

    @torch.no_grad()
    def learn(self, h, s):
        self.learned_pseudo_inverse_hs(
            input=(
                torch.sigmoid(self.hidden_hs @ s)
                if self.hidden_layer_factor != 0
                else s
            ),
            output=h,
        )
        self.learned_pseudo_inverse_sh(
            input=(
                torch.sigmoid(self.hidden_sh @ h)
                if self.hidden_layer_factor != 0
                else h
            ),
            output=s,
        )

    @torch.no_grad()
    def sensory_from_hippocampal(self, H):
        if H.ndim == 1:
            H = H.unsqueeze(0)

        if self.hidden_layer_factor != 0:
            hidden = torch.sigmoid(H @ self.hidden_sh.T)
            return hidden @ self.W_sh.T
        else:
            # return H. @ self.W_sh.T
            return H @ self.W_sh.T
    @torch.no_grad()
    def hippocampal_from_sensory(self, S):
        if S.ndim == 1:
            S = S.unsqueeze(0)

        if self.hidden_layer_factor != 0:
            hidden = torch.sigmoid(S @ self.hidden_hs.T)
            return torch.relu(hidden @ self.W_hs.T)
        else:
            return torch.relu(
                S @ self.W_hs.T
            )  # to relu or not to relu, that is the question.

