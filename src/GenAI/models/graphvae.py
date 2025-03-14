class GraphVAE(nn.Module):
    def __init__(self, latent_features: int, max_nodes: int, node_feature_dim: int = 1,
                 position_dim: int = 3, hidden_dim: int = 64):
        """
        A simple Graph VAE.
        """
        super().__init__()
        self.latent_features = latent_features
        self.max_nodes = max_nodes
        self.input_dim = node_feature_dim + position_dim  # e.g. 1+3 = 4
        # Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_features)
        self.fc_logvar = nn.Linear(hidden_dim, latent_features)
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes * self.input_dim)
        )

    def encode(self, node_features, node_positions, mask):
        x = torch.cat([node_features, node_positions], dim=-1)  # (B, N, 4)
        B, N, _ = x.size()
        h = self.node_encoder(x.view(B * N, self.input_dim))
        h = h.view(B, N, -1)
        mask_expanded = mask.unsqueeze(-1)
        h = h * mask_expanded
        sum_h = h.sum(dim=1)
        counts = mask_expanded.sum(dim=1)
        pooled = sum_h / (counts + 1e-6)
        pooled = self.graph_encoder(pooled)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        B = z.size(0)
        out = self.decoder_fc(z)
        out = out.view(B, self.max_nodes, self.input_dim)
        return out

    def forward(self, node_features, node_positions, mask):
        mu, logvar = self.encode(node_features, node_positions, mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

