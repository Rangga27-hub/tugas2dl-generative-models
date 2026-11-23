from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Decoder(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

    def forward(self, z):
        return self.net(z).reshape(-1, 1, 28, 28)


class Autoencoder(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                                     nn.Linear(256, latent), nn.ReLU())
        self.decoder = Decoder(latent)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class VAE(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU())
        self.mu = nn.Linear(256, latent)
        self.logvar = nn.Linear(256, latent)
        self.decoder = Decoder(latent)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def forward(self, x, deterministic=False):
        mu, logvar = self.encode(x)
        z = mu if deterministic else mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z), mu, logvar


class Classifier(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, latent)
        self.fc3 = nn.Linear(latent, 10)

    def encode(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x.flatten(1)))))

    def forward(self, x):
        return self.fc3(self.encode(x))

    def transpose_latent(self, z):
        # For row vectors, z @ W2 then @ W1 is the column-vector W1^T ReLU(W2^T z).
        return torch.sigmoid(F.relu(z @ self.fc2.weight) @ self.fc1.weight).reshape(-1, 1, 28, 28)

    def transpose_class(self, one_hot):
        return self.transpose_latent(F.relu(one_hot @ self.fc3.weight))


class VanillaGAN(nn.Module):
    def __init__(self, noise=100):
        super().__init__()
        self.generator = nn.Sequential(nn.Linear(noise, 256), nn.LeakyReLU(0.2),
                                       nn.Linear(256, 512), nn.LeakyReLU(0.2),
                                       nn.Linear(512, 784), nn.Tanh(), nn.Unflatten(1, (1, 28, 28)))
        self.discriminator = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.LeakyReLU(0.2),
                                           nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1))

    def forward(self, z):
        return self.generator(z)


class DCGAN(nn.Module):
    def __init__(self, noise=100):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(noise, 128 * 7 * 7), nn.Unflatten(1, (128, 7, 7)),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())
        self.discriminator = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(128 * 7 * 7, 1))
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(module.weight, 0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1, 0.02)
            nn.init.zeros_(module.bias)

    def forward(self, z):
        return self.generator(z)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, t):
        freq = torch.exp(torch.arange(self.dim // 2, device=t.device) * (-math.log(10000) / (self.dim // 2 - 1)))
        angles = t.float()[:, None] * freq[None]
        return self.net(torch.cat([angles.sin(), angles.cos()], dim=1))


class TimeBlock(nn.Module):
    def __init__(self, inp, out, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out)
        self.time = nn.Linear(time_dim, out)
        self.conv2 = nn.Conv2d(out, out, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out)
        self.skip = nn.Conv2d(inp, out, 1) if inp != out else nn.Identity()

    def forward(self, x, emb):
        h = F.silu(self.norm1(self.conv1(x))) + self.time(emb)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class Denoiser(nn.Module):
    """Small U-Net: 28 -> 14 -> 7 -> 14 -> 28; timestep injection in every block."""
    def __init__(self, channels=32):
        super().__init__()
        c = channels
        self.time = TimeEmbedding(c * 4)
        self.down1 = TimeBlock(1, c, c * 4)
        self.down2 = TimeBlock(c, c * 2, c * 4)
        self.mid = TimeBlock(c * 2, c * 4, c * 4)
        self.up2 = TimeBlock(c * 6, c * 2, c * 4)
        self.up1 = TimeBlock(c * 3, c, c * 4)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x, t):
        e = self.time(t)
        a = self.down1(x, e)
        b = self.down2(F.avg_pool2d(a, 2), e)
        h = self.mid(F.avg_pool2d(b, 2), e)
        h = self.up2(torch.cat([F.interpolate(h, size=b.shape[-2:], mode="nearest"), b], 1), e)
        h = self.up1(torch.cat([F.interpolate(h, size=a.shape[-2:], mode="nearest"), a], 1), e)
        return self.out(h)


class DDPM(nn.Module):
    def __init__(self, steps=200, channels=32, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.denoiser = Denoiser(channels)
        self.steps = steps
        beta = torch.linspace(beta_start, beta_end, steps)
        alpha = 1 - beta
        abar = torch.cumprod(alpha, 0)
        prev = F.pad(abar[:-1], (1, 0), value=1)
        for name, value in {"beta": beta, "alpha": alpha, "abar": abar,
                            "posterior_variance": beta * (1 - prev) / (1 - abar)}.items():
            self.register_buffer(name, value)

    def q_sample(self, x, t, noise):
        a = self.abar[t, None, None, None]
        return a.sqrt() * x + (1 - a).sqrt() * noise

    def forward(self, x, t):
        return self.denoiser(x, t)

    @torch.no_grad()
    def sample(self, x, generator=None, snapshots=False):
        chain = {self.steps: x[:1].detach().cpu()} if snapshots else {}
        keep = {0, self.steps // 4, self.steps // 2, 3 * self.steps // 4}
        for i in reversed(range(self.steps)):
            t = torch.full((len(x),), i, device=x.device, dtype=torch.long)
            eps = self.denoiser(x, t)
            # Predict/clamp x0 and use the exact DDPM posterior mean.
            x0 = ((x - (1 - self.abar[i]).sqrt() * eps) / self.abar[i].sqrt()).clamp(-1, 1)
            prev = self.abar[i - 1] if i else torch.ones_like(self.abar[i])
            mean = (prev.sqrt() * self.beta[i] / (1 - self.abar[i])) * x0
            mean += ((1 - prev) * self.alpha[i].sqrt() / (1 - self.abar[i])) * x
            noise = torch.randn(x.shape, device=x.device, generator=generator) if i else torch.zeros_like(x)
            x = mean + self.posterior_variance[i].sqrt() * noise
            if snapshots and i in keep:
                chain[i] = x[:1].detach().cpu()
        return (x, chain) if snapshots else x


def build_model(name, config):
    latent, noise = config["latent_dim"], config["noise_dim"]
    if name in ("ae", "sparse_ae"):
        return Autoencoder(latent)
    if name == "vae":
        return VAE(latent)
    if name == "mlp":
        return Classifier(latent)
    if name == "decoder":
        return Decoder(latent)
    if name == "gan":
        return VanillaGAN(noise)
    if name == "dcgan":
        return DCGAN(noise)
    if name == "ddpm":
        return DDPM(config["ddpm_steps"], config["ddpm_channels"], config["beta_start"], config["beta_end"])
    raise ValueError(name)
