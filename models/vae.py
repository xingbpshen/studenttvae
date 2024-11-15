import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderPaper(nn.Module):
    def __init__(self, config, dist):
        super(EncoderPaper, self).__init__()
        self.config = config
        self.dist = dist
        self.fc1 = torch.nn.Linear(config.data.features, 500)
        self.fc2 = torch.nn.Linear(500, 500)
        if dist in ["gaussian", "student-t"]:
            self.fc_mu = torch.nn.Linear(500, config.model.latent_dim)
            self.fc_logvar = torch.nn.Linear(500, config.model.latent_dim)
        else:
            raise ValueError(f"Invalid distribution {dist}")

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        if self.dist in ["gaussian", "student-t"]:
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            return mu, logvar
        else:
            raise ValueError(f"Invalid distribution {self.dist}")


class DecoderPaper(nn.Module):
    def __init__(self, config, dist):
        super(DecoderPaper, self).__init__()
        self.config = config
        self.dist = dist
        self.fc1 = torch.nn.Linear(config.model.latent_dim, 500)
        self.fc2 = torch.nn.Linear(500, 500)
        if dist == "gaussian":
            self.fc_mu = torch.nn.Linear(500, config.data.features)
            self.fc_logvar = torch.nn.Linear(500, config.data.features)
        elif dist == "student-t":
            self.fc_mu = torch.nn.Linear(500, config.data.features)
            self.fc_lamb = torch.nn.Linear(500, config.data.features)
            self.fc_v = torch.nn.Linear(500, config.data.features)
        else:
            raise ValueError(f"Invalid distribution {dist}")

    def forward(self, z):
        z = torch.tanh(self.fc1(z))
        z = torch.tanh(self.fc2(z))
        if self.dist == "gaussian":
            mu = self.fc_mu(z)
            logvar = self.fc_logvar(z)
            return mu, logvar
        elif self.dist == "student-t":
            mu = self.fc_mu(z)
            log_lamb = self.fc_lamb(z)
            log_v = self.fc_v(z)
            return mu, log_lamb, log_v
        else:
            raise ValueError(f"Invalid distribution {self.dist}")


# Define a basic ResNet block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# EncoderResNet for Student-t VAE
class EncoderResNet(nn.Module):
    def __init__(self, config, dist):
        super(EncoderResNet, self).__init__()
        latent_dim = config.model.latent_dim  # specify the dimension of the latent space
        in_channels = config.model.in_channels
        self.dist = dist

        # Initial convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)  # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 32x32 -> 16x16
        self.res_block1 = ResNetBlock(128, 128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 16x16 -> 8x8
        self.res_block2 = ResNetBlock(256, 256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # 8x8 -> 4x4

        if dist in ["gaussian", "student-t"]:
            # Fully connected layers for `mu` and `logvar`
            self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
            self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        else:
            raise ValueError(f"Invalid distribution {dist}")

    def forward(self, x):
        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.res_block1(x)
        x = F.relu(self.conv3(x))
        x = self.res_block2(x)
        x = F.relu(self.conv4(x))

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        if self.dist in ["gaussian", "student-t"]:
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            return mu, logvar
        else:
            raise ValueError(f"Invalid distribution {self.dist}")


# DecoderResNet for Student-t VAE
class DecoderResNet(nn.Module):
    def __init__(self, config, dist):
        super(DecoderResNet, self).__init__()
        latent_dim = config.model.latent_dim  # specify the dimension of the latent space
        out_channels = config.model.out_channels
        self.dist = dist

        # Fully connected layer to project latent space back to feature map size
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        # Transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 4x4 -> 8x8
        self.res_block1 = ResNetBlock(256, 256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 8x8 -> 16x16
        self.res_block2 = ResNetBlock(128, 128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 16x16 -> 32x32
        self.deconv4 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)  # 32x32 -> 64x64

        if dist == "gaussian":
            # Output layers for `mu` and `logvar`
            self.fc_mu = nn.Conv2d(out_channels, out_channels, kernel_size=1)
            self.fc_logvar = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        elif dist == "student-t":
            # Output layers for `mu`, `lambda`, and `v`
            self.fc_mu = nn.Conv2d(out_channels, out_channels, kernel_size=1)  # Output `mu` for Student-t distribution
            self.fc_lambda = nn.Conv2d(out_channels, out_channels, kernel_size=1)  # Output `lambda` (scale)
            self.fc_v = nn.Conv2d(out_channels, out_channels, kernel_size=1)  # Output `v` (degrees of freedom)

    def forward(self, z):
        # Project latent variable `z` back to feature map size
        x = self.fc(z)
        x = x.view(x.size(0), 512, 4, 4)

        # Pass through transposed convolutional layers
        x = F.relu(self.deconv1(x))
        x = self.res_block1(x)
        x = F.relu(self.deconv2(x))
        x = self.res_block2(x)
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))  # Output is in [0, 1] for images

        if self.dist == "gaussian":
            # Output parameters for Gaussian distribution
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            return mu, logvar
        elif self.dist == "student-t":
            # Output parameters for Student-t distribution
            mu = self.fc_mu(x)
            log_lamb = self.fc_lambda(x)
            log_v = self.fc_v(x)
            return mu, log_lamb, log_v


def get_encoder_decoder(config, dist):
    if config.model.type == "paper":
        return EncoderPaper(config, dist), DecoderPaper(config, dist)
    elif config.model.type == "resnet":
        return EncoderResNet(config, dist), DecoderResNet(config, dist)
    else:
        raise ValueError(f"Invalid model type {config.model.type}")


class VAE(nn.Module):
    def __init__(self, config, dist):
        super(VAE, self).__init__()
        self.config = config
        self.dist = dist
        self.encoder, self.decoder = get_encoder_decoder(config, dist)

    def forward(self, x):
        mu_p, logvar_p = self.encoder(x)
        z = self.reparameterize(mu_p, logvar_p)
        if self.dist == "gaussian":
            mu_l, logvar_l = self.decoder(z)
            return mu_p, logvar_p, mu_l, logvar_l
        elif self.dist == "student-t":
            mu_l, log_lambda_l, log_v_l = self.decoder(z)
            return mu_p, logvar_p, mu_l, log_lambda_l, log_v_l

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
