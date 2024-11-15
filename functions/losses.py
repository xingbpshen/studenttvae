import torch
from models.vae import VAE
from functions import gaussian_log_likelihood, student_t_log_likelihood


def kl_divergence_gaussian(mu_z, log_var_z):
    # KL divergence for Gaussian q(z|x) ~ N(mu_z, sigma^2) and p(z) ~ N(0, 1)
    kl_div = -0.5 * torch.sum(1 + log_var_z - mu_z ** 2 - log_var_z.exp(), dim=-1)
    return kl_div


def n_elbo_st(x, mu_z, log_var_z, mu_x, log_lambda_x, log_v_x):
    # Reconstruction log-likelihood (log p(x|z))
    recon_log_likelihood = student_t_log_likelihood(x, mu_x, log_lambda_x, log_v_x)

    # KL divergence (KL(q(z|x) || p(z)))
    kl_div = kl_divergence_gaussian(mu_z, log_var_z)

    # ELBO
    elbo = recon_log_likelihood - kl_div
    return -elbo.mean()


def n_elbo_g(x, mu_z, log_var_z, mu_x, log_var_x):
    # Reconstruction log-likelihood (log p(x|z))
    recon_log_likelihood = gaussian_log_likelihood(x, mu_x, log_var_x)

    # KL divergence (KL(q(z|x) || p(z)))
    kl_div = kl_divergence_gaussian(mu_z, log_var_z)

    # ELBO
    elbo = recon_log_likelihood - kl_div
    return -elbo.mean()


def calculate_loss(model: VAE, x):
    if model.dist == "student-t":
        mu_p, logvar_p, mu_l, lamb_l, v_l = model(x)
        return n_elbo_st(x, mu_p, logvar_p, mu_l, lamb_l, v_l)
    elif model.dist == "gaussian":
        mu_p, logvar_p, mu_l, logvar_l = model(x)
        return n_elbo_g(x, mu_p, logvar_p, mu_l, logvar_l)
    else:
        raise ValueError(f"Invalid distribution {model.dist}")
