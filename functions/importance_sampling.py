import torch
from torch.utils.data import DataLoader
from functions import gaussian_log_likelihood, student_t_log_likelihood


def importance_sample_log_likelihood(model, test_dataset, K=100, batch_size=64, device='cpu'):
    """
    Estimate the marginal log-likelihood of a test dataset using importance sampling.
    Returns:
    - Mean marginal log-likelihood estimate for the test dataset.
    """
    model.eval()  # Set model to evaluation mode
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    total_log_px = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            batch_size = x.size(0)

            # Encode to get parameters for q(z|x)
            mu_z, log_var_z = model.encoder(x)
            std_z = torch.exp(0.5 * log_var_z)

            # Initialize list to hold importance weights for the batch
            log_weights = []

            for _ in range(K):
                # Sample z ~ q(z|x)
                eps = torch.randn_like(std_z)
                z = mu_z + eps * std_z  # Reparameterization trick

                # Decode to get parameters for p(x|z)
                if model.dist == "gaussian":
                    mu_x, log_var_x = model.decoder(z)
                    # Compute log p(x|z) using the Gaussian decoder likelihood
                    log_p_x_given_z = gaussian_log_likelihood(x, mu_x, log_var_x)

                elif model.dist == "student-t":
                    mu_x, log_lamb_x, log_v_x = model.decoder(z)
                    # Compute log p(x|z) using the Student-t decoder likelihood
                    log_p_x_given_z = student_t_log_likelihood(x, mu_x, log_lamb_x, log_v_x)
                else:
                    raise ValueError(f"Unknown distribution type '{model.dist}'")

                # Compute log p(z) (prior on z, assuming standard normal prior)
                log_p_z = gaussian_log_likelihood(z, torch.zeros_like(z), torch.zeros_like(z))

                # Compute log q(z|x)
                log_q_z_given_x = gaussian_log_likelihood(z, mu_z, log_var_z)

                # Calculate the importance weight: log w^(k) = log p(x, z^(k)) - log q(z^(k) | x)
                log_w = log_p_x_given_z + log_p_z - log_q_z_given_x
                log_weights.append(log_w)

            # Stack log weights and compute log mean of the weights for marginal log-likelihood estimate
            log_weights = torch.stack(log_weights, dim=0)  # Shape: [K, batch_size]
            log_px = torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(K, dtype=torch.float, device=device))

            # Accumulate results
            total_log_px += log_px.sum().item()
            total_samples += batch_size

    # Compute the mean marginal log-likelihood over the dataset
    mean_log_px = total_log_px / total_samples
    return mean_log_px
