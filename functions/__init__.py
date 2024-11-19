import torch.optim as optim
import torch
import torch.special as sp


# Code adapted from DDIM by Song et al. (https://github.com/ermongroup/ddim)
def get_optimizer(config, parameters):
    if config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    else:
        raise NotImplementedError(f"Invalid optimizer {config.optim.optimizer}.")


def student_t_log_likelihood(x, mu, log_lamb, log_v):   # lamb = 1 / scale^2
    lamb = log_lamb.exp()
    v = log_v.exp()
    dim = tuple(range(1, len(x.shape)))
    term1 = sp.gammaln((v + 1) / 2) - sp.gammaln(v / 2)
    term2 = -0.5 * torch.log(v * torch.pi)
    term3 = 0.5 * torch.log(lamb)
    term4 = -((v + 1) / 2) * torch.log(1 + (lamb * (x - mu) ** 2) / v)
    return (term1 + term2 + term3 + term4).sum(dim=dim)


def gaussian_log_likelihood(x, mu_x, log_var_x):
    # Calculate the log-likelihood for a Gaussian distribution
    log_likelihood = -0.5 * (log_var_x + (x - mu_x) ** 2 / log_var_x.exp())
    # Sum over the image dimensions and channels, but not batch
    return log_likelihood.view(log_likelihood.size(0), -1).sum(dim=-1)
