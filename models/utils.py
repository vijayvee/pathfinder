# Don't remove this file and don't change the imports of load_state_dict_from_url
# from other files. We need this so that we can swap load_state_dict_from_url with
# a custom internal version in fbcode.
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# Don't remove this file and don't change the imports of load_state_dict_from_url
# from other files. We need this so that we can swap load_state_dict_from_url with
# a custom internal version in fbcode.
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from numpy.core.numeric import True_
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
import numpy as np
import math

def genGabor(sz, theta, gamma, sigma, sf, phi=0, contrast=2):
    """Generate gabor filter based on argument parameters."""
    location = (sz[0] // 2, sz[1] // 2)
    [x, y] = np.meshgrid(np.arange(sz[0])-location[0],
                         np.arange(sz[1])-location[1])

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    envelope = .5 * contrast * \
        np.exp(-(x_theta**2 + (y_theta * gamma)**2)/(2 * sigma**2))
    gabor = envelope * np.cos(2 * math.pi * x_theta * sf + phi)
    return gabor

def generate_gabor_filter_weights(sz, l_theta, l_sfs,
                                  l_phase, gamma=1,
                                  contrast=1, return_dict=False):
    """Generate a bank of gabor filter weights.
    Args:
      sz: (filter height, filter width), +-2 SD of gaussian envelope
      l_theta: List of gabor orientations
      l_sfs: List of spatial frequencies, cycles per SD of envelope
      l_phase: List of gabor phase
    Returns:
      gabor filter weights with parameters sz X l_theta X l_sfs X l_phase
    """
    gabor_bank = []
    theta2filter = {}
    for theta in l_theta:
        curr_filters = []
        for sf in l_sfs:
            for phase in l_phase:
                g = genGabor(sz=(sz, sz), theta=theta,
                             gamma=gamma, sigma=sz/4,
                             sf=sf/sz, phi=phase,
                             contrast=contrast)
                gabor_bank.append(g)
                curr_filters.append(g)
        theta2filter[theta] = torch.from_numpy(
            np.array(curr_filters, dtype=np.float32))
    theta2filter = {t: torch.unsqueeze(g_b, 1)
                    for t, g_b in theta2filter.items()}
    gabor_bank = np.array(gabor_bank, dtype=np.float32)
    gabor_bank = np.expand_dims(gabor_bank, 1)
    if return_dict:
        return gabor_bank, theta2filter
    return gabor_bank

# define normalized 2D gaussian
def get_gaussian_filterbank(n_filters, f_sz, device='cuda'):
    """Generate a torch tensor for conv2D weights resembling 2D gaussian"""
    def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
        return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

    def normalize(x):
        mid_x, mid_y = x.shape[0] // 2, x.shape[1] // 2
        x = x / x[mid_x][mid_y]
        return x
    filters = []
    x = np.linspace(-1, 1, num=f_sz)
    y = np.linspace(-1, 1, num=f_sz)
    x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
    z_narrow = normalize(gaus2d(x, y, sx=.5, sy=.5))
    z_low = normalize(gaus2d(x, y, sx=1., sy=1.))
    z_mid = normalize(gaus2d(x, y, sx=2, sy=2))
    z_wide = normalize(gaus2d(x, y, sx=3, sy=3))
    filters = [z_narrow] * (n_filters // 4) + [z_low] * (n_filters // 4) + [z_mid] * (n_filters // 4) + [z_wide] * (n_filters // 4)
    filters = np.array(filters)
    filters = np.random.permutation(filters)
    filters = filters.reshape((n_filters, 1, f_sz, f_sz))
    filters = torch.Tensor(filters).float().to(device)
    return filters

def nonnegative_weights_init(m):
    """Non-negative initialization of weights."""
    if isinstance(m, nn.Conv2d):
        nn.init.uniform_(m.weight)
    else:
        m.data.fill_(0.1)
        
def orthogonal_weights_init(m):
    """Orthogonal initialization of weights."""
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)
        m.weight.data.clamp_(0)
        m.bias.data.fill_(0.)
    else:
        m.data.fill_(0.)

def gaussian_weights_init(m):
    """Initialize weights using 2d Gaussian."""
    if isinstance(m, nn.Conv2d):
        n_filters, _, f_sz, _ = m.weight.shape
        weights = get_gaussian_filterbank(n_filters, f_sz)
        m.weight.data = weights

def get_gabor_conv(in_channels, out_channels, f_size, stride):
    """Get a gabor-initialized convolution layer"""
    l_theta = np.linspace(0, np.pi, out_channels // 2)
    filter_weights = generate_gabor_filter_weights(f_size, l_theta,
                                                    [2], [0, np.pi],
                                                    contrast=1.)
    conv_layer = nn.Conv2d(in_channels=in_channels,
                            out_channels=filter_weights.shape[0],
                            kernel_size=f_size, stride=stride,
                            padding=(f_size - 1) // 2,
                            bias=False).to('cuda')
    with torch.no_grad():
        conv_layer.weight.copy_(
            torch.from_numpy(filter_weights).float())
        conv_layer.weight.requires_grad = False
    return conv_layer