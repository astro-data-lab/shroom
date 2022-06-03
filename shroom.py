import numpy as np
import scipy
import matplotlib.pyplot as plt
from minisom import MiniSom
import corner

objective = lambda data: data.std() / data.mean()

def optimize_som(train, bounds):


    # minimize std of SOM activation
    def std_sigma(params):
        L1, L2, sigma, lr = params
        L1, L2 = int(L1), int(L2)
        som = MiniSom(L1, L2, D, sigma=sigma, learning_rate=lr)
        sel = np.random.choice(N, size=10000)
        som.train(train[sel], 10000)
        freq = som.activation_response(train)
        obj = objective(freq)
        return obj

    result = scipy.optimize.differential_evolution(std_sigma, bounds, tol=0.1, maxiter=100)
    return result

def get_som(data, optimize=False):
    N, D = data.shape
    # find good SOM parameters for reference
    # 1) number of cells, based on #nodes = 5 * sqrt(N)
    # use smalles N from test and train
    L = int(np.sqrt(5 * np.sqrt(N)))

    # best quick guesses
    if not optimize:
        # elongated maps easier for alongated data distributions
        L1, L2 = L, L//2
        sigma = (L1+L2) / 20
        lr = 0.25
    else:
        bounds = [(L//2, L*1.5), (L//2, L*1.5), (1, L / 5), (0.1, 1)]
        result = optimize_som(data)
        L1, L2, sigma, lr = result.x
        L1, L2 = int(L1), int(L2)

    som = MiniSom(L1, L2, D, sigma=sigma, learning_rate=lr)
    som.train_random(data, N)

    return som


# bootstrap resampling
def get_diff_dispersion(som, data, R=10):
    N = len(data)
    freq = som.activation_response(data)
    diffs = []
    for _ in range(R):
        bootstrap_idx = np.random.choice(N, size=N, replace=True)
        data_ = data[bootstrap_idx]
        freq_ = som.activation_response(data_)
        diff_ = (freq_ - freq)/N
        diffs.append(diff_)
    diff_std = np.array(diffs).std()
    return diff_std

def get_maps(som, train, test, sig=2):

    # get direct projections
    freq = som.activation_response(train)
    freq_test = som.activation_response(test)
    print (f"SOM objective: { objective(freq) }")

    # difference map: difference of normalized distributions
    diff = freq_test / freq_test.sum() -  freq / freq.sum()

    # find cells significantly below or above the mean in the difference map
    train_std = get_diff_dispersion(som, train)
    test_std = get_diff_dispersion(som, test)
    above = diff > diff.mean() + sig*(train_std + test_std)
    below = diff < diff.mean() - sig*(train_std + test_std)

    return freq, freq_test, diff, above, below


def connect_cells(detection, minimum=5, structure=None):
    labeled_im, num_features = scipy.ndimage.label(detection, structure=structure)
    idx = np.indices(detection.shape)
    groups = []
    for i in range(num_features):
        cells = labeled_im == i+1
        cell_idxs = np.array((idx[0][cells], idx[1][cells])).T
        groups.append(cell_idxs)

    # order by largest group
    sizes = tuple(len(cells) for cells in groups)
    order = np.argsort(sizes)
    groups = tuple(groups[i] for i in order[::-1] if sizes[i] >= minimum)

    # update labeled image
    labeled_im = np.zeros(detection.shape, dtype=int)
    for i, group in enumerate(groups):
        for cell in group:
            labeled_im[tuple(cell)] = i+1

    return groups, labeled_im

def aggregate(som, data, cells, weight=None, win_map=None):

    if win_map is None:
        # get all cell ids for data
        win_map = som.win_map(data, return_indices=True)

    # for every group in below:
    # aggregate all samples in train that belong to this group
    samples = np.zeros(data.shape[1])
    total = 0
    for cell in cells:
        cell_samples = data[win_map[tuple(cell)]]
        if weight is None:
            w = 1
        else:
            w = weight[tuple(cell)]
        samples += cell_samples.sum(axis=0) * w
        total += len(cell_samples) * w
        sample_mean = samples / total
    return sample_mean, total

def aggregate_groups(som, data, groups, weight=None):
    # get all cell ids for data
    win_map = som.win_map(data, return_indices=True)
    return tuple(aggregate(som, data, group, weight=weight, win_map=win_map) for group in groups)

def plot_maps(freq, freq_test, diff, labeled_above, labeled_below):
    fig, axes = plt.subplots(1, 5, figsize=(12,4))
    axes[0].imshow(freq, cmap='Greys')
    axes[0].set_title("Train")
    axes[1].imshow(freq_test, cmap='Greys')
    axes[1].set_title("Test")
    axes[2].imshow(diff, cmap='RdBu_r')
    axes[2].set_title("Difference")
    axes[3].imshow(diff, cmap='Greys')
    axes[3].imshow(np.ma.masked_where(labeled_above == 0, labeled_above), cmap='Reds_r', alpha=0.7, vmax=labeled_above.max()+1)
    axes[3].set_title("Groups Above")
    axes[4].imshow(diff, cmap='Greys')
    axes[4].imshow(np.ma.masked_where(labeled_below == 0, labeled_below), cmap='Blues_r', alpha=0.7, vmax=labeled_below.max()+1)
    axes[4].set_title("Groups Below");
    return fig


def plot_data(train, test, **kwargs):
    fig = corner.corner(train, hist_kwargs={"density": True});
    corner.corner(test, fig=fig, color='C1', hist_kwargs={"density": True}, **kwargs);
    return fig

def analyze(som, train, test, sig=3, min_cells=5, plot=True, labels=None):
    # make frequency maps and their differences
    freq, freq_test, diff, above, below = get_maps(som, train, test, sig=sig)
    # find connected groups: overdensities often ridges, so use broader
    # structure kernel (connected to all 8 neighbor cells)
    groups_above, labeled_above = connect_cells(above, minimum=min_cells, structure=np.ones((3,3)))
    groups_below, labeled_below = connect_cells(below, minimum=min_cells)
    print (f"Detected {len(groups_above)} groups above threshold")
    print (f"Detected {len(groups_below)} groups below threshold")

    if plot:
        plot_maps(freq, freq_test, diff, labeled_above, labeled_below)

    # aggregate train where it is more abundant than test: what's missing from test
    aggregates_below = aggregate_groups(som, train, groups_below, weight=-diff)

    # aggregate in reverse: where test is overdense compared to training
    aggregates_above = aggregate_groups(som, test, groups_above, weight=diff)

    if plot:
        for mean, sig in aggregates_above:
            plot_data(train, test, labels=labels, truths=mean, truth_color='C3')
        for mean, sig in aggregates_below:
            plot_data(train, test, labels=labels, truths=mean, truth_color='C0')

    return aggregates_above, aggregates_below
