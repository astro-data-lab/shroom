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

from matplotlib.lines import Line2D

def plot_data(train, test, subset=None, subset_label='Subset', **kwargs):
    fig = corner.corner(train, **kwargs);

    # get bins of 1d histograms:
    # bit of a hack, but we want to make these histograms as comparable as possible
    D = train.shape[1]
    bins = []
    legend_bits = []
    for d in range(D):
        panel = d*D + d # diagonal panels in corner plot
        ax = fig.axes[panel]
        hist = ax.get_children()[0]
        bins_ = np.unique(hist.get_xy()[:,0])
        bins.append(bins_)
        lw = hist.get_lw()
        color = hist.get_ec()
        if d == 0:
            legend_bits += [(Line2D([0], [0], color=color, lw=lw), 'Train')]

    # instead of regular histogram, need to go through 1D smoothing to provide exact bin edges:
    # see https://github.com/dfm/corner.py/issues/173
    # match the sample sizes via weights
    weights = len(train) / len(test) * np.ones(len(test))
    corner.corner(test, bins=bins, smooth1d=1e-6, weights=weights, fig=fig, color='C1', hist_kwargs={"lw": lw});
    legend_bits += [(Line2D([0], [0], color='C1', lw=lw), 'Test')]

    if subset is not None:
        # show relative abundance of subset vs test
        weights = len(subset) / len(test) * len(train) / len(test) * np.ones(len(subset))
        corner.corner(subset, bins=bins, smooth1d=1e-6, weights=weights, fig=fig, color='C3', hist_kwargs={"lw": lw});
        legend_bits += [(Line2D([0], [0], color='C3', lw=lw), subset_label)]

    # add legend to first panel
    ax = fig.axes[0]
    lines, names = zip(*legend_bits)
    ax.legend(lines, names, frameon=False)

    return fig



class Shroom():

    def __init__(self, train, test, som=None, verbose=True):
        self._train = train
        self._test = test
        if som is None:
            som = get_som(self._train)
        self._som = som
        self.verbose = verbose
        self._make_maps()

    def _make_maps(self):
        # get direct projections
        self.map_train = self._som.activation_response(self._train)
        self.map_test = self._som.activation_response(self._test)

        if self.verbose:
            print (f"SOM objective: { objective(self.map_train) } (lower is better)")

        # difference map: difference of normalized distributions
        self.map_diff = self.map_test / self.map_test.sum() -  self.map_train / self.map_train.sum()

        # win map: sample index -> map cell
        self._win_map = {}
        self._win_map["train"] = self._som.win_map(self._train, return_indices=True)
        self._win_map["test"] = self._som.win_map(self._test, return_indices=True)

    def find_discrepancies(self, sig=3, min_cells=5, plot=True):

        # find cells significantly below or above the mean in the difference map
        if not hasattr(self, "_train_std"):
            self._train_std = self._get_diff_dispersion(self._som, self._train)
        if not hasattr(self, "_test_std"):
            self._test_std = self._get_diff_dispersion(self._som, self._test)

        threshold = sig*(self._train_std + self._test_std)
        map_above = self.map_diff > self.map_diff.mean() + threshold
        map_below = self.map_diff < self.map_diff.mean() - threshold

        # find connected groups: overdensities often ridges, so use broader
        # structure kernel (connected to all 8 neighbor cells)
        groups_above, labeled_above = self._connect_cells(map_above, minimum=min_cells, structure=np.ones((3,3)))
        groups_below, labeled_below = self._connect_cells(map_below, minimum=min_cells)

        if self.verbose:
            print (f"Detected {len(groups_above)} groups above threshold")
            print (f"Detected {len(groups_below)} groups below threshold")

        if plot:
            plot_maps(self.map_train, self.map_test, self.map_diff, labeled_above, labeled_below)

        return groups_above, groups_below

    def show(self, group, label="Group", **kwargs):
        data = self._train
        win_map = self._som.win_map(data, return_indices=True)
        samples = np.concatenate(tuple(data[win_map[tuple(cell)]] for cell in group), axis=0)
        return plot_data(self._train, self._test, subset=samples, subset_label=label, **kwargs)

    # bootstrap resampling
    def _get_diff_dispersion(self, som, data, R=10):
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

    def _connect_cells(self, detection, minimum=5, structure=None):
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

    def aggregate(self, group, data, weight=None, plot=False, **kwargs):
        # TODO: win map should be cached if it's only for train and test
        win_map = self._som.win_map(data, return_indices=True)
        samples = np.zeros(data.shape[1])
        total = 0
        for cell in group:
            cell_samples = data[win_map[tuple(cell)]]
            if weight is None:
                w = 1
            else:
                w = weight[tuple(cell)]
            samples += cell_samples.sum(axis=0) * w
            total += len(cell_samples) * w
            sample_mean = samples / total

        if plot:
            plot_data(self._train, self._test, truths=sample_mean, **kwargs)

        return sample_mean, total

    def pick(self, group, data, N=1, method="max", plot=False, **kwargs):
        assert method in ["max", "sample"]

        win_map = self._som.win_map(data, return_indices=True)

        if method == "max":
            # select cell with largest discrepancy
            max_idx = np.argmax(np.abs(self.map_diff[group[:,0], group[:,1]]))
            max_cell = group[max_idx]

            # select the most central N examples from this cell
            cell_samples = data[win_map[tuple(max_cell)]]
            m = cell_samples.mean(axis=0)
            min_idx = np.argsort(np.linalg.norm(cell_samples - m, axis=0))[:N]
            exemplars = cell_samples[min_idx]
        else:
            # sample from cells according to their discrepancy
            p = np.abs(self.map_diff[group[:,0], group[:,1]])
            p /= p.sum()
            cell_ids = np.random.choice(len(group), size=N, p=p)

            exemplars = np.empty((N, data.shape[1]))
            for i, cell_id in enumerate(cell_ids):
                # select sample randomly from cell
                cell = group[cell_id]
                cell_samples = data[win_map[tuple(cell)]]
                idx = np.random.choice(len(cell_samples), size=1)
                exemplars[i] = cell_samples[idx]

        if plot:
            for exemplar in exemplars:
                plot_data(self._train, self._test, truths=exemplar, **kwargs)

        return exemplars
