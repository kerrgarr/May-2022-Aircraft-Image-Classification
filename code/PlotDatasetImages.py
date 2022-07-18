import argparse
import matplotlib.pyplot as plt
from .utils import FAMILY_NAMES, AircraftDataset


def PlotDatasetImages(args):
    dataset = AircraftDataset(args.dataset)

    f, axes = plt.subplots(args.n, len(FAMILY_NAMES))

    counts = [0]*len(FAMILY_NAMES)

    for img, label in dataset:
        c = counts[label]
        if c < args.n:
            ax = axes[c][label]
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.axis('off')
            ax.set_title(FAMILY_NAMES[label])
            counts[label] += 1
        if sum(counts) >= args.n * len(FAMILY_NAMES):
            break

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('-n', type=int, default=5)
    args = parser.parse_args()

    PlotDatasetImages(args)
