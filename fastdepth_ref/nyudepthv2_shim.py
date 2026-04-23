"""
Shim module exposed as ``nyudepthv2`` so that torch.load can unpickle the
Hagaik92/FastDepth checkpoints, which embed the original NYUDataset class and
a full argparse.Namespace as part of the saved dict.

We don't actually need those objects at inference time — only the
``model_state_dict`` tensor dict — but the pickle machinery still has to
resolve the class references or the load will raise ModuleNotFoundError.
This provides dummy placeholders with the right qualified names.
"""
from torch.utils.data import Dataset


def h5_loader(path):  # referenced in ctor
    return None


class NYUDataset(Dataset):
    """Dummy stand-in for Hagaik's NYUDataset (only used to satisfy pickle)."""

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("placeholder dataset")

    # The original Hagaik class binds these methods; pickle will look them up
    # when rehydrating saved instances, so we provide stubs.
    def train_transform(self, rgb, depth):
        return rgb, depth

    def val_transform(self, rgb, depth):
        return rgb, depth

    def build_dataset(self, root_dir, class_to_idx):
        return []

    def get_classes(self, root_dir):
        return [], {}

    def __getraw__(self, index):
        return None, None


def create_data_loaders(*args, **kwargs):  # also referenced in pickle
    return None, None
