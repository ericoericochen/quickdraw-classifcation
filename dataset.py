import os
import numpy as np
import torch
from torch.utils.data import Dataset
import gcp


class QuickDrawDataset(Dataset):
    _categories = None

    @staticmethod
    def categories():
        """Get labels for the quickdraw dataset"""
        if QuickDrawDataset._categories:
            return QuickDrawDataset._categories

        filename = "categories.txt"

        with open(filename, "r", encoding="utf-8") as file:
            # each line is a category
            categories = [line.strip() for line in file]

        # cache categories so we don't have to read file again
        QuickDrawDataset._categories = categories
        return categories

    def __init__(self, root: str, train: bool = True, download: bool = False):
        print(f"[quickdraw dataset: root={root}, train={train}, download={download}]")

        self._root = root
        self._train = train
        self._bucket_name = "quickdraw_dataset"
        self._source_blob_prefix = "full/numpy_bitmap"
        self._random = np.random.default_rng(5)
        if download:
            self._dataset = self._download_dataset()
            print(self._dataset[0].shape, self._dataset[1].shape)
        else:
            self._dataset = self._load_dataset()

        # normalize dataset -> divide by 255
        self._dataset[0] /= 255

    def _size_per_label(self):
        return 100 if self._train else 10

    def _size(self, categories):
        return len(categories) * self._size_per_label()

    def _download_dataset(self):
        """
        Donwload dataset and returns a tuple (features, labels)
        """
        # categories = ["aircraft carrier", "airplane"]
        categories = QuickDrawDataset.categories()
        dataset_size = self._size(categories)

        # dataset location
        dataset_path = self._dataset_path(self._root, self._train)

        # create dataset folder
        os.makedirs(dataset_path, exist_ok=True)

        # collect all features and labels
        curr_count = 0
        drawings = []
        labels = []

        size = self._size_per_label()
        for i, category in enumerate(categories):
            xs, ys = self._load_from_category(
                category=category, label=i, to=dataset_path, size=size
            )

            drawings.append(xs)
            labels.append(ys)

            curr_count += len(xs)
            print(f"[{curr_count}/{dataset_size} downloaded]")

        # concenate results from each category
        drawings = np.concatenate(drawings)
        drawings = drawings.reshape(len(drawings), 28, 28)
        labels = np.concatenate(labels)

        # save in .npy files
        drawings_path = os.path.join(dataset_path, "drawings.npy")
        labels_path = os.path.join(dataset_path, "labels.npy")

        np.save(drawings_path, drawings)
        np.save(labels_path, labels)

        drawings = np.expand_dims(drawings, axis=1)

        # convert to tensor
        drawings = torch.from_numpy(drawings).type(torch.float)
        labels = torch.from_numpy(labels)

        return [drawings, labels]

    def _load_dataset(self):
        """
        Load pre-downloaded dataset from path
        """
        dataset_path = self._dataset_path(self._root, self._train)
        drawings_path = os.path.join(dataset_path, "drawings.npy")
        labels_path = os.path.join(dataset_path, "labels.npy")

        drawings: np.ndarray = np.load(drawings_path)

        # add 1 channel
        drawings = np.expand_dims(drawings, axis=1)
        labels: np.ndarray = np.load(labels_path)

        drawings = torch.from_numpy(drawings).type(torch.float)
        labels = torch.from_numpy(labels)

        return [drawings, labels]

    def _dataset_path(self, root, train=False):
        root = root.lstrip("/")
        dataset_dir = "train" if train else "test"
        dataset_path = os.path.join(root, dataset_dir)

        return dataset_path

    def _load_from_category(self, category, label, to, size):
        """
        Download dataset for category from GCP to a temporary location `to` and randomly sample
        `size` of them, returning a tuple X, Y where X is a 3D tensor containing images and Y
        is a 1D tensor containing `label`
        """
        print(f"[loading {category}, label={label}, size={size}]")

        npy_filename = f"{category}.npy"
        npy_path = os.path.join(to, npy_filename)

        # download .npy files from gcp
        source_blob_name = os.path.join(self._source_blob_prefix, npy_filename)
        gcp.download_public_file(
            bucket_name=self._bucket_name,
            source_blob_name=source_blob_name,
            destination_file_name=npy_path,
        )

        data: np.ndarray = np.load(npy_path)
        n = data.shape[0]

        # delete .npy file
        os.remove(npy_path)

        # randomly sample
        random_indices = self._random.choice(n, size=size, replace=False)

        features = data[random_indices]
        labels = np.full(size, label)

        return features, labels

    def __len__(self):
        return len(self._dataset[0])

    def __getitem__(self, index):
        return self._dataset[0][index], self._dataset[1][index]
