import os
import numpy as np
import gcp


class QuickDrawDataset:
    def __init__(self, root: str, train: bool = True, download: bool = False):
        print(f"[quickdraw dataset: root={root}, train={train}, download={download}]")

        self._bucket_name = "quickdraw_dataset"
        self._source_blob_prefix = "full/numpy_bitmap"
        self._random = np.random.default_rng(5)

        # dataset location
        root = root.lstrip("/")
        dataset_dir = "train" if train else "test"
        dataset_path = os.path.join(root, dataset_dir)

        label = "airplane"
        index = 0

        # create dataset folder
        os.makedirs(dataset_path, exist_ok=True)

        # .npy file location
        npy_filename = f"{label}.npy"
        npy_path = os.path.join(dataset_path, npy_filename)

        # download .npy files from gcp if download=True
        if download:
            source_blob_name = os.path.join(self._source_blob_prefix, npy_filename)
            gcp.download_public_file(
                bucket_name=self._bucket_name,
                source_blob_name=source_blob_name,
                destination_file_name=npy_path,
            )

        # parse data from .npy file
        data: np.ndarray = np.load(npy_path)
        print(data)

        num_items = 1
        random_indices = self._random.choice(
            data.shape[0], size=num_items, replace=False
        )

        selected = data[random_indices]
        print(selected)

        items = []

        for image in selected:
            items.append((image, index))

        print(items)

    def get(self):
        return []
