import os
import glob
import numpy as np

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils import data

from torch.utils.data.distributed import DistributedSampler

from .dataset import Dataset
from .manifest_dataset import ManifestDataset


class MultiEpochsDataLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def split_ids(len_ids):
    train_size = int(round((80 / 100) * len_ids))
    valid_size = int(round((10 / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))

    train_indices, test_indices = train_test_split(
        np.linspace(0, len_ids - 1, len_ids).astype("int"),
        test_size=test_size,
        random_state=42,
    )

    train_indices, val_indices = train_test_split(
        train_indices, test_size=test_size, random_state=42
    )

    return train_indices, test_indices, val_indices


def get_dataloaders(
    rank,
    world_size,
    input_paths=None,
    targets=None,
    batch_size=1,
    workers=8,
    prefetch_factor=2,
    pin_memory=True,
    persistent_workers=True,
    seed=0,
    train_paths=None,
    train_labels=None,
    train_meta=None,
    val_paths=None,
    val_labels=None,
    val_meta=None,
    test_paths=None,
    test_labels=None,
    test_meta=None,
):

    transform_input4train = transforms.Compose(
        [
            transforms.ColorJitter(
                brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01
            ),
            transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def _normalize_paths(paths):
        if paths is None:
            return None
        if isinstance(paths, str) and os.path.isdir(paths):
            return sorted(glob.glob(os.path.join(paths, "*.jpg")))
        return paths

    train_paths = _normalize_paths(train_paths)
    val_paths = _normalize_paths(val_paths)
    test_paths = _normalize_paths(test_paths)

    if (
        train_paths is not None
        and val_paths is not None
        and test_paths is not None
        and train_labels is not None
        and val_labels is not None
        and test_labels is not None
    ):
        train_dataset = ManifestDataset(
            train_paths,
            train_labels,
            train_meta,
            transform_input=transform_input4train,
        )

        train_sampler = DistributedSampler(
            train_dataset,
            rank=rank,
            num_replicas=world_size,
            shuffle=True,
            drop_last=True,
            seed=seed,
        )

        train_dataloader = MultiEpochsDataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            drop_last=True,
            num_workers=workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        if rank == 0:
            transform_input4test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            val_dataset = ManifestDataset(
                val_paths,
                val_labels,
                val_meta,
                transform_input=transform_input4test,
            )
            test_dataset = ManifestDataset(
                test_paths,
                test_labels,
                test_meta,
                transform_input=transform_input4test,
            )

            val_dataloader = MultiEpochsDataLoader(
                dataset=val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )

            test_dataloader = MultiEpochsDataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )
        else:
            test_dataloader = None
            val_dataloader = None

        return train_dataloader, test_dataloader, val_dataloader, train_sampler

    train_indices, test_indices, val_indices = split_ids(len(input_paths))

    train_dataset = Dataset(
        input_paths=input_paths,
        targets=targets,
        transform_input=transform_input4train,
    )

    train_dataset = data.Subset(train_dataset, train_indices)

    train_sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True, drop_last=True
    )

    train_dataloader = MultiEpochsDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    if rank == 0:

        transform_input4test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        test_dataset = Dataset(
            input_paths=input_paths,
            targets=targets,
            transform_input=transform_input4test,
        )

        val_dataset = Dataset(
            input_paths=input_paths,
            targets=targets,
            transform_input=transform_input4test,
        )
        val_dataset = data.Subset(val_dataset, val_indices)
        test_dataset = data.Subset(test_dataset, test_indices)

        test_dataloader = MultiEpochsDataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        val_dataloader = MultiEpochsDataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    else:
        test_dataloader = None
        val_dataloader = None

    return train_dataloader, test_dataloader, val_dataloader, train_sampler


def get_test_dataloader(
    input_paths,
    targets=None,
    meta=None,
    workers=8,
    prefetch_factor=2,
    pin_memory=True,
    persistent_workers=True,
    batch_size=1,
):

    transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = ManifestDataset(
        input_paths,
        targets,
        meta,
        transform_input=transform_input4test,
        return_meta=meta is not None,
    )

    test_dataloader = MultiEpochsDataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return test_dataloader

