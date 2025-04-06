import torch
from opacus import PrivacyEngine

import collections.abc as abc
from functools import partial
from torch.utils.data._utils.collate import default_collate
from typing import Union, Callable
import copy
from opacus.optimizers import DPOptimizer
from opacus.utils.uniform_sampler import (
    DistributedUniformWithReplacementSampler,
    UniformWithReplacementSampler,
)
from torch.utils.data import BatchSampler, DataLoader, Sampler
import math
import numpy as np

from typing import List


class BatchSplittingSampler(Sampler[List[int]]):
    """
    Samples according to the underlying instance of ``Sampler``, but splits
    the index sequences into smaller chunks.

    Used to split large logical batches into physical batches of a smaller size,
    while coordinating with DPOptimizer when the logical batch has ended.
    """

    def __init__(
        self,
        *,
        sampler: Sampler[List[int]],
        max_batch_size: int,
        optimizer: DPOptimizer,
    ):
        """

        Args:
            sampler: Wrapped Sampler instance
            max_batch_size: Max size of emitted chunk of indices
            optimizer: optimizer instance to notify when the logical batch is over
        """
        self.sampler = sampler
        self.max_batch_size = max_batch_size
        self.optimizer = optimizer

    def __iter__(self):
        for batch_idxs in self.sampler:
            if len(batch_idxs) == 0:
                self.optimizer.signal_skip_step(do_skip=False)
                yield []
                continue

            split_idxs = np.array_split(
                batch_idxs, math.ceil(len(batch_idxs) / self.max_batch_size)
            )
            split_idxs = [s.tolist() for s in split_idxs]
            for x in split_idxs[:-1]:
                self.optimizer.signal_skip_step(do_skip=True)
                yield x
            self.optimizer.signal_skip_step(do_skip=False)
            yield split_idxs[-1]

    def __len__(self):
        if isinstance(self.sampler, BatchSampler):
            return math.ceil(
                len(self.sampler) * (self.sampler.batch_size / self.max_batch_size)
            )
        elif isinstance(self.sampler, UniformWithReplacementSampler) or isinstance(
            self.sampler, DistributedUniformWithReplacementSampler
        ):
            expected_batch_size = self.sampler.sample_rate * self.sampler.num_samples
            return math.ceil(
                len(self.sampler) * (expected_batch_size / self.max_batch_size)
            )

        return len(self.sampler)


class OpacusAnonymizationManager:
    def __init__(self, params):
        self.params = params

    def apply_privacy(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
    ):
        model, optimizer, train_dataloader, privacy_engine = self._apply_privacy(
            model, optimizer, train_dataloader
        )
        train_dataloader.collate_fn = self._empty_collate(train_dataloader.dataset[0])
        max_physical_batch_size = self.params["differential_privacy"].get(
            "max_physical_batch_size", self.params["batch_size"]
        )
        if max_physical_batch_size != self.params["batch_size"]:
            train_dataloader = self._wrap_data_loader(
                data_loader=train_dataloader,
                max_batch_size=max_physical_batch_size,
                optimizer=optimizer,
            )

        return model, optimizer, train_dataloader, privacy_engine

    def _apply_privacy(self, model, optimizer, train_dataloader):
        privacy_engine = PrivacyEngine(
            accountant=self.params["differential_privacy"]["accountant"],
            secure_mode=self.params["differential_privacy"]["secure_mode"],
        )
        epsilon = self.params["differential_privacy"].get("epsilon")

        if epsilon is not None:
            (
                model,
                optimizer,
                train_dataloader,
            ) = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                max_grad_norm=self.params["differential_privacy"]["max_grad_norm"],
                epochs=self.params["num_epochs"],
                target_epsilon=self.params["differential_privacy"]["epsilon"],
                target_delta=self.params["differential_privacy"]["delta"],
            )
        else:
            model, optimizer, train_dataloader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                noise_multiplier=self.params["differential_privacy"][
                    "noise_multiplier"
                ],
                max_grad_norm=self.params["differential_privacy"]["max_grad_norm"],
            )
        return model, optimizer, train_dataloader, privacy_engine

    def _empty_collate(
        self,
        item_example: Union[
            torch.Tensor, np.ndarray, abc.Mapping, abc.Sequence, int, float, str
        ],
    ) -> Callable:
        """
        Creates a new collate function that behave same as default pytorch one,
        but can process the empty batches.

        Args:
            item_example (Union[torch.Tensor, np.ndarray, abc.Mapping, abc.Sequence, int, float, str]): An example item from the dataset.

        Returns:
            Callable: function that should replace dataloader collate: `dataloader.collate_fn = empty_collate(...)`
        """

        def custom_collate(batch, _empty_batch_value):
            if len(batch) > 0:
                return default_collate(batch)  # default behavior
            else:
                return copy.copy(_empty_batch_value)

        empty_batch_value = self._build_empty_batch_value(item_example)

        return partial(custom_collate, _empty_batch_value=empty_batch_value)

    def _build_empty_batch_value(
        self,
        sample: Union[
            torch.Tensor, np.ndarray, abc.Mapping, abc.Sequence, int, float, str
        ],
    ):
        """
        Build an empty batch value from a sample. This function is used to create a placeholder for empty batches in an iteration. Inspired from https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/collate.py#L108. The key difference is that pytorch `collate` has to traverse batch of objects AND unite its fields to lists, while this function traverse a single item AND creates an "empty" version of the batch.

        Args:
            sample (Union[torch.Tensor, np.ndarray, abc.Mapping, abc.Sequence, int, float, str]): A sample from the dataset.

        Raises:
            TypeError: If the data type is not supported.

        Returns:
            Union[torch.Tensor, np.ndarray, abc.Mapping, abc.Sequence, int, float, str]: An empty batch value.
        """
        if isinstance(sample, torch.Tensor):
            # Create an empty tensor with the same shape except for the zeroed batch dimension.
            return torch.empty((0,) + sample.shape)
        elif isinstance(sample, np.ndarray):
            # Create an empty tensor from a numpy array, also with the zeroed batch dimension.
            return torch.empty(
                (0,) + sample.shape, dtype=torch.from_numpy(sample).dtype
            )
        elif isinstance(sample, abc.Mapping):
            # Recursively handle dictionary-like objects.
            return {
                key: self._build_empty_batch_value(value)
                for key, value in sample.items()
            }
        elif isinstance(sample, tuple) and hasattr(sample, "_fields"):  # namedtuple
            return type(sample)(
                *(self._build_empty_batch_value(item) for item in sample)
            )
        elif isinstance(sample, abc.Sequence) and not isinstance(sample, str):
            # Handle lists and tuples, but exclude strings.
            return [self._build_empty_batch_value(item) for item in sample]
        elif isinstance(sample, (int, float, str)):
            # Return an empty list for basic data types.
            return []
        else:
            raise TypeError(f"Unsupported data type: {type(sample)}")

    def _wrap_data_loader(
        self, data_loader: DataLoader, max_batch_size: int, optimizer: DPOptimizer
    ):
        """
        Replaces batch_sampler in the input data loader with ``BatchSplittingSampler``

        Args:
            data_loader: Wrapper DataLoader
            max_batch_size: max physical batch size we want to emit
            optimizer: DPOptimizer instance used for training

        Returns:
            New DataLoader instance with batch_sampler wrapped in ``BatchSplittingSampler``
        """

        return DataLoader(
            dataset=data_loader.dataset,
            batch_sampler=BatchSplittingSampler(
                sampler=data_loader.batch_sampler,
                max_batch_size=max_batch_size,
                optimizer=optimizer,
            ),
            num_workers=data_loader.num_workers,
            collate_fn=data_loader.collate_fn,
            pin_memory=data_loader.pin_memory,
            timeout=data_loader.timeout,
            worker_init_fn=data_loader.worker_init_fn,
            multiprocessing_context=data_loader.multiprocessing_context,
            generator=data_loader.generator,
            prefetch_factor=data_loader.prefetch_factor,
            persistent_workers=data_loader.persistent_workers,
        )
