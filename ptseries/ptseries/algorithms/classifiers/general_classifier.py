from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    import os

    import torch
    from torch.utils.data import DataLoader

    from ptseries.common.logger import Logger

    FILE_LIKE: TypeAlias = str | os.PathLike


from ptseries.optimizers import HybridOptimizer


class VariationalClassifier:
    """General class for a quantum variational classifier.

    The user specifies the model and the loss function when instantiating this class,
    and specifies the training data in the train function

    Args:
        model: model used for describing the architecture (classical network + PTLayer) of the variational
            classifier.
        loss_function: loss function used for training.
        optimizer: a PyTorch optimizer used for training. If None, HybridOptimizer is used.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_function: Callable,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        self.model = model
        self.loss_function = loss_function

        # Initialize optimizers with default learning rates, which are overridden in the train method
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = HybridOptimizer(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Function that performs inference.

        Args:
            x: PyTorch tensor for inference

        Returns:
            Tensor containing the inference result
        """
        return self.model(x)

    def train(
        self,
        train_dataloader: DataLoader,
        learning_rate: float = 1e-3,
        epochs: int = 5,
        val_dataloader: DataLoader | None = None,
        val_frequency: float | None = None,
        val_filename: FILE_LIKE | None = None,
        print_frequency: int = 100,
        verbose: bool = True,
        logger: Logger | None = None,
    ):
        """Trains the classifier.

        Args:
            train_dataloader: PyTorch dataloader containing the training data
            learning_rate: learning rate
            epochs: number of epochs to train for
            val_dataloader: PyTorch dataloader containing the validation data
            val_frequency: number of epochs after which to measure performance on val set.
                Can be fractional.
            val_filename: filename for saving the best validation model as a pickle file.
                If a logger is specified, this is optional and best left empty
            print_frequency: how often to print the loss, measured in batches
            verbose: whether to print information
            logger: Logger instance to which training data and metadata is saved
        """
        # Make sure that val parameters are correctly specified
        if val_dataloader is not None or val_frequency is not None or val_filename is not None:
            if val_dataloader is None or val_frequency is None:
                raise Exception("To validate during training, please enter validation dataloader and frequency.")
            else:
                if val_filename is None and logger is None:
                    raise Exception("For validation, please specify a val_filename, a logger, or both")
                # This allows for non-integer val frequencies, for example val_frequency=0.5 runs twice per epoch
                val_batch_frequency = int(val_frequency * len(train_dataloader))

        if logger is not None:
            metadata = {
                "learning_rate": learning_rate,
                "epochs": epochs,
            }
            logger.register_metadata(metadata)
            # Redirect val filename to logger folder
            if val_filename is None:
                val_filename = logger.log_folder + "/val_model.pt"
            else:
                val_filename = logger.log_folder + "/" + val_filename

        # Override default learning rates
        self.optimizer.set_initial_lr(learning_rate)

        # Keep track of total number of batches processed and validation score
        batch_number = 0
        val_best = np.inf

        print("Starting training...")

        self.model.train()

        for epoch in range(epochs):
            if verbose:
                print("Training on epoch {}...".format(epoch + 1))

            for batch_idx, (X, Y) in enumerate(train_dataloader):
                if verbose:
                    self._estimate_running_time(batch_number, epochs, len(train_dataloader))

                output = self.model(X)
                loss = self.loss_function(output, Y)

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_number += 1

                if logger is not None:
                    logger.log("train_losses", batch_number, loss.item())

                if batch_number % print_frequency == print_frequency - 1 and verbose:
                    print("Batch {}: loss for this batch is {:.2f}".format(batch_number + 1, loss.detach().numpy()))

                # Perform validation
                if val_dataloader is not None and batch_number % val_batch_frequency == val_batch_frequency - 1:
                    self.model.eval()

                    if verbose:
                        print("Starting validation...")
                    val_loss = []
                    with torch.no_grad():
                        for X_val, Y_val in val_dataloader:
                            loss = self.loss_function(self.forward(X_val), Y_val)
                            val_loss.append(loss.mean())
                    val_loss = torch.stack(val_loss).mean()
                    if verbose:
                        print("Current validation loss is {:.2f}".format(val_loss))
                    if logger is not None:
                        logger.log("val_losses", batch_number, val_loss.item())

                    if val_loss < val_best:
                        val_best = val_loss

                        self.save(val_filename)
                        if verbose:
                            print("Model saved.")

                    self.model.train()

        if logger is not None:
            if logger.log_dir is not None:
                logger.save()

        self.model.eval()

    def _estimate_running_time(self, batch_number: int, epochs: int, n_batch_per_epoch: int):
        """Estimates and prints the training time by extrapolating from the first 25 batches."""
        if batch_number == 5:
            self.time_5 = time.time()
        elif batch_number == 25:
            time_25 = time.time()
            time_single_batch = (time_25 - self.time_5) / 20
            n_batch_total = epochs * n_batch_per_epoch
            tot_time_min = n_batch_total * time_single_batch / 60
            msg = "Estimated total training time : {} minutes".format(int(tot_time_min))
            print(msg)

    def save(self, filename: FILE_LIKE):
        """Save the model of the classifier as a .pt file.

        Args:
            filename: name of the .pt file
        """
        torch.save(self.model.to("cpu").state_dict(), filename)

    def load(self, filename: FILE_LIKE):
        """Load the specified model."""
        self.model.load_state_dict(torch.load(filename))
