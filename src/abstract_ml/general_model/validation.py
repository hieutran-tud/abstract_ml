from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar
import numpy as np
from ..utils.data_handler import TrainingData

R = TypeVar('R')


class EarlyStopper:
    """
    Patience-based early stopper driven solely by validation loss.

    Attributes (config):
        max_epochs: hard cap on training epochs.
        enabled: if False, only max_epochs bound applies.
        warmup_epochs: ignore decisions during the first W epochs.
        patience: stop after this many consecutive non-improving epochs.
        ema_alpha: if not None, use EMA smoothing of val loss for decisions.
        min_delta_abs: absolute improvement threshold.
        min_delta_rel: relative improvement threshold in [0,1).
        adapt_to_noise_window: if set (m>=2), adapt 
                               min_delta_abs := max(min_delta_abs, kappa * std(window)).
        adapt_kappa: multiplier for adaptive absolute threshold.
        nan_is_worse: treat NaN/Inf val losses as non-improvements (clamped to +inf).
        validation_losses: list[float] of raw per-epoch validation losses.
        smoothed_loss: the current decision signal (EMA or raw).
        best_smoothed: best decision-signal value seen after warmup.
        best_params: snapshot of best model parameters.
        best_epoch: epoch index (1-based) where best was observed.
        no_improve_count: consecutive non-improvement counter.
        stopped: whether patience condition has been met.
    """

    def __init__(
        self,
        max_epochs: int,
        enabled: bool = True,
        warmup_epochs: int = 3,
        patience: int = 15,
        ema_alpha: float | None = 0.9,
        min_delta_abs: float = 0.0,
        min_delta_rel: float = 0.001,          # 0.1%
        adapt_to_noise_window: int | None = None,
        adapt_kappa: float = 0.3,
        nan_is_worse: bool = True,
    ) -> None:
        self.max_epochs = int(max_epochs)
        self.enabled = bool(enabled)
        self.warmup_epochs = int(warmup_epochs)
        self.patience = int(patience)
        self.ema_alpha = ema_alpha
        self.min_delta_abs = float(min_delta_abs)
        self.min_delta_rel = float(min_delta_rel)
        self.adapt_to_noise_window = adapt_to_noise_window
        self.adapt_kappa = float(adapt_kappa)
        self.nan_is_worse = bool(nan_is_worse)

        self.validation_losses: list[float] = []
        self.smoothed_loss: float | None = None
        self.best_smoothed: float = np.inf
        self.best_params: Any = None
        self.best_epoch: int | None = None  # 1-based
        self.no_improve_count: int = 0
        self.stopped: bool = False

    @property
    def epoch_num(self) -> int:
        """
        Number of epochs processed so far (== len(validation_losses)).
        
        Returns:
            int: Number of epochs processed so far.
        """
        return len(self.validation_losses)

    def get_best_params(self) -> Any:
        """
        Return best parameter snapshot (or None if never set).
        
        Returns:
            Any: Best parameter snapshot.
        """
        return self.best_params

    def initialize(self, model: ValidatableTrainingModel[R]) -> None:
        """
        Reset all runtime state, including the losses and early-stopping counters.
        Use this at the start of a training run.
        
        Args:
            model (ValidatableTrainingModel[R]): The model whose parameters are to be tracked.
        """
        self.validation_losses = []
        self.smoothed_loss = None
        self.best_smoothed = np.inf
        self.best_params = model.get_parameters()
        self.best_epoch = None
        self.no_improve_count = 0
        self.stopped = False

    def log_validation_loss(self, loss: float) -> None:
        """
        Append a new raw validation loss for the current epoch and update the
        decision signal (EMA or raw). Non-finite values are handled according to
        `nan_is_worse`.
        
        Args:
            loss (float): The validation loss to log.
        """
        if not np.isfinite(loss):
            if self.nan_is_worse:
                loss = float('inf')
            else:
                raise ValueError(
                    "Validation loss is NaN/Inf and nan_is_worse=False.")

        self.validation_losses.append(loss)

        # Update EMA (decision signal) or use raw loss
        if self.ema_alpha is None:
            self.smoothed_loss = loss
        else:
            if self.smoothed_loss is None:
                self.smoothed_loss = loss
            else:
                a = self.ema_alpha
                self.smoothed_loss = a * self.smoothed_loss + (1.0 - a) * loss

    def _adaptive_abs_threshold(self) -> float:
        thr = self.min_delta_abs
        m = self.adapt_to_noise_window
        if m is not None and m >= 2 and len(self.validation_losses) >= m:
            window = self.validation_losses[-m:]
            mean = float(np.mean(window))
            var = float(np.mean([(x - mean) ** 2 for x in window]))
            std = np.sqrt(max(0.0, var))
            if np.isfinite(std) and std > 0.0:
                thr = max(thr, self.adapt_kappa * std)
        return thr

    def log_new_param(self, params: Any) -> None:
        """
        Consider the just-finished epoch (the last logged loss) and decide whether the
        provided params are a new best; update early-stopping counters accordingly.
        
        Args:
            params (Any): The model parameters after the just-finished epoch.
        """
        # If no loss has been logged yet, nothing to do.
        if self.epoch_num == 0 or self.smoothed_loss is None:
            return

        t = self.epoch_num  # 1-based epoch count
        s = self.smoothed_loss

        # During warmup: set initial best on the boundary epoch, but do not count patience.
        if t <= self.warmup_epochs:
            if t == self.warmup_epochs:
                self.best_smoothed = s
                self.best_params = params
                self.best_epoch = t
                self.no_improve_count = 0
            return

        # After warmup: evaluate improvement
        improved = False

        # Absolute improvement
        abs_thr = self._adaptive_abs_threshold()
        if s < (self.best_smoothed - abs_thr):
            improved = True

        # Relative improvement (OR with absolute)
        if (not improved) and (self.min_delta_rel is not None) and (self.min_delta_rel > 0.0):
            rel_target = self.best_smoothed * (1.0 - self.min_delta_rel)
            if s < rel_target:
                improved = True

        if improved:
            self.best_smoothed = s
            self.best_params = params
            self.best_epoch = t
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
            if self.no_improve_count >= self.patience:
                self.stopped = True

    def stop_training(self) -> bool:
        """
        Check if training should stop based on patience or max_epochs.
        
        Returns:
            bool: Whether training should stop.
        """
        if self.epoch_num >= self.max_epochs:
            return True
        if not self.enabled:
            return False  # only the max_epochs bound applies above
        return self.stopped

class ValidatableTrainingModel(ABC, Generic[R]):
    """
    An abstract base class for models that can be validated on a validation dataset.
    """

    @abstractmethod
    def get_parameters(self) -> Any:
        """
        Get the model (set of) parameters.

        Returns:
            Any: The model set parameters
        """

    @abstractmethod
    def set_parameters(self, params: Any) -> None:
        """
        Set the model (set of) parameters.

        Args:
            params (Any): The new model parameters
        """

    @abstractmethod
    def validate(self, training_data_handler: TrainingData) -> np.floating:
        """
        Validate the model using the validation dataset.

        Args:
            training_data_handler (TrainingData): an object for managing the training data.

        Returns:
            float: The validation loss.
        """

    @abstractmethod
    def train_epoch(
        self,
        training_data_handler: TrainingData,
        batch_size: int,
        **kwargs
    ) -> R:
        """
        Train the model for one epoch.
        The training data is shuffled and divided into batches. 
        For each batch, the model performs a forward pass,
        computes the loss and its gradient, and updates the model parameters using the optimizer.
        The average loss over the epoch is returned.

        Args:
            training_data_handler (TrainingData): an object for managing the training data.
            batch_size (int): Size of each training batch.
            optimizer (GradientOptimizer): Optimizer for updating model parameters.

        Returns:
            float: outcome of training epoch
        """

    def train(self,
              training_data_handler: TrainingData,
              early_stopper: EarlyStopper,
              batch_size: int,
              verbose: Callable | None = None,
              /, **kwargs) -> tuple[list[R], list[float]]:
        """
        Train the model for a specified number of epochs.
        After each epoch, the model is validated using the provided training data handler.
        The method returns a list of training outcome [R] and validation losses for each epoch.

        Args:
            training_data_handler (TrainingData): an object for managing the training data.
            epochs (int): Number of training epochs.
            batch_size (int): Minibatch size for real data and latent noise.

        Returns:
            tuple[list[R], list[float]]: A tuple containing:
                - training_outcomes (list[R]): List of training outcomes for each epoch.
                - validation_losses (list[float]): List of validation losses for each epoch.
        """
        training_outcomes: list[R] = []
        early_stopper.initialize(self)
        while not early_stopper.stop_training():
            outcome = self.train_epoch(
                training_data_handler, batch_size, **kwargs)
            training_outcomes.append(outcome)
            validation_loss = self.validate(training_data_handler)
            early_stopper.log_validation_loss(float(validation_loss))
            current_params = self.get_parameters()
            early_stopper.log_new_param(current_params)
            if verbose is not None:
                verbose(model = self,
                        epoch = early_stopper.epoch_num,
                        training_outcome = outcome,
                        validation_loss = validation_loss)
        best_params = early_stopper.get_best_params()
        self.set_parameters(best_params)
        validation_losses = early_stopper.validation_losses
        return training_outcomes, validation_losses
