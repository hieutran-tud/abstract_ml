from abc import ABC, abstractmethod
from typing import override
import numpy as np
from ..general_model.optimizer import GradientOptimizer
from ..general_model.parameterized_model import ParameterizedModel
from ..general_model.validation import ValidatableTrainingModel
from ..utils.data_handler import TrainingData
from ..general_model.validation import EarlyStopper

rng = np.random.default_rng(123411)

def _softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def _log_softmax(logits: np.ndarray):
    max_logit = np.max(logits, axis=-1, keepdims=True)
    log_sum_exp = np.log(
        np.sum(np.exp(logits - max_logit), axis=-1, keepdims=True))
    return logits - max_logit - log_sum_exp


class ClassificationModel(ABC):
    """
    Abstract base class for classification models.
    
    Attributes:
        num_classes (int): Number of label classes in the classification task.
    """
    num_classes: int

    @abstractmethod
    def predict_label(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the given input data.
        The labels are represented as integers from 0 to (num_classes - 1).
        
        Args:
            x (np.ndarray): Input data of shape (n_samples, n_features), dtype float.
        
        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,), dtype int.
        """

    def accuracy(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Calculate the model accuracy:
        Accuracy = (Number of correct predictions) / (Total number of predictions)
    
        Args:
            x_test (np.ndarray): Input data of shape (n_samples, n_features), dtype float.
            y_test (np.ndarray): True class labels of shape (n_samples,), dtype int.

        Returns:
            float: The accuracy of the model on the test data.
        """
        y_pred = self.predict_label(x_test)
        return np.mean(y_pred == y_test)

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes


class ProbabilisticClassificationModel(ClassificationModel, ValidatableTrainingModel[np.floating]):
    """
    A probabilistic classification model 
    that uses a parameterized model through a softmax output layer:
    C_theta(x) = softmax(M_theta(x))
    where M_theta is a parameterized logits function.
    
    Attributes:
        model (ParameterizedModel): The underlying parameterized model for logits computation.
        optimizer (GradientOptimizer): The gradient descent optimizer used for training.
    """



    @override
    def get_parameters(self) -> np.ndarray:
        """
        Get the model (set of) parameters.

        Returns:
            np.ndarray: The model set parameters
        """
        return self.model.get_parameter()


    @override
    def set_parameters(self, params: np.ndarray) -> None:
        """
        Set the model (set of) parameters.

        Args:
            params (np.ndarray): The new model parameters
        """
        self.model.set_parameter(params)


    def __init__(self,
                 model: ParameterizedModel,
                 optimizer: GradientOptimizer
                 ) -> None:
        self.model = model
        self.optimizer = optimizer
        super().__init__(model.output_dim())

    @staticmethod
    def grad_given_logits(
        y_true: np.ndarray,
        logits: np.ndarray
    ) -> np.ndarray:
        """
        Compute the Loss gradient given the true distribution, 
        usually in one-hot encoded format, and the logits:
            L = - (1/N) * Σ Σ y_true * log(softmax(logits))
            dL/dlogits = (softmax(logits) - y_true) / N

        Args:
            y_true (np.ndarray): True distribution, shape (num_samples, output_dim), dtype float
            logits (np.ndarray): Logits from the model, shape (num_samples, output_dim), dtype float

        Returns:
            np.ndarray: Gradient of the loss with respect to logits, shape (num_samples, output_dim)
        """
        num_samples = y_true.shape[0]
        distr = _softmax(logits)
        grad_logits = (distr - y_true) / num_samples
        return grad_logits

    @staticmethod
    def compute_loss_by_logits(
        y_true: np.ndarray,
        logits: np.ndarray,
    ) -> np.floating:
        """
        Compute the cross-entropy loss given logits and true distributions,
        usually in one-hot encoded format.
        
        Args:
            y_true (np.ndarray): True distribution, shape (n_samples, n_classes), dtype float.
            logits (np.ndarray): (Estimated) logits from the model, 
                                 shape (n_samples, n_classes), dtype float.
            
        Returns:
            float: The computed cross-entropy loss.
        """
        num_samples = y_true.shape[0]
        log_probs = _log_softmax(logits)
        loss = -np.sum(y_true * log_probs) / num_samples
        return loss

    def _label_to_distribution(self, train_label: np.ndarray) -> np.ndarray:
        """
        Convert integer class labels to one-hot encoded class distributions.
        
        Args:
            train_label (np.ndarray): Class labels of shape (n_samples,), dtype int.
            
        Returns:
            np.ndarray: One-hot encoded class distributions 
                        of shape (n_samples, n_classes), dtype float.
        """
        train_distribution = np.eye(self.num_classes)[train_label]
        return train_distribution

    def _distribution_to_label(self, class_distribution: np.ndarray) -> np.ndarray:
        """
        Convert class distributions to integer class labels.
        
        Args:
            class_distribution (np.ndarray): Class distributions of shape 
                                             (n_samples, n_classes), dtype float.
            
        Returns:
            np.ndarray: Integer class labels of shape (n_samples,), dtype int.
        """
        return np.argmax(class_distribution, axis=1)

    def predict_distribution(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class distributions for the given input data.
        
        Args:
            x (np.ndarray): Input data of shape (n_samples, n_features), dtype float.
            
        Returns:
            np.ndarray: Predicted class distributions of shape (n_samples, n_classes), dtype float.
        """
        logits = self.model.forward(x)
        return _softmax(logits)

    def train_epoch(self,
                    training_data_handler: TrainingData,
                    batch_size: int,
                    *,
                    learning_rate: float = 0.001,
                    **_) -> np.floating:
        """
        Train the model for one epoch.
        The training data is shuffled and divided into batches. 
        For each batch, the model performs a forward pass,
        computes the loss and its gradient, and updates the model parameters using the optimizer.
        The average loss over the epoch is returned.

        Args:
            training_data_handler (TrainingData): an object for managing the training data.
            batch_size (int): Size of each training batch.
            learning_rate (float, optional): Learning rate for the optimizer.
                                             Defaults to 0.001.
            
        Returns:
            float: Average loss over the epoch.
        """
        total_loss: np.floating = np.float64(0)
        shuffled_batches = training_data_handler.shuffle_and_divide(batch_size)
        for (x_data, y_true) in shuffled_batches:
            logits = self.model.forward(x_data)
            loss = self.compute_loss_by_logits(y_true, logits)
            dloss_dlogits = ProbabilisticClassificationModel.grad_given_logits(
                y_true, logits)
            grad_params = self.model.loss_gradient_by_param(
                x_data, dloss_dlogits)
            self.optimizer.stepwise_update(learning_rate, self.model, grad_params)
            total_loss += loss
        return total_loss / len(shuffled_batches)

    def train_with_labels(self,
                          x_train: np.ndarray,
                          y_train: np.ndarray,
                          epochs: int,
                          batch_size: int,
                          learning_rate: float = 0.001,
                          validation_ratio: float = 0,
                          rand_gen: np.random.Generator = rng
                          ) -> tuple[list[np.floating], list[float]]:
        """
        Train the model using input data and integer class labels.
        The integer class labels are converted to one-hot encoded distributions before training.
        
        Args:
            x_train (np.ndarray): Training input data of shape (n_samples, n_features), dtype float.
            y_train (np.ndarray): Integer class labels of shape (n_samples,), dtype int.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
            optimizer (GradientOptimizer): Optimizer for updating model parameters.
            validation_ratio (float, optional): Ratio of data to use for validation. 
                                                Defaults to 0.

        Returns:
            tuple[list[np.floating], list[float]]: Training and validation losses for each epoch.
        """
        y_train_distribution = self._label_to_distribution(y_train)

        training_data_handler = TrainingData(
            x_train, y_train_distribution, validation_ratio, rand_gen=rand_gen)
        early_stopper = EarlyStopper(max_epochs=epochs, patience=max(5, epochs//10))
        return self.train(
            training_data_handler,
            early_stopper,
            batch_size,
            learning_rate=learning_rate
        )

    @override
    def validate(self, training_data_handler: TrainingData) -> np.floating:
        """
        Validate the model using the validation dataset.
        
        Args:
            training_data_handler (TrainingData): an object for managing the training data.

        Returns:
            float: The validation loss.
        """
        validation_x, validation_y = training_data_handler.get_validation_data()
        validation_loss = self.compute_loss_by_input(
            validation_y, validation_x)
        return validation_loss

    @override
    def predict_label(self, x: np.ndarray) -> np.ndarray:
        class_distribution = self.predict_distribution(x)
        return self._distribution_to_label(class_distribution)

    def compute_loss(self, y_prob_pred: np.ndarray,
                     y_true: np.ndarray,
                     epsilon: float = 1e-15) -> np.floating:
        """
        Compute the cross-entropy loss given predicted probabilities and true labels.
        
        Args:
            y_prob_pred (np.ndarray): Predicted class probabilities, shape (n_samples, n_classes).
            y_true (np.ndarray): True distribution, usually in one-hot encoded format,
                                 shape (n_samples, n_classes).
            epsilon (float, optional): Small constant to avoid log(0). 
                                       Defaults to 1e-15.
        
        Returns:
            float: The computed cross-entropy loss.
        """
        num_samples = y_true.shape[0]
        log_probs = np.log(np.clip(y_prob_pred, epsilon, 1.0-epsilon))
        loss = -np.sum(y_true * log_probs) / num_samples
        return loss

    def compute_loss_by_input(self, y_true: np.ndarray, x: np.ndarray) -> np.floating:
        """Compute the cross-entropy loss given input data and true labels.
        
        Args:
            y_true (np.ndarray): True labels in one-hot encoded format, 
                                 shape (n_samples, n_classes).
            x (np.ndarray): Input data, shape (n_samples, n_features).
            
        Returns:
            float: The computed cross-entropy loss.
        """
        logits = self.model.forward(x)
        log_probs = _log_softmax(logits)
        num_samples = y_true.shape[0]
        loss = -np.sum(y_true * log_probs) / num_samples
        return loss
