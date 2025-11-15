from typing import Callable, override
import numpy as np
from ..general_model.parameterized_model import ParameterizedModel
from .gan_components import Generator, Discriminator
from ..general_model.optimizer import Adam, GradientOptimizer
from ..general_model.validation import ValidatableTrainingModel
from .noise_generator import NoiseSampler
from ..utils.function_collections import logistic, softplus
from ..utils.helpers import fid
from ..utils.data_handler import TrainingData


# ----------------------------------------
#            GAN TRAINER
# ----------------------------------------

class GANTrainer(ValidatableTrainingModel[tuple[float, float]]):
    """    
    A trainer class for Generative Adversarial Networks (GANs).

    This class orchestrates the training process for a GAN,
    including managing the generator and discriminator,
    sampling noise, computing losses and gradients, and updating model parameters.

    Attributes:
        generator (Generator): The generator model.
        discriminator (Discriminator): The discriminator model.
        noise_sampler (NoiseSampler): The noise sampler for generating latent vectors.
    """

    generator: Generator
    discriminator: Discriminator
    noise_sampler: NoiseSampler

    @staticmethod
    def discriminator_grads_given_logits(
        logits_real: np.ndarray,
        logits_fake: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the discriminator gradients given the logits for real and fake data.

        Args:
            logits_real (np.ndarray): Logits for real data, shape (batch_size, 1), dtype float.
            logits_fake (np.ndarray): Logits for fake data, shape (batch_size, 1), dtype float.
        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - grad_real (np.ndarray): dLoss/dlogits_real, shape (batch_size, 1), dtype float.
                - grad_fake (np.ndarray): dLoss/dlogits_fake, shape (batch_size, 1), dtype float.
        """
        d_real = logistic(logits_real)
        d_fake = logistic(logits_fake)

        m1 = logits_real.shape[0]
        m2 = logits_fake.shape[0]
        grad_real = - (1 - d_real) / m1
        grad_fake = d_fake / m2

        return grad_real, grad_fake

    @staticmethod
    def discriminator_loss_given_logits(
        logits_real: np.ndarray,
        logits_fake: np.ndarray
    ) -> float:
        """
        Compute the discriminator loss given the logits for real and fake data.

        Args:
            logits_real (np.ndarray): Logits for real data, shape (batch_size, 1), dtype float.
            logits_fake (np.ndarray): Logits for fake data, shape (batch_size, 1), dtype float.
        Returns:
            float: The total discriminator loss.
        """
        loss = np.mean(softplus(-logits_real)) + np.mean(softplus(logits_fake))
        return loss

    @staticmethod
    def generator_grads_given_logits(
        logits_fake: np.ndarray
    ) -> np.ndarray:
        """
        Compute the generator gradients given the logits for fake data.

        Args:
            logits_fake (np.ndarray): Logits for fake data, shape (batch_size, 1), dtype float
        Returns:
            np.ndarray: dLoss/dlogits_fake, shape (batch_size, 1), dtype float
        """
        m2 = logits_fake.shape[0]
        grad_fake = - (1 - logistic(logits_fake)) / m2
        return grad_fake

    @staticmethod
    def generator_loss_given_logits(
        logits_fake: np.ndarray
    ) -> float:
        """
        Compute the generator loss given the logits for fake data.

        Args:
            logits_fake (np.ndarray): Logits for fake data., shape (batch_size, 1), dtype float
        Returns:
            float: The total generator loss.
        """
        loss = np.mean(softplus(-logits_fake))
        return loss

    def __init__(self, gen_model: ParameterizedModel,
                 disc_model: ParameterizedModel,
                 noise_sampler: NoiseSampler) -> None:
        self.generator = Generator(gen_model)
        self.discriminator = Discriminator(disc_model)
        self.noise_sampler = noise_sampler

    @override
    def get_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.discriminator.model.get_parameter(),
                self.generator.model.get_parameter())

    @override
    def set_parameters(self, params: tuple[np.ndarray, np.ndarray]) -> None:
        self.discriminator.model.set_parameter(params[0])
        self.generator.model.set_parameter(params[1])


    def fid_score(self, real_data: np.ndarray,
                  feature_extractor: Callable[[
                      np.ndarray], np.ndarray] | None = None
                  ) -> np.floating:
        """
        Compute the Fréchet Inception Distance (FID) between real and generated data by the model.

        Args:
            real_data (np.ndarray): Real data samples, shape (num_samples, data_dim), dtype float.
            feature_extractor (Callable[[np.ndarray], np.ndarray] | None, optional): 
                A function to extract features from the data.
                If None, the raw data will be used. 
                Defaults to None.

        Returns:
            float: The FID score between real and generated data.
        """
        num_samples = real_data.shape[0]
        generated_data = self.generate(num_samples)
        real_features = feature_extractor(
            real_data) if feature_extractor else real_data
        generated_features = feature_extractor(
            generated_data) if feature_extractor else generated_data
        mu_real = np.mean(real_features, axis=0)
        mu_gen = np.mean(generated_features, axis=0)
        cov_real = np.cov(real_features, rowvar=False)
        cov_gen = np.cov(generated_features, rowvar=False)
        return fid(mu_real, mu_gen, cov_real, cov_gen)

    def train_epoch(
        self,
        training_data_handler: TrainingData,
        batch_size: int,
        *,
        gen_optimizer: GradientOptimizer | None = None,
        disc_optimizer: GradientOptimizer | None = None,
        d_steps_per_one_g_step: int = 1,
        **_
    ) -> tuple[float, float]:
        """
        Perform one epoch of adversarial training. Each epoch processes d_steps_per_one_g_step
        minibatches for the discriminator followed by one minibatch for the generator. The method
        returns the average losses for the discriminator and generator over the epoch.

        Args:
            training_data_handler (TrainingData): an object for managing the training data
            batch_size (int): Minibatch size for real data and latent noise
            gen_optimizer (GradientOptimizer | None): Optimizer for the generator.
                                                      If None, defaults to Adam with lr=0.001.
                                                      Defaults to None.
            disc_optimizer (GradientOptimizer | None): Optimizer for the discriminator.
                                                       If None, defaults to Adam with lr=0.001.
                                                       Defaults to None.
            d_steps_per_one_g_step (int, optional): Number of discriminator steps per generator step
                                                    Defaults to 1.

        Returns:
            tuple[float, float]: A tuple containing:
                - avg_disc_loss (float): The average discriminator loss for the epoch.
                - avg_gen_loss (float): the average generator loss for the epoch.
        """

        disc_losses: list[float] = []
        gen_losses: list[float] = []

        shuffled_batches = training_data_handler.shuffle_and_divide(batch_size)

        if disc_optimizer is None:
            disc_optimizer = Adam(0.001)
        if gen_optimizer is None:
            gen_optimizer = Adam(0.001)

        for (x_real,) in shuffled_batches:
            batch_size = x_real.shape[0]
            # Discriminator steps
            for _ in range(d_steps_per_one_g_step):
                x_fake = self.generate(batch_size)
                logits_real = self.discriminator.logits(x_real)
                logits_fake = self.discriminator.logits(x_fake)
                disc_grad_real, disc_grad_fake = GANTrainer.discriminator_grads_given_logits(
                    logits_real, logits_fake
                )
                disc_loss = GANTrainer.discriminator_loss_given_logits(
                    logits_real, logits_fake
                )
                disc_losses.append(disc_loss)

                grads_phi_real = self.discriminator.model.loss_gradient_by_param(
                    x_real, disc_grad_real)
                grads_phi_fake = self.discriminator.model.loss_gradient_by_param(
                    x_fake, disc_grad_fake)

                total_grads_phi = grads_phi_real + grads_phi_fake
                disc_optimizer.stepwise_update(
                    self.discriminator.model,
                    total_grads_phi
                )

            # Generator step
            z = self.noise_sampler.sample(batch_size)
            x_fake = self.generator.generate(z)
            logits_fake = self.discriminator.logits(x_fake)
            gen_loss = GANTrainer.generator_loss_given_logits(
                logits_fake
            )
            gen_grad_fake = GANTrainer.generator_grads_given_logits(
                logits_fake
            )
            gen_losses.append(gen_loss)

            grad_disc_input = self.discriminator.model.loss_gradient_by_input(
                x_fake, gen_grad_fake)
            grad_theta = self.generator.model.loss_gradient_by_param(
                z, grad_disc_input)

            gen_optimizer.stepwise_update(
                self.generator.model,
                grad_theta
            )

        avg_disc_loss = float(np.mean(disc_losses)) if disc_losses else np.nan
        avg_gen_loss = float(np.mean(gen_losses)) if gen_losses else np.nan

        return avg_disc_loss, avg_gen_loss

    def validate(self, training_data_handler: TrainingData) -> np.floating:
        """
        Validate the model using the validation dataset.

        Args:
            training_data_handler (TrainingData): an object for managing the training data.

        Returns:
            float: The FID score on the validation dataset.
        """
        real_validation_data = training_data_handler.get_validation_data()[0]
        return self.fid_score(real_validation_data)

    def generate(self, num_samples: int = 1) -> np.ndarray:
        """
        generate synthetic data, should only be executed after training.

        Args:
            num_samples (int, optional): number of samples to generate.
                                         Defaults to 1.

        Returns:
            np.ndarray: Generated samples., shape (num_samples, data_dim), dtype float
        """
        z = self.noise_sampler.sample(num_samples)
        return self.generator.generate(z)
