# ---------------------------------
# Put this WGANTrainer class (e.g., in gan_trainer.py)
# ---------------------------------
from __future__ import annotations

from typing import Callable, override

import numpy as np

from ..general_model.optimizer import GradientOptimizer
from ..general_model.parameterized_model import ParameterizedModel
from ..general_model.validation import TrainingData, ValidatableTrainingModel
from ..utils.helpers import fid
from .gan_components import Critic  # the class defined above
from .gan_components import Generator
from .noise_generator import NoiseSampler


class WGANTrainer(ValidatableTrainingModel[tuple[float, float]]):
    """
    A trainer class for Wasserstein GAN (WGAN) with parameter-space 1-Lipschitz enforcement.

    Attributes:
        generator (Generator): The generator model.
        critic (Critic): The critic model (no logistic; outputs real scores).
        noise_sampler (NoiseSampler): The noise sampler for generating latent vectors.
        gen_optimizer (GradientOptimizer): Optimizer for the generator.
        critic_optimizer (GradientOptimizer): Optimizer for the critic.
    """

    @staticmethod
    def critic_loss_given_scores(
        scores_real: np.ndarray,
        scores_fake: np.ndarray
    ) -> float:
        """
        Compute the critic loss to minimize:
        L_Critic = mean(C_fake) - mean(C_real)

        Args:
            scores_real (np.ndarray): Critic scores for real data, shape (batch_size, 1).
            scores_fake (np.ndarray): Critic scores for fake data, shape (batch_size, 1).

        Returns:
            float: The critic loss value.
        """
        return float(np.mean(scores_fake) - np.mean(scores_real))

    @staticmethod
    def critic_grads_given_scores(
        scores_real: np.ndarray,
        scores_fake: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradients of the critic loss with respect to the scores.
        dL_Critic/dC_real = -1/m
        dL_Critic/dC_fake = +1/m

        Args:
            scores_real (np.ndarray): Critic scores for real data, shape (batch_size, 1).
            scores_fake (np.ndarray): Critic scores for fake data, shape (batch_size, 1).

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing;
                - Gradient w.r.t. real scores, shape (batch_size, 1).
                - Gradient w.r.t. fake scores, shape (batch_size, 1).
        """
        m_r = scores_real.shape[0]
        m_f = scores_fake.shape[0]
        grad_real = -np.ones_like(scores_real) / m_r
        grad_fake = +np.ones_like(scores_fake) / m_f
        return grad_real, grad_fake

    @staticmethod
    def generator_loss_given_scores(
        scores_fake: np.ndarray
    ) -> float:
        """
        Compute the generator loss to minimize:
        L_Generator = -mean(C_fake)

        Args:
            scores_fake (np.ndarray): Critic scores for fake data, shape (batch_size, 1).

        Returns:
            float: The generator loss value.
        """
        return float(-np.mean(scores_fake))

    @staticmethod
    def generator_grads_given_scores(
        scores_fake: np.ndarray
    ) -> np.ndarray:
        """
        Compute the gradient of the generator loss with respect to the fake scores.
        dL_G/dC_fake = -1/m

        Args:
            scores_fake (np.ndarray): Critic scores for fake data, shape (batch_size, 1).

        Returns:
            np.ndarray: Gradient w.r.t. fake scores, shape (batch_size, 1).
        """
        m_f = scores_fake.shape[0]
        return -np.ones_like(scores_fake) / m_f

    def __init__(self,
                 gen_model: ParameterizedModel,
                 critic_model: ParameterizedModel,
                 noise_sampler: NoiseSampler,
                 gen_optimizer: GradientOptimizer,
                 critic_optimizer: GradientOptimizer
                 ) -> None:
        self.generator = Generator(gen_model)
        self.critic = Critic(critic_model)
        self.noise_sampler = noise_sampler
        self.gen_optimizer = gen_optimizer
        self.critic_optimizer = critic_optimizer

    @override
    def get_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.critic.model.get_parameter(),
                self.generator.model.get_parameter())

    @override
    def set_parameters(self, params: tuple[np.ndarray, np.ndarray]) -> None:
        self.critic.model.set_parameter(params[0])
        self.generator.model.set_parameter(params[1])

    def fid_score(self, real_data: np.ndarray,
                  feature_extractor: Callable[[
                      np.ndarray], np.ndarray] | None = None
                  ) -> np.floating:
        """
        Compute the Fréchet Inception Distance (FID) between real and generated data.
        """
        num_samples = real_data.shape[0]
        generated_data = self.generate(num_samples)
        real_features = feature_extractor(
            real_data) if feature_extractor else real_data
        gen_features = feature_extractor(
            generated_data) if feature_extractor else generated_data
        mu_real = np.mean(real_features, axis=0)
        mu_gen = np.mean(gen_features, axis=0)
        cov_real = np.cov(real_features, rowvar=False)
        cov_gen = np.cov(gen_features, rowvar=False)
        return fid(mu_real, mu_gen, cov_real, cov_gen)

    def train_epoch(
        self,
        training_data_handler: TrainingData,
        batch_size: int,
        *,
        gen_learning_rate: float = 0.0001,
        disc_learning_rate: float = 0.0001,
        d_steps_per_one_g_step: int = 5,
        **_
    ) -> tuple[float, float]:
        """
        Perform one epoch of adversarial training. Each epoch processes d_steps_per_one_g_step
        minibatches for the critic followed by one minibatch for the generator. The method
        returns the average losses for the critic and generator over the epoch.

        Args:
            training_data_handler (TrainingData): an object for managing the training data
            batch_size (int): Minibatch size for real data and latent noise
            gen_learning_rate (float): learning rate for the generator's optimizer
                                       Defaults to 0.0001
            disc_learning_rate (float): learning rate for the critic's optimizer
                                        Defaults to 0.0001
            d_steps_per_one_g_step (int, optional): Number of critic steps per generator step
                                                    Defaults to 5.

        Returns:
            tuple[float, float]: A tuple containing:
                - avg_c_loss (float): The average critic loss for the epoch.
                - avg_g_loss (float): the average generator loss for the epoch.
        """
        critic_losses: list[float] = []
        gen_losses: list[float] = []

        shuffled_batches = training_data_handler.shuffle_and_divide(batch_size)

        for (x_real,) in shuffled_batches:
            b = x_real.shape[0]

            for _ in range(d_steps_per_one_g_step):
                z_d = self.noise_sampler.sample(b)
                x_fake_d = self.generator.generate(z_d)

                scores_real = self.critic.score(x_real)
                scores_fake = self.critic.score(x_fake_d)

                c_loss = WGANTrainer.critic_loss_given_scores(
                    scores_real, scores_fake)
                critic_losses.append(c_loss)

                crit_grad_real, crit_grad_fake = WGANTrainer.critic_grads_given_scores(
                    scores_real, scores_fake)

                grads_phi_real = self.critic.model.loss_gradient_by_param(
                    x_real, crit_grad_real)
                grads_phi_fake = self.critic.model.loss_gradient_by_param(
                    x_fake_d, crit_grad_fake)
                total_grads_phi = grads_phi_real + grads_phi_fake

                self.critic_optimizer.stepwise_update(disc_learning_rate,
                    self.critic.model, total_grads_phi)

                self.critic.model.lipschitz_normalize()

            z_g = self.noise_sampler.sample(b)
            x_fake_g = self.generator.generate(z_g)

            scores_fake_g = self.critic.score(x_fake_g)

            g_loss = WGANTrainer.generator_loss_given_scores(scores_fake_g)
            gen_losses.append(g_loss)

            gen_grad_fake = WGANTrainer.generator_grads_given_scores(
                scores_fake_g)

            dloss_dx_fake = self.critic.model.loss_gradient_by_input(
                x_fake_g, gen_grad_fake)
            grad_theta = self.generator.model.loss_gradient_by_param(
                z_g, dloss_dx_fake)

            self.gen_optimizer.stepwise_update(gen_learning_rate,self.generator.model, grad_theta)

        avg_c_loss = float(np.mean(critic_losses)) if critic_losses else np.nan
        avg_g_loss = float(np.mean(gen_losses)) if gen_losses else np.nan

        return avg_c_loss, avg_g_loss


    @override
    def validate(self, training_data_handler: TrainingData) -> np.floating:
        real_validation_data = training_data_handler.get_validation_data()[0]
        return self.fid_score(real_validation_data)

    def generate(self, num_samples: int = 1) -> np.ndarray:
        """
        Generate synthetic samples via the generator.
        
        Args:
            num_samples (int): Number of samples to generate. Defaults to 1.

        Returns:
            np.ndarray: Generated samples of shape (num_samples, data_dim).
        """
        z = self.noise_sampler.sample(num_samples)
        return self.generator.generate(z)
