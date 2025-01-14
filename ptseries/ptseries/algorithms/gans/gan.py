import time
from math import ceil

import torch

from ptseries.algorithms.gans.utils import infiniteloop


class WGANGP:
    """General class implementing a Wasserstein GAN with Gradient Penalty (WGAN-GP).

    See https://arxiv.org/pdf/1704.00028.pdf for more details.

    Agnostic regarding the dimension of the data, as long as it is consistent with the provided model.

    Args:
        latent (nn.Module): Class instance that has a `generate` method to produce samples from the
            latent space, such as a PTGenerator.
        generator (nn.Module): PyTorch model used to describe the architecture of the
            (quantum/classical/hybrid) generator.
        critic (nn.Module): PyTorch model used to describe the architecture of the
            (quantum/classical/hybrid) critic.
    """

    def __init__(self, latent, generator, critic):
        super(WGANGP, self).__init__()

        self.latent = latent
        self.generator = generator
        self.critic = critic

        self.gen_data = []
        self.time_50 = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.generator = self.generator.to(self.device)
        self.critic = self.critic.to(self.device)

        self.opt_gen = torch.optim.Adam(self.generator.parameters(), betas=[0.0, 0.9])
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), betas=[0.0, 0.9])

    def train(
        self,
        train_dataloader,
        learning_rate=5e-4,
        n_iter=1,
        print_frequency=100,
        penalty=10,
        critic_iter=5,
        verbose=True,
        logger=None,
        save_frequency=None,
        learning_rate_decay=False,
        latent_update_frequency=None,
        latent_learning_rate=1e-2,
    ):
        """Trains the generator and the critic by implementing a WGAN-GP.

        Args:
            train_dataloader (torch.utils.data.DataLoader): PyTorch dataloader containing the training data
            learning_rate (float, optional): learning rate
            n_iter (int, optional): number of iterations to train for
            print_frequency (int, optional): how often to print the loss, measured in batches
            penalty (float, optional): gradient penalty hyperparameter
            critic_iter (int, optional): number of critic gradient descent steps per generator steps
            verbose (bool, optional): whether to print information
            logger (Logger, optional): Logger instance to which training data and metadata is saved
            save_frequency (int, optional): frequency at which we save the models
            learning_rate_decay (bool): if true, implement a linear decay of the learning rate from learning_rate to 0.
            latent_update_frequency (int, optional): frequency at which the latent space is trained
            latent_learning_rate (float, optional): learning rate for the latent space
        """

        if logger is not None:
            metadata = {
                "learning_rate": learning_rate,
                "n_iter": n_iter,
                "gradient_penaly": penalty,
                "critic_iter": critic_iter,
                "device": str(self.device),
                "learning_rate_decay": learning_rate_decay,
                "latent_update_frequency": latent_update_frequency,
            }
            logger.register_metadata(metadata)

        for param in self.opt_gen.param_groups:
            param["lr"] = learning_rate
        for param in self.opt_critic.param_groups:
            param["lr"] = learning_rate

        if learning_rate_decay:
            self.scheduler_gen = torch.optim.lr_scheduler.LambdaLR(self.opt_gen, lambda step: 1 - step / n_iter)
            self.scheduler_critic = torch.optim.lr_scheduler.LambdaLR(self.opt_critic, lambda step: 1 - step / n_iter)

        if latent_update_frequency is not None:
            self.opt_latent = torch.optim.Adam(self.latent.parameters(), lr=latent_learning_rate, betas=[0.0, 0.9])

            self.latent_prev = None
            self.losses_prev = None

        print("Starting training...")

        looper = infiniteloop(train_dataloader)

        batch_size = next(looper).to(self.device).shape[0]

        for step in range(n_iter):
            if verbose:
                self._estimate_running_time(step, n_iter)

            latent = self.latent.generate((1 + critic_iter) * batch_size).to(self.device)
            latent = torch.chunk(latent, 1 + critic_iter, dim=0)

            for i in range(critic_iter):
                # Train the critic
                data_real = next(looper).to(self.device)

                with torch.no_grad():
                    data_fake = self.generator(latent[i]).detach()

                loss_critic_real = self.critic(data_real)
                loss_critic_fake = self.critic(data_fake)
                penalty_value = self._gradient_penalty(data_real, data_fake)
                loss_critic = loss_critic_fake.mean() - loss_critic_real.mean() + penalty * penalty_value

                if latent_update_frequency is not None:
                    self._accumulate_latents_and_losses(-loss_critic_fake, latent[i])

                self.critic.zero_grad()
                loss_critic.backward()
                self.opt_critic.step()

            # Train the generator
            for p in self.critic.parameters():
                p.requires_grad_(False)

            data_fake = self.generator(latent[critic_iter])
            losses_gen = -self.critic(data_fake)
            loss_gen = losses_gen.mean()

            self.generator.zero_grad()
            loss_gen.backward()
            self.opt_gen.step()

            if latent_update_frequency is not None:
                self._accumulate_latents_and_losses(losses_gen, latent[critic_iter])

            # train the latent space
            self._train_latent(step, latent_update_frequency)

            for p in self.critic.parameters():
                p.requires_grad_(True)

            if logger is not None:
                logger.log("loss_critic_real", step, loss_critic_real.mean().item())
                logger.log("loss_critic_fake", step, loss_critic_fake.mean().item())
                logger.log("loss_gen_mean", step, loss_gen.mean().item())

                if latent_update_frequency is not None:
                    logger.log("thetas", step, self.latent.theta_trainable.tolist())

            if verbose and step % print_frequency == print_frequency - 1:
                print(f"Batch {step + 1}: ", end="")
                print(f"Wasserstein loss is {loss_critic_real.mean() - loss_critic_fake.mean():.2f}, ", end="")
                print(f"the generator loss is {loss_gen.mean():.2f}", end="")
                if learning_rate_decay:
                    print(f", the learning rate is {self.opt_gen.param_groups[0]['lr']:2f}")
                else:
                    print("")

            if save_frequency is not None and step % save_frequency == 0 and step > 0 and logger.log_dir is not None:
                save_file_critic_i = logger.log_folder + "/critic" + "_" + str(step) + ".pt"
                save_file_gen_i = logger.log_folder + "/gen" + "_" + str(step) + ".pt"
                torch.save(self.critic.state_dict(), save_file_critic_i)
                torch.save(self.generator.state_dict(), save_file_gen_i)

            if learning_rate_decay:
                self.scheduler_critic.step()
                self.scheduler_gen.step()

        if logger is not None:
            if logger.log_dir is not None:
                logger.save()

            # Save the last model
            save_file_critic_lastepoch = logger.log_folder + "/critic_final.pt"
            save_file_gen_lastepoch = logger.log_folder + "/gen_final.pt"
            torch.save(self.critic.state_dict(), save_file_critic_lastepoch)
            torch.save(self.generator.state_dict(), save_file_gen_lastepoch)

    def _gradient_penalty(self, real, fake):
        """Calculates the gradient penalty in a WGAN-GP"""
        batch_size = real.size(0)

        # Sample epsilon from the uniform distribution, with shape matching the data
        eps = torch.rand(batch_size, *tuple([1] * (len(real.shape) - 1)), device=self.device)
        eps = eps.expand_as(real)

        # Interpolation between real data and fake data.
        interpolation = eps * real + (1 - eps) * fake.requires_grad_()

        mixed_scores = self.critic(interpolation)
        grad_outputs = torch.ones_like(mixed_scores, device=self.device)

        # Compute Gradients
        gradients = torch.autograd.grad(
            outputs=mixed_scores, inputs=interpolation, grad_outputs=grad_outputs, create_graph=True
        )[0]

        # Compute and return gradient norm
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)

    def _accumulate_latents_and_losses(self, losses, latent):
        if self.latent_prev is None:
            self.latent_prev = latent
            self.losses_prev = losses
        else:
            self.latent_prev = torch.cat((self.latent_prev, latent), dim=0)
            self.losses_prev = torch.cat((self.losses_prev, losses), dim=0)

    def _reset_latents_and_losses(self):
        self.latent_prev = None
        self.losses_prev = None

    def _train_latent(self, step, latent_update_frequency):
        if latent_update_frequency is not None and step % latent_update_frequency == 0:
            self.opt_latent.zero_grad()
            self.latent.backward(self.latent_prev, self.losses_prev)
            self.opt_latent.step()

            self._reset_latents_and_losses()

    def generate_data(self, num=25, batch_size=None):
        """Generates num fake data from the generator

        Args:
            num (int, optional): total number of data elements to generate
            batch_size (int, optional): batch size for generating the data
        """

        self.generator.eval()

        with torch.no_grad():
            if batch_size is None:
                batch_size = num

            data_fake = []
            for _ in range(int(num / batch_size)):
                latent = self.latent.generate(batch_size).to(self.device)
                data = self.generator(latent).detach()
                data_fake.append(data.detach())

            data_fake = torch.cat(data_fake, dim=0).cpu().numpy()

        self.generator.train()

        return data_fake

    def _check_latent(self):
        with torch.no_grad():
            try:
                check = self.latent.generate(10)  # test with a batch size of 10
            except:
                raise Exception("Error while running latent")

    def _estimate_running_time(self, batch_number, n_iter):
        """Estimates and prints the training time by extrapolating from the first 150 batches"""
        if batch_number == 50:
            self.time_50 = time.time()
        elif batch_number == 150:
            time_150 = time.time()
            time_single_batch = (time_150 - self.time_50) / 100
            tot_time_min = n_iter * time_single_batch / 60
            msg = f"Estimated total training time: {ceil(tot_time_min)} minutes"
            print(msg)
