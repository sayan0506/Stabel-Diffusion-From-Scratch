import torch
import numpy as np


class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps = 1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        # all the naming conventions and parameters refer to DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)
        # the parameter for the transition kernel
        # we are using linear scheduler, thus we are using a linear space
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype = torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        # for calculation simplicity using gp series we do calculate the alpha_cumprod over all the possible timesteps
        self.alphas_cumprod = torch.cumprod(self.alphas, dim = 0)
        self.one = torch.tensor(1.0)

        self.generator = generator

        self.num_train_timesteps = num_training_steps
        # this creates the array of a timestep for each of the training steps
        # np.arange(0, num_training_steps)[::-1].copy(), here [::-1] reverse the array
        # as we are dealing with the timesteps correponding to reverse diffusion
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())


    def set_inference_timesteps(self, num_inference_steps = 50):
        self.num_inference_steps = num_inference_steps
        # 999, 998, ... 0 -> 1000 steps
        # 999, 999 - 20, 999 - 40, ... 0 -> 50 steps where 20 = step ratio(1000//50)
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        # we multiply using step ratio bcz
        # np.arange() return 0, 1, ... 49
        # but the reverse of the above list multiplied by step_ratio will return 999, 999 -20, 999 - 40 ... 0
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        # obtain torch tensor
        self.timesteps = torch.from_numpy(timesteps)


    def _get_previous_timestep(self, timestep:int) -> int:
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t
    

    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        # if prev_t is 0 then we take timestep = 1
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >=0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev 

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        variance = torch.clamp(variance, min=1e-20)

        return variance


    def set_strength(self, strength = 1):
        """
           Set how much noise to add to the input image
           More noise (strength ~ 1) means output will be further from input image 
           Less noise (strength ~ 0) means output will be closer to input image
        """
        # start_step is the number of noise levels to skip
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # 1. Compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >=0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        # previous values of alpha and beta prod used to evaluate current value or alpha and beta
        current_beta_t = 1 - current_alpha_t 

        # 2. Compute predicted original sample from predicted noise
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / (alpha_prod_t ** 0.5)


        # 3. Compute coefficients of pred_original_sample_x0 and current sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = (current_alpha_t ** 0.5 * beta_prod_t_prev/beta_prod_t)

        # 4. Compute predicted previous sample mu_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # 5. Add noise
        variance = 0
        # if t=0, x = mu

        if t>0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device = device, dtype = model_output.dtype)
            # variance computed previously
            # x = mu + sigma * noise
            variance = (self._get_variance(t) ** 0.5) * noise 

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
        ) -> torch.FloatTensor:

        # make sure both original samples and alpha_cumprod saem device and dtype, else we cant aplly noise
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype = original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        # obtain the geometric product of alphas from 0 to T, which means we need cummulative product at all timesteps
        # implies, 0-1, 0-2, 0-3, ... 0-T
        # for the mean term
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # for the variance term in the transition kernel
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # now add noise, also sample noise from the distribution q(x_t|x_0)
        # the sample obtained from N(mu, sigma) obtained using x = mu + sigma*N(0,1)
        # mu = sqrt_alpha_prod * x_0
        # sample noise of the shape of shape of original image
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

        return noisy_samples



        



    






    