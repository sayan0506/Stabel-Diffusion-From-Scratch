# -----------------------------------------------------------
# File: pipeline.py
# Author: Sayan Hazra
# Description: This file contains the pipeline for generating images from the diffusion model.
# -----------------------------------------------------------
# The code is adapted from the SD implementation of umar jamil
# https://github.com/umarjamil/stable-diffusion
# -----------------------------------------------------------

"""
This file contains the pipeline for generating images from the diffusion model.
"""

import torch
import numpy as np
import tqdm as tqdm
from ddpm import DDPMSampler

# width of image corresponding to the original distribution
# (512, 512, 3)
HEIGHT = 512
WIDTH = 512
LATENTS_WIDTH = 512//8
LATENTS_HEIGHT = 512//8

def generate(
        prompt, # condinational prompt
        uncond_prompt=None, # unconditional prompt or negative prompting, usually refers the context which the model should avoid while generating
        input_image = None, # for image conditioning
        strength = 0.8, # strength for noising input image which controls the model creativity or in general we often see it as temparature 
        do_cfg = True, # whether to do classifier free guidance or not
        cfg_scale = 7.5, # how much to follow the conditional prompt, weightage while cfg
        sampler_name = "ddpm", # name of the sampler
        n_inference_steps = 50, #number of denoising steps or numer of steps during reverse diffusion
        models = {}, # passes the model dictonary
        seed = None, # to control the randomness
        device = None, # device to run the model on
        idle_device = None, # the device which is idle, device to move when the work is done
        tokenizer = None, # tokenize the given conditional text
    ):

    # with torch.no_grad() it will not calculate the gradients, during the model operations
    # if its not mentioned it generally calculats the gradients and stores inside the memory 
    with torch.no_grad():
        if not 0 < strength <=1:
            # raise value error raises the error of that value category
            raise ValueError("strength must be between 0 and 1")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle =  lambda x: x

        # initialize the random number generator according to the seed
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed() # initializes seed with a random number
        else:
            generator.manual_seed(seed) # initializes seed with the given number

        '''
        TEXT CONDITIONING
        '''
        # get the clip model
        clip = models["clip"]
        # load the model to the device
        clip.to(device)

        # get the context for the diffusion model
        # using the prompt and clip trained text encoder
        if do_cfg:
            # convert the text into tokens
            # max length of our sequence is 77, if the seq length<77, it does padding with dummy tokens
            # returns the input_ids or ids of the tokens
            # generate tokens for batch of data
            # it basically returns the dictionary id
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding = "max_length", max_length = 77
            ).input_ids # input_ids are the cond_tokens
            # (batch_size, seq_lengths)
            # converts the list of tokens to tensors having each element converted to dtype torch.long
            # then moves the tensor to the device
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device= device)
            # returns the token embeddings
            # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
            cond_context = clip(cond_tokens)
            # does the above thing for uncond_tokens
            # (batch_size, seq_len)
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding = "max_length", max_length = 77
            )
            # create tensor of same size
            uncond_tokens = torch.tensor(uncond_tokens, dtype = torch.long, device = device)
            # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
            uncond_context = clip(uncond_tokens)
            # (batch_size, seq_len, embedding_dim) + (batch_size, seq_len, embedding_dim) -> 2 * (batch_size, seq_len, embedding_dim)
            context = torch.concat([cond_context, uncond_context])
        
        else:
            # converts a list of input ids for a sequence of 77 tokens
            # batch list of tokens each elemnet in the list is also a list of 77 tokens
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_padding", max_length  = 77
            )
            # creates a tensor
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
            context = clip(tokens)

        # after the embeddings are obtained we need to detach the model and store it to idle device
        to_idle(clip)

        if sampler_name == "ddpm":
            # used to sample
            sampler = DDPMSampler(generator)
            # it takes inference timesteps
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # imga econditioning
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # (HEIGHT, WIDTH, CHANNEL)
            input_image_tensor = np.array(input_image_tensor)
            # obtain torch tensor
            input_image_tensor = torch.tensor(input_image_tensor, dtype = torch.float32, device= device)
            # rescale the image pixels
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # add batch dimension
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (batch_size, height, width, channel) -> (batch_size, channel, height, width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # (batch_size, 4, latents_height, latents_width)
            # this noise is used for the reparameterization trick
            encoder_noise = torch.randn(latents_shape, generator = generator, device = device)
            # (batch_size, 4, latents_height, latents_width)
            latents = encoder(input_image_tensor, encoder_noise)

            # add noise to latents for the generation so that DDPM unet shows creativeity while reverse diffusion
            # based on strenggth it skips the noise step
            sampler.set_strength(strength = strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)

        else:
            # if image conditioning is not none
            # generator is the random number generator of torch module
            # having a seed 
            # generate an image of latents shape completely noisy image each pixel sampled from gaussian distribution
            latents = torch.randn(latents_shape, generator = generator, device = device)


        diffusion = models["diffusion"]
        diffusion.to(device)
        
        # using the sampler, we sample the timesteps for all the reverse diffusion steps
        timesteps = tqdm(sampler.timesteps)
        
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (batch_size, 4, latents_height, latents_width)
            model_input = latents

            if do_cfg:
                # increases the batch size to 2
                model_input = model_input.repeat(2,1,1,1)

            # model output is predicted noise
            # (batch_size, 4, latnts_height, latents_width) -> (batch_size, 4, latents_height, latents_width)
            # context = torch.concat([cond_context, uncond_context])
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                # as we passed two context one for conditioned
                # another for negative prompting
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # (batch_size, 4, latents_height, latents_width) -> (batch_size, 4, latents_height, latents_width)
            # from predicted noise it obtains the latents
            # using the sampler
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)
        
        # (batch_size, 4, latents_height, latents_width) -> (batch_size, 3, height, width)
        decoder = models["decoder"]
        decoder.to(device)
        # output images
        # (batch_size, 4, latents_height, latents_width) -> (batch_size, 3, height, width)
        images = decoder(latents)
        to_idle(decoder)

        # this restricts or clamps the values of the image pixels within a limit
        images = rescale(images, (-1,1), (0, 255), clamp = True) 
        # (batch_size, 3, height, width) -> (batch_size, height, width, 3)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)
    
    return x


# for time embedding it uses the formula for positional encoding
def get_time_embedding(timestep):
    # shape: (160,)
    # dimension of timesteps = 160
    # pos is the timestep position, i is the ith dimension
    # pos / 10000^(2i / 160)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # shape: (1, 160)
    x = torch.tensor([timestep], dtype = torch.float32)[:, None] * freqs[None]
    # shape: (1, 320)
    # for both sin and cost concat
    # across the time_emb_dim concat for all timesteps
    return torch.cat([torch.cos(x), torch.sin(x)], dim = -1)
 












