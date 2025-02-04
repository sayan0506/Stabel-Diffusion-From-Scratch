import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

'''
Design VAE Encoder
'''
class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            
            ## we will define the UNET encoder here
            # it starts with a conv2d layer
            # we use 128 kernels of kernel size (3,3)
            # we use passing  1
            # we are passing folowing argument to super class
            
            # (batch, channel, height, width) -> (batch, 128, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # now starts the VAE residual block

            # (batch, 128, height, width) -> (batch, 128, height, width)
            VAE_ResidualBlock(128,128),

            # (batch, 128, height, width) -> (batch, 256, height, width)
            VAE_ResidualBlock(128,128),

            # this is the downscale block
            # (batch, 128, height, width) -> (batch, 128, height/2, width/2)
            nn.Conv2d(128,128, kernel_size=3, stride=2, padding=0),

            # as dimension is reduced the kernel numbers get increases
            # (batch, 128, height/2, width/2) -> (batch, 256, height/2, width/2)
            VAE_ResidualBlock(128, 256),

            # (batch, 128, height/2, width/2) -> (batch, 256, height/2, width/2)
            VAE_ResidualBlock(256, 256),

            # downscale block
            # (batch, 256, height/2, width/2) -> (batch, 256, height/4, width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (batch, 128, height/4, width/4) -> (batch, 512, height/4, width/4)
            VAE_ResidualBlock(256, 512),

            # (batch, 512, height/4, width/4) -> (batch, 512, height/4, width/4)
            VAE_ResidualBlock(512, 512),

            # down block
            #(batch, 512, height/4, width/4) -> (batch, 512, height/8, width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 

            # mid block
            #(batch, 512, height/8, width/8) -> (batch, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            #(batch, 512, height/8, width/8) -> (batch, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            #(batch, 512, height/8, width/8) -> (batch, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            # attention block
            #(batch, 512, height/8, width/8) -> (batch, 512, height/8, width/8)
            VAE_AttentionBlock(512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 

            #(batch, 512, height/8, width/8) -> (batch, 512, height/8, width/8)
            # the number of groups it choose for group normalization is 32
            nn.GroupNorm(32,512),
            
            #(batch, 512, height/8, width/8) -> (batch, 512, height/8, width/8)
            nn.SiLU(),

            # padding=1 is like padding "same", which compensated the reduction in size
            # reduces the number of features, which is the bottleneck or latent space of the encoder
            #(batch, 512, height/8, width/8) -> (batch, 8, height/8, width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # it does channelwise aggregation
            #(batch, 8, height/8, width/8) -> (batch, 8, height/8, width/8)
            nn.Conv2d(8,8, kernel_size=1, padding=0)
        
        )

    def forward(self, x:torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (batchs_size, channel_size, height, width)
        # noise: (batch_size, out_channels, height, width)

        # we wrote or passed list of arguments
        # or layer details inside the sequential block
        # so use for loop to pass the images through the encoder layers
        for module in self:
            # whenever there is stride = 2, we do assymmetric padding
            # no exact reason but found working while experimenting
            if getattr(module, 'stride', None) == (2,2):
                # padding right, bottom only     
                x = F.pad(x, (0,1,0,1))

            x = module(x)

        # obtain the mean and log variance 
        # obtain 2 chunks(mean, log_variance), split across dim=1 i.e. channel dimension
        # (batch, 8, height/8, width/8) -> 2 * (batch, 4, height/8, width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # clamp log variance within a certain range to maintain numerical stability
        log_variance = torch.clamp(log_variance, -30, 20) 

        # do exponent of log to obtain the value of variance
        variance = log_variance.exp()

        # now we obtain sigma
        stddev = variance.sqrt()

        # now we reparameterize using the noise
        # z = mean + stddev * noise
        x = mean + stddev * noise

        # scale the output by a constant
        x *= 0.18215

        return x
                

        