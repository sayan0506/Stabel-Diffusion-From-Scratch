import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import SelfAttention

# VAE attention block
class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # as we know the attention block consusts of norm
        # followed by self attention
        self.groupnorm = nn.GroupNorm(32, channels)
        # we are considering single attention head
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        # x: (batch_size, channels/features, height, width)
        residue = x 

        # (batch_size, features, height, width) -> (batch_size, features, height*width)
        x = self.groupnorm(x)

        n, c, h, w = x.shape

        # flatten the feature maps
        x = x.view((n,c,h*w))

        # (batch_size, features, height*width) -> (batch_size, height*width, features)
        # as the required shape for attention (batch_size, seq_len, emb_dim)
        # in our case (batch_size, pixel_list_len, channels)
        # channels hold the feature extracted for each pixel
        x = x.transpose(-1, -2)

        # apply self attention to the feature maps without causal mask
        x = self.attention(x)

        # reverse process
        # (batch_size, height*width, features) -> (batch_size, features, height*width)
        x = x.transpose(-1,-2)

        # (batch_size, features, height*width) -> (batch_size, features, height, width)
        x = x.view((n,c,h,w))

        # residual
        x += residue

        return x


class VAE_ResidualBlock(nn.Module):
    # here the object is not sequential, the object is a nn module
    # we are defining the residual block module
    # which will be used by sequential block
    # where at constructor we passed the arguments to sequencial block super class
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Across the residual block the resolution of the feature map remains constant

        # our ssequence is 
        # residual in -> norm -> SiLU -> conv -> norm -> conv -> residual out
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        # here the channels are changed

        # here channels are changed and for second one the size remains same but we do feature extraction on the modified feature map
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        # no change in channels input and output channels are same but size reduces
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            # if we are adding both feature map during residual out
            # both channels are same,
            # we are taking residual layer as identity
            # as we are going to addd that feature map for the current residual block
            self.residual_layer = nn.Identity()
        else:
            # is both channels are not same, then we are skipping that layer
            # then we are adding conv to the residual branch, we are using 1*1 conv layer
            # where no paddding needed
            # both the channels are not same then we are not using residual connection for that block
            # 1*1 is channelwise aggregation
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x) -> torch.Tensor:
        # x: (Batch, in_channels, height, width)

        # variable for residue layer
        residue = x.copy()

        # first norm layer
        # (batch_size, in_channels, h, w) -> (batch_size, in_channels, h, w)
        x = self.groupnorm_1(x)
        
        # silu activation
        x = F.silu(x)

        # first conv layer
        # (batch_size, in_channels, h, w) -> (batch_size, out_channels, h, w)
        # channel changed to out_channels
        x = self.conv_1(x)

        # second norm layer
        # (batch_size, out_channels, h, w) -> (batch_size, out_channels, h, w)
        x = self.groupnorm_2(x)

        # silu activation
        # (batch_size, out_channels, h, w) -> (batch_size, out_channels, h, w)
        x = F.silu(x)

        # second conv 1*1
        # (batch_size, out_channels, h, w) -> (batch_size, out_channels, h, w)
        x = self.conv_2(x)

        # now add the residual output
        x = x + self.residual_layer(residue)

        return x


# define VAE decoder
class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (batch_size, 4, h/8, w/8) -> (batch_size, 4, h/8, w/8)
            # padding same no resolution change
            nn.Conv2d(4,4, kernel_size=3, padding=1),

            # (batch_size, 4, h/8, w/8) -> (batch_size, 512, h/8, w/8)
            # increase channel
            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            # residual block
            # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/8, w/8)
            VAE_ResidualBlock(512, 512),

            # VAE attention block
            # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/8, w/8)
            VAE_AttentionBlock(512),

            # (batch_size, 512, h/8, w/8) -> (batch_size, 256, h/8, w/8)
            # residual block
            VAE_ResidualBlock(512,512),

            # (batch_size, 256, h/8, w/8) -> (batch_size, 256, h/8, w/8)
            # residual block
            # same as encoder branch at same level
            VAE_ResidualBlock(512,512),

            # in encoder at end 3 consequetive residual used
            # (batch_size, 256, h/8, w/8) -> (batch_size, 256, h/8, w/8)
            VAE_ResidualBlock(512,512),

            # in encoder at end 3 consequetive residual used
            # (batch_size, 256, h/8, w/8) -> (batch_size, 256, h/8, w/8)
            VAE_ResidualBlock(512,512),

            # upsample
            # repeats the rows and columns of the data by scale_factor(like resize the image by doubling its size)
            # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/4, w/4)
            nn.Upsample(scale_factor=2),

            # (batch_size, 512, h/4, w/4) -> (batch_size, 512, h/4, w/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # (batch_size, 512, h/4, w/4) -> (batch_size, 512, h/4, w/4)
            VAE_ResidualBlock(512,512),

            # (batch_size, 512, h/4, w/4) -> (batch_size, 512, h/4, w/4)
            VAE_ResidualBlock(512,512),

            # (batch_size, 512, h/4, w/4) -> (batch_size, 512, h/4, w/4)
            VAE_ResidualBlock(512,512),

            # 2nd upscale
            # (batch_size, 512, h/4, w/4) -> (batch_size, 512, h/2, w/2
            nn.Upsample(scale_factor=2),

            # (batch_size, 512, h/2, w/2) -> (batch_size, 512, h/2, w/2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # (batch_size, 512, h/2, w/2) -> (batch_size, 256, h/2, w/2)
            VAE_ResidualBlock(512,256),

            # (batch_size, 256, h/2, w/2) -> (batch_size, 256, h/2, w/2)
            VAE_ResidualBlock(256,256),

            # (batch_size, 256, h/2, w/2) -> (batch_size, 256, h/2, w/2)
            VAE_ResidualBlock(256,256),

            # 3rd upscale
            # (batch_size, 256, h/2, w/2) -> (batch_size, 256, h, w)
            nn.Upsample(scale_factor=2),

            # (batch_size, 512, h, w) -> (batch_size, 512, h, w)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            # (batch_size, 256, h, w) -> (batch_size, 128, h, w)
            VAE_ResidualBlock(256,128),

            # (batch_size, 128, h, w) -> (batch_size, 128, h, w)
            VAE_ResidualBlock(128,128),

            # (batch_size, 128, h, w) -> (batch_size, 128, h, w)    
            VAE_ResidualBlock(128,128),

            # (batch_size, 128, h, w) -> (batch_size, 128, h, w)
            nn.GroupNorm(32,128),

            # (batch_size, 128, h, w) -> (batch_size, 128, h, w)
            nn.SiLU(),

            # (batch_size, 128, h, w) -> (batch_size, 3, h, w)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),            
        )


    def forward(self, x):
        # x: (Batch_size, 4, height/8, width/8)

        # remove the scaling addded by the encoder
        x /= 0.18215

        for module in self:
            x = module(x)

        # (Batch_size, 3, h, w)
        return x







