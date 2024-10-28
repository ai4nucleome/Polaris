import torch
import torch.nn as nn
from operator import itemgetter

from typing import Type, Callable, Tuple, Optional, Set, List, Union
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath
from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv

# helper functions

def exists(val):
    return val is not None

def map_el_ind(arr, ind):
    '''
    To extract the sorted list and the corresponding original index respectively.
    '''
    return list(map(itemgetter(ind), arr))

def sort_and_return_indices(arr):
    '''
    Sorts the input list and returns the sorted list and its original index
    
    [0, 3, 2, 1] -> ([0, 1, 2, 3], [0, 3, 2, 1])
    
    '''
    # create indices list
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)

# calculates the permutation to bring the input tensor to something attend-able
# also calculates the inverse permutation to bring the tensor back to its original shape

def calculate_permutations(num_dimensions, emb_dim):
    '''
    
    To get the reshape index while calculating axial attention
    
    For [B, C, H, W] -> [[0, 3, 2, 1], [0, 2, 3, 1]]
    
    '''
    total_dimensions = num_dimensions + 2
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)
      
    return permutations

# helper classes

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    '''
    LayerNorm before axial attention calculation
    '''
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class PermuteToFrom(nn.Module):

    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

        # merge all but axial dimension
        axial = axial.reshape(-1, t, d)

        # attention
        axial = self.fn(axial, **kwargs)

        # restore to original shape and permutation
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial

# axial pos emb

class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape, emb_dim_index = 1):
        super().__init__()
        parameters = []
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]

        self.num_axials = len(shape)

        for i, (axial_dim, axial_dim_index) in enumerate(zip(shape, ax_dim_indexes)):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[axial_dim_index] = axial_dim
            parameter = nn.Parameter(torch.randn(*shape))
            setattr(self, f'param_{i}', parameter)

    def forward(self, x):
        # print("pos_embed")
        for i in range(self.num_axials):
            x = x + getattr(self, f'param_{i}')
        return x

# attention

class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads=None, drop=0):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads
        self.drop_rate = drop

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias = False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

        self.proj_drop = DropPath(drop)

    def forward(self, x, kv = None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        out = self.proj_drop(out)
        return out

# axial transformer block

class AxialTransformerBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 axial_pos_emb_shape, 
                 pos_embed,
                 heads = 8, 
                 dim_heads = None,
                 drop = 0.,
                 drop_path_rate=0.,
    ):
        super().__init__()
        dim_index = 1
        permutations = calculate_permutations(2, dim_index)
        # position embedding 
        self.pos_emb = AxialPositionalEmbedding(dim, axial_pos_emb_shape, dim_index) if pos_embed else nn.Identity()
        # Attention
        self.height_attn, self.width_attn = nn.ModuleList([PermuteToFrom(permutation, PreNorm(dim, SelfAttention(dim, heads, dim_heads, drop=drop))) for permutation in permutations])
        # FFN in attention
        self.FFN = nn.Sequential(
            ChanLayerNorm(dim),
            nn.Conv2d(dim, dim * 4, 3, padding = 1),
            nn.GELU(),
            DropPath(drop),
            nn.Conv2d(dim * 4, dim, 3, padding = 1),
            DropPath(drop),
            
            ChanLayerNorm(dim),
            nn.Conv2d(dim, dim * 4, 3, padding = 1),
            nn.GELU(),
            DropPath(drop),
            nn.Conv2d(dim * 4, dim, 3, padding = 1),
            DropPath(drop),
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
    def forward(self, x):
        x = self.pos_emb(x)
        x = x + self.drop_path(self.height_attn(x))
        x = x + self.drop_path(self.width_attn(x))
        x = x + self.drop_path(self.FFN(x))
        return x
    
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def _gelu_ignore_parameters(*args, **kwargs) -> nn.Module:
    """ Bad trick to ignore the inplace=True argument in the DepthwiseSeparableConv of Timm.
    Args:
        *args: Ignored.
        **kwargs: Ignored.
    Returns:
        activation (nn.Module): GELU activation function.
    """
    activation = nn.GELU()
    return activation

class DoubleConv(nn.Module):
    """ DoubleConv: MBConv block as described in: https://arxiv.org/pdf/2204.01697.pdf.

        Without downsampling:
        x ← x + Proj(SE(DWConv(Conv(Norm(x)))))

        With downsampling:
        x ← Proj(Pool2D(x)) + Proj(SE(DWConv ↓(Conv(Norm(x))))).

        Conv is a 1 X 1 convolution followed by a Batch Normalization layer and a GELU activation.
        SE is the Squeeze-Excitation layer.
        Proj is the shrink 1 X 1 convolution.

        Note: This implementation differs slightly from the original MobileNet implementation!

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downscale (bool, optional): If true downscale by a factor of two is performed. Default: False
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        drop_path (float, optional): Dropout rate to be applied during training. Default 0.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downscale: bool = False,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            drop_path: float = 0.,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(DoubleConv, self).__init__()
        # Save parameter
        self.drop_path_rate: float = drop_path
        # Ignore inplace parameter if GELU is used
        if act_layer == nn.GELU:
            act_layer = _gelu_ignore_parameters
        # Make main path
        self.main_path = nn.Sequential(
            norm_layer(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1)),
            DepthwiseSeparableConv(in_chs=in_channels, out_chs=out_channels, stride=2 if downscale else 1,
                                   act_layer=act_layer, norm_layer=norm_layer, drop_path_rate=drop_path),
            SqueezeExcite(in_chs=out_channels, rd_ratio=0.25),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1))
        )
        # Make skip path
        if downscale:
            self.skip_path = nn.Sequential(
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
            )
        else:
            self.skip_path = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)) 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)] (downscaling is optional).
        """
        output = self.main_path(x)
        if self.drop_path_rate > 0.:
            output = drop_path(output, self.drop_path_rate, self.training)
        x = output + self.skip_path(x)
        return x


class DeconvModule(nn.Module):
    """Deconvolution upsample module in decoder for UNet (2X upsample).

    This module uses deconvolution to upsample feature map in the decoder
    of UNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_layer: Normalization layer to use.
            Default: nn.BatchNorm2d.
        act_layer (dict | None): Activation layer to use.
            Default: nn.Mish.
        kernel_size (int): Kernel size of the convolutional layer. Default: 4.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.Mish,
                 kernel_size=4,
                 scale_factor=2):
        super(DeconvModule, self).__init__()

        assert (kernel_size - scale_factor >= 0) and\
               (kernel_size - scale_factor) % 2 == 0,\
               f'kernel_size should be greater than or equal to scale_factor '\
               f'and (kernel_size - scale_factor) should be even numbers, '\
               f'while the kernel size is {kernel_size} and scale_factor is '\
               f'{scale_factor}.'

        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

        norm = norm_layer(out_channels)
        activate = act_layer()
        self.deconv_upsamping = nn.Sequential(deconv, norm, activate)

    def forward(self, x):
        """Forward function."""
        out = self.deconv_upsamping(x)
        return out

class Stage(nn.Module):
    """Downscaling with DoubleConv then Axial Attention
    
    Args:
        image_size (int): input data size
        depth (int): Depth of the stage.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_heads (int, optional): Number of attention heads. Default 32
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        
    """

    def __init__(self,
            image_size: int,
            depth: int,
            in_channels: int,
            out_channels: int,
            type_name: str,
            pos_embed: bool,
            num_heads: int = 32,
            drop: float = 0.,
            drop_path: Union[List[float], float] = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__()
        self.type_name = type_name
        
        if self.type_name == "encoder":
            # Downsampling for encoder
            self.conv = DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                downscale=True,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_path=drop_path[0],
            )
            # Init blocks 
            self.blocks = nn.Sequential(*[
                AxialTransformerBlock(
                    dim=out_channels, 
                    axial_pos_emb_shape=pair(image_size), 
                    heads = num_heads, 
                    drop = drop,
                    drop_path_rate=drop_path[index],
                    dim_heads = None,
                    pos_embed=pos_embed
                )
                for index in range(depth)
            ])
            
        elif self.type_name == "decoder":
            # Deconv for upsampling bottom connection
            self.upsample = DeconvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_layer=norm_layer,
                act_layer=act_layer
                )
            # Downsampling for decoder
            self.conv = DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                downscale=False,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_path=drop_path[0],
            )
            # Init blocks
            self.blocks = nn.Sequential(*[
                AxialTransformerBlock(
                    dim=out_channels, 
                    axial_pos_emb_shape=pair(image_size), 
                    heads = num_heads, 
                    drop = drop,
                    drop_path_rate=drop_path[index],
                    dim_heads = None,
                    pos_embed=pos_embed
                )
                for index in range(depth)
            ])
            
    def forward(self, x, skip=None):
        # [B, C_in, H, W] -> [B, C_out, H // 2, W // 2]
        if self.type_name == "encoder":
            x = self.conv(x)
            x = self.blocks(x)
        elif self.type_name == "decoder":
            x = self.upsample(x)
            x = torch.cat([skip, x], dim=1)
            x = self.conv(x)
            x = self.blocks(x)
        return x

class FinalExpand(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_dim,
        out_channels,
        norm_layer,
        act_layer,
        ):
        super().__init__()
        self.upsample = DeconvModule(
                in_channels=in_channels,
                out_channels=embed_dim,
                norm_layer=norm_layer,
                act_layer=act_layer
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim*2, out_channels=embed_dim, kernel_size=3, stride=1, padding=1),
            act_layer(),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=1, padding=1),
            act_layer(),
        )
       
    def forward(self, skip, x):
        x = self.upsample(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)

        return x

class polarisnet(nn.Module):
    def __init__(
        self, 
        image_size=224, 
        in_channels=1, 
        out_channels=1,
        embed_dim=64, 
        depths=[2,2,2,2],
        channels=[64,128,256,512], 
        num_heads = 16, 
        drop=0., 
        drop_path=0.1, 
        act_layer=nn.GELU, 
        norm_layer=nn.BatchNorm2d,
        pos_embed=False
        ):
        
        super(polarisnet, self).__init__()
        self.num_stages = len(depths)
        self.num_features = channels[-1]
        self.embed_dim = channels[0]
        
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            act_layer(),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            act_layer(),
        )
        
        # Init blocks
        drop_path = torch.linspace(0.0, drop_path, sum(depths)).tolist()
        encoder_stages = []
        for index in range(self.num_stages):
            encoder_stages.append(
                Stage(
                    image_size=image_size//(pow(2,1+index)),
                    depth=depths[index],
                    in_channels=embed_dim if index == 0 else channels[index - 1],
                    out_channels=channels[index],
                    num_heads=num_heads,
                    drop=drop,
                    drop_path=drop_path[sum(depths[:index]):sum(depths[:index + 1])],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    type_name = "encoder",
                    pos_embed=pos_embed
                )
            )
        self.encoder_stages = nn.ModuleList(encoder_stages)
        
        decoder_stages = []
        for index in range(self.num_stages-1):
            decoder_stages.append(
                Stage(
                    image_size=image_size//(pow(2,self.num_stages-index-1)),
                    depth=depths[self.num_stages - index - 2],
                    in_channels=channels[self.num_stages - index - 1],
                    out_channels=channels[self.num_stages - index - 2],
                    num_heads=num_heads,
                    drop=drop,
                    drop_path=drop_path[sum(depths[:(self.num_stages-2-index)]):sum(depths[:(self.num_stages-2-index) + 1])],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    type_name = "decoder",
                    pos_embed=pos_embed
                )
            )
        self.decoder_stages = nn.ModuleList(decoder_stages)

        self.norm = norm_layer(self.num_features)
        self.norm_up= norm_layer(self.embed_dim)
        
        self.up = FinalExpand(
            in_channels=channels[0],
            embed_dim=embed_dim,
            out_channels=embed_dim,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
        self.output = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

    def encoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        """ encoder_forward pass

        Args:
            input (torch.Tensor): Input of the shape [B, C, H, W].

        Returns:
            output (torch.Tensor): Dense prediction output of the shape [B, C, H, W].
        """
        outs = []
        x = self.conv_first(x)
        for stage in self.encoder_stages:
            outs.append(x)
            x = stage(x)
            
        x = self.norm(x) 
         
        return x, outs
    
    def decoder_forward(self, x: torch.Tensor, x_downsample: list) -> torch.Tensor:
        """ encoder_forward pass

        Args:
            input (torch.Tensor): Input of the shape [B, C, H, W].

        Returns:
            output (torch.Tensor): Dense prediction output of the shape [B, C, H, W].
        """
        for inx, stage in enumerate(self.decoder_stages):
            x = stage(x, x_downsample[len(x_downsample)-1-inx])

        x = self.norm_up(x)  # B L C
  
        return x
 
    def up_x4(self, x: torch.Tensor, x_downsample: list):
        x = self.up(x_downsample[0],x)
        x = self.output(x)
            
        return x
    
    def forward(self, x):
        x, x_downsample = self.encoder_forward(x) 
        x = self.decoder_forward(x,x_downsample)
        x = self.up_x4(x,x_downsample)
        
        return x 
    
if __name__ == '__main__':
    net = polarisnet(in_channels=1, embed_dim=64, pos_embed=True).cuda()
    # print(net)
    X = torch.randn(5, 1, 224, 224).cuda()
    y = net(X)
    print(y.shape)

