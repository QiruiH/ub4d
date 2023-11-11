import torch
import torch.nn as nn


# Positional encoding embedding. Code is borrowed taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    # 这返回的就是一个embed，然后一个out_dim
    return embed, embedder_obj.out_dim

# import torch

################################
#      hash embedding          #
################################

import numpy as np
import tinycudann as tcnn


def get_encoder(input_dim=3,
                degree=4, n_bins=16, n_frequencies=12,
                n_levels=16, level_dim=2, 
                base_resolution=16, log2_hashmap_size=19, 
                desired_resolution=512):
    
    # Sparse grid encoding
    # 我们先只保留hash的部分
    print('Hash size', log2_hashmap_size)
    per_level_scale = np.exp2(np.log2(desired_resolution  / base_resolution) / (n_levels - 1))
    embed = tcnn.Encoding(
        n_input_dims=input_dim,
        encoding_config={
            "otype": 'HashGrid',
            "n_levels": n_levels,
            "n_features_per_level": level_dim,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "per_level_scale": per_level_scale
        },
        dtype=torch.float
    )
    out_dim = embed.n_output_dims

    return embed, out_dim