import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Union, List


class CausalConv1D(nn.Module):
    features: int
    kernel_size: int
    dilation: int = 1

    @nn.compact
    def __call__(self, x):
        padding = (self.kernel_size - 1) * self.dilation
        x = jnp.pad(x, [(0, 0), (0, 0), (padding, 0)], mode="constant")
        conv = nn.Conv(self.features, self.kernel_size, kernel_dilation=self.dilation)
        return conv(x)


class BlockLinear(nn.Module):
    in_features: int
    out_features: int
    num_blocks: int

    @nn.compact
    def __call__(self, x):
        block_in_features = self.in_features // self.num_blocks
        block_out_features = self.out_features // self.num_blocks
        x_split = jnp.split(x, self.num_blocks, axis=-1)
        y_split = [nn.Dense(block_in_features, block_out_features)(x_i) for x_i in x_split]
        return jnp.concatenate(y_split, axis=-1)


class sLSTM(nn.Module):
    inp_dim: int
    head_dim: int
    head_num: int
    ker_size: int = 4
    p_factor: float = 4 / 3

    def setup(self):
        self.inp_norm = nn.LayerNorm(self.inp_dim)
        self.hid_norm = nn.GroupNorm(num_groups=self.head_num)

        self.W_z = nn.Dense(features=self.head_num * self.head_dim)
        self.W_i = nn.Dense(features=self.head_num * self.head_dim)
        self.W_o = nn.Dense(features=self.head_num * self.head_dim)
        self.W_f = nn.Dense(features=self.head_num * self.head_dim)

        self.R_z = BlockLinear(self.head_dim, self.head_dim, self.head_num)
        self.R_i = BlockLinear(self.head_dim, self.head_dim, self.head_num)
        self.R_f = BlockLinear(self.head_dim, self.head_dim, self.head_num)
        self.R_o = BlockLinear(self.head_dim, self.head_dim, self.head_num)

        proj_dim = int(self.p_factor * self.head_num * self.head_dim)
        self.up_proj = nn.Dense(features=2 * proj_dim)
        self.down_proj = nn.Dense(features=self.inp_dim)

    @staticmethod
    def init_hidden(
        batch_size: int, head_num: int, head_dim: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        c_0 = jnp.zeros((batch_size, head_num * head_dim))
        n_0 = jnp.ones((batch_size, head_num * head_dim))
        h_0 = jnp.zeros((batch_size, head_num * head_dim))
        m_0 = jnp.zeros((batch_size, head_num * head_dim))
        return c_0, n_0, h_0, m_0

    def __call__(
        self,
        seq: jnp.ndarray,
        hid: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        use_conv: bool = False,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        c_tm1, n_tm1, h_tm1, m_tm1 = hid

        x_t = self.inp_norm(seq)

        if use_conv:
            x_c = CausalConv1D(features=seq.shape[-1], kernel_size=self.ker_size)(x_t)
            x_c = nn.silu(x_c).squeeze()
        else:
            x_c = x_t

        h_tm1_reshaped = h_tm1.reshape((seq.shape[0], self.head_num, self.head_dim))
        i_t = self.W_i(x_c) + jax.vmap(self.R_i)(h_tm1_reshaped).reshape(seq.shape[0], -1)
        f_t = self.W_f(x_c) + jax.vmap(self.R_f)(h_tm1_reshaped).reshape(seq.shape[0], -1)
        z_t = self.W_z(x_t) + jax.vmap(self.R_z)(h_tm1_reshaped).reshape(seq.shape[0], -1)
        o_t = self.W_o(x_t) + jax.vmap(self.R_o)(h_tm1_reshaped).reshape(seq.shape[0], -1)
        m_t = jnp.maximum(f_t + m_tm1, i_t)
        i_t = jnp.exp(i_t - m_t)
        f_t = jnp.exp(f_t - m_t + m_tm1)
        z_t = jnp.tanh(z_t)
        o_t = jax.nn.sigmoid(o_t)
        c_t = f_t * c_tm1 + i_t * z_t
        n_t = f_t * n_tm1 + i_t
        h_t = o_t * (c_t / n_t)

        out = self.hid_norm(h_t.reshape(seq.shape[0], self.head_num, self.head_dim)).reshape(
            seq.shape[0], -1
        )

        out1, out2 = jnp.split(self.up_proj(out), 2, axis=-1)
        out = out1 + jax.nn.gelu(out2)
        out = self.down_proj(out)

        return out + seq, (c_t, n_t, h_t, m_t)
