import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Union, List


def block_diag(*arrs):
    shapes = jnp.array([a.shape for a in arrs])
    out = jnp.zeros(jnp.sum(shapes, axis=0))
    r, c = 0, 0
    for i, arr in enumerate(arrs):
        rr, cc = shapes[i]
        out = out.at[r : r + rr, c : c + cc].set(arr)
        r += rr
        c += cc
    return out


class BlockLinear(nn.Module):
    block_dims: List[Union[int, List[int]]]
    bias: bool = False

    @nn.compact
    def __call__(self, inp: jnp.ndarray) -> jnp.ndarray:
        blocks = [
            self.param(f"block_{i}", jax.random.normal, (dim, dim))
            for i, dim in enumerate(self.block_dims)
        ]

        full = block_diag(*blocks)
        out = jnp.matmul(full, inp)

        if self.bias:
            bias = self.param("bias", jax.random.normal, (sum(self.block_dims),))
            out = out + bias
        return out


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
        
        # TODO: setup BlockLinear
        self.R_z = nn.Dense(features=self.head_dim)
        self.R_i = nn.Dense(features=self.head_dim)
        self.R_o = nn.Dense(features=self.head_dim)
        self.R_f = nn.Dense(features=self.head_dim)

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
            x_c = nn.Conv(features=1, kernel_size=(self.ker_size,))(x_t)
            x_c = nn.silu(x_c).squeeze()
        else:
            x_c = x_t

        h_tm1_reshaped = h_tm1.reshape((seq.shape[0], -1, self.head_dim))

        i_t = self.W_i(x_c) + jax.vmap(self.R_i, in_axes=(0,))(h_tm1_reshaped).reshape(
            seq.shape[0], -1
        )
        f_t = self.W_f(x_c) + jax.vmap(self.R_f, in_axes=(0,))(h_tm1_reshaped).reshape(
            seq.shape[0], -1
        )
        z_t = self.W_z(x_t) + jax.vmap(self.R_z, in_axes=(0,))(h_tm1_reshaped).reshape(
            seq.shape[0], -1
        )
        o_t = self.W_o(x_t) + jax.vmap(self.R_o, in_axes=(0,))(h_tm1_reshaped).reshape(
            seq.shape[0], -1
        )

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
