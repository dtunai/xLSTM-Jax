import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

# sLSTM block import
from xlstm_jax.lstm_blocks import sLSTM

# Define test parameters (input seq dim, dim of heads, num of heads
seq_len = 32
batch_size = 4
inp_dim = 16
head_num = 4
head_dim = 8

# Mocking-up input sequence
seq = jax.random.normal(jax.random.PRNGKey(0), (batch_size, inp_dim))
s_lstm = sLSTM(inp_dim=inp_dim, head_dim=head_dim, head_num=head_num, p_factor=4 / 3)
hidden_state = sLSTM.init_hidden(batch_size=batch_size, head_num=head_num, head_dim=head_dim)
params = s_lstm.init(jax.random.PRNGKey(1), seq, hidden_state)
output, out_hidden_state = s_lstm.apply(params, seq, hidden_state)

print("Out", output)
print("Out dim", output.shape)
print("Hidden state dim", [h.shape for h in out_hidden_state])
