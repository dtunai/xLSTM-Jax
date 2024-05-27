## x-LSTM: Extended Long Short-Term Memory (Jax / Flax)

This package provides Jax + Flax Linen implementation of the paper "xLSTM: Extended Long Short-Term Memory" et al Beck (2024). Paper introduces two main modifications to the LSTM architectures. Those modifications – exponential gating and novel memory structures – enrich the LSTM family by two members: (i) the new sLSTM with a scalar memory, a scalar update, and memory mixing, and (ii) the new mLSTM with a matrix memory and a covariance (outer product) update rule, which is fully parallelizable. My repository, for now, just implements sLSTM with a scalar memory, a scalar update, and new memory mixing, mLSTM will be added later.

## Getting Started

**Requirements**

```bash
jaxlib==0.4.25
jax==0.4.25
numpy==1.25.2
flax==0.8.4
```

You can install the package using `pip3 install -e .`:

```bash
git clone https://github.com/dtunai/xLSTM-Jax
cd xLSTM-Jax
pip3 install -e .
```

## Usage

Instantiate the model:

```python
s_lstm = sLSTM(inp_dim=inp_dim, head_dim=head_dim, head_num=head_num, p_factor=4 / 3)
```

or modify the `usage.py`.


## TODOs

[ ] Add training code for sLSTM with proper example
[ ] Add mLSTM block for parallelization and covariance update rule

## License

This package is licensed under the Apache License - see the LICENSE file for details.