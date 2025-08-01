import jax, jax.numpy as jnp
from jax import random
jax.config.update('jax_use_magma', 'on')
key = random.PRNGKey(0)
a = random.normal(key, (2048, 2048))
# 일반 eig. MAGMA 미링크 시 GPU → CPU fallback
w, _, _ = jax.lax.linalg.eig(a, use_magma=True)
print("backend:", jax.default_backend(), "eigenvals shape:", w.shape)
