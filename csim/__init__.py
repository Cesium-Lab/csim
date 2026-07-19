import jax

# Orbital/rigid-body dynamics need float64: jax defaults to float32, which is
# nowhere near precise enough for the tolerances (down to 1e-12) used throughout
# csim's math/physics modules.
jax.config.update("jax_enable_x64", True)
