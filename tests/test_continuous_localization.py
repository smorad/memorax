from memax.datasets.continuous_localization import step
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
import jax

def test_step():
    dx = jnp.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    x = jnp.array([[1, 0, 0], [2, 0, 0], [2, 1, 0]])
    drot = Rotation.from_quat(jnp.stack([
        Rotation.from_euler('z', jnp.pi/2).as_quat(),
        Rotation.from_euler('y', -jnp.pi/2).as_quat(),
        Rotation.from_euler('YZ', (jnp.pi/2, -jnp.pi/2)).as_quat()
    ], axis=0))
    rot = Rotation.from_quat(jnp.stack([
        Rotation.from_euler('z', jnp.pi / 2).as_quat(),
        Rotation.from_euler('zx', (jnp.pi/2, jnp.pi/2)).as_quat(),
        Rotation.identity().as_quat(),
    ], axis=0))

    x_start = jnp.zeros((1,3))
    rot_start = jax.vmap(Rotation.identity, axis_size=1)()

    _, (x_pred, rot_pred) = jax.lax.scan(step, (x_start, rot_start), (dx[:,None], drot[:,None]))

    pred_rot = rot_pred.as_matrix()[:,0]
    true_rot = rot.as_matrix()
    pred_x = x_pred[:,0]
    true_x = x

    assert jnp.allclose(pred_rot, true_rot)
    assert jnp.allclose(pred_x, true_x)


if __name__ == "__main__":
    test_step()