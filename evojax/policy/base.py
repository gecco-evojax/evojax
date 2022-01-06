# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from abc import abstractmethod
import jax.numpy as jnp


class PolicyNetwork(ABC):
    """Interface for all policy networks in EvoJAX."""

    num_params: int

    @abstractmethod
    def get_actions(self,
                    vec_obs: jnp.ndarray,
                    params: jnp.ndarray) -> jnp.ndarray:
        """Get vectorized actions for the corresponding (obs, params) pair.

        Args:
            vec_obs - Vectorized observations of shape (num_envs, *obs_shape).
            params - A batch of parameters, shape is (num_envs, param_size).
        Returns:
            jnp.ndarray. Vectorized actions.
        """
        raise NotImplementedError()
