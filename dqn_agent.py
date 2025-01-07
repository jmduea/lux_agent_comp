from functools import partial
import os

import jax
import optax
from core.base_agent import DQN, BaseAgent, ReplayBuffer, TrainStateDQN
import jax.numpy as jnp
from flax.training import checkpoints


class DQNAgent(BaseAgent):
    def __init__(self, player, env_cfg):
        super().__init__(player, env_cfg)

        self.checkpoint_dir = "./checkpoints"
        self.checkpoint_dir = os.path.abspath(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # DQN Parameters
        self.state_size = 6
        self.action_size = 6
        self.hidden_size = 128
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-4

        # Initialize replay buffer
        self.memory = ReplayBuffer(10000)

        self.policy_params = None
        self.target_params = None
        self.policy_state = None
        self.target_state = None
        self.init_networks(self.key)
        if not self.state.training:
            self.load_model()
            self.epsilon = 0.0
        else:
            self.load_model()

    @partial(jax.jit, static_argnums=(0,))
    def _get_closest_relic(self, unit_pos, relic_nodes, relic_mask):
        """
        Jittable method for finding the closest relic to a given unit's position
        """
        distances = jnp.where(
            relic_mask, jnp.linalg.norm(relic_nodes - unit_pos, axis=1), jnp.inf
        )

        closest_idx = jnp.argmin(distances)

        no_relic = jnp.isinf(distances[closest_idx])

        closest_relic = jnp.where(
            no_relic,
            jnp.array([-1, -1], dtype=relic_nodes.dtype),
            relic_nodes[closest_idx],
        )

        return closest_relic

    @partial(jax.jit, static_argnums=(0,))
    def _create_unit_state(self, unit_pos, unit_energy, relic_nodes, step, relic_mask):
        closest_relic = self._get_closest_relic(unit_pos, relic_nodes, relic_mask)

        state = jnp.concatenate(
            [
                unit_pos,
                closest_relic,
                jnp.atleast_1d(unit_energy),
                jnp.atleast_1d(step / 505.0),
            ]
        )

        return state

    def init_networks(self, rng_key):
        dqn_model = DQN(hidden_size=self.hidden_size, output_size=self.action_size)

        dummy_x = jnp.zeros((1, self.state_size), dtype=jnp.float32)

        init_params = dqn_model.init(rng_key, dummy_x)

        tx = optax.adam(self.learning_rate)

        self.policy_state = TrainStateDQN.create(
            apply_fn=dqn_model.apply, params=init_params, tx=tx
        )

        self.target_state = self.policy_state.replace(
            params=jax.tree_util.tree_map(lambda x: x, self.policy_state.params)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _forward(self, params, x):
        return self.policy_state.apply_fn(params, x)

    @partial(jax.jit, static_argnums=(0,))
    def _get_q_values(self, params, x):
        return self._forward(params, x)

    @partial(jax.jit, static_argnums=(0,))
    def _get_available_units(self, unit_mask):
        """
        Jittable method to get available unit ids from unit mask
        """
        return jnp.nonzero(unit_mask, size=self.max_units)[0]

    @partial(jax.jit, static_argnums=(0,))
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        obs = self._preprocess_obs(obs)
        actions = jnp.zeros((self.max_units, 3), dtype=jnp.int32)

        unit_mask = obs["units_mask"][self.team_id]
        available_units = self._get_available_units(unit_mask)

        self.score = obs["team_points"][self.team_id]

        for unit in available_units:
            state = self._create_unit_state(
                obs["units"]["position"][self.team_id][unit][0],
                obs["units"]["energy"][self.team_id][unit][0],
                obs["relic_nodes"][0],
                obs["steps"],
                obs["relic_nodes_mask"][0],
            )

            def select_action(self, state):
                self.key, subkey = jax.random.split(self.key)
                sample_val = jax.random.uniform(subkey)
                condition = jnp.logical_and(
                    sample_val < self.epsilon, jnp.array(self.training)
                )
                del subkey

                def random_action(_):
                    self.key, subkey = jax.random.split(self.key)
                    rand_act = jax.random.randint(
                        subkey, shape=(), minval=0, maxval=self.action_size
                    )
                    del subkey
                    return rand_act

                def greedy_action(_):
                    q_vals = self._get_q_values(
                        self.policy_state.params, state[None, :]
                    )
                    return jnp.argmax(q_vals[0], axis=0)

                action_type = jax.lax.cond(
                    condition, random_action, greedy_action, None
                )

                return action_type

            action_type = select_action(self, state)

            def get_sap_action(obs, unit, action_type, actions):
                opp_positions = jnp.array(obs["units"]["position"][self.opp_team_id])
                opp_mask = jnp.array(obs["units_mask"][self.opp_team_id])
                valid_mask = (opp_mask == 1) & (opp_positions[:, :] != -1)

                target_idx = jnp.where(valid_mask, size=1, fill_value=-1)[0]
                has_valid_target = target_idx != -1
                target_pos = jax.lax.select(
                    has_valid_target, opp_positions[target_idx[0]], jnp.array([0, 0])
                )

                unit_pos = obs["units"]["position"][self.team_id][unit]
                deltas = jax.lax.select(
                    has_valid_target, target_pos - unit_pos, jnp.array([0, 0])
                )

                # Update actions array
                action_values = jnp.array([action_type, deltas[0], deltas[1]])
                actions = actions.at[unit].set(action_values)

                return actions

            actions = jax.lax.cond(
                action_type == 5,
                lambda: get_sap_action(obs, unit, action_type, actions),
                lambda: actions.at[unit].set(jnp.array([action_type, 0, 0])),
            )
        return actions

    def learn(self, rng, step, last_obs, actions, obs, rewards, dones):
        if not self.training or len(self.memory) < self.batch_size:
            return rng

        rng, subkey = jax.random.split(rng)
        batch = self.memory.sample(self.batch_size)
        (states, actions_, rewards_, next_states, dones_) = zip(*batch)

        states = jnp.stack(states, axis=0)
        actions_ = jnp.array(actions_, dtype=jnp.int32)
        rewards_ = jnp.array(rewards_, dtype=jnp.float32)
        next_states = jnp.stack(next_states, axis=0)
        dones_ = jnp.array(dones_, dtype=jnp.float32)

        def loss_fn(
            params, target_params, states, actions_, rewards_, next_states, dones_
        ):
            q_values = self._get_q_values(params, states)
            q_taken = jax.vmap(lambda qv, a: qv[a])(q_values, actions_)

            next_q_values = self._get_q_values(target_params, next_states)
            next_q_max = jnp.max(next_q_values, axis=1)

            target = rewards_ + (1.0 - dones_) * self.gamma * next_q_max

            loss = jnp.mean((q_taken - target) ** 2)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)

        def train_step_fn(
            policy_state: TrainStateDQN,
            target_params,
            states,
            actions_,
            rewards_,
            next_states,
            dones_,
        ):
            loss_val, grads = grad_fn(
                policy_state.params,
                target_params,
                states,
                actions_,
                rewards_,
                next_states,
                dones_,
            )
            policy_state = policy_state.apply_gradients(grads=grads)
            return policy_state, loss_val

        self.policy_state, loss_val = train_step_fn(
            self.policy_state,
            self.target_state.params,
            states,
            actions_,
            rewards_,
            next_states,
            dones_,
        )

        if step % 100 == 0:
            self.target_state = self.target_state.replace(
                params=self.policy_state.params
            )

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return rng

    def save_model(self, step=0):
        abs_path = os.path.abspath(self.checkpoint_dir)
        checkpoints.save_checkpoint(
            ckpt_dir=abs_path,
            target=self.policy_state.params,
            step=step,
            prefix=f"dqn_model_{self.player}_",
            overwrite=True,
        )
        print(f"Model saved for {self.player} at step {step}")

    def load_model(self):
        abs_path = os.path.abspath(self.checkpoint_dir)
        loaded_params = checkpoints.restore_checkpoint(
            ckpt_dir=abs_path,
            target=None,
            prefix=f"dqn_model_{self.player}_",
        )
        if loaded_params:
            self.policy_state = self.policy_state.replace(params=loaded_params)
            self.target_state = self.target_state.replace(params=loaded_params)
            print(f"Model loaded for {self.player}")
        else:
            print(f"No model found for {self.player}")
            print(f"No model found for {self.player}")
