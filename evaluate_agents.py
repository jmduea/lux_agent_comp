import time

import jax
import numpy as np
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

# from saved_agents.relic_bot_defensive_sapper.agent import Agent as AgentToTest
from matplotlib import pyplot as plt

from agent import Agent as AgentToTest
from core.debug import SpaceVisualizer
from core.space import Space, update_space


def evaluate_agents(
    agent_1_cls,
    agent_2_cls,
    seed=42,
    games_to_play=3,
    replay_save_dir="replays",
    training=False,
    debug=False,
):
    env = RecordEpisode(
        LuxAIS3GymEnv(numpy_output=True),
        save_on_close=True,
        save_on_reset=True,
        save_dir=replay_save_dir,
    )
    visualizer_1 = SpaceVisualizer()
    # visualizer_2 = SpaceVisualizer()
    for i in range(games_to_play):
        seed = np.random.randint(0, 2**16 - 1)
        # obs, info = env.reset(seed=4255)
        # obs, info = env.reset(seed=51780)
        obs, info = env.reset(seed=seed)
        player_0_space = Space.create()
        # player_1_space = Space.create()
        env_cfg = info["params"]  # only contains observable game parameters
        player_0 = agent_1_cls("player_0", env_cfg)
        player_1 = agent_2_cls("player_1", env_cfg)

        rng, init_key0 = jax.random.split(jax.random.PRNGKey(seed))

        # main game loop
        game_done = False
        step = 0
        last_obs = None
        last_actions = None
        print(f"Running game {i}")
        while not game_done:
            actions = dict()

            if training:
                last_obs = {
                    "player_0": obs["player_0"].copy(),
                    "player_1": obs["player_1"].copy(),
                }

            for agent in [player_0, player_1]:
                actions[agent.player] = agent.act(step=step, obs=obs[agent.player])
            if training:
                last_actions = {
                    "player_0": actions["player_0"].copy(),
                    "player_1": actions["player_1"].copy(),
                }
            obs, rewards, terminated, truncated, info = env.step(actions)
            player_0_space = update_space(player_0_space, obs["player_0"])
            # player_1_space = update_space(player_1_space, obs["player_1"])
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            rewards = {
                "player_0": obs["player_0"]["team_points"][player_0.team_id],
                "player_1": obs["player_1"]["team_points"][player_1.team_id],
            }
            if training and last_obs is not None:
                for agent in [player_0, player_1]:
                    for unit_id in range(env_cfg["max_units"]):
                        if obs[agent.player]["units_mask"][agent.team_id][unit_id]:
                            current_state = agent._create_unit_state(
                                last_obs[agent.player]["units"]["position"][
                                    agent.team_id
                                ][unit_id],
                                last_obs[agent.player]["units"]["energy"][
                                    agent.team_id
                                ][unit_id],
                                last_obs[agent.player]["relic_nodes"],
                                last_obs[agent.player]["steps"],
                                last_obs[agent.player]["relic_nodes_mask"],
                            )

                        next_state = agent._create_unit_state(
                            obs[agent.player]["units"]["position"][agent.team_id][
                                unit_id
                            ],
                            obs[agent.player]["units"]["energy"][agent.team_id][
                                unit_id
                            ],
                            obs[agent.player]["relic_nodes"],
                            obs[agent.player]["steps"],
                            obs[agent.player]["relic_nodes_mask"],
                        )
                        agent.memory.push(
                            current_state,
                            last_actions[agent.player][unit_id][0],
                            rewards[agent.player],
                            next_state,
                            dones[agent.player],
                        )
                agent.learn(
                    rng,
                    step,
                    last_obs[agent.player],
                    actions[agent.player],
                    obs[agent.player],
                    rewards[agent.player],
                    dones[agent.player],
                )

            if dones["player_0"] or dones["player_1"]:
                game_done = True
                if training:
                    player_0.save_model()
                    player_1.save_model()

            step += 1
            if debug:
                # visualize_space(player_0.agent_state.space)
                visualizer_1.update(player_0_space)
                # visualizer_1.update(player_0_space)
                # visualizer_2.update(player_1_space)
                time.sleep(0.00005)
    env.close()  # free up resources and save final replay
    print(
        f"Obstacle Movement Period: {player_0.state.space.drift_speed_found}, {player_0.state.space.drift_speed}, {player_0.state.space.ALL_RELICS_FOUND}"
    )
    plt.ioff()
    plt.show()
    # visualize_space(player_0.agent_state.space)


if __name__ == "__main__":
    evaluate_agents(
        agent_1_cls=AgentToTest,
        agent_2_cls=AgentToTest,
        games_to_play=1,
        replay_save_dir="replays",
        debug=True,
    )
