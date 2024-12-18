import sys

from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

from agent import Agent as AgentToTest
from saved_agents.relic_bot_base.agent import Agent as BaseAgent


def evaluate_agents(
    agent_1_cls, agent_2_cls, seed=42, games_to_play=3, replay_save_dir="replays"
):
    with open("error.log", "w") as log_file:
        original_stderr = sys.stderr
        sys.stderr = log_file
        try:
            env = RecordEpisode(
                LuxAIS3GymEnv(numpy_output=True),
                save_on_close=True,
                save_on_reset=True,
                save_dir=replay_save_dir,
            )
            obs, info = env.reset(seed=seed)
            for i in range(games_to_play):
                obs, info = env.reset()
                env_cfg = info["params"]  # only contains observable game parameters
                player_0 = agent_1_cls("player_0", env_cfg)
                player_1 = agent_2_cls("player_1", env_cfg)

                # main game loop
                game_done = False
                step = 0
                print(f"Running game {i}")
                while not game_done:
                    actions = dict()
                    for agent in [player_0, player_1]:
                        actions[agent.player] = agent.act(
                            step=step, obs=obs[agent.player]
                        )
                    obs, reward, terminated, truncated, info = env.step(actions)
                    # info["state"] is the environment state object, you can inspect/play around with it to e.g. print
                    # unobservable game data that agents can't see
                    dones = {k: terminated[k] | truncated[k] for k in terminated}
                    if dones["player_0"] or dones["player_1"]:
                        game_done = True
                    # if step % 50 == 0 and step > 0:
                    # breakpoint()
                    step += 1
            env.close()  # free up resources and save final replay
        finally:
            sys.stderr = original_stderr


if __name__ == "__main__":
    evaluate_agents(agent_1_cls=AgentToTest, agent_2_cls=BaseAgent)
