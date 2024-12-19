import numpy as np
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

from saved_agents.q_learning_agent.agent import Agent as AgentToTest
from saved_agents.relic_bot_base.agent import Agent as BaseAgent


def evaluate_dqn_agents(
    agent_1_cls, agent_2_cls, seed=42, games_to_play=1, replay_save_dir="replays"
):
    """
    Evaluate DQN agents against each other or against a base agent.

    Args:
        agent_1_cls: Class of first agent (DQN agent)
        agent_2_cls: Class of second agent (usually base agent)
        games_to_play: Number of games to play
    """
    # Initialize environment
    env = RecordEpisode(
        LuxAIS3GymEnv(numpy_output=True),
        save_on_close=True,
        save_on_reset=True,
        save_dir=replay_save_dir,
    )
    obs, info = env.reset(seed=seed)

    for i in range(games_to_play):
        if i % 5 == 0:
            seed = np.random.randint(0, 2**16 - 1)
        obs, info = env.reset(seed=seed)
        env_cfg = info["params"]  # Get environment configuration from params
        print(f"Game {i + 1}/{games_to_play}")
        print("-" * 50)
        # Initialize agents
        agent_1 = agent_1_cls("player_0", env_cfg)
        agent_2 = agent_2_cls("player_1", env_cfg)

        # Statistics
        agent_1_wins = 0
        agent_2_wins = 0
        draws = 0

        game_done = False
        step = 0

        while not game_done:
            # Get actions from both agents
            actions = dict()
            # Get player 0's action
            actions["player_0"] = agent_1.act(step=step, obs=obs["player_0"])
            # Get player 1's action
            actions["player_1"] = agent_2.act(step=step, obs=obs["player_1"])

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            if dones["player_0"] or dones["player_1"]:
                game_done = True
            step += 1

        # Get final scores from rewards
        score_1 = rewards["player_0"]
        score_2 = rewards["player_1"]

        # Determine winner
        if score_1 > score_2:
            agent_1_wins += 1
            print(f"Agent 1 wins! Score: {score_1} vs {score_2}")
        elif score_2 > score_1:
            agent_2_wins += 1
            print(f"Agent 2 wins! Score: {score_2} vs {score_1}")
        else:
            draws += 1
            print(f"Draw! Score: {score_1} vs {score_2}")
        # Save DQN model after each game
        agent_1.save_model()

    env.close()

    # Print final statistics
    print("\nFinal Statistics:")
    print("-" * 50)
    print(f"Agent 1 wins: {agent_1_wins}")
    print(f"Agent 2 wins: {agent_2_wins}")
    print(f"Draws: {draws}")
    print(f"Win rate: {agent_1_wins / games_to_play * 100:.2f}%")


if __name__ == "__main__":
    evaluate_dqn_agents(
        agent_1_cls=AgentToTest, agent_2_cls=BaseAgent, games_to_play=50
    )
