import time

import numpy as np
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from tqdm import tqdm, trange

from rl_agent import Agent, config


def train(
    agent_1_cls,
    agent_2_cls,
    seed=42,
    games_to_play=1,
    seed_variability=1,
    replay_save_dir="replays",
):
    #####################################################
    # ----------------- Initial Setup ----------------- #
    #####################################################
    env = RecordEpisode(
        LuxAIS3GymEnv(numpy_output=True),
        save_on_close=True,
        save_on_reset=True,
        save_dir=replay_save_dir,
    )
    obs, info = env.reset(seed=seed)
    env_cfg = info["params"]
    agent_1 = agent_1_cls("agent_1", env_cfg)
    agent_2 = agent_2_cls("agent_2", env_cfg)
    learn_iters = 0

    ########################################################
    # ----------------- Training Metrics ----------------- #
    ########################################################
    episode_rewards = []
    episode_lengths = []
    avg_scores = {agent_1.player: np.zeros(2), agent_2.player: np.zeros(2)}
    score_history = {
        agent_1.player: [np.zeros(games_to_play) for _ in range(2)],
        agent_2.player: [np.zeros(games_to_play) for _ in range(2)],
    }
    total_steps = 0
    #####################################################
    # ----------------- Training Loop ----------------- #
    #####################################################
    # Main progress bar for all games
    game_pbar = trange(games_to_play, desc="Training Progress")
    for i in game_pbar:
        # Alternate positions each game
        if i % 2 == 0:
            # Even games: agent_1 is player_0, agent_2 is player_1
            agent_1.player = "player_0"
            agent_2.player = "player_1"
            player_0, player_1 = agent_1, agent_2
        else:
            # Odd games: agent_1 is player_1, agent_2 is player_0
            agent_1.player = "player_1"
            agent_2.player = "player_0"
            player_0, player_1 = agent_2, agent_1

        # Change the seed every seed_variability number of games to give the agents a diverse learning experience
        if i % seed_variability == 0 and i != 0:
            seed = np.random.randint(0, 2**16 - 1)
            obs, info = env.reset(seed=seed)
            for agent in [agent_1, agent_2]:
                if isinstance(agent, Agent):
                    agent.update_env_config(info["params"])
        else:
            obs, info = env.reset(seed=seed)
        ###################################################
        # ----------------- Game Loop ------------------- #
        ###################################################
        done = False
        step = 0
        # tracking points for reward calculation
        previous_points = [0, 0]
        current_points = [0, 0]
        total_points = [0, 0]
        team_wins = [0, 0]

        # Progress bar for steps within the game
        step_pbar = tqdm(total=505, desc=f"Game {i+1}", leave=False)
        game_start_time = time.time()

        episode_reward = 0
        episode_length = 0

        while not done:
            actions = {}
            # Get actions, and store state, action, probs, vals, reward, and done in memory
            for idx, agent in enumerate([player_0, player_1]):
                if isinstance(agent, Agent):
                    action, prob, val = agent.choose_action(obs[agent.player])
                    actions[agent.player] = action
                    # Calc points gained this step
                    # point_gain will be stored in the replay buffer as the reward
                    # this is done so that we aren't relying solely on the sparse reward from the environment at the end of each match
                    current_points[idx] = obs[agent.player]["team_points"][idx]
                    point_gain = current_points[idx] - previous_points[idx]
                    total_points[idx] += point_gain if point_gain > 0 else 0
                    agent.remember(
                        obs[agent.player], action, prob, val, point_gain, done
                    )
                else:
                    # fall back in case we're training against a rule-based agent
                    actions[agent.player] = agent.act(step=step, obs=obs[agent.player])
            obs_, reward, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            done = any(dones.values())

            # Update points and rewards
            for idx, agent in enumerate([player_0, player_1]):
                if isinstance(agent, Agent):
                    # Store current points as previous for next step
                    previous_points[idx] = current_points[idx]
                    # Get new current points
                    current_points[idx] = obs_[agent.player]["team_points"][idx]
                    # Calculate point gain
                    point_gain = current_points[idx] - previous_points[idx]
                    # Add to running total if positive
                    if point_gain > 0:
                        total_points[idx] += point_gain
                        episode_reward += (
                            point_gain  # Track episode reward based on point gains
                        )

                    if obs_[agent.player]["match_steps"] == 100:
                        team_wins[idx] += obs_[agent.player]["team_wins"][idx]
                    # Update the last stored reward if game is over
                    if done:
                        # scale reward based on the points difference
                        # wins with a larger gap in points should be rewarded more heavily
                        # this should also by proxy reward the agent for learning
                        # to minimize the number of points the opponent earns
                        # without having to reward the agent for actions taken to stop the opponent from getting points
                        point_gain += (
                            current_points[idx] - current_points[1 - idx]
                        ) * 2 + obs_[agent.player]["team_wins"][idx] * 100

                        agent.memory.rewards[-1] = point_gain
                        agent.memory.dones[-1] = done
            # grab match step
            match_step = obs_["player_0"]["match_steps"]

            # Learn in three scenarios:
            # 1. At the end of each match (every 101 steps)
            # 2. When we have accumulated N steps
            # 3. At the end of the game
            should_learn = (
                (match_step == 101) or (step > 0 and step % config["N"] == 0) or done
            )
            if should_learn:
                for agent in [player_0, player_1]:
                    if isinstance(agent, Agent):
                        learn_result = agent.learn()
                        learn_iters += 1
                        # Log training metrics
                        if (
                            learn_result is not None
                        ):  # Only log when learning actually happened
                            actor_loss, critic_loss, value, advantage = learn_result
                            metrics = {
                                "losses/actor_loss": actor_loss,
                                "losses/critic_loss": critic_loss,
                                "losses/total_loss": actor_loss + critic_loss,
                                "values/value_estimate": value,
                                "values/advantages": advantage,
                            }
                            agent.actor.log_metrics(metrics, total_steps)
            obs = obs_
            step += 1
            episode_length += 1
            total_steps += 1

            # Update step progress bar
            step_pbar.update(1)

            # Update game progress bar description with current scores
            game_pbar.set_description(
                f"Training Progress: | "
                f"P0: {total_points[0]} | "
                f"P1: {total_points[1]} | "
                f"Steps: {step} | "
                f"Learn Iters: {learn_iters} | "
                f"Time: {int(time.time() - game_start_time)}s | "
            )

        # Close step bar
        step_pbar.close()
        game_duration = time.time() - game_start_time
        # Update game scores and save models

        for agent in [agent_1, agent_2]:
            if isinstance(agent, Agent):
                player_idx = 0 if agent.player == "player_0" else 1
                score = total_points[player_idx]

                agent_id = "agent_1" if agent == agent_1 else "agent_2"
                score_history[agent_id][player_idx][i] = score
                avg_scores[agent_id][player_idx] = np.mean(
                    score_history[agent_id][player_idx][: i + 1]
                )

                if score > avg_scores[agent_id][player_idx]:
                    # performed better than average, save the model
                    agent.save_models()
                print(
                    f"Epoch {i+1}, Agent: {agent_id},\n"
                    f"game_score: {score},\n"
                    f"avg_score: {avg_scores[agent_id][player_idx]},\n"
                    f"best_score: {np.max(score_history[agent_id][player_idx])},\n"
                    f"time_steps: {total_steps},\n"
                    f"learn_iters: {learn_iters}\n"
                )

        # Log episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        metrics = {
            "episode/reward": episode_reward,
            "episode/length": episode_length,
            "episode/avg_reward_100": np.mean(episode_rewards[-100:]),
            "episode/avg_length_100": np.mean(episode_lengths[-100:]),
        }
        agent_1.actor.log_metrics(metrics, total_steps)

        # Log network distributions periodically
        if i % 50 == 0:
            agent_1.actor.log_distributions(total_steps)

        if i % 10 == 0:
            print(
                f'Game {i}/{games_to_play}, '
                f'Avg Reward (last 100): {metrics["episode/avg_reward_100"]:.2f}, '
                f'Avg Length (last 100): {metrics["episode/avg_length_100"]:.2f}'
            )

        game_pbar.set_postfix(
            {
                "Duration": f"{game_duration:.1f}s",
                "P0_Score": total_points[0],
                "P1_Score": total_points[1],
            }
        )
    env.close()  # free up resources and save final replay


if __name__ == "__main__":
    train(Agent, Agent, seed=42, games_to_play=500, seed_variability=50)
