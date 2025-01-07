from typing import List

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from core.space import Space
from tests.sample_input.baseline_agent_game import game_env_cfg, obs_game_step_95

nebula_tile_color = (166, 177, 225, 255)
asteroid_tile_color = (51, 56, 68, 255)


class SpaceVisualizer:
    def __init__(
        self,
        space_size: int = 24,
        num_attrs: int = 9,
        title: str = "Space Visualization",
    ):
        """
        Initializes the SpaceVisualizer with predefined subplots and color maps.

        Args:
            space_size (int, optional): Size of the space grid. Defaults to 24.
            num_attrs (int, optional): Number of attributes to visualize. Defaults to 9.
            title (str, optional): Main title for the visualization. Defaults to "Space Visualization".
        """
        self.space_size = space_size
        self.num_attrs = num_attrs
        self.title = title

        # Define color maps
        self.node_type_cmap = mcolors.ListedColormap(
            ["grey", "white", "violet", "midnightblue"]
        )
        self.energy_cmap = "viridis"
        self.binary_cmap = "gray"
        self.mask_cmap = "Reds"
        self.heatmap_cmap = "hot"

        # Define attributes
        self.attributes = [
            ("Node Type", None, self.node_type_cmap, "Categorical"),
            ("Energy", None, self.energy_cmap, "Continuous"),
            (
                "Is Visible with Node Type",
                None,
                mcolors.ListedColormap(
                    ["grey"] + ["grey", "white", "violet", "midnightblue"]
                ),
                "Categorical",
            ),
            # ("Is Unknown", None, self.binary_cmap, "Binary"),
            # ("Is Walkable", None, self.binary_cmap, "Binary"),
            ("Relic Mask", None, self.mask_cmap, "Binary"),
            # ("Reward Heatmap", None, self.heatmap_cmap, "Continuous"),
            # ("Explored for Relic", None, self.binary_cmap, "Binary"),
            # ("Explored for Reward", None, self.binary_cmap, "Binary"),
        ]

        # Initialize the plot
        self.fig, self.axs = plt.subplots(
            3,
            3,
            figsize=(12, 12),
            sharex=True,
            sharey=True,
        )
        self.fig.suptitle(self.title, fontsize=24)

        # Initialize image objects list
        self.im_objects: List[plt.Artist] = []

        # Initialize subplots with empty data
        for idx, (attr_name, _, cmap, attr_type) in enumerate(self.attributes):
            row = idx // 3
            col = idx % 3
            ax = self.axs[row, col]

            # Initialize with zeros or appropriate default
            initial_data = np.zeros((self.space_size, self.space_size))

            # Create the imshow object and store it
            if attr_name == "Is Visible with Node Type":
                im = ax.matshow(
                    initial_data,
                    cmap=mcolors.ListedColormap(
                        ["grey", "white", "violet", "midnightblue"][:4]
                    ),
                    origin="upper",
                )
            else:
                im = ax.matshow(
                    initial_data, cmap=cmap, origin="upper", interpolation="nearest"
                )
            self.im_objects.append(im)

            # Set titles and grid
            ax.set_title(attr_name, fontsize=14)
            ax.set_xticks(np.arange(-0.5, self.space_size, 1))
            ax.set_yticks(np.arange(-0.5, self.space_size, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(which="major", color="black", linewidth=1, mouseover=True)

            # Add colorbar
            cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if attr_type == "Binary":
                cbar.set_ticks([0, 1])
                cbar.set_ticklabels(["False", "True"])
            elif attr_type == "Categorical":
                # Handle specific categorical cases
                if attr_name == "Is Visible with Node Type":
                    cbar.set_ticks([-1, 0, 1, 2])
                    cbar.set_ticklabels(["Not Visible", "Empty", "Nebula", "Asteroid"])
                else:
                    # General categorical handling
                    unique_vals = [0, 1, 2, 3]  # Adjust based on actual categories
                    cbar.set_ticks(unique_vals)
                    cbar.set_ticklabels([str(val) for val in unique_vals])
            elif attr_type == "Continuous":
                cbar.set_label(attr_name)

        # Hide any unused subplots (if any)
        total_subplots = 9
        if len(self.attributes) < total_subplots:
            for idx in range(len(self.attributes), total_subplots):
                row = idx // 3
                col = idx % 3
                self.axs[row, col].axis("off")

        self.step_text = self.fig.text(
            0.5,
            0.95,
            "Step: 0",
            ha="center",
            va="center",
            fontsize=16,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        plt.tight_layout()
        plt.ion()  # Enable interactive mode
        plt.show()

    def update(self, space: "Space", env_idx: int = 0):
        """
        Updates the visualization with the latest space data.

        Args:
            space (Space): The space instance to visualize
        """
        for idx, (attr_name, _, cmap, attr_type) in enumerate(self.attributes):
            # Retrieve the corresponding data based on attribute name
            if attr_name == "Node Type":
                data = space.tile_type
            elif attr_name == "Energy":
                data = space.energy
            elif attr_name == "Is Visible with Node Type":
                data = jnp.where(space.is_visible, space.tile_type, -1)
            # elif attr_name == "Is Unknown":
            #     data = space.is_unknown
            # elif attr_name == "Is Walkable":
            #     data = space.is_walkable
            elif attr_name == "Relic Mask":
                data = space.relic_mask
            # elif attr_name == "Reward Heatmap":
            #     data = space.reward_heatmap
            # elif attr_name == "Explored for Relic":
            #     data = space.explored_for_relic
            # elif attr_name == "Explored for Reward":
            #     data = space.explored_for_reward
            else:
                data = np.zeros((self.space_size, self.space_size))  # Fallback

            if data.ndim == 3:
                data = data[env_idx, ...]

            # Convert JAX arrays to NumPy if necessary
            if isinstance(data, jnp.ndarray):
                data = np.array(data)

            # Update the image data
            self.im_objects[idx].set_data(data)

            # For categorical data, update the color limits if necessary
            if attr_type == "Categorical":
                self.im_objects[idx].set_clim(vmin=-1, vmax=2)
            elif attr_type == "Binary":
                self.im_objects[idx].set_clim(vmin=0, vmax=1)
            elif attr_type == "Continuous":
                self.im_objects[idx].autoscale()

        current_step = space.step
        if current_step.ndim == 1:
            current_step = current_step[env_idx]
        self.step_text.set_text(f"Step: {current_step}")

        self.fig.canvas.draw_idle()
        plt.pause(0.0001)  # Small pause to allow the plot to update


def visualize_space(space: Space, title: str = "Space Visualization"):
    """Visualize the various attributes of the Space object

    Args:
        space (Space): The space instance to visualize
        title (str, optional): Main Title for the visualization. Defaults to "Space Visualization".
    """
    space_size = space.tile_type.shape[0]
    # num_attrs = 8

    node_type_cmap = mcolors.ListedColormap(["grey", "white", "violet", "midnightblue"])
    energy_cmap = "viridis"
    binary_cmap = "gray"
    mask_cmap = "Reds"
    heatmap_cmap = "hot"

    fig, axs = plt.subplots(
        3,
        3,
        figsize=(12, 12),
        sharex=True,
        sharey=True,
    )

    fig.suptitle(title, fontsize=24)
    attributes = [
        ("Node Type", space.tile_type, node_type_cmap, "Categorical"),
        ("Energy", space.energy, energy_cmap, "Continuous"),
        (
            "Is Visible with Node Type",
            jnp.where(space.is_visible, space.tile_type, -1),
            mcolors.ListedColormap(
                ["grey"] + ["grey", "white", "violet", "midnightblue"]
            ),
            "Categorical",
        ),
        ("Is Unknown", space.is_unknown, binary_cmap, "Binary"),
        # ("Is Walkable", space.is_walkable, binary_cmap, "Binary"),
        ("Relic Mask", space.relic_mask, mask_cmap, "Binary"),
        # ("Reward Heatmap", space.reward_heatmap, heatmap_cmap, "Continuous"),
        ("Explored for Relic", space.explored_for_relic, binary_cmap, "Binary"),
        # ("Explored for Reward", space.explored_for_reward, binary_cmap, "Binary"),
    ]

    for idx, (attr_name, attr_data, cmap, attr_type) in enumerate(attributes):
        row = idx // 3
        col = idx % 3
        ax = axs[row, col]

        if attr_type == "Binary":
            im = ax.imshow(attr_data, cmap=cmap, origin="upper")
        elif attr_type == "Categorical":
            if attr_name == "Is Visible with Node Type":
                unique_vals = jnp.unique(attr_data)
                im = ax.imshow(attr_data, cmap=cmap, origin="upper")
            unique_vals = jnp.unique(attr_data)
            num_unique = unique_vals.size
            cmap = mcolors.ListedColormap(
                ["grey", "white", "violet", "midnightblue"][:num_unique]
            )
            im = ax.imshow(attr_data, cmap=cmap, origin="upper")
        elif attr_type == "Continuous" and attr_name != "Energy":
            im = ax.imshow(
                attr_data, cmap=cmap, origin="upper", interpolation="nearest"
            )
        else:
            im = ax.imshow(
                attr_data, cmap=cmap, origin="upper", interpolation="nearest"
            )

        ax.set_title(attr_name, fontsize=14)
        ax.set_xticks(np.arange(-0.5, space_size, 1))
        ax.set_yticks(np.arange(-0.5, space_size, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(which="major", color="black", linewidth=1)
        ax.tick_params(axis="both", which="both", length=0)  # Remove tick marks

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if attr_type == "Binary":
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["False", "True"])
        elif attr_type == "Categorical":
            if attr_name == "Is Visible with Node Type":
                cbar.set_ticks([-1, 0, 1, 2])
                cbar.set_ticklabels(["Not Visible", "Empty", "Nebula", "Asteroid"])
            cbar.set_ticks(unique_vals)
            cbar.set_ticklabels([str(val) for val in unique_vals])
        elif attr_type == "Continuous" and attr_name != "Energy":
            cbar.set_label(attr_name)
        else:
            cbar.set_label(attr_name)

    # Hide any unused subplots (if any)
    total_subplots = 9
    if len(attributes) < total_subplots:
        for idx in range(len(attributes), total_subplots):
            row = idx // 3
            col = idx % 3
            axs[row, col].axis("off")
    plt.tight_layout()
    plt.show(block=True)

    # _, ax = plt.subplots()
    # extent = [0, space.node_type.shape[1], space.node_type.shape[0], 0]
    # ax.imshow(
    #     space.node_type,
    #     cmap=node_type_cmap,
    #     extent=extent,
    #     aspect="equal",
    # )
    # ax.set_xticks(np.arange(-0, space_size, 1))
    # ax.set_yticks(np.arange(-0, space_size, 1))
    # ax.set_xticklabels(
    #     np.arange(0, space_size, 1), fontsize=8, rotation=90, ha="center", va="top"
    # )
    # ax.set_yticklabels(
    #     np.arange(0, space_size, 1), fontsize=8, rotation=90, va="center", ha="right"
    # )
    # ax.grid(which="major", color="black")

    # plt.pcolormesh(space.node_type, cmap=node_type_cmap)
    # plt.colorbar()
    # ax2 = plt.gca()
    # ax2.set_aspect("equal")

    # plt.show(block=True)


if __name__ == "__main__":
    env_cfg = game_env_cfg
    obs = obs_game_step_95
    # agent = Agent("player_0", env_cfg)
    # actions = agent.act(95, obs)

    # space = agent.state.space
    # visualize_space(space)
    # visualize_space(space)
