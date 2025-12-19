from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection


class MatrixGame:
    """
    Encapsulates common 2x2 symmetric and asymmetric games.
    Payoff matrices are shaped (2, 2, 2), where payoff[i, j, k] gives the payoff for player i
    when player 0 chooses action j and player 1 chooses action k.
    """

    STAG_HUNT = np.array([
        [[1.0, 0.0], [2/3, 2/3]],
        [[1.0, 2/3], [0.0, 2/3]]
    ], dtype=np.float64)

    SUBSIDY_GAME = np.array([
        [[12, 0], [11, 10]],
        [[12, 11], [0, 10]] 
    ], dtype=np.float64)

    MATCHING_PENNIES = np.array([
        [[0, 1], [1, 0]], 
        [[1, 0], [0, 1]] 
    ], dtype=np.float64)

    PRISONERS_DILEMMA = np.array([
        [[3, 0], [5, 1]],
        [[3, 5], [0, 1]]
    ], dtype=np.float64)

    @staticmethod
    def generate_policy(x: float, y: float) -> np.ndarray:
        """
        Generates a mixed policy matrix given probabilities x and y.
        Returns a 2x2 matrix:
            [[x, 1 - x],  (player 0 strategy)
             [y, 1 - y]]  (player 1 strategy)
        """
        return np.array([[x, 1 - x], [y, 1 - y]], dtype=np.float64)

    @staticmethod
    def generate_policy_grid(grid_shape: tuple[int, int]) -> np.ndarray:
        """
        Generates a grid of policy matrices over [0,1]x[0,1] space.
        Returns a numpy array of shape (rows * cols, 2, 2).
        """
        grid = DiGrid.generate_grid_centers(grid_shape)
        return np.array([MatrixGame.generate_policy(x, y) for x, y in grid])


class DiGrid:
    """
    Utility functions for generating and transforming 2D grid-based strategy data.
    """

    @staticmethod
    def generate_grid_centers(grid_shape: tuple[int, int]) -> list[tuple[float, float]]:
        """
        Computes center coordinates for each cell in a uniform [0,1] x [0,1] grid.

        Parameters:
            grid_shape (tuple[int, int]): Grid dimensions (rows, cols).

        Returns:
            list of (x, y): Center coordinates of each grid cell.
        """
        rows, cols = grid_shape
        return [((2 * col + 1) / (2 * cols), (2 * row + 1) / (2 * rows))
                for row in range(rows) for col in range(cols)]

    @staticmethod
    def unify_vector(dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalizes 2D vectors (dx, dy) to unit length in-place. Zero-magnitude vectors remain unchanged.

        Parameters:
            dx, dy (np.ndarray): Arrays representing vector components.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Normalized dx, dy.
        """
        assert dx.shape == dy.shape
        magnitude = np.sqrt(dx**2 + dy**2)
        magnitude[magnitude == 0] = 1  # Prevent division by zero
        dx /= magnitude
        dy /= magnitude
        return dx, dy

    @staticmethod
    def replicator_dynamics(
        payoff: np.ndarray,
        grid_shape: tuple[int, int],
        *,
        use_unit_vector: bool = False
    ) -> np.ndarray:
        """
        Computes the replicator dynamics vector field for a symmetric 2x2 game.

        Parameters:
            payoff (np.ndarray): (2, 2, 2) payoff matrix. payoff[0] for player 1, payoff[1] for player 2.
            grid_shape (tuple[int, int]): Grid resolution (rows, cols).
            use_unit_vector (bool): Normalize the output vectors to unit length.

        Returns:
            np.ndarray: Array of shape (4, *grid_shape): [x, y, dx, dy].
        """
        assert payoff.shape == (2, 2, 2), "Payoff matrix must be shape (2, 2, 2)"

        x, y = np.meshgrid(*[np.linspace(1/(2*s), (2*s-1)/(2*s), s) for s in grid_shape])

        a, b = payoff[0], payoff[1]
        a11, a12, a21, a22 = a[0, 0], a[0, 1], a[1, 0], a[1, 1]
        b11, b12, b21, b22 = b[0, 0], b[0, 1], b[1, 0], b[1, 1]

        dx = x * (1 - x) * ((a11 - a21) * y + (a12 - a22) * (1 - y))
        dy = y * (1 - y) * ((b11 - b12) * x + (b21 - b22) * (1 - x))

        if use_unit_vector:
            dx, dy = DiGrid.unify_vector(dx, dy)

        return np.array([x, y, dx, dy])

    @staticmethod
    def boltzmann_replicator_dynamics(
        payoff: np.ndarray,
        grid_shape: tuple[int, int],
        *,
        temperature: float = 0.1,
        use_unit_vector: bool = False
    ) -> np.ndarray:
        """
        Computes the Boltzmann Replicator Dynamics for a 2x2 game.

        Parameters:
            payoff (np.ndarray): (2, 2, 2) payoff matrix.
            grid_shape (tuple[int, int]): Grid resolution.
            temperature (float): Softmax temperature. T→0 recovers standard replicator dynamics.
            use_unit_vector (bool): Normalize output vectors.

        Returns:
            np.ndarray: Array of shape (4, *grid_shape): [x, y, dx, dy].
        """
        assert payoff.shape == (2, 2, 2)
        assert temperature >= 0

        x, y = np.meshgrid(*[np.linspace(1/(2*s), (2*s-1)/(2*s), s) for s in grid_shape])
        eps = 1e-10  # numerical stability

        a, b = payoff[0], payoff[1]
        a11, a12, a21, a22 = a[0, 0], a[0, 1], a[1, 0], a[1, 1]
        b11, b12, b21, b22 = b[0, 0], b[0, 1], b[1, 0], b[1, 1]

        dx = x * (1 - x) * ((a11 - a21) * y + (a12 - a22) * (1 - y) - temperature * np.log((x + eps) / (1 - x + eps)))
        dy = y * (1 - y) * ((b11 - b12) * x + (b21 - b22) * (1 - x) - temperature * np.log((y + eps) / (1 - y + eps)))

        if use_unit_vector:
            dx, dy = DiGrid.unify_vector(dx, dy)

        return np.array([x, y, dx, dy])

    @staticmethod
    def lenient_boltzmann_replicator_dynamics(
        payoff: np.ndarray,
        grid_shape: tuple[int, int],
        *,
        temperature: float = 0.1,
        kappa: int = 3,
        use_unit_vector: bool = False
    ) -> np.ndarray:
        """
        Computes Lenient Boltzmann Replicator Dynamics for a 2x2 game.

        Parameters:
            payoff (np.ndarray): (2, 2, 2) payoff matrix.
            grid_shape (tuple[int, int]): Grid resolution.
            temperature (float): Softmax temperature.
            kappa (int): Leniency exponent. Higher = more tolerant.
            use_unit_vector (bool): Normalize output vectors.

        Returns:
            np.ndarray: Array of shape (4, *grid_shape): [x, y, dx, dy].
        """
        assert payoff.shape == (2, 2, 2)
        assert temperature >= 0
        eps = 1e-10

        x, y = np.meshgrid(*[np.linspace(1/(2*s), (2*s-1)/(2*s), s) for s in grid_shape])
        x1, x2, y1, y2 = x, 1 - x, y, 1 - y

        a = payoff[0]
        b = payoff[1].T
        a11, a12, a21, a22 = a[0, 0], a[0, 1], a[1, 0], a[1, 1]
        b11, b12, b21, b22 = b[0, 0], b[0, 1], b[1, 0], b[1, 1]
        
        u1 = (
            a11 * y1 * ((y1 + y2 * (a11>=a12))**kappa - (y2 * (a11 > a12))**kappa) / (y1 + y2 * (a11 == a12) + eps)
            + a12 * y2 * ((y2 + y1 * (a11<=a12))**kappa - (y1 * (a11 < a12))**kappa) / (y2 + y1 * (a11 == a12) + eps)
        )
        u2 = (
            a21 * y1 * ((y1 + y2 * (a21>=a22))**kappa - (y2 * (a21 > a22))**kappa) / (y1 + y2 * (a21 == a22) + eps)
            + a22 * y2 * ((y2 + y1 * (a21<=a22))**kappa - (y1 * (a21 < a22))**kappa) / (y2 + y1 * (a21 == a22) + eps)
        )
        w1 = (
            b11 * x1 * ((x1 + x2 * (b11>=b12))**kappa - (x2 * (b11 > b12))**kappa) / (x1 + x2 * (b11 == b12) + eps)
            + b12 * x2 * ((x2 + x1 * (b11<=b12))**kappa - (x1 * (b11 < b12))**kappa) / (x2 + x1 * (b11 == b12) + eps)
        )
        w2 = (
            b21 * x1 * ((x1 + x2 * (b21>=b22))**kappa - (x2 * (b21 > b22))**kappa) / (x1 + x2 * (b21 == b22) + eps)
            + b22 * x2 * ((x2 + x1 * (b21<=b22))**kappa - (x1 * (b21 < b22))**kappa) / (x2 + x1 * (b21 == b22) + eps)
        )

        dx = x * (1 - x) * (u1 - u2 - temperature * np.log((x + eps) / (1 - x + eps)))
        dy = y * (1 - y) * (w1 - w2 - temperature * np.log((y + eps) / (1 - y + eps)))

        if use_unit_vector:
            dx, dy = DiGrid.unify_vector(dx, dy)

        return np.array([x, y, dx, dy])


class QL:

    class QLog:
        """
        Logs the Q-values and policies for each iteration of learning.

        Attributes:
            q_values (np.ndarray): Array of shape (num_iterations + 1, 2, 2) storing Q-values.
            policy (np.ndarray): Array of shape (num_iterations + 1, 2, 2) storing computed policies.
            _len (int): The current number of logged iterations.
        """
        def __init__(self, num_iterations: int):
            self.q_values: np.ndarray = np.zeros((num_iterations + 1, 2, 2), dtype=np.float64)
            self.policy: np.ndarray = np.zeros((num_iterations + 1, 2, 2), dtype=np.float64)
            self._len: int = 0

        def log(self, q_values: np.ndarray, policy: np.ndarray) -> None:
            """
            Logs the current Q-values and policy, then increments the counter.

            Args:
                q_values (np.ndarray): Q-values for the current iteration.
                policy (np.ndarray): Policy for the current iteration.
            """
            self.q_values[self._len] = q_values
            self.policy[self._len] = policy
            self._len += 1


    class QLogList:
        """
        Container for a list of QLog objects with methods to compute statistics (mean, median) over the logs.
        """
        def __init__(self, logs: list[QL.QLog] = None):
            if logs is None:
                logs = []
            self.logs: list[QL.QLog] = logs

        def append(self, q_log: QL.QLog) -> None:
            """Appends a QLog instance to the list."""
            self.logs.append(q_log)

        def mean(self) -> QL.QLog:
            """Aggregates the Q_logs using the mean method."""
            return self.aggregate(method="mean")

        def median(self) -> QL.QLog:
            """Aggregates the Q_logs using the median method."""
            return self.aggregate(method="median")

        def aggregate(self, method: str = "mean") -> QL.QLog:
            """
            Aggregates the Q_logs in this list using the specified method ('mean' or 'median').

            Assumes that all Q_logs have the same number of iterations and matching array shapes.

            Args:
                method (str): Aggregation method to use, either 'mean' (default) or 'median'.

            Returns:
                QL.QLog: A new QLog object with each logged entry as the aggregated (mean or median)
                          corresponding entry from the list of Q_logs.
            """
            if not self.logs:
                raise ValueError("No Q_logs to compute aggregation.")

            if method not in {"mean", "median"}:
                raise ValueError(f"Unsupported method '{method}'. Use 'mean' or 'median'.")

            # Stack logged arrays: shape becomes (num_logs, num_iterations+1, 2, 2)
            q_values_stack = np.stack([log.q_values for log in self.logs], axis=0)
            policy_stack = np.stack([log.policy for log in self.logs], axis=0)

            # Choose aggregation function
            agg_func = np.mean if method == "mean" else np.median

            # Compute aggregation over the first axis (across logs)
            agg_q_values = agg_func(q_values_stack, axis=0)
            agg_policy = agg_func(policy_stack, axis=0)

            # Create a new QLog to hold the aggregated result.
            num_iterations = agg_q_values.shape[0] - 1
            agg_log = QL.QLog(num_iterations=num_iterations)
            agg_log.q_values = agg_q_values.copy()
            agg_log.policy = agg_policy.copy()
            agg_log._len = agg_q_values.shape[0]  # Set the log length appropriately

            return agg_log


    class QPlot:
        """
        Contains static methods related to plotting Q-learning progress.
        """
        @staticmethod
        def trace(ax: Axes, q_log: QL.QLog, *, cmap: str = 'viridis') -> None:
            """
            Draws a trace line of policy changes over iterations on the provided axes.

            Args:
                ax (Axes): The matplotlib axes to draw on.
                q_log (QL.QLog): The log containing Q-values and policies.
                cmap (str): The colormap to use for the trace. Defaults to 'viridis'.
            """
            t_x, t_y = q_log.policy[:, 0, 0], q_log.policy[:, 1, 0]
            
            # Create line segments from the x and y coordinates
            points = np.array([t_x, t_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            norm = plt.Normalize(0, len(t_x) - 1)
            
            # Reverse specific colormaps for better tail/head visualization
            if cmap in ['viridis', 'plasma', 'inferno', 'magma', 'cividis']:
                cmap = cmap + "_r"
                
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(np.arange(len(t_x)))
            lc.set_linewidth(1)

            ax.add_collection(lc)

        @staticmethod
        def summary(qLog: QL.QLog):
            q_values_history, policy_history = qLog.q_values, qLog.policy
            fig, axes = plt.subplots(2, 2, figsize=(8, 6))

            # --- Agent 1 Q-values (top-left) ---
            axes[0, 0].plot(q_values_history[:, 0, 0], label='Q(action 1)')
            axes[0, 0].plot(q_values_history[:, 0, 1], label='Q(action 2)', linestyle='dotted')
            axes[0, 0].set_title('Agent 1 Q-values')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Q-value')
            axes[0, 0].legend()

            # --- Agent 1 Policy (top-right) ---
            # policy_history[:, 0] has shape (time, action), so we plot each action's probability
            axes[0, 1].plot(policy_history[:, 0, 0], label='P(action 1)')
            axes[0, 1].plot(policy_history[:, 0, 1], label='P(action 2)', linestyle='dotted')
            axes[0, 1].set_title('Agent 1 Policy')
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Probability')
            axes[0, 1].legend()

            # --- Agent 2 Q-values (bottom-left) ---
            axes[1, 0].plot(q_values_history[:, 1, 0], label='Q(action 1)')
            axes[1, 0].plot(q_values_history[:, 1, 1], label='Q(action 2)', linestyle='dotted')
            axes[1, 0].set_title('Agent 2 Q-values')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Q-value')
            axes[1, 0].legend()

            # --- Agent 2 Policy (bottom-right) ---
            axes[1, 1].plot(policy_history[:, 1, 0], label='P(action 1)')
            axes[1, 1].plot(policy_history[:, 1, 1], label='P(action 2)', linestyle='dotted')
            axes[1, 1].set_title('Agent 2 Policy')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Probability')
            axes[1, 1].legend()

            plt.tight_layout()

    @staticmethod
    def epsilon_greedy_q_learning(
            payoff: np.ndarray,
            num_iterations: int = 100, 
            alpha: float = 0.1, 
            epsilon: float = 0.3, 
            init_q_values: np.ndarray | None = None,
            adjust_frequencey: bool = False,
        ) -> QL.QLog:
        """
        Runs epsilon-greedy Q-learning given a payoff matrix.

        Args:
            payoff (np.ndarray): The payoff matrix.
            num_iterations (int): Number of learning iterations.
            alpha (float): Learning rate.
            epsilon (float): Exploration factor.
            init_q_values (np.ndarray, optional): Initial Q-values array of shape (2, 2). 
                                                  Initialized to zeros if None.
            adjust_frequencey (bool): If True, adjusts updates by the policy probabilities.

        Returns:
            QL.QLog: A log containing Q-values and policies for each iteration.
        """
        q_values = np.zeros((2, 2)) if init_q_values is None else init_q_values.copy()
        assert q_values.shape == (2, 2), "init_q_values must have shape (2, 2)"

        q_log = QL.QLog(num_iterations=num_iterations)
        for i in range(num_iterations):
            # Determine policy using epsilon-greedy strategy
            best_actions = np.argmax(q_values, axis=1)
            policy = np.full((2, 2), epsilon / 2)
            policy[[0, 1], best_actions] += (1 - epsilon)
            
            # Log current Q-values and policy
            q_log.log(q_values=q_values, policy=policy)

            # Sample actions using the cumulative distribution method
            cum_policy = np.cumsum(policy, axis=1)
            random_vals = np.random.rand(policy.shape[0])
            actions = (cum_policy > random_vals[:, None]).argmax(axis=1)

            # Get rewards for the chosen actions
            rewards = payoff[[0, 1], actions[0], actions[1]]

            # Update Q-values
            delta_q = alpha * (rewards - q_values[[0, 1], actions])
            if adjust_frequencey:
                delta_q /= policy[[0, 1], actions]
            q_values[[0, 1], actions] += delta_q

        else:
            # Log final values after last update to ensure proper log length
            q_log.log(q_values=q_values, policy=policy)

        return q_log


    @staticmethod
    def boltzmann_q_learning(
            payoff: np.ndarray,
            num_iterations: int = 100, 
            alpha: float = 0.1, 
            temperature: float = 1, 
            init_q_values: np.ndarray | None = None,
            adjust_frequencey: bool = True,
        ) -> QL.QLog:
        """
        Runs Boltzmann Q-learning given a payoff matrix.

        Args:
            payoff (np.ndarray): ndarray of shape (2, 2, 2) where payoff[p, a, b] is the reward for player p.
            num_iterations (int): Number of iterations for learning.
            alpha (float): Learning rate.
            temperature (float): Parameter for Boltzmann (softmax) exploration.
            init_q_values (np.ndarray, optional): Initial Q-values array of shape (2, 2). Defaults to zeros if None.
            adjust_frequencey (bool): If True, adjusts updates by the policy probabilities.

        Returns:
            QL.QLog: A log with the Q-values and policies for each iteration.
        """
        # Initialize Q-values
        q_values = QL.generate_mean_q_values(payoff, temperature, np.ones((2, 2)) / 2.0) if init_q_values is None else init_q_values.copy()
        assert q_values.shape == (2, 2), "init_q_values must have shape (2, 2)"

        q_log = QL.QLog(num_iterations=num_iterations)
        for i in range(num_iterations):
            # Compute policy using softmax on each player's Q-values
            q_values_shifted = q_values - np.max(q_values, axis=1, keepdims=True)
            exp_q = np.exp(q_values_shifted / temperature)
            sum_exp_q = np.sum(exp_q, axis=1, keepdims=True)
            policy = exp_q / sum_exp_q

            # Log current state
            q_log.log(q_values=q_values.copy(), policy=policy.copy())

            # Determine actions using cumulative distribution method
            cum_policy = np.cumsum(policy, axis=1)
            random_vals = np.random.rand(policy.shape[0])
            actions = (cum_policy > random_vals[:, np.newaxis]).argmax(axis=1)

            # Get rewards corresponding to chosen actions
            rewards = payoff[[0, 1], actions[0], actions[1]]

            # Update Q-values for chosen actions
            delta_q = alpha * (rewards - q_values[[0, 1], actions])
            if adjust_frequencey:
                delta_q /= policy[[0, 1], actions]
            q_values[[0, 1], actions] += delta_q
        else:
            # Log final state
            q_log.log(q_values=q_values, policy=policy)

        return q_log

    
    @staticmethod
    def lenient_boltzmann_q_learning(
            payoff: np.ndarray,
            num_iterations: int = 100, 
            alpha: float = 0.1,
            temperature: float = 1, 
            kappa: int = 3,
            init_q_values: np.ndarray | None = None,
            adjust_frequencey: bool = True,
        ) -> QL.QLog:
        """
        Runs lenient Boltzmann Q-learning given a payoff matrix.

        Args:
            payoff (np.ndarray): ndarray of shape (2, 2, 2) where payoff[p, a, b] is the reward for player p.
            num_iterations (int): Number of iterations for the learning process.
            alpha (float): Learning rate.
            temperature (float): Parameter for Boltzmann (softmax) exploration.
            kappa (int): Leniency parameter for scaling the decay of negative updates.
            init_q_values (np.ndarray, optional): Initial Q-values array of shape (2, 2). Defaults to zeros if None.
            adjust_frequencey (bool): If True, adjusts updates based on policy probabilities.

        Returns:
            QL.QLog: A log containing the Q-values and policies for each iteration.
        """
        # Initialize Q-values
        q_values = QL.generate_mean_q_values(payoff, temperature, np.ones((2, 2)) / 2.0) if init_q_values is None else init_q_values.copy()
        assert q_values.shape == (2, 2), "init_q_values must have shape (2, 2)"
        
        # Initialize the reward buffer for each (player, action)
        action_reward_buffer = np.empty((2, 2, kappa), dtype=np.float64)
        action_reward_buffer[:] = np.max(payoff)
        action_reward_buffer_index = np.zeros((2, 2), dtype=np.uint)

        q_log = QL.QLog(num_iterations=num_iterations)
        for i in range(num_iterations):
            # Compute policy using softmax for each player's Q-values
            q_values_shifted = q_values - np.max(q_values, axis=1, keepdims=True)
            exp_q = np.exp(q_values_shifted / temperature)
            sum_exp_q = np.sum(exp_q, axis=1, keepdims=True)
            policy = exp_q / sum_exp_q

            # Log current state
            q_log.log(q_values=q_values.copy(), policy=policy.copy())

            # Sample actions using the cumulative distribution method
            cum_policy = np.cumsum(policy, axis=1)
            random_vals = np.random.rand(policy.shape[0])
            actions = (cum_policy > random_vals[:, np.newaxis]).argmax(axis=1)

            # Obtain rewards from the payoff matrix
            rewards = payoff[[0, 1], actions[0], actions[1]]

            # Update the reward buffer for the chosen actions
            buffer_index = action_reward_buffer_index[[0, 1], actions]
            action_reward_buffer[[0, 1], actions, buffer_index] = rewards
            action_reward_buffer_index[[0, 1], actions] += 1
            action_reward_buffer_index[[0, 1], actions] %= kappa

            # Determine the maximum reward stored in the buffer for the chosen actions
            max_rewards = np.max(action_reward_buffer[[0, 1], actions], axis=1)
            update_signal = rewards >= max_rewards

            # Update Q-values only for those actions meeting the update condition
            delta_q = update_signal * alpha * (rewards - q_values[[0, 1], actions])
            if adjust_frequencey:
                delta_q /= policy[[0, 1], actions]
            q_values[[0, 1], actions] += delta_q
        else:
            # Log final state after last update
            q_log.log(q_values=q_values, policy=policy)

        return q_log

    
    @staticmethod
    def generate_mean_q_values(
            payoff: np.ndarray, 
            temperature: float, 
            init_policy: np.ndarray
        ) -> np.ndarray:
        """
        Generates initial Q-values based on a payoff matrix, temperature parameter, and initial policy.

        Args:
            payoff (np.ndarray): A 3D array of shape (2, 2, 2) representing the payoff values.
            temperature (float): A positive scalar that scales the influence of the log odds.
            init_policy (np.ndarray): A 2D array of shape (2, 2) representing the initial policy probabilities.
                                      Each element must be in the open interval (0, 1).

        Returns:
            np.ndarray: A 2D array of shape (2, 2) containing the computed initial Q-values.
        """
        # Validate inputs
        if payoff.shape != (2, 2, 2):
            raise ValueError("`payoff` must have shape (2, 2, 2).")
        if temperature <= 0:
            raise ValueError("`temperature` must be greater than 0.")
        if init_policy.shape != (2, 2) or not np.all((init_policy > 0) & (init_policy < 1)):
            raise ValueError("`init_policy` must have shape (2, 2) with all entries in (0, 1).")

        # Compute the scaled log odds ratio.
        log_ratio = np.log(init_policy[:, 0] / init_policy[:, 1])
        delta_q = temperature * log_ratio / 2.0

        # Compute the mean Q-values using Einstein summation.
        mean_q = np.einsum("pxy,x,y->p", payoff, init_policy[0], init_policy[1])

        # Form the initial Q-values combining the mean and delta adjustments.
        init_q_values = np.vstack((mean_q + delta_q, mean_q - delta_q)).T

        return init_q_values


