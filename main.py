class Easy21:
    def __init__(self):
        # print("Let's start to play a game of Easy21!")
        self.player_sum = random.randint(1, 10)
        self.dealer_sum = random.randint(1, 10)

    def reset(self):
        self.__init__()

    def step(self, action):
        # print("PLAYER - " + action)

        def hit_me():
            card_number = random.randint(1, 10)
            card_color = random.choice(("red", "black", "black"))
            # print("\tCARD IS:" + card_color + str(card_number))
            if card_color == "red":
                return -card_number
            elif card_color == "black":
                return card_number
            else:
                print("Illegal card color")
                exit(1)

        def is_bust(x):
            return not (1 <= x <= 21)

        if action == "hit":
            self.player_sum += hit_me()
            if is_bust(self.player_sum):
                return "LOSE", -1
            else:
                return "TURN", 0
        elif action == "stick":
            while (self.dealer_sum < 17) and (not is_bust(self.dealer_sum)):
                # print("Dealer - HIT")
                self.dealer_sum += hit_me()
                # print("P:", self.player_sum, "D:", self.dealer_sum)
            if is_bust(self.dealer_sum):
                return "WIN", 1
            else:
                # print("Dealer - STICK")
                if self.player_sum > self.dealer_sum:
                    return "WIN", 1
                elif self.player_sum < self.dealer_sum:
                    return "LOSE", -1
                elif self.player_sum == self.dealer_sum:
                    return "DRAW", 0
                else:
                    print("shouldn't be here.")
                    exit(1)
        else:
            print("Illegal Action")
            exit(1)


class Agent:
    def __init__(self, constant_eps=None, constant_step=None):
        self.state_actions = self.get_empty_sa_dictionary()
        self.e_traces = self.get_empty_sa_dictionary()
        self.N0 = 100
        self.constant_eps = constant_eps
        self.constant_step = constant_step
        self.total_reward = 0
        self.total_episodes = 0

    def plot_value_function(self, title="qvalue_surface", show_result=False):
        # This import registers the 3D projection, but is otherwise unused.
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import numpy as np

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        X = np.arange(1, 22, 1.0)
        Y = np.arange(1, 11, 1.0)
        X, Y = np.meshgrid(X, Y)
        Z = X.copy()

        for i in range(21):
            for j in range(10):
                best_action = self.get_greedy_action((i + 1, j + 1))
                if best_action == "hit":
                    value = 1
                else:
                    value = 0
                hit_value = self.state_actions[(i + 1, j + 1)]["hit"]["value"]
                stick_value = self.state_actions[(i + 1, j + 1)]["stick"]["value"]
                Z[j][i] = max(hit_value, stick_value)
                # print(i + 1, j + 1, Z[j][i], hit_value, stick_value, best_action)
        # print(Z)
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(-1.01, 1.01)
        ax.set_xlim(1, 21)
        ax.set_xticks([i for i in range(1, 22, 2)])
        ax.set_yticks([i for i in range(1, 11, 1)])
        ax.set_ylim(1, 10)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_ylabel("Dealer showing")
        ax.set_xlabel("Player's sum")
        ax.set_zlabel("Value")

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.view_init(15, 120)
        plt.savefig(open(title + ".svg", "wb"), format="svg")
        if show_result:
            plt.show()
        return

    @staticmethod
    def get_empty_sa_dictionary():
        d = {}
        for i in range(1, 22, 1):
            for j in range(1, 11, 1):
                d[(i, j)] = {}
                for k in ["hit", "stick"]:
                    d[(i, j)][k] = {"value": 0, "count": 0}
        return d.copy()

    def count_state_visits(self, state):
        state_value = self.state_actions[state]
        return state_value["hit"]["count"] + state_value["stick"]["count"]

    def get_action(self, state, learn):
        if self.constant_eps is None:
            # epsilon greedy
            cnt = self.count_state_visits(state)
            eps = (1.0 * self.N0) / (self.N0 + cnt)
        else:
            eps = self.constant_eps
        rnd = random.random()
        # print("EPS", rnd, eps)
        if random.random() < eps and learn:
            return RandomAgent.get_random_action()
        else:
            return self.get_greedy_action(state)

    def get_greedy_action(self, state):
        # state_value = self.state_actions[state]
        if self.get_state_action_value(state, "hit") >= self.get_state_action_value(state, "stick"):
            return "hit"
        else:
            return "stick"

    @staticmethod
    def get_random_action():
        return random.choice(("hit", "stick"))

    def episode_init(self):
        return

    def get_state_action_value(self, state, action):
        return self.state_actions[state][action]["value"]

    def play_one_episode(self, game, learn=True):
        game.reset()
        self.episode_init()
        moves = []
        status = "TURN"
        start_state = (game.player_sum, game.dealer_sum)
        # print(start_state)
        start_action = self.get_action(start_state, learn=learn)

        while True:
            # print("P:", E.player_sum, "D:", E.dealer_sum)
            status, reward = game.step(start_action)
            if status == "TURN":
                next_state = (game.player_sum, game.dealer_sum)
                if learn:
                    next_action = self.get_action(next_state, learn=learn)
                    self.update_values(start_state, start_action, reward, next_state, next_action)
                else:
                    next_action = self.get_action(start_state, learn=learn)
                start_state = next_state
                start_action = next_action
            else:
                if learn:
                    self.update_values(start_state, start_action, reward, None, None)
                break

        # print("Game result is:", status, " with reward of:", reward)
        return reward

    def update_values(self, start_state, start_action, reward, next_state, next_action):
        return

    def count_state_visits(self, state):
        state_value = self.state_actions[state]
        return state_value["hit"]["count"] + state_value["stick"]["count"]

    def simulate(self, game, episodes):
        for i in range(episodes):
            self.total_reward += self.play_one_episode(game=game, learn=True)
            self.total_episodes += 1

    def evaluate_agent(self, game, episodes):
        score = 0
        for i in range(episodes):
            score += self.play_one_episode(game=game, learn=False)
        print(repr(self))
        print("Total score:" + str(score), "#Episodes:", episodes, " AVG:", score * 1.0 / episodes)

    @staticmethod
    def compute_mse(a, b):
        count = 0.0
        sum = 0.0
        for i in range(21):
            for j in range(10):
                for k in ["hit", "stick"]:
                    count += 1
                    value_a = a.get_state_action_value((i + 1, j + 1), k)
                    value_b = b.get_state_action_value((i + 1, j + 1), k)
                    if abs(value_a - value_b) > 4:
                        print("error is too large!", value_a, value_b)

                    sum = sum + ((value_a - value_b) ** 2)
        return sum / count

    def __repr__(self):
        return "this is an agent"


class RandomAgent(Agent):
    def get_action(self, state, learn):
        return super().get_random_action()

    def __repr__(self):
        return "this is a random acting agent"


class DealerAgent(Agent):
    def get_action(self, state, learn):
        if state[0] >= 17:
            return "stick"
        else:
            return "hit"

    def __repr__(self):
        return "this is an agent who acts like a dealer"


class AlwaysHit(Agent):
    def get_action(self, state, learn):
        return "hit"

    def __repr__(self):
        return "always hits"


class AlwaysStick(Agent):
    def get_action(self, state, learn):
        return "stick"

    def __repr__(self):
        return "always sticks"


class MonteCarloAgent(Agent):

    def update_values(self, state, action, reward, next_state, next_action):
        count = self.state_actions[state][action]["count"]
        value = self.state_actions[state][action]["value"]
        alpha = 1.0 / (1.0 + count)
        new_value = (1 - alpha) * value + (alpha * reward)
        self.state_actions[state][action]["count"] = count + 1
        self.state_actions[state][action]["value"] = new_value

    def play_one_episode(self, game, learn=True):
        game.reset()
        moves = []
        status = "TURN"
        while status == "TURN":
            state = (game.player_sum, game.dealer_sum)
            if learn:
                action = self.get_action(state, learn=learn)
            else:
                action = self.get_greedy_action(state)
            status, reward = game.step(action)
            moves.append((state, action))
        if learn:
            for m in moves:
                self.update_values(m[0], m[1], reward, None, None)
        return reward

    def __repr__(self):
        return "this is a monte-carlo trained agent"


class SARSA(Agent):

    def __init__(self, labmda_value, gamma_value, linear_approx=False, constant_eps=None, constant_step=None):
        super().__init__(constant_eps=constant_eps, constant_step=constant_step)
        self.lambda_value = labmda_value
        self.gamma_value = gamma_value
        self.linear_approx = linear_approx

        if linear_approx:
            self.feature_values = [random.random() for i in range(3 * 6 * 2)]
            self.e_traces = [0 for i in range(3 * 6 * 2)]

    def state_action_to_features(self, state, action):
        d = [0 for i in range(3)]
        p = [0 for i in range(6)]
        a = [0 for i in range(2)]

        if 1 <= state[1] <= 4:
            d[0] = 1
        if 4 <= state[1] <= 7:
            d[1] = 1
        if 7 <= state[1] <= 10:
            d[2] = 1

        if 1 <= state[0] <= 4:
            p[0] = 1
        if 4 <= state[0] <= 9:
            p[1] = 1
        if 7 <= state[0] <= 12:
            p[2] = 1
        if 10 <= state[0] <= 15:
            p[3] = 1
        if 13 <= state[0] <= 18:
            p[4] = 1
        if 16 <= state[0] <= 21:
            p[5] = 1

        if action == "hit":
            a[0] = 1
        if action == "stick":
            a[1] = 1

        features = d + p + a
        return features

    def get_state_action_value(self, state, action):
        if self.linear_approx:
            features = self.state_action_to_features(state, action)
            # print(features)
            s = 0
            for i in range(len(features)):
                s += (features[i] * self.feature_values[i])
            return s
        else:
            return super().get_state_action_value(state, action)

    def update_values(self, start_state, start_action, reward, next_state, next_action):
        if self.constant_step is None:
            count = self.state_actions[start_state][start_action]["count"]
            alpha = 1.0 / (1.0 + count)
        else:
            alpha = self.constant_step
        gamma = self.gamma_value
        s_q = self.get_state_action_value(start_state, start_action)
        if next_state is not None:
            n_q = self.get_state_action_value(next_state, next_action)
        else:
            n_q = 0
        # print(reward,gamma,n_q,s_q)
        delta = reward + gamma * n_q - s_q

        if self.linear_approx:
            features = self.state_action_to_features(start_state, start_action)
            for i in range(len(features)):
                self.e_traces[i] += features[i]
            for i in range(len(features)):
                q_update = alpha * delta * self.e_traces[i]
                if abs(q_update) > 2:
                    print(q_update, alpha, delta, self.e_traces[i])
                    exit()
                self.feature_values[i] += q_update
                self.e_traces[i] *= gamma * self.lambda_value
        else:
            self.e_traces[start_state][start_action]["value"] += 1
            for i in range(21):
                for j in range(10):
                    for k in ["hit", "stick"]:
                        q_update = alpha * delta * self.e_traces[(i + 1, j + 1)][k]["value"]
                        self.state_actions[(i + 1, j + 1)][k]["value"] += q_update
                        self.e_traces[(i + 1, j + 1)][k]["value"] *= gamma * self.lambda_value
            self.state_actions[start_state][start_action]["count"] = count + 1

        return

    def episode_init(self):
        if self.linear_approx:
            self.e_traces = [0 for i in range(36)]
        else:
            self.e_traces = super().get_empty_sa_dictionary()

    def __repr__(self):
        return "this is a SARSA trained agent." + " lambda:" + str(self.lambda_value) + " approx.:" + str(
            self.linear_approx)


def Q1():
    game = Easy21()
    R = RandomAgent()
    D = DealerAgent()
    H = AlwaysHit()
    S = AlwaysStick()

    R.evaluate_agent(game, 100 * epoch)
    D.evaluate_agent(game, 100 * epoch)
    H.evaluate_agent(game, 100 * epoch)
    S.evaluate_agent(game, 100 * epoch)


def Q2():
    game = Easy21()
    MCC = MonteCarloAgent()
    MCC.simulate(game=game, episodes=1000 * epoch)
    MCC.plot_value_function("MC_optimal_value_function", show_result=False)
    MCC.evaluate_agent(game, 100 * epoch)
    return MCC


def Q3(MCC, linear_approximation=False):
    game = Easy21()
    lambdas = [0.1 * i for i in range(11)]
    mses = []
    episodes = [10 * i for i in range(2000)]

    for l in lambdas:
        if linear_approximation:
            S = SARSA(labmda_value=l, gamma_value=1, linear_approx=True, constant_eps=0.05, constant_step=0.01)
        else:
            S = SARSA(labmda_value=l, gamma_value=1)
        S.simulate(game=Easy21(), episodes=epoch)
        err = Agent.compute_mse(MCC, S)
        mses.append(err)
    plt.figure()
    ax = plt.axes()
    ax.set_xlabel("SARSA(Lambda) - Lambda value")
    ax.set_ylabel("MSE after 1000 episodes")
    ax.plot(lambdas, mses)
    plt.savefig(open("SARSA_mse_vs_lambda_approx_" + str(linear_approximation) + ".svg", "wb"), format="svg")

    for l in [0, 1]:
        if linear_approximation:
            S = SARSA(labmda_value=l, gamma_value=1, linear_approx=True, constant_eps=0.05, constant_step=0.01)
        else:
            S = SARSA(labmda_value=l, gamma_value=1)
        mse_l = []
        for i in range(len(episodes)):
            S.simulate(game=Easy21(), episodes=10)
            err = Agent.compute_mse(MCC, S)
            mse_l.append(err)
        if l == 0 or l == 1:
            plt.figure()
            ax = plt.axes()
            ax.set_xlabel("SARSA(Lambda) - episodes")
            ax.set_ylabel("MSE")
            ax.plot(episodes, mse_l)
            plt.savefig(
                open("SARSA_mse_vs_episode_lambda_" + str(int(l)) + "_approx_" + str(linear_approximation) + ".svg",
                     "wb"),
                format="svg")


if __name__ == '__main__':
    import random
    import numpy as np
    import matplotlib.pyplot as plt

    plt.style.use('seaborn-whitegrid')
    epoch = 1000

    ###Q1 -  Implementation of Easy21###
    Q1()

    ###Q2 - Monte-Carlo Control in Easy21###
    MCC = Q2()

    ###Q3 - TD Learning in Easy21###
    Q3(MCC, linear_approximation=False)
    ###Q4 - Linear Function Approximation in Easy21###
    Q3(MCC, linear_approximation=True)
