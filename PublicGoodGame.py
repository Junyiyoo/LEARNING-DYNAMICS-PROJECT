from StochasticGame import StochasticGame


class PublicGoodGame(StochasticGame):
    def __init__(self, rs: list[float]):
        super()
        self.rs = rs

        # Definition of transition function by a vector
        self.q[0] = 1 # only if all players cooperate the game stay in State 1
        for i in range(len(self.N)):
            self.q[i] = 0

    def payoff_function(self, state: int, actions: list[int], amount: list[float]) -> list[float]:
        cooperating_players = actions.count(True)
        rewards = []
        for i,action in enumerate(actions):
            if action: # True = Cooperation
                reward = ((cooperating_players + 1) / len(self.N)) * self.rs[state] * amount[i] - amount[i]
            else:
                reward = (cooperating_players / len(self.N)) * self.rs[state] * amount[i]
            rewards.append(reward)
        return rewards