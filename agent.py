import os.path

import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self, load:bool = False):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft if used
        self.model = Linear_QNet(13, 500, 3)
        if load:
            self.model.load_state_dict(torch.load(os.path.join('./model', 'model.pth')))
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y) # 20 is block size
        point_r = Point(head.x + 20, head.y)
        point_d = Point(head.x, head.y + 20)
        point_u = Point(head.x, head.y - 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        def danger_straight(incr=0):
            return ((dir_r and game.is_collision(point_r)) or
                (dir_d and game.is_collision(point_d)) or
                (dir_u and game.is_collision(point_u)) or
                (dir_l and game.is_collision(point_l)))

        def snake_on_side(action):
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            if action == [0, 1, 0]:
                side = clock_wise[(clock_wise.index(game.direction)+1) % 4]
            if action == [0, 0, 1]:
                side = clock_wise[(clock_wise.index(game.direction)-1) % 4]
            boundary = Point(0, 0)
            start = Point(0, 0)

            if side == Direction.UP:
                start = Point(game.head.x, game.head.y-20)
                boundary = Point(game.head.x, 0)
            if side == Direction.DOWN:
                start = Point(game.head.x, game.head.y+20)
                boundary = Point(game.head.x, game.h)
            if side == Direction.LEFT:
                start = Point(game.head.x-20, game.head.y)
                boundary = Point(0, game.head.y)
            if side == Direction.UP:
                start = Point(game.head.x+20, game.head.y)
                boundary = Point(game.w, game.head.y)

            if side in (Direction.UP, Direction.DOWN):
                for y in range(int(start[1]), int(boundary[1]), 20):
                    pt = Point(head.x, y)
                    if pt in game.snake:
                        return True

            if side in (Direction.LEFT, Direction.RIGHT):
                for x in range(int(start[0]), int(boundary[0]), 20):
                    pt = Point(x, head.y)
                    if pt in game.snake:
                        return True
            return False

        state = [
            # Danger straight
            danger_straight(),

            # Danger right
            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),

            # Danger left
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,

            danger_straight() and snake_on_side([0, 1, 0]),
            danger_straight() and snake_on_side([0, 0, 1])
        ]

        # print(state)

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

        # alternative way, but slower with pytorch
        # for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model( state0.to(torch.device('cuda:0')))
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(load: bool):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(False if load is False else True)
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        #get move
        final_move = agent.get_action(state_old)

        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            agent.model.save()
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    print('Load the previous model? (y/n)')
    load = False if input() == 'n' else True
    train(load)
