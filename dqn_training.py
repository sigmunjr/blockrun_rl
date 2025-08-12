import sys

import torch
from torch import nn

from blockrun import BlockRun, visualize, visualize_course, play_game


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, (5, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 64, (5, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, (5, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 128, (13, 5), padding='valid'),
            nn.ReLU(),
            nn.Conv2d(128, 4, (1, 1)),
        )

    def forward(self, x):
        return self.network(x).mean(dim=(2, 3))


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0

    def add(self, experience):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.buffer_size


def train(should_visualize=False):
    q_network = QNetwork().cuda()
    target_network = QNetwork().cuda()

    block_run = BlockRun(2_000_000)

    optimizer = torch.optim.Adam(q_network.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()

    rewards_last_n_steps = 0
    last_x_pos = 5
    discount_factor = 0.99
    for i in range(1000_000):
        view = block_run.render()
        view = view.float()[None]
        q_values = q_network(view)

        if torch.rand(1) < 0.1:
            action = torch.randint(0, 4, (1,))
        else:
            action = torch.argmax(q_values)

        block_run.step(action.squeeze())
        next_view = block_run.render()
        next_view = next_view.float()[None]
        next_q_values = target_network(next_view)
        target_q_values = q_values.clone()
        target_q_values[0, action] = block_run.current_reward + discount_factor * torch.max(next_q_values)
        rewards_last_n_steps += block_run.current_reward
        loss = loss_fn(q_values, target_q_values)
        optimizer.zero_grad()
        # Loss clipping
        # loss = torch.clamp(loss, -1, 1)
        loss.backward()
        optimizer.step()
        if should_visualize:
            visualize(view[0].to(bool))
            print(f'Q values for actions:')
            print('UP:', q_values[0, 0].item(), )
            print('RIGHT:', q_values[0, 1].item(), )
            print('DOWN:', q_values[0, 2].item(), )
            print('LEFT:', q_values[0, 3].item(), )
        if i % 1000 == 0:
            target_network.load_state_dict(q_network.state_dict())
            torch.save(q_network.state_dict(), 'q_network.pt')
            print(
                f'Iteration: {i}, Loss: {loss.item()}, rewards_last_n_steps: {rewards_last_n_steps} x_pos steps: {block_run.position[0] - last_x_pos}')
            last_x_pos = block_run.position[0]
            rewards_last_n_steps = 0
        if i % 1000 == 0:
            print(
                f'Iteration: {i}, Loss: {loss.item()} score: {block_run.score} x pos: {block_run.position[0]} y pos: {block_run.position[1]}')


if __name__ == '__main__':
    if sys.argv[1] == 'viz':
        visualize_course()
    elif sys.argv[1] == 'train_viz':
        train(should_visualize=True)
    elif sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'play':
        play_game()
