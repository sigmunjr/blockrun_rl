import torch
try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV not available in minimal environments
    cv2 = None
import os
import numpy as np
from PIL import Image
import io
import cairosvg


ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')
TILE_SIZE = 32


def load_svg_image(name, size=TILE_SIZE):
    path = os.path.join(ASSET_DIR, name)
    png_data = cairosvg.svg2png(url=path, output_width=size, output_height=size)
    return Image.open(io.BytesIO(png_data)).convert('RGBA')


BACKGROUND_IMG = load_svg_image('background.svg')
OBSTACLE_IMG = load_svg_image('obstacle.svg')
BIRD_IMG = load_svg_image('bird.svg')
REWARD_IMG = load_svg_image('reward.svg')


class BlockRun:
    POSITION = 0
    OBSTACLE = 1
    REWARD = 2
    HISTORY = 3
    def __init__(self, length, device='cuda:0'):
        self.length = length
        self.device = device
        self.view_range = 6
        self.position = torch.tensor([self.view_range, 3], dtype=int, device=device)
        self.actions_to_step = torch.tensor(
            [[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=int, device=device
        )
        self.obstacle_prob = 0.7
        self.reward_interval = 6
        self.course = self.generate_course()
        self.score = 0
        self.current_reward = 0

    def generate_course(self):
        course = torch.zeros((4, self.length, 5), dtype=bool, device=self.device, requires_grad=False)
        obstacle_indices_full = torch.arange(1, self.length - 1, step=2, device=self.device)
        selected_obstacles = torch.rand(obstacle_indices_full.shape[0], device=self.device) < self.obstacle_prob
        obstacle_indices = obstacle_indices_full[selected_obstacles]
        obstacle = torch.ones((obstacle_indices.shape[0], 5), dtype=bool, device=self.device)
        obstacle_erase_steps = 3
        for _ in range(obstacle_erase_steps):
            obstacle_whole = torch.randint(0, 5, (obstacle_indices.shape[0],), device=self.device)
            obstacle[torch.arange(obstacle_indices.shape[0], device=self.device), obstacle_whole] = False
        course[BlockRun.OBSTACLE, obstacle_indices, :] = obstacle
        avg_nr_rewards = self.length // self.reward_interval
        nr_rewards = torch.randint(int(0.8 * avg_nr_rewards), int(1.2 * avg_nr_rewards), (1,), device=self.device)
        reward_indices = torch.randperm(obstacle_indices.shape[0], device=self.device)[:nr_rewards]
        course[BlockRun.REWARD, obstacle_indices[reward_indices], obstacle_whole[reward_indices]] = True
        return course

    def step(self, action):
        with torch.no_grad():
            position = self.position + self.actions_to_step[action]
            # print(f'Position 1: {position}')
            position[0] = torch.clamp(position[0], torch.tensor(self.view_range, device=self.device), torch.tensor(self.length - self.view_range, device=self.device))
            # print(f'Position 2: {position}')
            position[1] = torch.clamp(position[1],
                                           0, 4)
            if action == 1:
                self.current_reward = 0
            else:
                self.current_reward = -1
            # print(f'Position 3: {position}')
            if self.course[BlockRun.HISTORY, position[0], position[1]]:
                self.score -= 1
                self.current_reward -= 1
            if self.course[BlockRun.OBSTACLE, position[0], position[1]]:
                return
            if self.course[BlockRun.REWARD, position[0], position[1]]:
                self.course[BlockRun.REWARD, position[0], position[1]] = False
                self.score += 20
                self.current_reward = 20
            self.course[BlockRun.HISTORY, self.position[0], self.position[1]] = True
            self.position = position
            return self.position

    def render(self):
        view = self.course[:, self.position[0] - self.view_range:self.position[0] + self.view_range + 1, :].clone()
        view[0, self.view_range, self.position[1]] = 1
        return view

def map_view_to_image(view):
    width = view.shape[1]
    height = view.shape[2]
    canvas = Image.new('RGBA', (width * TILE_SIZE, height * TILE_SIZE))
    for x in range(width):
        for y in range(height):
            pos = (x * TILE_SIZE, y * TILE_SIZE)
            canvas.paste(BACKGROUND_IMG, pos, BACKGROUND_IMG)
            if view[BlockRun.OBSTACLE, x, y]:
                canvas.paste(OBSTACLE_IMG, pos, OBSTACLE_IMG)
            if view[BlockRun.REWARD, x, y]:
                canvas.paste(REWARD_IMG, pos, REWARD_IMG)
            if view[BlockRun.POSITION, x, y]:
                canvas.paste(BIRD_IMG, pos, BIRD_IMG)
    np_img = np.array(canvas.convert('RGB'), dtype=np.uint8)
    np_img = np.transpose(np_img, (1, 0, 2))
    return torch.from_numpy(np_img)

def visualize_course():
    import matplotlib.pyplot as plt
    block_run = BlockRun(2_000_000)
    view = block_run.render()
    image = map_view_to_image(view)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()

def play_game():
    block_run = BlockRun(2_000_000)
    keymap = {
        'w': 2,
        'd': 1,
        's': 0,
        'a': 3
    }
    while True:
        view = block_run.render()
        key = visualize(view)
        while chr(key) not in keymap:
            key = cv2.waitKey(1)
        action = keymap[chr(key)]
        block_run.step(action)
        print(f'Action: {action} score: {block_run.score}')


def visualize(view):
    image = map_view_to_image(view)
    show_img = cv2.resize(cv2.cvtColor(image.transpose(1, 0).numpy(), cv2.COLOR_RGB2BGR), (512, 512),
                          interpolation=cv2.INTER_NEAREST)
    cv2.imshow('BlockRun', show_img)
    key = cv2.waitKey(1)
    return key


if __name__ == '__main__':
    # visualize_course()
    play_game()
