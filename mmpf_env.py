from scipy import spatial
from skimage import io
import numpy as np
import time
import sys
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import copy
from skimage.measure import block_reduce
from sensor import *
from parameter import *
class Env():
    def __init__(self, map_index, plot=False, test=False, num_agents=2):
        self.test = test
        self.num_agents = num_agents
        if self.test:
            self.map_dir = f'DungeonMaps/test'
        else:
            self.map_dir = f'DungeonMaps/train'
        self.map_list = os.listdir(self.map_dir)
        self.map_list.sort(reverse=True)
        self.map_index = map_index % np.size(self.map_list)
        self.ground_truth, self.robot_positions = self.import_ground_truth(self.map_dir + '/' + self.map_list[self.map_index])
        self.ground_truth_size = np.shape(self.ground_truth) # (480, 640)
        self.robot_belief = np.ones(self.ground_truth_size) * 127 # unexplored 127
        
        self.finish_percent = 0.8
        self.resolution = 4
        self.sensor_range = 80
        self.old_robot_belief = copy.deepcopy(self.robot_belief)

        self.plot = plot
        self.frame_files = []
        if self.plot:
            # initialize the route for each agent
            self.xPoints = [[pos[0]] for pos in self.robot_positions]
            self.yPoints = [[pos[1]] for pos in self.robot_positions]

        self.travel_dists = [0] * self.num_agents
        self.explored_rate = 0
        self.route_nodes = [[pos] for pos in self.robot_positions]
        self.frontiers = None
        self.downsampled_belief = None
        self.obstacles = None

    def begin(self):
        for pos in self.robot_positions:
            self.robot_belief = sensor_work(pos, self.sensor_range, self.robot_belief, self.ground_truth)
        self.downsampled_belief = block_reduce(self.robot_belief.copy(), block_size=(self.resolution, self.resolution), func=np.min)
        self.frontiers = self.find_frontier()
        self.old_robot_belief = copy.deepcopy(self.robot_belief)
        self.obstacles = self.robot_belief == 1

    def step(self, next_node_coords):
        for i, coords in enumerate(next_node_coords):
            dist = np.linalg.norm(coords - self.robot_positions[i])
            self.travel_dists[i] += dist
            self.robot_positions[i] = coords
            self.route_nodes[i].append(coords)
            self.robot_belief = sensor_work(coords, self.sensor_range, self.robot_belief, self.ground_truth)

        self.downsampled_belief = block_reduce(self.robot_belief.copy(), block_size=(self.resolution, self.resolution), func=np.min)
        self.frontiers = self.find_frontier()
        self.explored_rate = self.evaluate_exploration_rate()
        self.obstacles = self.robot_belief == 1

        if self.plot:
            for i, pos in enumerate(self.robot_positions):
                self.xPoints[i].append(pos[0])
                self.yPoints[i].append(pos[1])

        done = self.check_done()
        return done

    def import_ground_truth(self, map_index):
        ground_truth = (io.imread(map_index, 1) * 255).astype(int)
        robot_locations = np.array(np.where(ground_truth == 208)).T
        robot_locations = robot_locations[:self.num_agents, [1, 0]]  # Limit to num_agents and swap x, y
        ground_truth = (ground_truth > 150)
        ground_truth = ground_truth * 254 + 1
        return ground_truth, robot_locations

    def free_cells(self):
        index = np.where(self.ground_truth == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def check_done(self):
        return np.sum(self.ground_truth == 255) - np.sum(self.robot_belief == 255) <= 250

    def evaluate_exploration_rate(self):
        return np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)

    def calculate_new_free_area(self):
        old_free_area = self.old_robot_belief == 255
        current_free_area = self.robot_belief == 255
        new_free_area = (current_free_area.astype(np.int) - old_free_area.astype(np.int)) * 255
        return new_free_area, np.sum(old_free_area)

    def find_frontier(self):
        y_len, x_len = self.downsampled_belief.shape
        mapping = (self.downsampled_belief == 127) * 1
        mapping = np.pad(mapping, ((1, 1), (1, 1)), 'constant', constant_values=0)
        fro_map = mapping[2:][:, 1:x_len+1] + mapping[:y_len][:, 1:x_len+1] + mapping[1:y_len+1][:, 2:] + \
                  mapping[1:y_len+1][:, :x_len] + mapping[:y_len][:, 2:] + mapping[2:][:, :x_len] + mapping[2:][:, 2:] + \
                  mapping[:y_len][:, :x_len]
        ind_free = np.where(self.downsampled_belief.ravel(order='F') == 255)[0]
        ind_fron_1 = np.where(1 < fro_map.ravel(order='F'))[0]
        ind_fron_2 = np.where(fro_map.ravel(order='F') < 8)[0]
        ind_fron = np.intersect1d(ind_fron_1, ind_fron_2)
        ind_to = np.intersect1d(ind_free, ind_fron)

        points = np.array(np.meshgrid(np.arange(x_len), np.arange(y_len))).T.reshape(-1, 2)
        f = points[ind_to] * self.resolution
        return f

    def get_distance_map(self, goal):
        grid = np.ones_like(self.obstacles, dtype=float)
        grid[self.obstacles] = np.inf
        grid[tuple(goal)] = 0
        return ndimage.distance_transform_edt(grid)

    def update_combined_belief(self):
        self.obstacles = self.robot_belief == 1

    def plot_env(self, n, path, step, planned_routes=None):
        plt.switch_backend('agg')
        plt.cla()
        plt.imshow(self.robot_belief, cmap='gray')
        plt.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))
        
        if planned_routes:
            for route in planned_routes:
                planned_x, planned_y = zip(*route)
                plt.plot(planned_x, planned_y, c='r', linewidth=2, zorder=2)
        
        for i in range(self.num_agents):
            plt.plot(self.xPoints[i], self.yPoints[i], 'b', linewidth=2)
            plt.plot(self.robot_positions[i][0], self.robot_positions[i][1], 'mo', markersize=8)
            plt.plot(self.xPoints[i][0], self.yPoints[i][0], 'co', markersize=8)
        
        plt.scatter(self.frontiers[:, 0], self.frontiers[:, 1], c='r', s=2, zorder=3)
        plt.suptitle(f'Explored ratio: {self.explored_rate:.4g}  Travel distance: {sum(self.travel_dists):.4g}')
        plt.tight_layout()
        plt.savefig(f'{path}/{n}_{step}_samples.png', dpi=150)
        frame = f'{path}/{n}_{step}_samples.png'
        self.frame_files.append(frame)