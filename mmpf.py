# import numpy as np
# from scipy.spatial import distance
# from scipy.ndimage import distance_transform_edt
# import imageio
# import csv
# import os
# import copy
# import numpy as np
# import random
# import shapely.geometry
# import matplotlib.pyplot as plt
# from NBVP_env import Env
# from test_parameter import *

# gifs_path = f'results/mcts'

# class MMPF_worker:
#     def __init__(self, metaAgentID, global_step, num_agents=2, save_image=False):
#         print(f"Initializing MMPF worker with {num_agents} agents at step {global_step}")
#         self.metaAgentID = metaAgentID
#         self.global_step = global_step
#         self.save_image = save_image
#         self.num_agents = num_agents
#         self.env = Env(map_index=self.global_step, plot=save_image, test=True, num_agents=num_agents)
#         self.k = 1  # Constant for frontier potential field calculation
#         self.kr = 0.5  # Constant for robot repulsive potential field calculation
#         self.ds = 50  # Sensor range

#     def continuity_based_cluster(self, frontiers):
#         clusters = []
#         frontiers = frontiers.copy()  # Create a copy to avoid modifying the original list
        
#         while frontiers:
#             new_cluster = [frontiers.pop(0)]  # Start a new cluster with the first frontier
            
#             cluster_changed = True
#             while cluster_changed:
#                 cluster_changed = False
#                 frontiers_to_remove = []
                
#                 for frontier in frontiers:
#                     if any(self.is_neighbor(frontier, cluster_frontier) for cluster_frontier in new_cluster):
#                         new_cluster.append(frontier)
#                         frontiers_to_remove.append(frontier)
#                         cluster_changed = True
                
#                 for frontier in frontiers_to_remove:
#                     frontiers.remove(frontier)
            
#             clusters.append(new_cluster)
        
#         return clusters

#     def is_neighbor(self, frontier1, frontier2, threshold=1):
#         return distance.euclidean(frontier1, frontier2) <= threshold

#     def calculate_wave_front_distance(self, start, goal, obstacles):
#         grid = np.ones_like(obstacles, dtype=float)
#         grid[obstacles] = np.inf
#         grid[tuple(start)] = 0
#         return distance_transform_edt(grid)[tuple(goal)]

#     def calculate_frontier_potential(self, point, centroid, cluster_size):
#         rw = self.calculate_wave_front_distance(point, centroid, self.env.obstacles)
#         return -self.k * cluster_size / rw

#     def calculate_robot_repulsive_potential(self, point, robot_position):
#         rd = np.sum(np.abs(np.array(point) - np.array(robot_position)))
#         if rd < self.ds:
#             return self.kr * (self.ds - rd)
#         return 0

#     def find_next_best_viewpoints(self):
#         frontier_clusters = self.continuity_based_cluster(self.env.frontiers)
#         centroids = [np.mean(cluster, axis=0).astype(int) for cluster in frontier_clusters]
        
#         next_node_coords = []
#         for agent_id in range(self.num_agents):
#             current_pos = self.env.robot_positions[agent_id]
#             potentials = []
#             for centroid, cluster in zip(centroids, frontier_clusters):
#                 frontier_potential = self.calculate_frontier_potential(current_pos, centroid, len(cluster))
#                 robot_repulsive_potential = sum(
#                     self.calculate_robot_repulsive_potential(centroid, other_robot_pos)
#                     for other_id, other_robot_pos in enumerate(self.env.robot_positions)
#                     if other_id != agent_id
#                 )
#                 total_potential = frontier_potential + robot_repulsive_potential
#                 potentials.append(total_potential)
            
#             best_cluster_index = np.argmin(potentials)
#             next_node = centroids[best_cluster_index]
#             next_node_coords.append(next_node)
        
#         return next_node_coords

#     def run_episode(self, currEpisode):
#         perf_metrics = dict()
#         print(f"Running episode {currEpisode}")
#         done = False
#         self.env.begin()
#         i = 0

#         while not done:
#             i += 1
#             next_node_coords = self.find_next_best_viewpoints()
#             done = self.env.step(next_node_coords)

#             if self.save_image:
#                 if not os.path.exists(gifs_path):
#                     os.makedirs(gifs_path)
#                 self.env.plot_env(self.global_step, gifs_path, i, [])

#             if done:
#                 perf_metrics['travel_dist'] = sum(self.env.travel_dists)
#                 perf_metrics['explored_rate'] = self.env.explored_rate
#                 perf_metrics['success_rate'] = True
#                 perf_metrics['relax_success_rate'] = True if self.env.explored_rate > self.env.finish_percent else False
#                 break

#         if self.save_image:
#             path1 = gifs_path
#             self.make_gif(path1, currEpisode)

#         return perf_metrics

#     def work(self, currEpisode):
#         self.currEpisode = currEpisode
#         self.perf_metrics = self.run_episode(currEpisode)

#     def make_gif(self, path, n):
#         print(f"Creating GIF for episode {n}")
#         with imageio.get_writer('{}/{}_explored_rate_{:.4g}_length_{:.4g}.gif'.format(path, n, self.env.explored_rate, sum(self.env.travel_dists)), mode='I',
#                                 duration=0.5) as writer:
#             for frame in self.env.frame_files:
#                 image = imageio.imread(frame)
#                 writer.append_data(image)
#         print('GIF complete\n')

#         # Remove files
#         for filename in self.env.frame_files[:-1]:
#             os.remove(filename)

#work
# import numpy as np
# from scipy.spatial import distance
# from scipy.ndimage import distance_transform_edt
# import imageio
# import csv
# import os
# import copy
# import random
# import shapely.geometry
# import matplotlib.pyplot as plt
# from NBVP_env import Env
# from test_parameter import *

# gifs_path = f'results/mcts'

# class MMPF_worker:
#     def __init__(self, metaAgentID, global_step, num_agents=2, save_image=False):
#         print(f"Initializing MMPF worker with {num_agents} agents at step {global_step}")
#         self.metaAgentID = metaAgentID
#         self.global_step = global_step
#         self.save_image = save_image
#         self.num_agents = num_agents
#         self.env = Env(map_index=self.global_step, plot=save_image, test=True, num_agents=num_agents)
#         self.k = 1  # Constant for frontier potential field calculation
#         self.kr = 0.5  # Constant for robot repulsive potential field calculation
#         self.ds = 50  # Sensor range

#     def continuity_based_cluster(self, frontiers):
#         clusters = []
#         frontiers = frontiers.tolist()  # Convert numpy array to list
        
#         while frontiers:
#             new_cluster = [frontiers.pop(0)]
            
#             cluster_changed = True
#             while cluster_changed:
#                 cluster_changed = False
#                 frontiers_to_remove = []
                
#                 for frontier in frontiers:
#                     if any(self.is_neighbor(frontier, cluster_frontier) for cluster_frontier in new_cluster):
#                         new_cluster.append(frontier)
#                         frontiers_to_remove.append(frontier)
#                         cluster_changed = True
                
#                 for frontier in frontiers_to_remove:
#                     frontiers.remove(frontier)
            
#             clusters.append(new_cluster)
        
#         return clusters

#     def is_neighbor(self, frontier1, frontier2, threshold=1):
#         return distance.euclidean(frontier1, frontier2) <= threshold

#     # def calculate_wave_front_distance(self, start, goal, obstacles):
#     #  grid = np.ones_like(obstacles, dtype=float)
#     #  grid[obstacles] = np.inf

#     # # Ensure the start and goal positions are within the bounds of the grid
#     #  start = np.clip(start, [0, 0], np.array(grid.shape) - 1)
#     #  goal = np.clip(goal, [0, 0], np.array(grid.shape) - 1)

#     #  grid[tuple(start)] = 0

#     #  try:
#     #     return distance_transform_edt(grid)[tuple(goal)]
#     #  except IndexError as e:
#     #     print(f"Error: {e}, Start: {start}, Goal: {goal}, Grid shape: {grid.shape}")
#     #     return np.inf  # Return a high value if something goes wrong
#     def calculate_wave_front_distance(self, start, goal, obstacles):
#      grid = np.ones_like(obstacles, dtype=float)
#      grid[obstacles] = np.inf

#     # Ensure the start and goal positions are within the bounds of the grid
#      start = np.clip(start, [0, 0], np.array(grid.shape) - 1)
#      goal = np.clip(goal, [0, 0], np.array(grid.shape) - 1)

#     # Bresenham's line algorithm or any line drawing algorithm can be used to generate points on the path
#      line_points = self.bresenham_line(start, goal)
    
#      for point in line_points:
#         point = np.clip(point, [0, 0], np.array(grid.shape) - 1)  # Ensure all points are within bounds
#         grid[tuple(point)] = 0  # Mark these points on the grid
    
#      try:
#         distance_map = distance_transform_edt(grid)
#         return distance_map[tuple(goal)]
#      except IndexError as e:
#         print(f"Error: {e}, Start: {start}, Goal: {goal}, Grid shape: {grid.shape}")
#         return np.inf  # Return a high value if something goes wrong


#     def bresenham_line(self, start, goal):
   
#      x1, y1 = start
#      x2, y2 = goal
#      points = []

#      dx = abs(x2 - x1)
#      dy = abs(y2 - y1)
#      sx = 1 if x1 < x2 else -1
#      sy = 1 if y1 < y2 else -1
#      err = dx - dy

#      while (x1, y1) != (x2, y2):
#         points.append((x1, y1))
#         e2 = 2 * err
#         if e2 > -dy:
#             err -= dy
#             x1 += sx
#         if e2 < dx:
#             err += dx
#             y1 += sy

#      points.append((x2, y2))  # Add the last point
#      return points

#     def calculate_frontier_potential(self, point, centroid, cluster_size):
#         rw = self.calculate_wave_front_distance(point, centroid, self.env.obstacles)
#         return -self.k * cluster_size / rw

#     def calculate_robot_repulsive_potential(self, point, robot_position):
#         rd = np.sum(np.abs(np.array(point) - np.array(robot_position)))
#         if rd < self.ds:
#             return self.kr * (self.ds - rd)
#         return 0

#     def find_next_best_viewpoints(self):
#         frontier_clusters = self.continuity_based_cluster(self.env.frontiers)
#         centroids = [np.mean(cluster, axis=0).astype(int) for cluster in frontier_clusters]
        
#         next_node_coords = []
#         for agent_id in range(self.num_agents):
#             current_pos = self.env.robot_positions[agent_id]
#             potentials = []
#             for centroid, cluster in zip(centroids, frontier_clusters):
#                 frontier_potential = self.calculate_frontier_potential(current_pos, centroid, len(cluster))
#                 robot_repulsive_potential = sum(
#                     self.calculate_robot_repulsive_potential(centroid, other_robot_pos)
#                     for other_id, other_robot_pos in enumerate(self.env.robot_positions)
#                     if other_id != agent_id
#                 )
#                 total_potential = frontier_potential + robot_repulsive_potential
#                 potentials.append(total_potential)
            
#             best_cluster_index = np.argmin(potentials)
#             next_node = centroids[best_cluster_index]
#             next_node_coords.append(next_node)
        
#         return next_node_coords

#     def run_episode(self, currEpisode):
#         perf_metrics = dict()
#         print(f"Running episode {currEpisode}")
#         done = False
#         self.env.begin()
#         i = 0

#         while not done:
#             i += 1
#             next_node_coords = self.find_next_best_viewpoints()
#             done = self.env.step(next_node_coords)

#             if self.save_image:
#                 if not os.path.exists(gifs_path):
#                     os.makedirs(gifs_path)
#                 self.env.plot_env(self.global_step, gifs_path, i, [])

#             if done:
#                 perf_metrics['travel_dist'] = sum(self.env.travel_dists)
#                 perf_metrics['explored_rate'] = self.env.explored_rate
#                 perf_metrics['success_rate'] = True
#                 perf_metrics['relax_success_rate'] = True if self.env.explored_rate > self.env.finish_percent else False
#                 break

#         if self.save_image:
#             path1 = gifs_path
#             self.make_gif(path1, currEpisode)

#         return perf_metrics

#     def work(self, currEpisode):
#         self.currEpisode = currEpisode
#         self.perf_metrics = self.run_episode(currEpisode)

#     def make_gif(self, path, n):
#         print(f"Creating GIF for episode {n}")
#         with imageio.get_writer('{}/{}_explored_rate_{:.4g}_length_{:.4g}.gif'.format(path, n, self.env.explored_rate, sum(self.env.travel_dists)), mode='I',
#                                 duration=0.5) as writer:
#             for frame in self.env.frame_files:
#                 image = imageio.imread(frame)
#                 writer.append_data(image)
#         print('GIF complete\n')

#         # Remove files
#         for filename in self.env.frame_files[:-1]:
#             os.remove(filename)

# if __name__ == "__main__":
#     total_episode = 40
#     total_dist = 0
#     num_agents = 2  # Specify the number of agents here
    
#     for i in range(total_episode):
#         print(f"Starting episode {i+1}")
        
#         worker = MMPF_worker(metaAgentID=0, global_step=i, num_agents=num_agents, save_image=SAVE_GIFS)
#         performance = worker.run_episode(i)
#         total_dist += performance["travel_dist"]
#         mean_dist = total_dist / (i + 1)
#         print(f"Episode {i+1}/{total_episode}, Mean distance: {mean_dist:.2f}")

#     print(f"\nFinal mean distance over {total_episode} episodes: {mean_dist:.2f}")


# good

# import numpy as np
# from scipy.spatial import distance
# from scipy.ndimage import distance_transform_edt
# import imageio
# import os
# from NBVP_env import Env
# from test_parameter import *

# gifs_path = f'results/mcts'

# class MMPF_worker:
#     def __init__(self, metaAgentID, global_step, num_agents=2, save_image=False):
#         print(f"Initializing MMPF worker with {num_agents} agents at step {global_step}")
#         self.metaAgentID = metaAgentID
#         self.global_step = global_step
#         self.save_image = save_image
#         self.num_agents = num_agents
#         self.env = Env(map_index=self.global_step, plot=save_image, test=True, num_agents=num_agents)
#         self.k = 5.0  # Constant for frontier potential field calculation
#         self.kr = 2.0  # Constant for robot repulsive potential field calculation
#         self.ds = 50  # Sensor range
#         self.distance_maps = {}  # Store distance maps for each centroid
#         self.shared_map = np.zeros_like(self.env.ground_truth)  # Shared map between agents
#         self.shared_map = self.env.merge_robot_beliefs()
    
#     def update_shared_map(self):
#         """
#         Update the shared map using the merged robot beliefs.
#         """
#         self.shared_map = self.env.merge_robot_beliefs()
#     def clip_coordinates(self, coords):
#         return np.clip(coords, [0, 0], np.array(self.env.obstacles.shape) - 1)

#     # def continuity_based_cluster(self, frontiers):
#     #     clusters = []
#     #     frontiers = frontiers.tolist()
        
#     #     while frontiers:
#     #         new_cluster = [frontiers.pop(0)]
            
#     #         cluster_changed = True
#     #         while cluster_changed:
#     #             cluster_changed = False
#     #             frontiers_to_remove = []
                
#     #             for frontier in frontiers:
#     #                 if any(self.is_neighbor(frontier, cluster_frontier) for cluster_frontier in new_cluster):
#     #                     new_cluster.append(frontier)
#     #                     frontiers_to_remove.append(frontier)
#     #                     cluster_changed = True
                
#     #             for frontier in frontiers_to_remove:
#     #                 frontiers.remove(frontier)
            
#     #         clusters.append(new_cluster)
        
#     #     return clusters

#     def continuity_based_cluster(self, frontiers, robot_position):
#      clusters = []
#      frontiers = frontiers.tolist()
    
#      while frontiers:
#         new_cluster = [frontiers.pop(0)]
        
#         cluster_changed = True
#         while cluster_changed:
#             cluster_changed = False
#             frontiers_to_remove = []
            
#             for frontier in frontiers:
#                 if any(self.is_neighbor(frontier, cluster_frontier, robot_position) for cluster_frontier in new_cluster):
#                     new_cluster.append(frontier)
#                     frontiers_to_remove.append(frontier)
#                     cluster_changed = True
            
#             for frontier in frontiers_to_remove:
#                 frontiers.remove(frontier)
        
#         clusters.append(new_cluster)
    
#      return clusters


#     # def is_neighbor(self, frontier1, frontier2, threshold=1):
#     #     return distance.euclidean(frontier1, frontier2) <= threshold
#     def is_neighbor(self, frontier1, frontier2, robot_position=None, threshold=1):
#     # Adjust this function if robot_position should affect neighbor determination
#      return distance.euclidean(frontier1, frontier2) <= threshold


#     def calculate_wave_front_distance(self, start, goal):
#      start = self.clip_coordinates(start)
#      goal = self.clip_coordinates(goal)

#     # Calculate the sensor range bounds around the goal
#      min_bounds = np.maximum(goal - self.ds, 0)
#      max_bounds = np.minimum(goal + self.ds, np.array(self.env.obstacles.shape) - 1)
    
#     # Check if the distance map for the goal (centroid) already exists within the sensor range
#      if tuple(goal) not in self.distance_maps:
#         # Create a grid within the sensor range bounds
#         grid = np.ones_like(self.env.obstacles, dtype=float)
#         grid[self.env.obstacles] = np.inf  # Mark obstacles as infinity
        
#         # Restrict grid to the field of view of the sensor
#         grid[:min_bounds[0], :] = np.inf
#         grid[:, :min_bounds[1]] = np.inf
#         grid[max_bounds[0]+1:, :] = np.inf
#         grid[:, max_bounds[1]+1:] = np.inf
        
#         # Set the goal position within the sensor range to zero
#         grid[tuple(goal)] = 0

#         # Generate the wave-front distance map within the sensor range
#         self.distance_maps[tuple(goal)] = distance_transform_edt(grid)
    
#     # Return the precomputed distance from the start point to the goal
#      return self.distance_maps[tuple(goal)][tuple(start)]


#     def calculate_frontier_potential(self, point, centroid, cluster_size):
#         rw = self.calculate_wave_front_distance(point, centroid)
#         return -self.k * cluster_size / rw

#     def calculate_robot_repulsive_potential(self, point, robot_position):
#         rd = np.sum(np.abs(np.array(point) - np.array(robot_position)))  # Manhattan distance
#         if rd < self.ds:
#             return self.kr * (self.ds - rd)
#         return 0

#     def find_next_best_viewpoints(self):
#      next_node_coords = []
    
#      for agent_id in range(self.num_agents):
#         current_pos = self.clip_coordinates(self.env.robot_positions[agent_id])
        
#         # Each agent independently clusters frontiers based on its own map
#         agent_frontiers = self.env.get_agent_frontiers(agent_id)

#         frontier_clusters = self.continuity_based_cluster(agent_frontiers, current_pos)
        
#         centroids = [np.mean(cluster, axis=0).astype(int) for cluster in frontier_clusters]
#         potentials = []
        
#         for centroid, cluster in zip(centroids, frontier_clusters):
#             centroid = self.clip_coordinates(centroid)
#             frontier_potential = self.calculate_frontier_potential(current_pos, centroid, len(cluster))
#             robot_repulsive_potential = sum(
#                 self.calculate_robot_repulsive_potential(centroid, self.clip_coordinates(other_robot_pos))
#                 for other_id, other_robot_pos in enumerate(self.env.robot_positions)
#                 if other_id != agent_id
#             )
#             total_potential = frontier_potential + robot_repulsive_potential
#             potentials.append(total_potential)
        
#         best_cluster_index = np.argmin(potentials)
#         next_node = self.clip_coordinates(centroids[best_cluster_index])
#         next_node_coords.append(next_node)
    
#      return next_node_coords


#     def update_shared_map(self):
#         # Merge individual agent maps into the shared map
#         for agent_id in range(self.num_agents):
#             self.shared_map = np.logical_or(self.shared_map, self.env.get_agent_map(agent_id))

#     def run_episode(self, currEpisode):
#         perf_metrics = dict()
#         print(f"Running episode {currEpisode}")
#         done = False
#         self.env.begin()
#         i = 0

#         while not done:
#             i += 1
#             next_node_coords = self.find_next_best_viewpoints()
#             done = self.env.step(next_node_coords)

#             # Update shared map after each step
#             self.update_shared_map()

#             if self.save_image:
#                 if not os.path.exists(gifs_path):
#                     os.makedirs(gifs_path)
#                 self.env.plot_env(self.global_step, gifs_path, i, [])

#             if done:
#                 perf_metrics['travel_dist'] = sum(self.env.travel_dists)
#                 perf_metrics['explored_rate'] = self.env.explored_rate
#                 perf_metrics['success_rate'] = True
#                 perf_metrics['relax_success_rate'] = True if self.env.explored_rate > self.env.finish_percent else False
#                 break

#         if self.save_image:
#             path1 = gifs_path
#             self.make_gif(path1, currEpisode)

#         return perf_metrics

#     def work(self, currEpisode):
#         self.currEpisode = currEpisode
#         self.perf_metrics = self.run_episode(currEpisode)

#     def make_gif(self, path, n):
#         print(f"Creating GIF for episode {n}")
#         with imageio.get_writer('{}/{}_explored_rate_{:.4g}_length_{:.4g}.gif'.format(path, n, self.env.explored_rate, sum(self.env.travel_dists)), mode='I',
#                                 duration=0.5) as writer:
#             for frame in self.env.frame_files:
#                 image = imageio.imread(frame)
#                 writer.append_data(image)
#         print('GIF complete\n')

#         # Remove files
#         for filename in self.env.frame_files[:-1]:
#             os.remove(filename)

# if __name__ == "__main__":
#     total_episode = 40
#     total_dist = 0
#     num_agents = 2  # Specify the number of agents here
    
#     for i in range(total_episode):
#         print(f"Starting episode {i+1}")
        
#         worker = MMPF_worker(metaAgentID=0, global_step=i, num_agents=num_agents, save_image=SAVE_GIFS)
#         performance = worker.run_episode(i)
#         total_dist += performance["travel_dist"]
#         mean_dist = total_dist / (i + 1)
#         print(f"Episode {i+1}/{total_episode}, Mean distance: {mean_dist:.2f}")

#     print(f"\nFinal mean distance over {total_episode} episodes: {mean_dist:.2f}")




# import numpy as np
# from scipy.spatial import distance
# from scipy.ndimage import distance_transform_edt
# import imageio
# import os
# from NBVP_env import Env
# from test_parameter import *

# gifs_path = f'results/mcts'

# class MMPF_worker:
#     def __init__(self, metaAgentID, global_step, num_agents=1, save_image=False):
#         print(f"Initializing MMPF worker with {num_agents} agents at step {global_step}")
#         self.metaAgentID = metaAgentID
#         self.global_step = global_step
#         self.save_image = save_image
#         self.num_agents = num_agents
#         self.env = Env(map_index=self.global_step, plot=save_image, test=True, num_agents=num_agents)
#         self.k = 0.5  # Constant for frontier potential field calculation
#         self.kr = 2.0  # Constant for robot repulsive potential field calculation
#         self.ds = 60  # Sensor range
#         self.distance_maps = {}  
#         self.shared_map = np.zeros_like(self.env.ground_truth)  # Shared map between agents
#         self.shared_map = self.env.merge_robot_beliefs()
    
#     def update_shared_map(self):
#         """
#         Update the shared map using the merged robot beliefs.
#         """
#         self.shared_map = self.env.merge_robot_beliefs()
#     def clip_coordinates(self, coords):
#         return np.clip(coords, [0, 0], np.array(self.env.obstacles.shape) - 1)

#     def continuity_based_cluster(self, frontiers):
#         clusters = []
#         frontiers = frontiers.tolist()
        
#         while frontiers:
#             new_cluster = [frontiers.pop(0)]
            
            
#             cluster_changed = True
#             while cluster_changed:
#                 cluster_changed = False
#                 frontiers_to_remove = []
                
#                 for frontier in frontiers:
#                     if any(self.is_neighbor(frontier, cluster_frontier) for cluster_frontier in new_cluster):
#                         new_cluster.append(frontier)
#                         frontiers_to_remove.append(frontier)
#                         cluster_changed = True
                
#                 for frontier in frontiers_to_remove:
#                     frontiers.remove(frontier)
            
#             clusters.append(new_cluster)
        
#         return clusters

#     def is_neighbor(self, frontier1, frontier2, threshold=1):
#         return distance.euclidean(frontier1, frontier2) <= threshold

    
   
#     def is_valid_point(self, point):
#         """
#         Check if a point is within the map boundaries and not an obstacle.
#         """
#         x, y = point
#         return (0 <= x < self.env.obstacles.shape[0] and
#                 0 <= y < self.env.obstacles.shape[1] and
#                 not self.env.obstacles[x, y])

#     def calculate_wave_front_distance_map(self, goal):
#      goal = self.clip_coordinates(goal)
    
#     # Create a binary map where obstacles and unexplored areas are 0, and free space is 1
#      binary_map = ((self.env.obstacles == 0) & (self.env.robot_beliefs[0] == 255)).astype(float)
    
#     # Set the goal point to 2 (a value greater than 1) to ensure it's the starting point
#      binary_map[goal[0], goal[1]] = 2
    
#     # Calculate the distance transform
#      distance_map = distance_transform_edt(binary_map)
    
#     # Set obstacle and unexplored locations to infinity
#      distance_map[self.env.obstacles == 1] = np.inf
#      distance_map[self.env.robot_beliefs[0] != 255] = np.inf
     
#      return distance_map
#     def calculate_wave_front_distance(self, start, goal):
#         """
#         Calculate the wave-front distance between start and goal points, considering obstacles.
#         """
#         start = self.clip_coordinates(start)
#         goal = tuple(self.clip_coordinates(goal))
        
#         if goal not in self.distance_maps:
#             self.distance_maps[goal] = self.calculate_wave_front_distance_map(goal)
        
#         return self.distance_maps[goal][start[0], start[1]]

#     def find_next_best_viewpoints(self):
#      frontier_clusters = self.continuity_based_cluster(self.env.frontiers)
#      centroids = [np.mean(cluster, axis=0).astype(int) for cluster in frontier_clusters]
    
#     # Pre-calculate distance maps for all centroids
#      for centroid in centroids:
#         if tuple(centroid) not in self.distance_maps:
#             self.distance_maps[tuple(centroid)] = self.calculate_wave_front_distance_map(centroid)
    
#      next_node_coords = []
    
#      for agent_id in range(self.num_agents):
#         current_pos = self.clip_coordinates(self.env.robot_positions[agent_id])
#         potentials = []
        
#         for centroid, cluster in zip(centroids, frontier_clusters):
#             centroid = self.clip_coordinates(centroid)
#             wave_front_distance = self.calculate_wave_front_distance(current_pos, centroid)
            
#             if np.isinf(wave_front_distance):
#                 potentials.append(float('inf'))
#                 continue
            
#             cluster_size = len(cluster)
#             distance_to_centroid = np.linalg.norm(current_pos - centroid)
            
#             # Adjust the potential calculation
#             total_potential = -cluster_size + 0.1 * wave_front_distance + 0.05 * distance_to_centroid
#             potentials.append(total_potential)
        
#         if not potentials or all(np.isinf(p) for p in potentials):
#             # If no valid potentials, explore locally
#             next_node_coords.append(self.explore_locally(current_pos))
#         else:
#             best_cluster_index = np.argmin(potentials)
#             goal = self.clip_coordinates(centroids[best_cluster_index])
            
#             # Find the next point along the path to the goal
#             next_point = self.find_next_point_to_goal(current_pos, goal)
#             next_node_coords.append(next_point)
#         print(f"Generated {len(next_node_coords)} coordinates for {self.num_agents} agents")
#         return next_node_coords

#     def explore_locally(self, current_pos):
#     # Check all neighboring cells and move to the nearest unexplored one
#      for dx in [-1, 0, 1]:
#         for dy in [-1, 0, 1]:
#             if dx == 0 and dy == 0:
#                 continue
#             new_pos = (current_pos[0] + dx, current_pos[1] + dy)
#             if self.is_valid_point(new_pos) and self.env.robot_beliefs[0][new_pos] == 127:
#                 return new_pos
#      return current_pos  # If no unexplored neighbors, stay put

#     def find_next_point_to_goal(self, start, goal):
#      distance_map = self.distance_maps[tuple(goal)]
#      neighbors = [
#         (start[0] + dx, start[1] + dy)
#         for dx in [-1, 0, 1] for dy in [-1, 0, 1]
#         if (dx != 0 or dy != 0) and self.is_valid_point((start[0] + dx, start[1] + dy))
#      ]
    
#      if not neighbors:
#         return start
    
#      return min(neighbors, key=lambda p: distance_map[p[0], p[1]])

#     def calculate_frontier_potential(self, point, centroid, cluster_size):
#         rw = self.calculate_wave_front_distance(point, centroid)
#         return -self.k * cluster_size / rw

#     def calculate_robot_repulsive_potential(self, point, robot_position):
#         rd = np.sum(np.abs(np.array(point) - np.array(robot_position)))  # Manhattan distance
#         if rd < self.ds:
#             return self.kr * (self.ds - rd)
#         return 0

    

#     def update_shared_map(self):
#         # Merge individual agent maps into the shared map
#         for agent_id in range(self.num_agents):
#             self.shared_map = np.logical_or(self.shared_map, self.env.get_agent_map(agent_id))

#     def run_episode(self, currEpisode):
#         perf_metrics = dict()
#         print(f"Running episode {currEpisode}")
#         done = False
#         self.env.begin()
#         i = 0

#         while not done:
#             i += 1
#             next_node_coords = self.find_next_best_viewpoints()
#             done = self.env.step(next_node_coords)

#             # Update shared map after each step
#             self.update_shared_map()

#             if self.save_image:
#                 if not os.path.exists(gifs_path):
#                     os.makedirs(gifs_path)
#                 self.env.plot_env(self.global_step, gifs_path, i, [])

#             if done:
#                 perf_metrics['travel_dist'] = sum(self.env.travel_dists)
#                 perf_metrics['explored_rate'] = self.env.explored_rate
#                 perf_metrics['success_rate'] = True
#                 perf_metrics['relax_success_rate'] = True if self.env.explored_rate > self.env.finish_percent else False
#                 break

#         if self.save_image:
#             path1 = gifs_path
#             self.make_gif(path1, currEpisode)

#         return perf_metrics

#     def work(self, currEpisode):
#         self.currEpisode = currEpisode
#         self.perf_metrics = self.run_episode(currEpisode)

#     def make_gif(self, path, n):
#         print(f"Creating GIF for episode {n}")
#         with imageio.get_writer('{}/{}_explored_rate_{:.4g}_length_{:.4g}.gif'.format(path, n, self.env.explored_rate, sum(self.env.travel_dists)), mode='I',
#                                 duration=0.5) as writer:
#             for frame in self.env.frame_files:
#                 image = imageio.imread(frame)
#                 writer.append_data(image)
#         print('GIF complete\n')

#         # Remove files
#         for filename in self.env.frame_files[:-1]:
#             os.remove(filename)

# if __name__ == "__main__":
#     total_episode = 40
#     total_dist = 0
#     num_agents = 1  # Specify the number of agents here
    
#     for i in range(total_episode):
#         print(f"Starting episode {i+1}")
        
#         worker = MMPF_worker(metaAgentID=0, global_step=i, num_agents=num_agents, save_image=SAVE_GIFS)
#         performance = worker.run_episode(i)
#         total_dist += performance["travel_dist"]
#         mean_dist = total_dist / (i + 1)
#         print(f"Episode {i+1}/{total_episode}, Mean distance: {mean_dist:.2f}")

#     print(f"\nFinal mean distance over {total_episode} episodes: {mean_dist:.2f}")



#orig
import numpy as np
# from scipy.spatial import distance
# from scipy.ndimage import distance_transform_edt
# import imageio
# import os
# from NBVP_env import Env
# from test_parameter import *

# gifs_path = f'results/mcts'

# class MMPF_worker:
#     def __init__(self, metaAgentID, global_step, num_agents=2, save_image=False):
#         print(f"Initializing MMPF worker with {num_agents} agents at step {global_step}")
#         self.metaAgentID = metaAgentID
#         self.global_step = global_step
#         self.save_image = save_image
#         self.num_agents = num_agents
#         self.env = Env(map_index=self.global_step, plot=save_image, test=True, num_agents=num_agents)
#         self.k = 0.1  # Constant for frontier potential field calculation
#         self.kr = 1.0  # Constant for robot repulsive potential field calculation
#         self.ds = 50  # Sensor range
#         self.distance_maps = {}  # Store distance maps for each centroid

#     def clip_coordinates(self, coords):
#         return np.clip(coords, [0, 0], np.array(self.env.obstacles.shape) - 1)

#     def continuity_based_cluster(self, frontiers):
#         all_clusters = []
#         for agent_frontiers in frontiers:
#             clusters = []
#             agent_frontiers = agent_frontiers.tolist()
            
#             while agent_frontiers:
#                 new_cluster = [agent_frontiers.pop(0)]
                
#                 cluster_changed = True
#                 while cluster_changed:
#                     cluster_changed = False
#                     frontiers_to_remove = []
                    
#                     for frontier in agent_frontiers:
#                         if any(self.is_neighbor(frontier, cluster_frontier) for cluster_frontier in new_cluster):
#                             new_cluster.append(frontier)
#                             frontiers_to_remove.append(frontier)
#                             cluster_changed = True
                    
#                     for frontier in frontiers_to_remove:
#                         agent_frontiers.remove(frontier)
                
#                 clusters.append(new_cluster)
            
#             all_clusters.append(clusters)
        
#         return all_clusters

#     def is_neighbor(self, frontier1, frontier2, threshold=1):
#         return distance.euclidean(frontier1, frontier2) <= threshold

#     def calculate_wave_front_distance(self, start, goal):
#         start = self.clip_coordinates(start)
#         goal = self.clip_coordinates(goal)
        
#         if tuple(goal) not in self.distance_maps:
#             grid = np.ones_like(self.env.obstacles, dtype=float)
#             grid[self.env.obstacles] = np.inf
#             grid[tuple(goal)] = 0
#             self.distance_maps[tuple(goal)] = distance_transform_edt(grid)
        
#         return self.distance_maps[tuple(goal)][tuple(start)]

#     def calculate_frontier_potential(self, point, centroid, cluster_size):
#         rw = self.calculate_wave_front_distance(point, centroid)
#         return -self.k * cluster_size / rw

#     def calculate_robot_repulsive_potential(self, point, robot_position):
#         rd = np.sum(np.abs(np.array(point) - np.array(robot_position)))  # Manhattan distance
#         if rd < self.ds:
#             return self.kr * (self.ds - rd)
#         return 0

#     def find_next_best_viewpoints(self):
#         frontier_clusters = self.continuity_based_cluster(self.env.frontiers)
        
#         next_node_coords = []
#         for agent_id in range(self.num_agents):
#             current_pos = self.clip_coordinates(self.env.robot_positions[agent_id])
#             agent_clusters = frontier_clusters[agent_id]
            
#             if not agent_clusters:
#                 # If no frontiers for this agent, stay in place
#                 next_node_coords.append(current_pos)
#                 continue
            
#             centroids = [np.mean(cluster, axis=0).astype(int) for cluster in agent_clusters]
#             potentials = []
#             for centroid, cluster in zip(centroids, agent_clusters):
#                 centroid = self.clip_coordinates(centroid)
#                 frontier_potential = self.calculate_frontier_potential(current_pos, centroid, len(cluster))
#                 robot_repulsive_potential = sum(
#                     self.calculate_robot_repulsive_potential(centroid, self.clip_coordinates(other_robot_pos))
#                     for other_id, other_robot_pos in enumerate(self.env.robot_positions)
#                     if other_id != agent_id
#                 )
#                 total_potential = frontier_potential + robot_repulsive_potential
#                 potentials.append(total_potential)
            
#             best_cluster_index = np.argmin(potentials)
#             next_node = self.clip_coordinates(centroids[best_cluster_index])
#             next_node_coords.append(next_node)
        
#         return next_node_coords

#     def run_episode(self, currEpisode):
#         perf_metrics = dict()
#         print(f"Running episode {currEpisode}")
#         done = False
#         self.env.begin()
#         i = 0

#         while not done:
#             i += 1
#             next_node_coords = self.find_next_best_viewpoints()
#             done = self.env.step(next_node_coords)

#             if self.save_image:
#                 if not os.path.exists(gifs_path):
#                     os.makedirs(gifs_path)
#                 self.env.plot_env(self.global_step, gifs_path, i, [])

#             if done:
#                 perf_metrics['travel_dist'] = sum(self.env.travel_dists)
#                 perf_metrics['explored_rate'] = self.env.explored_rate
#                 perf_metrics['success_rate'] = True
#                 perf_metrics['relax_success_rate'] = True if self.env.explored_rate > self.env.finish_percent else False
#                 break

#         if self.save_image:
#             path1 = gifs_path
#             self.make_gif(path1, currEpisode)

#         return perf_metrics

#     def work(self, currEpisode):
#         self.currEpisode = currEpisode
#         self.perf_metrics = self.run_episode(currEpisode)

#     def make_gif(self, path, n):
#         print(f"Creating GIF for episode {n}")
#         with imageio.get_writer('{}/{}_explored_rate_{:.4g}_length_{:.4g}.gif'.format(path, n, self.env.explored_rate, sum(self.env.travel_dists)), mode='I',
#                                 duration=0.5) as writer:
#             for frame in self.env.frame_files:
#                 image = imageio.imread(frame)
#                 writer.append_data(image)
#         print('GIF complete\n')

#         # Remove files
#         for filename in self.env.frame_files[:-1]:
#             os.remove(filename)

# if __name__ == "__main__":
#     total_episode = 40
#     total_dist = 0
#     num_agents = 2  # Specify the number of agents here
    
#     for i in range(total_episode):
#         print(f"Starting episode {i+1}")
        
#         worker = MMPF_worker(metaAgentID=0, global_step=i, num_agents=num_agents, save_image=SAVE_GIFS)
#         performance = worker.run_episode(i)
#         total_dist += performance["travel_dist"]
#         mean_dist = total_dist / (i + 1)
#         print(f"Episode {i+1}/{total_episode}, Mean distance: {mean_dist:.2f}")

#     print(f"\nFinal mean distance over {total_episode} episodes: {mean_dist:.2f}")



# import numpy as np
# from scipy.spatial import distance
# from scipy.ndimage import distance_transform_edt
# import os
# import imageio

# class MMPF_worker:
#     def __init__(self, metaAgentID, global_step, num_agents=2, save_image=False):
#         print(f"Initializing MMPF worker with {num_agents} agents at step {global_step}")
#         self.metaAgentID = metaAgentID
#         self.global_step = global_step
#         self.save_image = save_image
#         self.num_agents = num_agents
#         self.env = Env(map_index=self.global_step, plot=save_image, test=True, num_agents=num_agents)
#         self.potential_scale = 1.0  # Equivalent to potential_scale_ in C++
#         self.gain_scale = 1.0  # Equivalent to gain_scale_ in C++
#         self.min_frontier_size = 3  # Equivalent to min_frontier_size_ in C++
#         self.distance_maps = {}  # Store distance maps for each centroid

#     def clip_coordinates(self, coords):
#         return np.clip(coords, [0, 0], np.array(self.env.obstacles.shape) - 1)

#     def search_frontiers(self, robot_position):
#         frontier_list = []
#         size_x, size_y = self.env.obstacles.shape
#         frontier_flag = np.zeros((size_x, size_y), dtype=bool)
#         visited_flag = np.zeros((size_x, size_y), dtype=bool)

#         # Find closest clear cell to start search
#         start_pos = self.find_nearest_clear_cell(robot_position)
#         if start_pos is None:
#             start_pos = robot_position

#         # BFS
#         queue = [start_pos]
#         visited_flag[start_pos] = True

#         while queue:
#             current = queue.pop(0)
#             for neighbor in self.get_neighbors(current):
#                 if not visited_flag[neighbor]:
#                     visited_flag[neighbor] = True
#                     if self.is_free_space(neighbor):
#                         queue.append(neighbor)
#                     elif self.is_new_frontier_cell(neighbor, frontier_flag):
#                         frontier_flag[neighbor] = True
#                         new_frontier = self.build_new_frontier(neighbor, robot_position, frontier_flag)
#                         if new_frontier.size * self.env.resolution >= self.min_frontier_size:
#                             frontier_list.append(new_frontier)

#         # Set costs of frontiers
#         for frontier in frontier_list:
#             frontier.cost = self.frontier_cost(frontier)
        
#         frontier_list.sort(key=lambda x: x.cost)
#         return frontier_list

#     def find_nearest_clear_cell(self, position):
#         # Implement logic to find nearest clear cell
#         # For simplicity, we'll just return the position if it's clear
#         if self.is_free_space(position):
#             return position
#         return None

#     def get_neighbors(self, cell):
#         x, y = cell
#         neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
#         return [n for n in neighbors if self.is_valid_cell(n)]

#     def is_valid_cell(self, cell):
#         x, y = cell
#         return 0 <= x < self.env.obstacles.shape[0] and 0 <= y < self.env.obstacles.shape[1]

#     def is_free_space(self, cell):
#      x, y = cell
#      return not self.env.obstacles[x, y]


#     def is_new_frontier_cell(self, cell, frontier_flag):
#         if self.env.obstacles[cell] or frontier_flag[cell]:
#             return False
#         return any(self.is_free_space(neighbor) for neighbor in self.get_neighbors(cell))

#     def build_new_frontier(self, initial_cell, reference, frontier_flag):
#         frontier = Frontier()
#         frontier.size = 1
#         frontier.min_distance = float('inf')

#         queue = [initial_cell]
#         while queue:
#             cell = queue.pop(0)
#             for neighbor in self.get_neighbors(cell):
#                 if self.is_new_frontier_cell(neighbor, frontier_flag):
#                     frontier_flag[neighbor] = True
#                     frontier.points.append(neighbor)
#                     frontier.size += 1
                    
#                     distance = np.linalg.norm(np.array(neighbor) - np.array(reference))
#                     if distance < frontier.min_distance:
#                         frontier.min_distance = distance
#                         frontier.middle = neighbor

#                     queue.append(neighbor)

#         frontier.centroid = np.mean(frontier.points, axis=0)
#         return frontier

#     def frontier_cost(self, frontier):
#         return (self.potential_scale * frontier.min_distance * self.env.resolution) - \
#                (self.gain_scale * frontier.size * self.env.resolution)

#     def find_next_best_viewpoints(self):
#         next_node_coords = []
#         for agent_id in range(self.num_agents):
#             current_pos = self.clip_coordinates(self.env.robot_positions[agent_id])
#             frontiers = self.search_frontiers(current_pos)
            
#             if not frontiers:
#                 next_node_coords.append(current_pos)
#                 continue
            
#             best_frontier = frontiers[0]  # The frontier with the lowest cost
#             next_node = self.clip_coordinates(best_frontier.middle)
#             next_node_coords.append(next_node)
        
#         return next_node_coords

#     def run_episode(self, currEpisode):
#         perf_metrics = dict()
#         print(f"Running episode {currEpisode}")
#         done = False
#         self.env.begin()
#         i = 0

#         while not done:
#             i += 1
#             next_node_coords = self.find_next_best_viewpoints()
#             done = self.env.step(next_node_coords)

#             if self.save_image:
#                 if not os.path.exists(gifs_path):
#                     os.makedirs(gifs_path)
#                 self.env.plot_env(self.global_step, gifs_path, i, [])

#             if done:
#                 perf_metrics['travel_dist'] = sum(self.env.travel_dists)
#                 perf_metrics['explored_rate'] = self.env.explored_rate
#                 perf_metrics['success_rate'] = True
#                 perf_metrics['relax_success_rate'] = True if self.env.explored_rate > self.env.finish_percent else False
#                 break

#         if self.save_image:
#             path1 = gifs_path
#             self.make_gif(path1, currEpisode)

#         return perf_metrics

#     def work(self, currEpisode):
#         self.currEpisode = currEpisode
#         self.perf_metrics = self.run_episode(currEpisode)

#     def make_gif(self, path, n):
#         print(f"Creating GIF for episode {n}")
#         with imageio.get_writer('{}/{}_explored_rate_{:.4g}_length_{:.4g}.gif'.format(path, n, self.env.explored_rate, sum(self.env.travel_dists)), mode='I',
#                                 duration=0.5) as writer:
#             for frame in self.env.frame_files:
#                 image = imageio.imread(frame)
#                 writer.append_data(image)
#         print('GIF complete\n')

#         # Remove files
#         for filename in self.env.frame_files[:-1]:
#             os.remove(filename)

# class Frontier:
#     def __init__(self):
#         self.points = []
#         self.centroid = None
#         self.size = 0
#         self.min_distance = float('inf')
#         self.middle = None
#         self.cost = 0

# if __name__ == "__main__":
#     total_episode = 40
#     total_dist = 0
#     num_agents = 2  # Specify the number of agents here
    
#     for i in range(total_episode):
#         print(f"Starting episode {i+1}")
        
#         worker = MMPF_worker(metaAgentID=0, global_step=i, num_agents=num_agents, save_image=SAVE_GIFS)
#         performance = worker.run_episode(i)
#         total_dist += performance["travel_dist"]
#         mean_dist = total_dist / (i + 1)
#         print(f"Episode {i+1}/{total_episode}, Mean distance: {mean_dist:.2f}")

#     print(f"\nFinal mean distance over {total_episode} episodes: {mean_dist:.2f}")












import numpy as np
from scipy.spatial import distance
from scipy.ndimage import distance_transform_edt
import imageio
import os
import random
import shapely.geometry
from NBVP_env import Env
from test_parameter import *

gifs_path = f'results/mmpf'

class TreeRRT:
    def __init__(self, start_coords, env):
        self.vertices = dict()
        self.env = env
        self.add_vertex(0, -1, start_coords)

    def add_vertex(self, vertex_id, parent_id, coords):
        self.vertices[vertex_id] = {'parent_id': parent_id, 'coords': coords}
    
    def get_nearest_vertex(self, sample_coords):
        vertices_coords = np.array([v['coords'] for v in self.vertices.values()])
        dist_list = np.linalg.norm(sample_coords - vertices_coords, axis=-1)
        nearest_vertex_id = np.argmin(dist_list)
        return nearest_vertex_id, vertices_coords[nearest_vertex_id]

class MMPF_RRT_worker:
    def __init__(self, metaAgentID, global_step, num_agents=2, save_image=False):
        print(f"Initializing MMPF-RRT worker with {num_agents} agents at step {global_step}")
        self.metaAgentID = metaAgentID
        self.global_step = global_step
        self.save_image = save_image
        self.num_agents = num_agents
        self.env = Env(map_index=self.global_step, plot=save_image, test=True, num_agents=num_agents)
        self.k = 5.0  # Constant for frontier potential field calculation
        self.kr = 1.0  # Constant for robot repulsive potential field calculation
        self.ds = 25 # Sensor range
        self.distance_maps = {}  # Store distance maps for each centroid
        self.step_length = 10  # RRT step length


    def clip_coordinates(self, coords):
        return np.clip(coords, [0, 0], np.array(self.env.obstacles.shape) - 1)

    def continuity_based_cluster(self, frontiers):
        all_clusters = []
        for agent_id in range(self.num_agents):
            clusters = []
            agent_frontiers = frontiers.tolist()
            
            while agent_frontiers:
                new_cluster = [agent_frontiers.pop(0)]
                
                cluster_changed = True
                while cluster_changed:
                    cluster_changed = False
                    frontiers_to_remove = []
                    
                    for frontier in agent_frontiers:
                        if any(self.is_neighbor(frontier, cluster_frontier) for cluster_frontier in new_cluster):
                            new_cluster.append(frontier)
                            frontiers_to_remove.append(frontier)
                            cluster_changed = True
                    
                    for frontier in frontiers_to_remove:
                        agent_frontiers.remove(frontier)
                
                clusters.append(new_cluster)
            
            all_clusters.append(clusters)
        
        return all_clusters

    def is_neighbor(self, frontier1, frontier2, threshold=1):
        return np.linalg.norm(np.array(frontier1) - np.array(frontier2)) <= threshold

    def calculate_wave_front_distance(self, start, goal):
        start = self.clip_coordinates(start)
        goal = self.clip_coordinates(goal)
        
        if tuple(goal) not in self.distance_maps:
            grid = np.ones_like(self.env.obstacles, dtype=float)
            grid[self.env.obstacles] = np.inf
            grid[tuple(goal)] = 0
            self.distance_maps[tuple(goal)] = distance_transform_edt(grid)
        
        return self.distance_maps[tuple(goal)][tuple(start)]

    def calculate_frontier_potential(self, point, centroid, cluster_size):
        rw = self.env.get_distance_map(centroid)[tuple(point)]
        return -self.k * cluster_size / rw if rw > 0 else float('inf')

    def calculate_robot_repulsive_potential(self, point, robot_position):
        rd = np.sum(np.abs(np.array(point) - np.array(robot_position)))  # Manhattan distance
        if rd < self.ds:
            return self.kr * (self.ds - rd)
        return 0

    def check_collision(self, start, end, robot_belief):
        collision = False
        line = shapely.geometry.LineString([start, end])

        sortx = np.sort([int(start[0]), int(end[0])])
        sorty = np.sort([int(start[1]), int(end[1])])

      #  robot_belief = robot_belief[sorty[0]:sorty[1] + 1, sortx[0]:sortx[1] + 1]
        robot_belief = self.env.robot_belief[sorty[0]:sorty[1] + 1, sortx[0]:sortx[1] + 1]

        occupied_area_index = np.where(robot_belief == 1)
        occupied_area_coords = np.asarray(
                [occupied_area_index[1] + sortx[0], occupied_area_index[0] + sorty[0]]).T
        unexplored_area_index = np.where(robot_belief == 127)
        unexplored_area_coords = np.asarray(
                [unexplored_area_index[1] + sortx[0], unexplored_area_index[0] + sorty[0]]).T
        unfree_area_coords = occupied_area_coords

        for i in range(unfree_area_coords.shape[0]):
            coords = ([(unfree_area_coords[i][0] -5, unfree_area_coords[i][1] -5),
                   (unfree_area_coords[i][0] + 5, unfree_area_coords[i][1] -5),
                   (unfree_area_coords[i][0] - 5, unfree_area_coords[i][1] + 5),
                   (unfree_area_coords[i][0] + 5, unfree_area_coords[i][1] + 5)])
            obstacle = shapely.geometry.Polygon(coords)
            if abs(end[0] - unfree_area_coords[i][0] <= 8) and abs(end[1] - unfree_area_coords[i][1] <= 8):
                collision = True
            if not collision:
                collision = line.intersects(obstacle)
            if collision:
                break

        if not collision:
            unfree_area_coords = unexplored_area_coords
            for i in range(unfree_area_coords.shape[0]):
                coords = ([(unfree_area_coords[i][0], unfree_area_coords[i][1]),
                           (unfree_area_coords[i][0] + 1, unfree_area_coords[i][1]),
                           (unfree_area_coords[i][0], unfree_area_coords[i][1]),
                           (unfree_area_coords[i][0] + 1, unfree_area_coords[i][1] + 1)])
                obstacle = shapely.geometry.Polygon(coords)
                collision = line.intersects(obstacle)
                if collision:
                    break

        return collision

    def steer(self, start_coords, goal_coords):
        direction = (goal_coords - start_coords)
        distance = np.linalg.norm(direction)
        step_size = min(self.step_length, distance)
        new_coords = start_coords + (direction / distance) * (step_size+1)
        return new_coords

    def extract_route(self, tree, vertex_id):
        route = []
        current_id = vertex_id
        while current_id != -1:
            route.append(tree.vertices[current_id]['coords'])
            current_id = tree.vertices[current_id]['parent_id']
        return route[::-1]

    def rrt_plan(self, start, goal):
        max_iter_steps = 50
        tree = TreeRRT(start, self.env)
        
        for _ in range(max_iter_steps):
            sample_coords = self.clip_coordinates(np.random.rand(2) * self.env.obstacles.shape)
            nearest_vertex_id, nearest_vertex_coords = tree.get_nearest_vertex(sample_coords)
            
            new_vertex_coords = self.steer(nearest_vertex_coords, sample_coords)
            if not self.check_collision(nearest_vertex_coords, new_vertex_coords, self.env.robot_belief):
                new_vertex_id = len(tree.vertices)
                tree.add_vertex(new_vertex_id, nearest_vertex_id, new_vertex_coords)
                
                if np.linalg.norm(new_vertex_coords - goal) < self.step_length:
                    return self.extract_route(tree, new_vertex_id)

        return None

    def find_next_best_viewpoints(self):
        frontier_clusters = self.continuity_based_cluster(self.env.frontiers)
        self.env.update_combined_belief()
        next_node_coords = []
        planned_paths = []
        for agent_id in range(self.num_agents):
            current_pos = self.clip_coordinates(self.env.robot_positions[agent_id])
            agent_clusters = frontier_clusters[agent_id]
            
            if not agent_clusters:
                next_node_coords.append(current_pos)
                planned_paths.append([current_pos])
                continue
            
            centroids = [np.mean(cluster, axis=0).astype(int) for cluster in agent_clusters]
            potentials = []
            for centroid, cluster in zip(centroids, agent_clusters):
                centroid = self.clip_coordinates(centroid)
                frontier_potential = self.calculate_frontier_potential(current_pos, centroid, len(cluster))
                robot_repulsive_potential = sum(
                    self.calculate_robot_repulsive_potential(centroid, self.clip_coordinates(other_robot_pos))
                    for other_id, other_robot_pos in enumerate(self.env.robot_positions)
                    if other_id != agent_id
                )
                total_potential = frontier_potential + robot_repulsive_potential
                potentials.append(total_potential)
            
            best_cluster_index = np.argmin(potentials)
            goal = self.clip_coordinates(centroids[best_cluster_index])
            
            path = self.rrt_plan(current_pos, goal)
            if path is not None and len(path) > 1:
                next_node = path[1]
                planned_path = path
            else:
                next_node = goal
                planned_path = [current_pos, goal]
            
            next_node_coords.append(next_node)
            planned_paths.append(planned_path)
        
        return next_node_coords, planned_paths

    def run_episode(self, currEpisode):
        perf_metrics = dict()
        print(f"Running episode {currEpisode}")
        done = False
        self.env.begin()
        i = 0

        while not done:
            i += 1
            next_node_coords, planned_paths = self.find_next_best_viewpoints()
            done = self.env.step(next_node_coords)

            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot_env(self.global_step, gifs_path, i, planned_paths)

            if done:
                perf_metrics['travel_dist'] = sum(self.env.travel_dists)
                perf_metrics['explored_rate'] = self.env.explored_rate
                perf_metrics['success_rate'] = True
                perf_metrics['relax_success_rate'] = True if self.env.explored_rate > self.env.finish_percent else False
                break

        if self.save_image:
            path1 = gifs_path
            self.make_gif(path1, currEpisode)

        return perf_metrics

    def work(self, currEpisode):
        self.currEpisode = currEpisode
        self.perf_metrics = self.run_episode(currEpisode)

    def make_gif(self, path, n):
        print(f"Creating GIF for episode {n}")
        with imageio.get_writer('{}/{}_explored_rate_{:.4g}_length_{:.4g}.gif'.format(path, n, self.env.explored_rate, sum(self.env.travel_dists)), mode='I',
                                duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('GIF complete\n')

        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)

if __name__ == "__main__":
    total_episode = 40
    total_dist = 0
    num_agents = 2  # Specify the number of agents here
    
    for i in range(total_episode):
        print(f"Starting episode {i+1}")
        
        worker = MMPF_RRT_worker(metaAgentID=0, global_step=i, num_agents=num_agents, save_image=SAVE_GIFS)
        performance = worker.run_episode(i)
        total_dist += performance["travel_dist"]
        mean_dist = total_dist / (i + 1)
        print(f"Episode {i+1}/{total_episode}, Mean distance: {mean_dist:.2f}")

    print(f"\nFinal mean distance over {total_episode} episodes: {mean_dist:.2f}")


    #working apf