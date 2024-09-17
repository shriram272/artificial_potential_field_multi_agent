This project was part of research done in collaboration with NUS Singapore for collecting baselines for multi-agent collaborative exploration with limited field of view sensors.

Here Artificial Potential Fields are used to guide the robots to explore while avoiding obstacles.It based on the following research paper cited below.


Yu, Jincheng & Tong, Jianming & Xu, Yuanfan & Xu, Zhilin & Dong, Haolin & Yang, Tianxiang & Wang, Yu. (2021). SMMR-Explore: SubMap-based Multi-Robot Exploration System with Multi-robot Multi-target Potential Field Exploration Method 

LOGIC and EXPLANATION


1. TreeRRT Class

This class handles the creation and management of the RRT.

    Vertices and Nearest Vertex Search:
        self.vertices stores nodes in the tree with their coordinates and parent relationships.
        get_nearest_vertex: Finds the nearest vertex to a sampled point by calculating the Euclidean distance to all vertices using:
        distance=(x2−x1)2+(y2−y1)2
        distance=(x2​−x1​)2+(y2​−y1​)2

        ​ This function uses np.linalg.norm to compute these distances efficiently.

2. MMPF_RRT_worker Class

This class implements the main exploration logic using RRT and APF.
a. Continuity-based Clustering

    Clustering Frontiers:
        The continuity_based_cluster method clusters frontiers (potential exploration targets) based on spatial continuity. It groups nearby frontiers into clusters using a simple distance threshold (e.g., neighbors within 1 unit distance are clustered together).

b. Potential Fields

The algorithm uses potential fields to guide the robots towards frontiers while avoiding other robots and obstacles.

    Frontier Potential:
        calculate_frontier_potential: Computes the attractive potential of a cluster centroid. This potential is inversely proportional to the distance (rw) to the frontier centroid:
        frontier potential=−k×cluster sizerw
        frontier potential=−rwk×cluster size​
            k is a constant that scales the potential.
            The greater the cluster size, the stronger the attraction, encouraging robots to explore larger unexplored areas.

    Robot Repulsive Potential:
        calculate_robot_repulsive_potential: Computes the repulsive potential from other robots using the Manhattan distance (rd):
        robot repulsive potential={kr×(ds−rd)if rd<ds0otherwise
        robot repulsive potential={kr​×(ds−rd)0​if rd<dsotherwise​
            kr is a constant for scaling the repulsive effect.
            ds is the sensor range; robots avoid getting too close to each other by adding repulsion if within this range.

c. Wavefront Distance Calculation

    Wavefront Propagation:
        calculate_wave_front_distance: Uses the Euclidean distance transform (distance_transform_edt) to compute the shortest path distances in a grid-like environment. This function generates a distance map where each cell contains the minimum distance to the goal.   


RESULTS


![0_1_samples](https://github.com/user-attachments/assets/195e3fcb-e3e7-4ff3-a220-89678832284b)

![0_9_samples](https://github.com/user-attachments/assets/b902fcbb-a70f-4474-9142-967117066ed7)

![0_16_samples](https://github.com/user-attachments/assets/07d1c5e1-e39e-447a-a50e-f6ab6edaf92c)

        
