import pybullet as p
import pybullet_data
import time
from useful_code import *
import random
from matplotlib import pyplot as plt   


def check_node_collision(robot_id, object_ids, joint_position):
    """
    Checks for collisions between a robot and an object in PyBullet. 

    Args:
        robot_id (int): The ID of the robot in PyBullet.
        object_id (int): The ID of the object in PyBullet.
        joint_position (list): List of joint positions. 

    Returns:
        bool: True if a collision is detected, False otherwise.
    """
    # set joint positions
    for joint_index, joint_pos in enumerate(joint_position):
        p.resetJointState(robot_id, joint_index, joint_pos)

    # Perform collision check for all links
    for object_id in object_ids:    # Check for each object
        for link_index in range(0, p.getNumJoints(robot_id)): # Check for each link of the robot
            contact_points = p.getClosestPoints(
                bodyA=robot_id, bodyB=object_id, distance=0.01, linkIndexA=link_index
            )
            if contact_points:  # If any contact points exist, a collision is detected
                return True # exit early
    return False


#################################################
#### YOUR CODE HERE: COLLISION EDGE CHECKING ####
#################################################
def check_edge_collision(robot_id, object_ids, joint_position_start, joint_position_end, discretization_step=0.01):
    """ 
    Checks for collision between two joint positions of a robot in PyBullet.
    Args:
        robot_id (int): The ID of the robot in PyBullet.
        object_ids (list): List of IDs of the objects in PyBullet.
        joint_position_start (list): List of joint positions to start from.
        joint_position_end (list): List of joint positions to get to.
        discretization_step (float): maximum interpolation distance before a new collision check is performed.
    Returns:
        bool: True if a collision is detected, False otherwise.
    """
    pass


# Provided
class Node:

    def __init__(self, joint_angles):
        self.joint_angles = np.array(joint_angles)  # joint angles of the node in n-dimensional space
        self.parent = None


######################################################################
##################### YOUR CODE HERE: RRT CLASS ######################
######################################################################
class RRT:

    def __init__(self, q_start, q_goal, robot_id, obstacle_ids, q_limits, max_iter=10000, step_size=0.5):
        """
        RRT Initialization.

        Parameters:
        - q_start: List of starting joint angles [x1, x2, ..., xn].
        - q_goal: List of goal joint angles [x1, x2, ..., xn].
        - obstacle_ids: List of obstacles, each as a tuple ([center1, center2, ..., centern], radius).
        - q_limits: List of tuples [(min_x1, max_x1), ..., (min_xn, max_xn)] representing the limits in each dimension.
        - max_iter: Maximum number of iterations.
        - step_size: Maximum step size to expand the tree.
        """
        self.q_start = Node(q_start)
        self.q_goal = Node(q_goal)
        self.obstacle_ids = obstacle_ids
        self.robot_id = robot_id
        self.q_limits = q_limits
        self.max_iter = max_iter
        self.step_size = step_size
        self.node_list = [self.q_start]

    def step(self, from_node, to_joint_angles):
        """Step from "from_node" to "to_joint_angles", that should
         (a) return the to_joint_angles if it is within the self.step_size or
         (b) only step so far as self.step_size, returning the new node within that distance"""
        pass

    def get_nearest_node(self, random_point):
        """Find the nearest node in the tree to a given point."""
        pass

    def plan(self):
        """Run the RRT algorithm to find a path of dimension Nx3. Limit the search to only max_iter iterations."""
        pass


#####################################################
##################### MAIN CODE #####################
#####################################################

if __name__ == "__main__":
    
    #######################
    #### PROBLEM SETUP ####
    #######################

    # Initialize PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) # For default URDFs
    p.setGravity(0, 0, -9.8)

    # Load the plane and robot arm
    ground_id = p.loadURDF("plane.urdf")
    arm_id = p.loadURDF("three_link_arm.urdf", [0, 0, 0], useFixedBase=True)

    # Add Collision Objects
    collision_ids = [ground_id] # add the ground to the collision list
    collision_positions = [[0.3, 0.5, 0.251], [-0.3, 0.3, 0.101], [-1, -0.15, 0.251], [-1, -0.15, 0.752], [-0.5, -1, 0.251], [0.5, -0.35, 0.201], [0.5, -0.35, 0.602]]
    collision_orientations =  [[0, 0, 0.5], [0, 0, 0.2], [0, 0, 0],[0, 0, 1], [0, 0, 0], [0, 0, .25], [0, 0, 0.5]]
    collision_scales = [0.5, 0.25, 0.5, 0.5, 0.5, 0.4, 0.4]
    for i in range(len(collision_scales)):
        collision_ids.append(p.loadURDF("cube.urdf",
            basePosition=collision_positions[i],  # Position of the cube
            baseOrientation=p.getQuaternionFromEuler(collision_orientations[i]),  # Orientation of the cube
            globalScaling=collision_scales[i]  # Scale the cube to half size
        ))

    # Goal Joint Positions for the Robot
    goal_positions = [[-2.54, 0.15, -0.15], [-1.82,0.15,-0.15],[0.5, 0.15,-0.15], [1.7,0.2,-0.15],[-2.54, 0.15, -0.15]]

    # Joint Limits of the Robot
    joint_limits = [[-np.pi, np.pi], [0, np.pi], [-np.pi, np.pi]]

    # A3xN path array that will be filled with waypoints through all the goal positions
    path_saved = np.array([[-2.54, 0.15, -0.15]]) # Start at the first goal position

    ####################################################################################################
    #### YOUR CODE HERE: RUN RRT MOTION PLANNER FOR ALL goal_positions (starting at goal position 1) ###
    ####################################################################################################
   


    ################################################################################
    ####  RUN THE SIMULATION AND MOVE THE ROBOT ALONG PATH_SAVED ###################
    ################################################################################

    # Set the initial joint positions
    for joint_index, joint_pos in enumerate(goal_positions[0]):
        p.resetJointState(arm_id, joint_index, joint_pos)

    # Move through the waypoints
    for waypoint in path_saved:
        # "move" to next waypoints
        for joint_index, joint_pos in enumerate(waypoint):
        # run velocity control until waypoint is reached
            while True:
                #get current joint positions
                goal_positions = [p.getJointState(arm_id, i)[0] for i in range(3)]
                # calculate the displacement to reach the next waypoint
                displacement_to_waypoint = waypoint-goal_positions
                # check if goal is reached
                max_speed = 0.05
                if(np.linalg.norm(displacement_to_waypoint) < max_speed):
                    break
                else:
                    # calculate the "velocity" to reach the next waypoint
                    velocities = np.min((np.linalg.norm(displacement_to_waypoint), max_speed))*displacement_to_waypoint/np.linalg.norm(displacement_to_waypoint)
                    for joint_index, joint_step in enumerate(velocities):
                        p.setJointMotorControl2(
                            bodyIndex=arm_id,
                            jointIndex=joint_index,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=joint_step,
                        )
                        
                #Take a simulation step
                p.stepSimulation()            
        time.sleep(1.0 / 240.0)


    # Disconnect from PyBullet
    time.sleep(100) # Remove this line -- it is just to keep the GUI open when you first run this starter code
    p.disconnect()
