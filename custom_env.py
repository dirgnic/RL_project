import gymnasium as gym
import numpy as np
import pygame
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, SineLane, CircularLane, LineType
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import IDMVehicle
class MyCustomEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {"type": "Kinematics"},
            "action": {"type": "ContinuousAction"},
            "simulation_frequency": 15,
            "policy_frequency": 1, 
            "duration": 500,
            "screen_width": 1280,
            "screen_height": 740,
            "scaling": 5.5,                 # Explicitly set scaling (pixels per meter)
            "centering_position": [0.3, 0.5] # Place ego-vehicle at 30% X, 50% Y of screen
        })
        return config
    

    def get_new_road_coords(self, connected_road, new_road_length):
        start_point_next = connected_road.position(connected_road.length, 0)
        
        heading_next = connected_road.heading_at(connected_road.length)
        next_road_length = 100

        end_point_next = start_point_next + new_road_length * np.array([
            np.cos(heading_next), 
            np.sin(heading_next)
        ])
        return start_point_next, end_point_next



    def _make_road(self):
        net = RoadNetwork()

        # 1. Create the first Straight Lane
        net.add_lane("a1", "merge", StraightLane([0, 0], [50, 0], speed_limit=30, width=15))
        net.add_lane("merge","a2",StraightLane([50, 0], [100, 0], speed_limit=30, width=15))
        net.add_lane("b0","b0_merge", StraightLane([50,-100],[50,-7.5],speed_limit=30, width=15))
        net.add_lane("b0_merge","merge", StraightLane([50,-7.5],[50,0],speed_limit=30, width=15,line_types=[LineType.NONE, LineType.NONE]))
        
        # 2. Create the Circular Lane Object (but keep it in a variable)
        # 270 to 360 is a bottom-to-right curve (Counter-Clockwise)
        turn_lane = CircularLane(
            center=[100, 60],
            radius=60,
            start_phase=np.radians(270),
            end_phase=np.radians(360),
            width=15
        )
        # Add it to the network
        net.add_lane("a2", "a3", turn_lane)
        start_point_next, end_point_next = self.get_new_road_coords(turn_lane,100)
        turn_lane = StraightLane(
            start_point_next, 
            end_point_next, 
            speed_limit=30, 
            width=15
        )
        net.add_lane("a3", "a4", turn_lane)
        start_point_next, end_point_next = self.get_new_road_coords(turn_lane,20)
        turn_lane = CircularLane(
            center = [start_point_next[0]-60,start_point_next[1]],
            radius=60,
            start_phase=np.radians(0),
            end_phase=np.radians(90),
            speed_limit=30,
            width=15
        )
        net.add_lane("a4","a5",turn_lane)
        start_point_next,end_point_next=self.get_new_road_coords(turn_lane,100)
        midpoint = start_point_next + (end_point_next - start_point_next) * 0.5
        turn_lane=StraightLane(
            start_point_next,
            midpoint,
            speed_limit=30,
            width=15
        )
        print(start_point_next,end_point_next)
        net.add_lane("a5","a5_merge",turn_lane)
        turn_lane=StraightLane(
            midpoint,
            end_point_next,
            speed_limit=30,
            width=15
        )
        net.add_lane("a5_merge","a6",turn_lane)
        aux=turn_lane
        '''ccc
        '''
        midpoint=midpoint+(end_point_next-midpoint)*0.5
        turn_lane=StraightLane(
            midpoint,
            [midpoint[0],midpoint[1]+7.5],
            speed_limit=30,
            width=15,
            line_types=[LineType.NONE,LineType.NONE]
        )
        net.add_lane("a5_merge","c0_merge",turn_lane)
        turn_lane=StraightLane(
            [midpoint[0],midpoint[1]+7.5],
            [midpoint[0],midpoint[1]+100],
            width=15
        )
        net.add_lane("c0_merge","c0",turn_lane)
        turn_lane=aux
        start_point_next,end_point_next=self.get_new_road_coords(turn_lane,20)
        turn_lane = CircularLane(
            center = [start_point_next[0],start_point_next[1]-60],
            radius=60,
            start_phase=np.radians(90),
            end_phase=np.radians(180),
            speed_limit=30,
            width=15
        )
        net.add_lane("a6","a7",turn_lane)
        start_point_next,end_point_next=self.get_new_road_coords(turn_lane,100)
        turn_lane=StraightLane(
            start_point_next,
            end_point_next,
            speed_limit=30,
            width=15
        )
        net.add_lane("a7","a8",turn_lane)
        start_point_next,end_point_next=self.get_new_road_coords(turn_lane,60)
        turn_lane=CircularLane(
            center = [start_point_next[0]+60,start_point_next[1]],
            radius=60,
            start_phase=np.radians(180),
            end_phase=np.radians(270),
            speed_limit=30,
            width=15            
        )
        net.add_lane("a8","a1",turn_lane)
        self.road = Road(
            network=net, 
            np_random=self.np_random, 
            record_history=self.config["show_trajectories"]
        )
    def _make_vehicles(self):
        self.vehicle = Vehicle(
            self.road,
            position=[0, 0],
            speed=2.0,
            heading=0
        )
        self.road.vehicles.append(self.vehicle)
        obstacle = IDMVehicle(self.road, position=[75, 0], speed=5)
        obstacle2 = IDMVehicle(self.road, position=[100, 200], speed=5)
        obstacle3 = IDMVehicle(self.road, position=[50, -75], speed=7)
        for i in range(0,10):
            vehicle=IDMVehicle(self.road, position=[np.random.randint(0,100), np.random.randint(0,200)],speed=np.random.randint(3,7))
            vehicle.lane_index=self.road.network.get_closest_lane_index(vehicle.position)
            self.road.vehicles.append(vehicle)
        obstacle.lane_index=self.road.network.get_closest_lane_index(obstacle.position)
        obstacle2.lane_index=self.road.network.get_closest_lane_index(obstacle2.position)
        obstacle3.lane_index=self.road.network.get_closest_lane_index(obstacle3.position)
        path = self.road.network.shortest_path("b0", "c0")
        obstacle3.plan_route_to("c0")
        self.road.vehicles.append(obstacle)
        self.road.vehicles.append(obstacle2)
        self.road.vehicles.append(obstacle3)
    def _reward(self, action):
        # 1. Weights (Tune these!)
        COLLISION_PENALTY = -1.0  # Normalized to -1 (easier for PPO than -1000)
        OFF_ROAD_PENALTY = -1.0
        HIGH_SPEED_REWARD = 0.5
        LANE_CENTERING_REWARD = 0.5


        # 2. Check Failures First (Terminal states)
        if self.vehicle.crashed:
            return COLLISION_PENALTY
        if not self.vehicle.on_road:
            return OFF_ROAD_PENALTY
        lane = self.vehicle.lane
        lane_heading = lane.heading_at(self.vehicle.position[0]) # approx using X or longitudinal
        heading_error = np.abs(self.vehicle.heading - lane_heading)
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        
        # Cosine Similarity: 
        # 1.0 if perfectly aligned
        # 0.0 if 90 degrees sideways
        # -1.0 if going backwards
        alignment_factor = np.cos(heading_error)
        
        # Scaled Speed Reward:
        # If we are going 30m/s aligned: Reward is high.
        # If we are going 30m/s backwards: Reward is negative (Punishment).
        speed_reward = (self.vehicle.speed / lane.speed_limit) * alignment_factor

        # 4. Calculate Lane Centering (Gaussian Bell Curve)
        # Using the vehicle's lateral position relative to the lane center
        # .local_coordinates returns [longitudinal, lateral]
        long, lat = lane.local_coordinates(self.vehicle.position)
        
        # This returns 1.0 if exactly in center, drops to 0.0 as we move away
        centering_reward = np.exp(-1 * (lat**2))
        total_reward = (HIGH_SPEED_REWARD * speed_reward) + \
                       (LANE_CENTERING_REWARD * centering_reward)        
        return total_reward
    
    def _reset(self) -> None :
        self._make_road()
        self._make_vehicles()
    def _is_terminated(self):
        # Stop if we crash or leave the road
        return self.vehicle.crashed or not self.vehicle.on_road

    def _is_truncated(self):
        # Truncate when the time limit specified in the config is reached
        return self.time >= self.config["duration"]


if __name__ == "__main__":
    # Manual keyboard control demo - only runs when executing this file directly
    import time
    
    # 1. Instantiate the environment class directly
    env = MyCustomEnv(render_mode='human') 

    # 2. Reset (Builds the road and vehicles)
    obs, info = env.reset()

    print("Environment created. Starting simulation...")
    failures = 0
    while failures < 1:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        
        # 3. Map Keys to Action Vector [Acceleration, Steering]
        # Range is usually [-1, 1] for ContinuousAction
        acceleration = 0.0
        steering = 0.0
        
        if keys[pygame.K_UP]:
            acceleration = 0.1   # Accelerate
        elif keys[pygame.K_DOWN]:
            acceleration = -0.05  # Brake/Reverse
            
        if keys[pygame.K_LEFT]:
            steering = -0.5      # Turn Left
        elif keys[pygame.K_RIGHT]:
            steering = 0.5       # Turn Right
        
        if keys[pygame.K_0]:
            failures = 10
        
        action = [acceleration, steering]
        obs, reward, terminated, truncated, info = env.step(action)
        # Render the graphics (AbstractEnv handles the PyGame window for you!)
        env.render()
        
        if terminated or truncated:
            failures += 1
            obs, info = env.reset()

    env.close()