import gymnasium as gym
import numpy as np
import pygame
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, SineLane, CircularLane, LineType
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import IDMVehicle
class MyCustomEnv(AbstractEnv):
    def _observe(self):
        obs = super()._observe()
        # Add extra features: fuel, passenger info, and traffic penalty
        fuel_level = getattr(self.vehicle, 'fuel', 1.0)
        passenger_count = len(getattr(self, 'passengers', []))
        passenger_onboard = int(self.vehicle.passenger is not None)
        traffic_penalty = int(self._on_traffic_tile(self.vehicle.position))
        # If obs is a dict, add fields; if array, append
        if isinstance(obs, dict):
            obs['fuel_level'] = fuel_level
            obs['passenger_count'] = passenger_count
            obs['passenger_onboard'] = passenger_onboard
            obs['traffic_penalty'] = traffic_penalty
        elif isinstance(obs, np.ndarray):
            obs = np.concatenate([obs, [fuel_level, passenger_count, passenger_onboard, traffic_penalty]])
        return obs
    def _init_taxi_upgrades(self):
        # Define map features for Taxi-v3 upgrades
        self.passengers = [
            {'pickup': (10, 10), 'dropoff': (90, 90), 'onboard': False, 'delivered': False},
            {'pickup': (80, 20), 'dropoff': (20, 80), 'onboard': False, 'delivered': False}
        ]
        self.fuel_capacity = 100.0
        self.fuel_consumption_per_step = 1.0
        self.refuel_stations = [(50, 0), (0, 50)]
        self.traffic_tiles = [(60, 60), (40, 40), (70, 30)]
        self.traffic_penalty = -0.5
        self.fuel_penalty = -1.0
        self.pickup_reward = 1.0
        self.dropoff_reward = 5.0
        self.refuel_amount = 100.0
        self.max_steps = 300
        self.current_step = 0

    def _on_traffic_tile(self, pos):
        # Simple check: if within 5 units of a traffic tile
        for t in self.traffic_tiles:
            if np.linalg.norm(np.array(pos) - np.array(t)) < 5.0:
                return True
        return False

    def _on_refuel_station(self, pos):
        for s in self.refuel_stations:
            if np.linalg.norm(np.array(pos) - np.array(s)) < 5.0:
                return True
        return False

    def _check_pickup(self):
        for p in self.passengers:
            if not p['onboard'] and not p['delivered'] and np.linalg.norm(np.array(self.vehicle.position) - np.array(p['pickup'])) < 5.0:
                p['onboard'] = True
                self.vehicle.passenger = p
                return self.pickup_reward
        return 0.0

    def _check_dropoff(self):
        if hasattr(self.vehicle, 'passenger') and self.vehicle.passenger is not None:
            p = self.vehicle.passenger
            if np.linalg.norm(np.array(self.vehicle.position) - np.array(p['dropoff'])) < 5.0:
                p['onboard'] = False
                p['delivered'] = True
                self.vehicle.passenger = None
                return self.dropoff_reward
        return 0.0

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {"type": "Kinematics"},
            "action": {"type": "ContinuousAction"},
            "simulation_frequency": 15,
            "policy_frequency": 1, 
            "duration": 300,
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
            speed=0.3,
            heading=0
        )
        self.vehicle.fuel = self.fuel_capacity
        self.vehicle.passenger = None
        self.road.vehicles.append(self.vehicle)
        # Add random traffic vehicles (obstacles)
        for i in range(0, 10):
            vehicle = IDMVehicle(self.road, position=[np.random.randint(0, 100), np.random.randint(0, 100)], speed=np.random.randint(3, 7))
            vehicle.lane_index = self.road.network.get_closest_lane_index(vehicle.position)
            self.road.vehicles.append(vehicle)
        path = self.road.network.shortest_path("b0", "c0")
        obstacle3.plan_route_to("c0")
        self.road.vehicles.append(obstacle)
        self.road.vehicles.append(obstacle2)
        self.road.vehicles.append(obstacle3)
    def _reward(self, action):
        # Taxi-v3 upgrades: reward for pickups, dropoffs, penalties for fuel, traffic, etc.
        reward = 0.0
        # Step fuel consumption
        self.vehicle.fuel -= self.fuel_consumption_per_step
        if self.vehicle.fuel <= 0:
            reward += self.fuel_penalty
        # Traffic penalty
        if self._on_traffic_tile(self.vehicle.position):
            reward += self.traffic_penalty
        # Pickup/dropoff
        reward += self._check_pickup()
        reward += self._check_dropoff()
        # Optionally add driving reward (centered, speed, etc.)
        lane = self.vehicle.lane
        lane_heading = lane.heading_at(self.vehicle.position[0])
        heading_error = np.abs(self.vehicle.heading - lane_heading)
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        alignment_factor = np.cos(heading_error)
        speed_reward = (self.vehicle.speed / lane.speed_limit) * alignment_factor
        long, lat = lane.local_coordinates(self.vehicle.position)
        centering_reward = np.exp(-1 * (lat**2))
        reward += 0.2 * speed_reward + 0.2 * centering_reward
        return reward
    
    def _reset(self) -> None:
        self._init_taxi_upgrades()
        self._make_road()
        self._make_vehicles()
        self.current_step = 0
    def _is_terminated(self):
        # Stop if we crash, run out of fuel, or all passengers delivered
        all_delivered = all(p['delivered'] for p in self.passengers)
        return self.vehicle.crashed or self.vehicle.fuel <= 0 or all_delivered

    def _is_truncated(self):
        # Stop if we run out of steps
        self.current_step += 1
        return self.current_step >= self.max_steps
# 1. Instantiate the environment
import time
# 1. Instantiate the environment class directly
env = MyCustomEnv(render_mode='human') 

# 2. Reset (Builds the road and vehicles)
obs, info = env.reset()

print("Environment created. Starting simulation...")
failures=0
while failures<1:
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
            failures=10
        action=[acceleration,steering]
        obs, reward, terminated, truncated, info = env.step(action)
        # Render the graphics (AbstractEnv handles the PyGame window for you!)
        env.render()
        
        if terminated or truncated:
            failures+=1
            obs, info = env.reset()

env.close()