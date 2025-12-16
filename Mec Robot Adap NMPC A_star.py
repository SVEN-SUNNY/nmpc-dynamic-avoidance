import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import heapq
import casadi as ca
import random
from scipy.ndimage import gaussian_filter

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    WIDTH = 12.0
    HEIGHT = 12.0
    RESOLUTION = 0.2
    
    ROBOT_WIDTH = 0.6
    ROBOT_LENGTH = 0.6
    
    # Adaptive Thresholds
    SAFETY_DIST = 1.5
    
    # Standard NMPC Params
    DT = 0.1          
    N = 8           # Reduced for faster computation
    
    SIM_TIME = 400
    
    # MANY DYNAMIC OBSTACLES
    NUM_DYNAMIC_OBSTACLES = 12
    OBSTACLE_RADIUS = 0.25
    
    # LOCAL COSTMAP PARAMETERS
    LOCAL_MAP_SIZE = 4.0
    LOCAL_MAP_RES = 0.15
    INFLATION_RADIUS = 0.6
    COSTMAP_MAX = 100.0
    COSTMAP_GOAL_ATTRACTION = 50.0
    
    # INTEGRATED PLANNER PARAMETERS
    NMPC_WEIGHT = 0.7
    COSTMAP_WEIGHT = 0.3
    BLEND_DISTANCE = 2.0

# ==========================================
# 2. MAPPING & A* 
# ==========================================
class OccupancyGrid:
    def __init__(self):
        self.cols = int(Config.WIDTH / Config.RESOLUTION)
        self.rows = int(Config.HEIGHT / Config.RESOLUTION)
        self.grid = np.zeros((self.rows, self.cols), dtype=int)

    def generate_navigable_obstacles(self, num_obs=8):
        """Generate obstacles that create a navigable environment"""
        count = 0
        
        # Create structured corridor walls
        main_obstacles = [
            # Left corridor wall (with gaps)
            (int(4/Config.RESOLUTION), int(2/Config.RESOLUTION), 1, int(3/Config.RESOLUTION)),
            (int(4/Config.RESOLUTION), int(7/Config.RESOLUTION), 1, int(3/Config.RESOLUTION)),
            
            # Right corridor wall (with gaps)
            (int(8/Config.RESOLUTION), int(2/Config.RESOLUTION), 1, int(3/Config.RESOLUTION)),
            (int(8/Config.RESOLUTION), int(7/Config.RESOLUTION), 1, int(3/Config.RESOLUTION)),
            
            # Top obstacles
            (int(2/Config.RESOLUTION), int(9/Config.RESOLUTION), 3, 1),
            (int(7/Config.RESOLUTION), int(9/Config.RESOLUTION), 3, 1),
            
            # Bottom obstacles
            (int(2/Config.RESOLUTION), int(3/Config.RESOLUTION), 3, 1),
            (int(7/Config.RESOLUTION), int(3/Config.RESOLUTION), 3, 1),
        ]
        
        for cx, cy, width, height in main_obstacles:
            self.grid[cy:cy+height, cx:cx+width] = 1
            count += 1
        
        # Add a few random obstacles (not blocking main corridors)
        attempts = 0
        while count < num_obs and attempts < 100:
            cx = random.randint(2, self.cols-3)
            cy = random.randint(2, self.rows-3)
            width = random.randint(1, 2)
            height = random.randint(1, 2)
            
            # Avoid central corridors
            in_corridor = False
            for i in range(width):
                for j in range(height):
                    grid_x = cx + i
                    grid_y = cy + j
                    world_x = grid_x * Config.RESOLUTION
                    world_y = grid_y * Config.RESOLUTION
                    
                    # Check if in main corridors
                    if (4 < world_x < 8 and (3 < world_y < 5 or 7 < world_y < 9)) or \
                       (4 < world_y < 8 and (3 < world_x < 5 or 7 < world_x < 9)):
                        in_corridor = True
                        break
            
            if not in_corridor:
                self.grid[cy:cy+height, cx:cx+width] = 1
                count += 1
            attempts += 1

def a_star_search(start, goal, grid_obj):
    def heuristic(a, b): 
        # Euclidean distance with tie-breaking
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return np.sqrt(dx*dx + dy*dy) * 1.001
    
    s_node = (int(start[1]/Config.RESOLUTION), int(start[0]/Config.RESOLUTION))
    g_node = (int(goal[1]/Config.RESOLUTION), int(goal[0]/Config.RESOLUTION))
    
    neighbors = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {s_node: 0}
    fscore = {s_node: heuristic(s_node, g_node)}
    oheap = []
    heapq.heappush(oheap, (fscore[s_node], s_node))
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        
        if current == g_node:
            path = []
            while current in came_from:
                r, c = current
                path.append((c * Config.RESOLUTION + Config.RESOLUTION/2, 
                             r * Config.RESOLUTION + Config.RESOLUTION/2))
                current = came_from[current]
            path.append(start)
            return np.array(path[::-1])
        
        close_set.add(current)
        
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            
            if 0 <= neighbor[0] < grid_obj.rows and 0 <= neighbor[1] < grid_obj.cols:
                if grid_obj.grid[neighbor[0]][neighbor[1]] == 1: 
                    continue
            else: 
                continue
            
            move_cost = 1.0 if i == 0 or j == 0 else 1.414
            tent_g = gscore[current] + move_cost
            
            if neighbor in close_set and tent_g >= gscore.get(neighbor, 0):
                continue
                
            if tent_g < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tent_g
                fscore[neighbor] = tent_g + heuristic(neighbor, g_node)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    
    return np.array([])

# ==========================================
# 3. LOCAL COSTMAP CLASS
# ==========================================
class LocalCostmap:
    def __init__(self):
        self.size = Config.LOCAL_MAP_SIZE
        self.res = Config.LOCAL_MAP_RES
        self.grid_size = int(self.size / self.res)
        self.center_idx = self.grid_size // 2
        self.costmap = np.zeros((self.grid_size, self.grid_size))
        self.gradient_x = np.zeros((self.grid_size, self.grid_size))
        self.gradient_y = np.zeros((self.grid_size, self.grid_size))
        self.inflation_kernel = self.create_inflation_kernel()
        
    def create_inflation_kernel(self):
        kernel_size = int(Config.INFLATION_RADIUS / self.res) * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                dist = np.sqrt(((i - center) * self.res)**2 + ((j - center) * self.res)**2)
                if dist <= Config.INFLATION_RADIUS:
                    kernel[i, j] = max(0, Config.COSTMAP_MAX * (1 - (dist / Config.INFLATION_RADIUS)**2))
        return kernel
    
    def update(self, robot_x, robot_y, robot_theta, goal_x, goal_y, 
               static_grid, dynamic_obstacles):
        """
        Update local costmap and compute gradient
        """
        # Reset costmap
        self.costmap = np.zeros((self.grid_size, self.grid_size))
        
        # Get world coordinates for each cell
        world_x = np.zeros((self.grid_size, self.grid_size))
        world_y = np.zeros((self.grid_size, self.grid_size))
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Convert from robot-centric to world coordinates
                local_x = (j - self.center_idx) * self.res
                local_y = (i - self.center_idx) * self.res
                
                # Rotate to align with robot orientation
                rot_x = local_x * np.cos(robot_theta) - local_y * np.sin(robot_theta)
                rot_y = local_x * np.sin(robot_theta) + local_y * np.cos(robot_theta)
                
                world_x[i, j] = robot_x + rot_x
                world_y[i, j] = robot_y + rot_y
        
        # 1. Add static obstacles
        static_cost_layer = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                wx, wy = world_x[i, j], world_y[i, j]
                
                if 0 <= wx < Config.WIDTH and 0 <= wy < Config.HEIGHT:
                    gx = int(wx / Config.RESOLUTION)
                    gy = int(wy / Config.RESOLUTION)
                    
                    if 0 <= gx < static_grid.cols and 0 <= gy < static_grid.rows:
                        if static_grid.grid[gy, gx] == 1:
                            static_cost_layer[i, j] = Config.COSTMAP_MAX
        
        # Inflate static obstacles
        from scipy.signal import convolve2d
        inflated_static = convolve2d(static_cost_layer, self.inflation_kernel, 
                                     mode='same', boundary='fill', fillvalue=0)
        self.costmap += np.minimum(inflated_static, Config.COSTMAP_MAX)
        
        # 2. Add dynamic obstacles
        dynamic_cost_layer = np.zeros((self.grid_size, self.grid_size))
        for obs in dynamic_obstacles:
            dist_sq = (world_x - obs.x)**2 + (world_y - obs.y)**2
            obs_cost = Config.COSTMAP_MAX * np.exp(-dist_sq / (Config.INFLATION_RADIUS**2))
            dynamic_cost_layer += obs_cost
        
        self.costmap += np.minimum(dynamic_cost_layer, Config.COSTMAP_MAX)
        
        # 3. Add goal attraction (negative gradient)
        goal_dist = np.sqrt((world_x - goal_x)**2 + (world_y - goal_y)**2)
        max_attraction_dist = self.size / 2
        
        goal_attraction = np.zeros_like(goal_dist)
        mask = goal_dist < max_attraction_dist
        goal_attraction[mask] = Config.COSTMAP_GOAL_ATTRACTION * (1 - goal_dist[mask]/max_attraction_dist)
        
        self.costmap -= goal_attraction
        
        # Ensure non-negative
        self.costmap = np.maximum(self.costmap, 0)
        
        # Smooth and compute gradient
        self.costmap = gaussian_filter(self.costmap, sigma=1.0)
        self.gradient_y, self.gradient_x = np.gradient(self.costmap)
        
        return world_x, world_y
    
    def get_gradient_force(self, robot_x, robot_y, robot_theta):
        """
        Get repulsive force from costmap gradient
        """
        # Find cell corresponding to robot position (center cell)
        center_i = self.center_idx
        center_j = self.center_idx
        
        # Get gradient at robot position
        grad_x = -self.gradient_x[center_i, center_j]  # Negative gradient = repulsive force
        grad_y = -self.gradient_y[center_i, center_j]
        
        # Rotate gradient to world frame
        force_x = grad_x * np.cos(robot_theta) - grad_y * np.sin(robot_theta)
        force_y = grad_x * np.sin(robot_theta) + grad_y * np.cos(robot_theta)
        
        # Normalize and scale
        force_mag = np.sqrt(force_x**2 + force_y**2)
        if force_mag > 0:
            force_x = force_x / force_mag
            force_y = force_y / force_mag
        
        return force_x, force_y

# ==========================================
# 4. DYNAMIC OBSTACLES
# ==========================================
class DynamicObstacle:
    def __init__(self, x, y, vx, vy):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.original_vx, self.original_vy = vx, vy
        self.bounce_randomness = random.uniform(0.8, 1.2)
        self.direction_change_timer = random.randint(30, 120)
        self.timer = 0
        
    def update(self, dt):
        self.timer += 1
        
        # Occasionally change direction
        if self.timer >= self.direction_change_timer:
            self.vx += random.uniform(-0.15, 0.15)
            self.vy += random.uniform(-0.15, 0.15)
            self.timer = 0
            self.direction_change_timer = random.randint(30, 120)
        
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Bounce off boundaries
        if self.x < Config.OBSTACLE_RADIUS:
            self.x = Config.OBSTACLE_RADIUS
            self.vx = abs(self.vx) * self.bounce_randomness
        elif self.x > Config.WIDTH - Config.OBSTACLE_RADIUS:
            self.x = Config.WIDTH - Config.OBSTACLE_RADIUS
            self.vx = -abs(self.vx) * self.bounce_randomness
            
        if self.y < Config.OBSTACLE_RADIUS:
            self.y = Config.OBSTACLE_RADIUS
            self.vy = abs(self.vy) * self.bounce_randomness
        elif self.y > Config.HEIGHT - Config.OBSTACLE_RADIUS:
            self.y = Config.HEIGHT - Config.OBSTACLE_RADIUS
            self.vy = -abs(self.vy) * self.bounce_randomness
            
        # Speed limits
        speed = np.sqrt(self.vx**2 + self.vy**2)
        max_speed = 0.5
        
        if speed > max_speed:
            self.vx = self.vx * max_speed / speed
            self.vy = self.vy * max_speed / speed
        elif speed < 0.05:
            self.vx = self.original_vx * 0.3
            self.vy = self.original_vy * 0.3

def create_navigable_dynamic_obstacles(num_obstacles):
    """Create dynamic obstacles that don't completely block paths"""
    obstacles = []
    
    for i in range(num_obstacles):
        # Divide map into zones
        zone = i % 4
        if zone == 0:  # Bottom-left
            x = random.uniform(2, 5)
            y = random.uniform(2, 5)
        elif zone == 1:  # Bottom-right
            x = random.uniform(7, 10)
            y = random.uniform(2, 5)
        elif zone == 2:  # Top-left
            x = random.uniform(2, 5)
            y = random.uniform(7, 10)
        else:  # Top-right
            x = random.uniform(7, 10)
            y = random.uniform(7, 10)
        
        # Avoid start and goal areas
        if np.sqrt((x-1.5)**2 + (y-1.5)**2) < 1.5 or np.sqrt((x-10.5)**2 + (y-10.5)**2) < 1.5:
            x = random.uniform(3, 9)
            y = random.uniform(3, 9)
        
        # Random velocity (slower)
        speed = random.uniform(0.2, 0.4)
        angle = random.uniform(0, 2*np.pi)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        
        obstacles.append(DynamicObstacle(x, y, vx, vy))
    
    return obstacles

# ==========================================
# 5. INTEGRATED PLANNER (NMPC + Costmap) - FIXED
# ==========================================
class IntegratedPlanner:
    def __init__(self):
        self.costmap = LocalCostmap()
        self.initialize_nmpc()
        
    def initialize_nmpc(self):
        """Initialize the NMPC controller - FIXED VERSION"""
        self.opti = ca.Opti()
        
        # State: [x, y, theta]
        self.X = self.opti.variable(3, Config.N + 1)
        
        # Control: [vx, vy, omega]
        self.U = self.opti.variable(3, Config.N)
        
        # Parameters
        self.P_init = self.opti.parameter(3)          # Initial state
        self.P_goal = self.opti.parameter(3)          # Goal state
        self.P_costmap_force = self.opti.parameter(2) # Costmap force [fx, fy]
        self.P_v_limit = self.opti.parameter(1)       # Velocity limit
        self.P_blend_weight = self.opti.parameter(1)  # NMPC vs Costmap blend
        
        # Closest obstacles for NMPC (for point-mass avoidance)
        self.P_obs = self.opti.parameter(2, 5)
        
        # Cost weights - FIXED: Use scalar weights instead of matrices for simplicity
        Q_pos_xy = 10.0    # Position tracking weight for x,y
        Q_pos_theta = 1.0  # Position tracking weight for theta
        Q_vel_xy = 0.5     # Control effort weight for vx,vy
        Q_vel_omega = 0.2  # Control effort weight for omega
        Q_costmap = 2.0    # Costmap alignment weight
        
        # Build cost function
        total_cost = 0
        
        for k in range(Config.N):
            # 1. Goal tracking cost - SIMPLIFIED TO AVOID MATRIX MULTIPLICATION
            state_error_xy = (self.X[0, k] - self.P_goal[0])**2 + (self.X[1, k] - self.P_goal[1])**2
            state_error_theta = (self.X[2, k] - self.P_goal[2])**2
            total_cost += Q_pos_xy * state_error_xy + Q_pos_theta * state_error_theta
            
            # 2. Control effort cost - SIMPLIFIED
            control_effort_xy = self.U[0, k]**2 + self.U[1, k]**2
            control_effort_omega = self.U[2, k]**2
            total_cost += Q_vel_xy * control_effort_xy + Q_vel_omega * control_effort_omega
            
            # 3. Costmap alignment cost - SIMPLIFIED
            # The costmap force points away from obstacles
            desired_vx = self.P_costmap_force[0] * self.P_v_limit
            desired_vy = self.P_costmap_force[1] * self.P_v_limit
            
            costmap_error = (self.U[0, k] - desired_vx)**2 + (self.U[1, k] - desired_vy)**2
            total_cost += self.P_blend_weight * Q_costmap * costmap_error
            
            # 4. Obstacle avoidance cost (point-mass) - SIMPLIFIED
            for j in range(5):
                obs_x = self.P_obs[0, j]
                obs_y = self.P_obs[1, j]
                
                # Skip invalid obstacles (negative values indicate no obstacle)
                if obs_x < -50 or obs_y < -50:
                    continue
                
                dist_sq = (self.X[0, k] - obs_x)**2 + (self.X[1, k] - obs_y)**2 + 0.1
                total_cost += 50.0 / dist_sq
        
        self.opti.minimize(total_cost)
        
        # Dynamics constraints
        for k in range(Config.N):
            dx = self.U[0, k]*ca.cos(self.X[2, k]) - self.U[1, k]*ca.sin(self.X[2, k])
            dy = self.U[0, k]*ca.sin(self.X[2, k]) + self.U[1, k]*ca.cos(self.X[2, k])
            dth = self.U[2, k]
            
            self.opti.subject_to(
                self.X[:, k+1] == self.X[:, k] + ca.vertcat(dx, dy, dth) * Config.DT
            )
            
            # Control constraints
            self.opti.subject_to(self.opti.bounded(-self.P_v_limit, self.U[0, k], self.P_v_limit))
            self.opti.subject_to(self.opti.bounded(-self.P_v_limit, self.U[1, k], self.P_v_limit))
            self.opti.subject_to(self.opti.bounded(-1.2, self.U[2, k], 1.2))
        
        # Initial state constraint
        self.opti.subject_to(self.X[:, 0] == self.P_init)
        
        # Solver options
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver('ipopt', opts)
    
    def plan(self, robot_state, goal_state, dynamic_obstacles, static_grid, closest_obstacle_dist):
        """
        Integrated planning combining NMPC and costmap
        """
        # 1. Update costmap
        world_x, world_y = self.costmap.update(
            robot_state[0], robot_state[1], robot_state[2],
            goal_state[0], goal_state[1], static_grid, dynamic_obstacles
        )
        
        # 2. Get costmap gradient force
        costmap_fx, costmap_fy = self.costmap.get_gradient_force(
            robot_state[0], robot_state[1], robot_state[2]
        )
        
        # 3. Adaptive parameters based on environment
        if closest_obstacle_dist < 0.8:
            v_limit = 0.3
            blend_weight = 0.8  # More weight to costmap (safety)
            robot_color = 'red'
        elif closest_obstacle_dist < Config.SAFETY_DIST:
            v_limit = 0.6
            blend_weight = 0.5  # Balanced
            robot_color = 'orange'
        else:
            v_limit = 1.2
            blend_weight = 0.3  # More weight to NMPC (efficiency)
            robot_color = 'blue'
        
        # 4. Get closest obstacles for NMPC point-mass avoidance
        obs_positions = np.ones((2, 5)) * -100  # Initialize far away
        
        if dynamic_obstacles:
            distances = []
            for i, obs in enumerate(dynamic_obstacles):
                dist = np.sqrt((robot_state[0] - obs.x)**2 + (robot_state[1] - obs.y)**2)
                distances.append((dist, i))
            
            distances.sort(key=lambda x: x[0])
            
            for idx, (_, obs_idx) in enumerate(distances[:5]):
                obs = dynamic_obstacles[obs_idx]
                obs_positions[0, idx] = obs.x
                obs_positions[1, idx] = obs.y
        
        # 5. Solve integrated NMPC
        try:
            # Set parameter values
            self.opti.set_value(self.P_init, robot_state)
            self.opti.set_value(self.P_goal, goal_state)
            self.opti.set_value(self.P_costmap_force, [costmap_fx, costmap_fy])
            self.opti.set_value(self.P_v_limit, v_limit)
            self.opti.set_value(self.P_blend_weight, blend_weight)
            self.opti.set_value(self.P_obs, obs_positions)
            
            # Set initial guess
            current_vx = v_limit * costmap_fx
            current_vy = v_limit * costmap_fy
            
            # Simple initial guess: move toward goal
            goal_dir = np.arctan2(goal_state[1] - robot_state[1], goal_state[0] - robot_state[0])
            theta_error = goal_dir - robot_state[2]
            # Normalize angle difference
            theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))
            
            self.opti.set_initial(self.X, np.tile(np.array(robot_state)[:, None], (1, Config.N+1)))
            self.opti.set_initial(self.U, np.array([[current_vx], [current_vy], [theta_error * 0.5]]))
            
            # Solve
            sol = self.opti.solve()
            control = sol.value(self.U[:, 0])
            
            # Get predicted trajectory for visualization
            predicted_traj = sol.value(self.X)[:2, :].T
            
        except Exception as e:
            # Fallback: simple gradient following
            print(f"NMPC failed: {e}, using fallback")
            control = np.array([
                v_limit * costmap_fx,
                v_limit * costmap_fy,
                0.0
            ])
            predicted_traj = None
        
        return control, robot_color, predicted_traj

# ==========================================
# 6. SIMULATION WITH INTEGRATED PLANNER - SIMPLIFIED
# ==========================================
def run_simulation():
    # Setup
    grid = OccupancyGrid()
    grid.generate_navigable_obstacles(8)
    
    start, goal = (1.5, 1.5), (10.5, 10.5)
    
    # Find path for visualization only
    path = a_star_search(start, goal, grid)
    if len(path) == 0:
        print("No path found!")
        return
    
    # Create dynamic obstacles
    dyn_obs = create_navigable_dynamic_obstacles(Config.NUM_DYNAMIC_OBSTACLES)
    
    # Initialize integrated planner
    planner = IntegratedPlanner()
    
    # Initial robot state
    true_state = np.array([start[0], start[1], np.pi/4])
    
    # Simulation control
    simulation_running = False
    simulation_paused = False
    
    # Visualization setup - SIMPLIFIED
    fig = plt.figure(figsize=(14, 8))
    
    # Main map
    ax_map = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
    
    # Local costmap
    ax_costmap = plt.subplot2grid((2, 3), (0, 2))
    
    # Info panel
    ax_info = plt.subplot2grid((2, 3), (1, 2))
    ax_info.axis('off')
    
    # Map setup
    ax_map.set_title("Integrated Planner: NMPC + Costmap (Press 's' to start/stop)")
    ax_map.imshow(grid.grid, cmap='binary', origin='lower', 
                  extent=[0, Config.WIDTH, 0, Config.HEIGHT], alpha=0.7)
    ax_map.plot(path[:,0], path[:,1], 'g--', alpha=0.5, label='Global Path')
    ax_map.plot(start[0], start[1], 'bo', markersize=12, label='Start')
    ax_map.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
    
    # Robot
    robot_patch = patches.Rectangle((0,0), Config.ROBOT_WIDTH, Config.ROBOT_LENGTH, 
                                    color='blue', alpha=0.8)
    ax_map.add_patch(robot_patch)
    
    # Dynamic obstacles
    obs_patches = []
    for obs in dyn_obs:
        patch = patches.Circle((obs.x, obs.y), Config.OBSTACLE_RADIUS, 
                              color='red', alpha=0.6)
        obs_patches.append(patch)
        ax_map.add_patch(patch)
    
    # Predicted trajectory
    predicted_line, = ax_map.plot([], [], 'y-', linewidth=2, alpha=0.6, label='Predicted')
    
    ax_map.legend(loc='upper right')
    ax_map.set_xlim(0, Config.WIDTH)
    ax_map.set_ylim(0, Config.HEIGHT)
    
    # Costmap visualization
    costmap_img = ax_costmap.imshow(np.zeros((planner.costmap.grid_size, planner.costmap.grid_size)), 
                                    cmap='RdYlGn_r', origin='lower',
                                    extent=[-planner.costmap.size/2, planner.costmap.size/2, 
                                            -planner.costmap.size/2, planner.costmap.size/2],
                                    vmin=0, vmax=Config.COSTMAP_MAX)
    ax_costmap.set_title("Local Costmap")
    ax_costmap.set_xlabel("Local X")
    ax_costmap.set_ylabel("Local Y")
    
    # Data storage
    h_t = []
    
    # Metrics
    total_distance = 0
    last_position = np.array(start)
    collision_count = 0
    success = False
    simulation_time = 0
    
    def get_lookahead_point(path, robot_pos, lookahead=1.0):
        """Get lookahead point on path"""
        if len(path) < 2:
            return path[-1] if len(path) > 0 else robot_pos[:2]
        
        # Find closest point on path
        dists = np.linalg.norm(path - robot_pos[:2], axis=1)
        closest_idx = np.argmin(dists)
        
        # Move forward along path
        target_idx = min(closest_idx + 3, len(path)-1)
        return np.array([path[target_idx][0], path[target_idx][1], 0.0])
    
    def update(frame):
        nonlocal true_state, simulation_running, simulation_paused
        nonlocal total_distance, last_position, collision_count, success, simulation_time
        
        if not simulation_running or simulation_paused:
            # Don't update simulation
            return_items = [robot_patch, costmap_img, predicted_line] + obs_patches
            return return_items
        
        # Increment simulation time
        simulation_time += Config.DT
        
        # 1. Update dynamic obstacles
        for obs, patch in zip(dyn_obs, obs_patches):
            obs.update(Config.DT)
            patch.set_center((obs.x, obs.y))
        
        # 2. Find closest obstacle
        min_dist = float('inf')
        for obs in dyn_obs:
            d = np.sqrt((true_state[0] - obs.x)**2 + (true_state[1] - obs.y)**2)
            if d < min_dist:
                min_dist = d
        
        # Check collision
        if min_dist < Config.ROBOT_WIDTH/2 + Config.OBSTACLE_RADIUS:
            collision_count += 1
        
        # 3. Check if goal reached
        dist_to_goal = np.sqrt((true_state[0] - goal[0])**2 + (true_state[1] - goal[1])**2)
        if dist_to_goal < 0.3 and not success:
            success = True
            simulation_running = False
            print(f"\n🎉 GOAL REACHED! 🎉")
            print(f"Time: {simulation_time:.1f}s")
            print(f"Distance: {total_distance:.1f}m")
            print(f"Collisions: {collision_count}")
        
        # 4. Get lookahead point for local goal
        local_goal = get_lookahead_point(path, true_state)
        
        # 5. Integrated planning
        control, robot_color, predicted_traj = planner.plan(
            true_state, local_goal, dyn_obs, grid, min_dist
        )
        
        # 6. Update distance traveled
        current_pos = np.array([true_state[0], true_state[1]])
        total_distance += np.linalg.norm(current_pos - last_position)
        last_position = current_pos.copy()
        
        # 7. Update robot state
        dx = control[0] * np.cos(true_state[2]) - control[1] * np.sin(true_state[2])
        dy = control[0] * np.sin(true_state[2]) + control[1] * np.cos(true_state[2])
        true_state += np.array([dx, dy, control[2]]) * Config.DT
        
        # Keep within bounds
        true_state[0] = np.clip(true_state[0], Config.ROBOT_WIDTH/2, Config.WIDTH - Config.ROBOT_WIDTH/2)
        true_state[1] = np.clip(true_state[1], Config.ROBOT_LENGTH/2, Config.HEIGHT - Config.ROBOT_LENGTH/2)
        
        # 8. Update robot visualization
        cx, cy, th = true_state
        w, l = Config.ROBOT_WIDTH, Config.ROBOT_LENGTH
        corner_x = cx - (w/2)*np.cos(th) + (l/2)*np.sin(th)
        corner_y = cy - (w/2)*np.sin(th) - (l/2)*np.cos(th)
        robot_patch.set_xy((corner_x, corner_y))
        robot_patch.angle = np.degrees(th)
        robot_patch.set_color(robot_color)
        
        # 9. Update predicted trajectory
        if predicted_traj is not None and len(predicted_traj) > 0:
            predicted_line.set_data(predicted_traj[:, 0], predicted_traj[:, 1])
        
        # 10. Update info panel
        ax_info.clear()
        ax_info.axis('off')
        
        status_color = 'green' if success else ('red' if min_dist < 0.8 else 'blue')
        status_text = "GOAL REACHED!" if success else "Running" if simulation_running else "Stopped"
        
        ax_info.text(0.1, 0.9, f"Status: {status_text}", 
                    fontsize=12, fontweight='bold', color=status_color)
        ax_info.text(0.1, 0.8, f"Closest Obstacle: {min_dist:.2f}m", 
                    fontsize=10, color='red' if min_dist < 0.8 else 'black')
        ax_info.text(0.1, 0.7, f"Goal Distance: {dist_to_goal:.2f}m", fontsize=10)
        ax_info.text(0.1, 0.6, f"Distance Traveled: {total_distance:.1f}m", fontsize=10)
        ax_info.text(0.1, 0.5, f"Collisions: {collision_count}", fontsize=10)
        ax_info.text(0.1, 0.4, f"Time: {simulation_time:.1f}s", fontsize=10)
        ax_info.text(0.1, 0.3, f"Controls:", fontsize=10, fontweight='bold')
        ax_info.text(0.15, 0.25, "'s' - Start/Stop", fontsize=9)
        ax_info.text(0.15, 0.20, "'r' - Reset", fontsize=9)
        ax_info.text(0.15, 0.15, "'q' - Quit", fontsize=9)
        
        # Store time for graphs
        h_t.append(simulation_time)
        
        return_items = [robot_patch, costmap_img, predicted_line] + obs_patches
        
        return return_items
    
    def on_key(event):
        nonlocal simulation_running, simulation_paused, true_state, total_distance
        nonlocal collision_count, success, simulation_time, last_position
        
        if event.key == 's':  # Start/Stop
            simulation_running = not simulation_running
            if simulation_running:
                print("Simulation STARTED")
                simulation_paused = False
            else:
                print("Simulation STOPPED")
        
        elif event.key == 'r':  # Reset
            # Reset simulation
            true_state = np.array([start[0], start[1], np.pi/4])
            simulation_running = False
            simulation_paused = False
            total_distance = 0
            collision_count = 0
            success = False
            simulation_time = 0
            last_position = np.array(start)
            
            # Reset robot position
            cx, cy, th = true_state
            w, l = Config.ROBOT_WIDTH, Config.ROBOT_LENGTH
            corner_x = cx - (w/2)*np.cos(th) + (l/2)*np.sin(th)
            corner_y = cy - (w/2)*np.sin(th) - (l/2)*np.cos(th)
            robot_patch.set_xy((corner_x, corner_y))
            robot_patch.angle = np.degrees(th)
            robot_patch.set_color('blue')
            
            # Clear predicted trajectory
            predicted_line.set_data([], [])
            
            print("Simulation RESET")
        
        elif event.key == 'q':  # Quit
            plt.close()
            print("Quitting...")
    
    # Connect keyboard events
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial instructions
    print("\n" + "="*50)
    print("INTEGRATED PLANNER SIMULATION")
    print("="*50)
    print("Controls:")
    print("  's' - Start/Stop simulation")
    print("  'r' - Reset simulation")
    print("  'q' - Quit")
    print("\nPress 's' to start the simulation...")
    print("="*50 + "\n")
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=Config.SIM_TIME, interval=50, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()