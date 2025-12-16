import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import heapq
import casadi as ca
import random
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SIMPLIFIED CONFIGURATION
# ==========================================
class Config:
    WIDTH = 15.0
    HEIGHT = 15.0
    RESOLUTION = 0.5
    
    ROBOT_WIDTH = 0.5
    ROBOT_LENGTH = 0.5
    
    # Safety parameters
    SAFETY_DIST = 0.8
    
    # NMPC Parameters
    DT = 0.1          
    N = 8             # Shorter horizon for faster computation
    
    SIM_TIME = 800
    
    # Dynamic obstacles
    NUM_DYNAMIC_OBSTACLES = 6
    OBSTACLE_RADIUS = 0.35
    
    # COSTMAP PARAMETERS
    LOCAL_MAP_SIZE = 5.0
    LOCAL_MAP_RES = 0.1
    INFLATION_RADIUS = 0.8
    COSTMAP_MAX = 50.0
    COSTMAP_GOAL_ATTRACTION = 80.0
    
    # DYNAMIC MODEL PARAMETERS
    MASS = 1.5
    INERTIA = 0.15
    MAX_FORCE = 4.0
    MAX_TORQUE = 2.5
    
    # Navigation parameters
    GOAL_TOLERANCE = 0.5

# ==========================================
# 2. MAPPING
# ==========================================
class OccupancyGrid:
    def __init__(self):
        self.cols = int(Config.WIDTH / Config.RESOLUTION)
        self.rows = int(Config.HEIGHT / Config.RESOLUTION)
        self.grid = np.zeros((self.rows, self.cols), dtype=int)

    def generate_structured_obstacles(self):
        """Create a structured obstacle course"""
        # Clear grid first
        self.grid.fill(0)
        
        # Create a simple structure
        obstacles = [
            # Border walls (with openings)
            (0, 0, 30, 1),      # Bottom wall
            (0, 29, 30, 1),     # Top wall  
            (0, 0, 1, 30),      # Left wall
            (29, 0, 1, 30),     # Right wall
            
            # Simple obstacles
            (10, 10, 2, 2),     # Left obstacle
            (20, 10, 2, 2),     # Right obstacle
            (15, 20, 2, 2),     # Top obstacle
        ]
        
        for cx, cy, width, height in obstacles:
            # Ensure within bounds
            cx = max(0, min(cx, self.cols - 1))
            cy = max(0, min(cy, self.rows - 1))
            width = min(width, self.cols - cx)
            height = min(height, self.rows - cy)
            
            if width > 0 and height > 0:
                self.grid[cy:cy+height, cx:cx+width] = 1
        
        # Create openings in border walls
        self.grid[0, 14:16] = 0   # Bottom opening
        self.grid[29, 14:16] = 0  # Top opening
        self.grid[14:16, 0] = 0   # Left opening
        self.grid[14:16, 29] = 0  # Right opening
        
        return len(obstacles)

def a_star_search(start, goal, grid_obj):
    def heuristic(a, b): 
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) * 1.001
    
    s_node = (int(start[1]/Config.RESOLUTION), int(start[0]/Config.RESOLUTION))
    g_node = (int(goal[1]/Config.RESOLUTION), int(goal[0]/Config.RESOLUTION))
    
    neighbors = [(0,1), (0,-1), (1,0), (-1,0)]
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
            
            tent_g = gscore[current] + 1.0
            
            if neighbor in close_set and tent_g >= gscore.get(neighbor, 0):
                continue
                
            if tent_g < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tent_g
                fscore[neighbor] = tent_g + heuristic(neighbor, g_node)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    
    return np.array([])

# ==========================================
# 3. SIMPLIFIED COSTMAP
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
        
    def update(self, robot_x, robot_y, robot_theta, goal_x, goal_y, 
               static_grid, dynamic_obstacles):
        self.costmap = np.zeros((self.grid_size, self.grid_size))
        
        # Create local grid
        xs = np.linspace(-self.size/2, self.size/2, self.grid_size)
        ys = np.linspace(-self.size/2, self.size/2, self.grid_size)
        local_x, local_y = np.meshgrid(xs, ys)
        
        # Rotate to world frame
        world_x = robot_x + local_x * np.cos(robot_theta) - local_y * np.sin(robot_theta)
        world_y = robot_y + local_x * np.sin(robot_theta) + local_y * np.cos(robot_theta)
        
        # 1. Goal attraction (strong)
        goal_dist = np.sqrt((world_x - goal_x)**2 + (world_y - goal_y)**2)
        goal_cost = -Config.COSTMAP_GOAL_ATTRACTION * (1 - goal_dist / self.size)
        goal_cost = np.clip(goal_cost, -Config.COSTMAP_GOAL_ATTRACTION, 0)
        self.costmap += goal_cost
        
        # 2. Static obstacles
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                wx, wy = world_x[i, j], world_y[i, j]
                
                if 0 <= wx < Config.WIDTH and 0 <= wy < Config.HEIGHT:
                    gx = int(wx / Config.RESOLUTION)
                    gy = int(wy / Config.RESOLUTION)
                    
                    if 0 <= gx < static_grid.cols and 0 <= gy < static_grid.rows:
                        if static_grid.grid[gy, gx] == 1:
                            # High cost at obstacle, decreasing with distance
                            dx = (i - self.center_idx) * self.res
                            dy = (j - self.center_idx) * self.res
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist < Config.INFLATION_RADIUS:
                                cost = Config.COSTMAP_MAX * (1 - dist/Config.INFLATION_RADIUS)
                                self.costmap[i, j] += cost
        
        # 3. Dynamic obstacles
        for obs in dynamic_obstacles:
            dist = np.sqrt((world_x - obs.x)**2 + (world_y - obs.y)**2)
            obs_cost = Config.COSTMAP_MAX * np.exp(-dist / 0.5)
            self.costmap += np.minimum(obs_cost, Config.COSTMAP_MAX)
        
        # Smooth and compute gradient
        self.costmap = gaussian_filter(self.costmap, sigma=0.5)
        self.gradient_y, self.gradient_x = np.gradient(-self.costmap)
        
        return world_x, world_y
    
    def get_gradient_force(self, robot_x, robot_y, robot_theta):
        center_i = self.center_idx
        center_j = self.center_idx
        
        # Get gradient at robot position
        grad_x = self.gradient_x[center_i, center_j]
        grad_y = self.gradient_y[center_i, center_j]
        
        # Normalize
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        if grad_mag > 1e-6:
            grad_x /= grad_mag
            grad_y /= grad_mag
        else:
            grad_x, grad_y = 1.0, 0.0
        
        # Rotate to world frame
        force_x = grad_x * np.cos(robot_theta) - grad_y * np.sin(robot_theta)
        force_y = grad_x * np.sin(robot_theta) + grad_y * np.cos(robot_theta)
        
        return force_x, force_y

# ==========================================
# 4. DYNAMIC OBSTACLES
# ==========================================
class DynamicObstacle:
    def __init__(self, x, y, vx, vy):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.radius = Config.OBSTACLE_RADIUS
        
    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Bounce off boundaries
        margin = self.radius
        if self.x < margin or self.x > Config.WIDTH - margin:
            self.vx *= -1
            self.x = np.clip(self.x, margin, Config.WIDTH - margin)
        if self.y < margin or self.y > Config.HEIGHT - margin:
            self.vy *= -1
            self.y = np.clip(self.y, margin, Config.HEIGHT - margin)

def create_dynamic_obstacles():
    obstacles = []
    
    # Create obstacles away from the direct path
    positions = [
        (4.0, 8.0), (11.0, 8.0),
        (4.0, 12.0), (11.0, 12.0),
        (8.0, 5.0), (8.0, 15.0),
    ]
    
    for x, y in positions[:Config.NUM_DYNAMIC_OBSTACLES]:
        # Slow movements
        speed = 0.1
        angle = random.uniform(0, 2*np.pi)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        
        obstacles.append(DynamicObstacle(x, y, vx, vy))
    
    return obstacles

# ==========================================
# 5. FIXED NMPC PLANNER (NO CASADI IF-ELSE ERRORS)
# ==========================================
class IntegratedPlanner:
    def __init__(self):
        self.costmap = LocalCostmap()
        self.initialize_nmpc()
        
    def initialize_nmpc(self):
        """Initialize NMPC without conditional statements that cause CasADi errors"""
        self.opti = ca.Opti()
        
        # State: [x, y, theta, vx, vy, omega]
        self.X = self.opti.variable(6, Config.N + 1)
        
        # Control: [Fx, Fy, Tau]
        self.U = self.opti.variable(3, Config.N)
        
        # Parameters
        self.P_init = self.opti.parameter(6)
        self.P_goal = self.opti.parameter(3)
        self.P_costmap_force = self.opti.parameter(2)
        self.P_max_force = self.opti.parameter(1)
        
        # Robot parameters
        m = Config.MASS
        I = Config.INERTIA
        
        # Cost weights
        Q_pos = ca.diag([50.0, 50.0, 5.0, 0.1, 0.1, 0.1])
        R_force = ca.diag([0.02, 0.02, 0.05])
        
        total_cost = 0
        
        for k in range(Config.N):
            # Goal tracking
            state_error = self.X[:3, k] - self.P_goal
            total_cost += ca.mtimes(state_error.T, ca.mtimes(Q_pos[:3, :3], state_error))
            
            # Velocity penalty
            vel_error = self.X[3:6, k]
            total_cost += ca.mtimes(vel_error.T, ca.mtimes(Q_pos[3:, 3:], vel_error))
            
            # Control effort
            control_effort = self.U[:, k]
            total_cost += ca.mtimes(control_effort.T, ca.mtimes(R_force, control_effort))
            
            # Costmap following
            theta = self.X[2, k]
            desired_Fx = self.P_costmap_force[0] * self.P_max_force
            desired_Fy = self.P_costmap_force[1] * self.P_max_force
            
            F_error_x = self.U[0, k] - desired_Fx
            F_error_y = self.U[1, k] - desired_Fy
            total_cost += 1.0 * (F_error_x**2 + F_error_y**2)
        
        self.opti.minimize(total_cost)
        
        # DYNAMICS EQUATIONS
        for k in range(Config.N):
            x = self.X[0, k]
            y = self.X[1, k]
            theta = self.X[2, k]
            vx = self.X[3, k]
            vy = self.X[4, k]
            omega = self.X[5, k]
            
            Fx = self.U[0, k]
            Fy = self.U[1, k]
            Tau = self.U[2, k]
            
            # Dynamic equations
            ax_robot = Fx / m
            ay_robot = Fy / m
            alpha = Tau / I
            
            # Rotate to world frame
            ax_world = ax_robot * ca.cos(theta) - ay_robot * ca.sin(theta)
            ay_world = ax_robot * ca.sin(theta) + ay_robot * ca.cos(theta)
            
            # State derivatives
            dx = vx
            dy = vy
            dtheta = omega
            dvx = ax_world
            dvy = ay_world
            domega = alpha
            
            # Euler integration
            state_dot = ca.vertcat(dx, dy, dtheta, dvx, dvy, domega)
            self.opti.subject_to(
                self.X[:, k+1] == self.X[:, k] + state_dot * Config.DT
            )
            
            # Control constraints
            self.opti.subject_to(self.opti.bounded(-Config.MAX_FORCE, self.U[0, k], Config.MAX_FORCE))
            self.opti.subject_to(self.opti.bounded(-Config.MAX_FORCE, self.U[1, k], Config.MAX_FORCE))
            self.opti.subject_to(self.opti.bounded(-Config.MAX_TORQUE, self.U[2, k], Config.MAX_TORQUE))
            
            # Velocity constraints
            max_speed = 1.0
            self.opti.subject_to(self.opti.bounded(-max_speed, self.X[3, k], max_speed))
            self.opti.subject_to(self.opti.bounded(-max_speed, self.X[4, k], max_speed))
            self.opti.subject_to(self.opti.bounded(-1.0, self.X[5, k], 1.0))
        
        # Initial state
        self.opti.subject_to(self.X[:, 0] == self.P_init)
        
        # Solver options
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
                'ipopt.max_iter': 50, 'ipopt.tol': 1e-2}
        self.opti.solver('ipopt', opts)
    
    def plan(self, robot_state, goal_state, dynamic_obstacles, static_grid, closest_obstacle_dist):
        # Update costmap
        world_x, world_y = self.costmap.update(
            robot_state[0], robot_state[1], robot_state[2],
            goal_state[0], goal_state[1], static_grid, dynamic_obstacles
        )
        
        # Get gradient force
        costmap_fx, costmap_fy = self.costmap.get_gradient_force(
            robot_state[0], robot_state[1], robot_state[2]
        )
        
        # Adaptive control based on obstacle distance
        current_speed = np.sqrt(robot_state[3]**2 + robot_state[4]**2)
        
        if closest_obstacle_dist < 0.5:
            max_force = Config.MAX_FORCE * 0.3
            robot_color = 'red'
            # Move perpendicular to obstacle if too close
            if len(dynamic_obstacles) > 0:
                closest = min(dynamic_obstacles, key=lambda o: np.sqrt((robot_state[0]-o.x)**2 + (robot_state[1]-o.y)**2))
                dx = robot_state[0] - closest.x
                dy = robot_state[1] - closest.y
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0:
                    # Perpendicular direction
                    costmap_fx = -dy / dist
                    costmap_fy = dx / dist
        elif closest_obstacle_dist < 0.8:
            max_force = Config.MAX_FORCE * 0.5
            robot_color = 'orange'
        else:
            max_force = Config.MAX_FORCE * 0.7
            robot_color = 'green'
        
        # Prepare full state
        if len(robot_state) < 6:
            full_state = np.zeros(6)
            full_state[:3] = robot_state[:3]
            full_state[3:6] = 0.0
        else:
            full_state = robot_state
        
        # Try to solve NMPC
        try:
            self.opti.set_value(self.P_init, full_state)
            self.opti.set_value(self.P_goal, goal_state[:3])
            self.opti.set_value(self.P_costmap_force, [costmap_fx, costmap_fy])
            self.opti.set_value(self.P_max_force, max_force)
            
            # Initial guess: move toward goal
            initial_control = np.zeros((3, Config.N))
            
            # Calculate direction to goal
            dx = goal_state[0] - robot_state[0]
            dy = goal_state[1] - robot_state[1]
            dist_to_goal = np.sqrt(dx**2 + dy**2)
            
            if dist_to_goal > 0.1:
                goal_dir_x = dx / dist_to_goal
                goal_dir_y = dy / dist_to_goal
                
                theta = robot_state[2]
                Fx_goal = goal_dir_x * np.cos(theta) + goal_dir_y * np.sin(theta)
                Fy_goal = -goal_dir_x * np.sin(theta) + goal_dir_y * np.cos(theta)
                
                initial_control[0, :] = max_force * 0.5 * Fx_goal
                initial_control[1, :] = max_force * 0.5 * Fy_goal
            
            # Initial state guess
            initial_state = np.zeros((6, Config.N + 1))
            for k in range(Config.N + 1):
                dt_k = Config.DT * k
                initial_state[0, k] = full_state[0] + full_state[3] * dt_k
                initial_state[1, k] = full_state[1] + full_state[4] * dt_k
                initial_state[2, k] = full_state[2] + full_state[5] * dt_k
                initial_state[3:, k] = full_state[3:]
            
            self.opti.set_initial(self.X, initial_state)
            self.opti.set_initial(self.U, initial_control)
            
            sol = self.opti.solve()
            control = sol.value(self.U[:, 0])
            
            # Convert to accelerations
            ax_robot = control[0] / Config.MASS
            ay_robot = control[1] / Config.MASS
            alpha = control[2] / Config.INERTIA
            
            # Rotate to world frame
            theta = robot_state[2]
            ax_world = ax_robot * np.cos(theta) - ay_robot * np.sin(theta)
            ay_world = ax_robot * np.sin(theta) + ay_robot * np.cos(theta)
            
            control_world = np.array([ax_world, ay_world, alpha])
            predicted_traj = sol.value(self.X)[:2, :].T
            
        except Exception as e:
            # Simple fallback: move toward goal
            print(f"NMPC failed: {str(e)[:50]}... Using fallback")
            
            # Calculate direction to goal
            dx = goal_state[0] - robot_state[0]
            dy = goal_state[1] - robot_state[1]
            dist_to_goal = np.sqrt(dx**2 + dy**2)
            
            if dist_to_goal > 0.1:
                goal_dir_x = dx / dist_to_goal
                goal_dir_y = dy / dist_to_goal
                
                # Apply force toward goal
                force_mag = max_force * 0.3
                ax_robot = force_mag * goal_dir_x / Config.MASS
                ay_robot = force_mag * goal_dir_y / Config.MASS
                
                theta = robot_state[2]
                ax_world = ax_robot * np.cos(theta) - ay_robot * np.sin(theta)
                ay_world = ax_robot * np.sin(theta) + ay_robot * np.cos(theta)
            else:
                ax_world, ay_world = 0, 0
            
            control_world = np.array([ax_world, ay_world, 0.0])
            predicted_traj = None
        
        return control_world, robot_color, predicted_traj

# ==========================================
# 6. SIMULATION
# ==========================================
def run_simulation():
    print("="*70)
    print("SIMPLIFIED MECANUM NAVIGATION")
    print("="*70)
    
    # Setup
    grid = OccupancyGrid()
    num_obstacles = grid.generate_structured_obstacles()
    print(f"Generated {num_obstacles} obstacle groups")
    
    # Start and goal
    start = (1.0, 1.0)
    goal = (14.0, 14.0)
    
    # Find global path
    path = a_star_search(start, goal, grid)
    if len(path) == 0:
        print("Warning: No global path found! Using straight line")
        path = np.array([start, goal])
    
    # Create obstacles
    dyn_obs = create_dynamic_obstacles()
    
    # Initialize planner
    planner = IntegratedPlanner()
    
    # Initial state
    true_state = np.array([start[0], start[1], np.pi/4, 0.0, 0.0, 0.0])
    
    # Stuck detection
    stuck_counter = 0
    last_positions = []
    
    # Simulation control
    simulation_running = False
    simulation_paused = False
    
    # Setup visualization
    fig = plt.figure(figsize=(16, 10))
    
    # Main map
    ax_map = plt.subplot2grid((2, 2), (0, 0), rowspan=2, colspan=1)
    
    # Plot obstacles
    ax_map.imshow(grid.grid, cmap='binary', origin='lower', 
                  extent=[0, Config.WIDTH, 0, Config.HEIGHT], alpha=0.9)
    
    # Plot path
    if len(path) > 0:
        ax_map.plot(path[:,0], path[:,1], 'g--', alpha=0.7, linewidth=2, label='Path')
    
    # Start and goal
    ax_map.plot(start[0], start[1], 'go', markersize=15, label='Start', markeredgecolor='black')
    ax_map.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal', markeredgecolor='black')
    
    # Goal region
    goal_circle = patches.Circle((goal[0], goal[1]), Config.GOAL_TOLERANCE, fill=False, 
                                 linestyle='--', edgecolor='red', alpha=0.5, linewidth=2)
    ax_map.add_patch(goal_circle)
    
    # Robot
    robot_patch = patches.Rectangle((0,0), Config.ROBOT_WIDTH, Config.ROBOT_LENGTH, 
                                    color='green', alpha=0.9, edgecolor='black')
    ax_map.add_patch(robot_patch)
    
    # Robot footprint
    robot_footprint = patches.Circle((0,0), Config.ROBOT_WIDTH/2,
                                     fill=False, linestyle=':', color='green', alpha=0.5)
    ax_map.add_patch(robot_footprint)
    
    # Dynamic obstacles
    obs_patches = []
    for obs in dyn_obs:
        patch = patches.Circle((obs.x, obs.y), Config.OBSTACLE_RADIUS, 
                              color='orange', alpha=0.5, edgecolor='darkred')
        obs_patches.append(patch)
        ax_map.add_patch(patch)
    
    # Predicted trajectory
    predicted_line, = ax_map.plot([], [], 'y-', linewidth=2, alpha=0.8, label='Predicted')
    
    ax_map.set_title("Mecanum Robot Navigation", fontsize=14, fontweight='bold')
    ax_map.legend(loc='upper right')
    ax_map.set_xlim(-0.5, Config.WIDTH + 0.5)
    ax_map.set_ylim(-0.5, Config.HEIGHT + 0.5)
    ax_map.set_aspect('equal')
    ax_map.grid(True, alpha=0.3)
    
    # Costmap
    ax_costmap = plt.subplot2grid((2, 2), (0, 1))
    costmap_img = ax_costmap.imshow(np.zeros((planner.costmap.grid_size, planner.costmap.grid_size)), 
                                    cmap='RdYlGn_r', origin='lower',
                                    extent=[-planner.costmap.size/2, planner.costmap.size/2, 
                                            -planner.costmap.size/2, planner.costmap.size/2])
    ax_costmap.set_title("Local Costmap", fontweight='bold')
    ax_costmap.set_xlabel("Local X (m)")
    ax_costmap.set_ylabel("Local Y (m)")
    plt.colorbar(costmap_img, ax=ax_costmap, label='Cost')
    
    # Info panel
    ax_info = plt.subplot2grid((2, 2), (1, 1))
    ax_info.axis('off')
    
    # Metrics
    total_distance = 0
    last_position = np.array(start)
    collision_count = 0
    success = False
    simulation_time = 0
    
    def get_current_waypoint(path, robot_pos, lookahead=2):
        """Get the next waypoint along the path"""
        if len(path) == 0:
            return goal
        
        # Find closest point on path
        dists = np.linalg.norm(path - robot_pos[:2], axis=1)
        closest_idx = np.argmin(dists)
        
        # Look ahead a bit
        lookahead_idx = min(closest_idx + 1, len(path)-1)
        
        return np.array([path[lookahead_idx][0], path[lookahead_idx][1], 0.0])
    
    def update(frame):
        nonlocal true_state, simulation_running, simulation_paused
        nonlocal total_distance, last_position, collision_count, success, simulation_time
        nonlocal stuck_counter, last_positions
        
        if not simulation_running or simulation_paused:
            return [robot_patch, robot_footprint, costmap_img, predicted_line] + obs_patches
        
        simulation_time += Config.DT
        
        # Update dynamic obstacles
        for obs, patch in zip(dyn_obs, obs_patches):
            obs.update(Config.DT)
            patch.set_center((obs.x, obs.y))
        
        # Find closest obstacle distance
        min_dist = float('inf')
        for obs in dyn_obs:
            d = np.sqrt((true_state[0] - obs.x)**2 + (true_state[1] - obs.y)**2)
            min_dist = min(min_dist, d)
        
        # Check collision with STATIC obstacles
        robot_grid_x = int(true_state[0] / Config.RESOLUTION)
        robot_grid_y = int(true_state[1] / Config.RESOLUTION)
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                gx = robot_grid_x + dx
                gy = robot_grid_y + dy
                if 0 <= gx < grid.cols and 0 <= gy < grid.rows:
                    if grid.grid[gy, gx] == 1:
                        obs_x = gx * Config.RESOLUTION + Config.RESOLUTION/2
                        obs_y = gy * Config.RESOLUTION + Config.RESOLUTION/2
                        dist = np.sqrt((true_state[0] - obs_x)**2 + (true_state[1] - obs_y)**2)
                        min_dist = min(min_dist, dist)
        
        # Check collision
        collision_threshold = Config.ROBOT_WIDTH/2 + Config.OBSTACLE_RADIUS + 0.1
        if min_dist < collision_threshold:
            collision_count += 1
        
        # Stuck detection
        current_speed = np.sqrt(true_state[3]**2 + true_state[4]**2)
        last_positions.append(np.array([true_state[0], true_state[1]]))
        if len(last_positions) > 20:
            last_positions.pop(0)
        
        # Check if stuck
        if len(last_positions) >= 20:
            movement = np.linalg.norm(last_positions[-1] - last_positions[0])
            if movement < 0.2 and current_speed < 0.1:
                stuck_counter += 1
                if stuck_counter > 30:
                    print(f"🚨 Robot stuck! Applying escape...")
                    # Push away from nearest obstacle
                    if dyn_obs:
                        closest = min(dyn_obs, key=lambda o: np.sqrt((true_state[0]-o.x)**2 + (true_state[1]-o.y)**2))
                        dx = true_state[0] - closest.x
                        dy = true_state[1] - closest.y
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist > 0:
                            true_state[3] += 0.5 * dx / dist
                            true_state[4] += 0.5 * dy / dist
                    stuck_counter = 0
            else:
                stuck_counter = max(0, stuck_counter - 1)
        
        # Check goal
        dist_to_goal = np.sqrt((true_state[0] - goal[0])**2 + (true_state[1] - goal[1])**2)
        
        if dist_to_goal < Config.GOAL_TOLERANCE and not success:
            success = True
            simulation_running = False
            print("\n" + "="*60)
            print("🎉 GOAL REACHED! 🎉")
            print(f"Time: {simulation_time:.1f}s")
            print(f"Distance: {total_distance:.1f}m")
            print(f"Collision warnings: {collision_count}")
            print("="*60)
            return [robot_patch, robot_footprint, costmap_img, predicted_line] + obs_patches
        
        # Get current waypoint
        if len(path) > 0:
            local_goal = get_current_waypoint(path, true_state)
        else:
            local_goal = np.array([goal[0], goal[1], 0.0])
        
        # Plan
        control, robot_color, predicted_traj = planner.plan(
            true_state, local_goal, dyn_obs, grid, min_dist
        )
        
        # Update distance
        current_pos = np.array([true_state[0], true_state[1]])
        total_distance += np.linalg.norm(current_pos - last_position)
        last_position = current_pos.copy()
        
        # Apply control (with limits)
        if not np.any(np.isnan(control)) and not np.any(np.isinf(control)):
            # Update velocities
            true_state[3] += control[0] * Config.DT
            true_state[4] += control[1] * Config.DT
            true_state[5] += control[2] * Config.DT
            
            # Velocity damping
            true_state[3] *= 0.97
            true_state[4] *= 0.97
            true_state[5] *= 0.94
            
            # Limit velocities
            max_speed = 0.6  # Slow but steady
            current_speed = np.sqrt(true_state[3]**2 + true_state[4]**2)
            if current_speed > max_speed:
                true_state[3] *= max_speed / current_speed
                true_state[4] *= max_speed / current_speed
            
            # Update position
            true_state[0] += true_state[3] * Config.DT
            true_state[1] += true_state[4] * Config.DT
            true_state[2] += true_state[5] * Config.DT
        
        # Keep within bounds
        margin = Config.ROBOT_WIDTH/2 + 0.1
        true_state[0] = np.clip(true_state[0], margin, Config.WIDTH - margin)
        true_state[1] = np.clip(true_state[1], margin, Config.HEIGHT - margin)
        
        # Update robot visualization
        cx, cy, th = true_state[:3]
        w, l = Config.ROBOT_WIDTH, Config.ROBOT_LENGTH
        
        corner_x = cx - (w/2)*np.cos(th) + (l/2)*np.sin(th)
        corner_y = cy - (w/2)*np.sin(th) - (l/2)*np.cos(th)
        robot_patch.set_xy((corner_x, corner_y))
        robot_patch.angle = np.degrees(th)
        robot_patch.set_color(robot_color)
        
        # Update footprint
        robot_footprint.center = (cx, cy)
        
        # Update predicted trajectory
        if predicted_traj is not None and len(predicted_traj) > 0:
            predicted_line.set_data(predicted_traj[:, 0], predicted_traj[:, 1])
        
        # Update costmap
        costmap_img.set_array(planner.costmap.costmap)
        costmap_img.autoscale()
        
        # Update info panel
        ax_info.clear()
        ax_info.axis('off')
        
        y_pos = 0.95
        # Status
        if success:
            status = "GOAL REACHED!"
            status_color = 'green'
        elif min_dist < 0.5:
            status = "DANGER"
            status_color = 'red'
        elif min_dist < 0.8:
            status = "CAUTION"
            status_color = 'orange'
        else:
            status = "MOVING"
            status_color = 'blue'
        
        ax_info.text(0.05, y_pos, f"Status: {status}", fontsize=14, 
                    fontweight='bold', color=status_color)
        y_pos -= 0.08
        
        # Metrics
        metrics = [
            (f"Goal Distance: {dist_to_goal:.2f} m", 'black'),
            (f"Closest Obstacle: {min_dist:.2f} m", 'red' if min_dist < 0.5 else 'orange' if min_dist < 0.8 else 'black'),
            (f"Position: ({true_state[0]:.2f}, {true_state[1]:.2f})", 'black'),
            (f"Heading: {np.degrees(true_state[2]):.0f}°", 'black'),
            (f"Speed: {current_speed:.2f} m/s", 'green' if current_speed > 0.1 else 'orange'),
            (f"Distance: {total_distance:.1f} m", 'black'),
            (f"Time: {simulation_time:.1f} s", 'black'),
            (f"Collisions: {collision_count}", 'red' if collision_count > 0 else 'black'),
            ("", 'black'),
            ("CONTROLS:", 'black', 'bold'),
            ("  [S] - Start/Stop", 'black'),
            ("  [P] - Pause/Resume", 'black'),
            ("  [R] - Reset", 'black'),
            ("  [Q] - Quit", 'black'),
        ]
        
        for text, color, *style in metrics:
            if 'bold' in style:
                ax_info.text(0.05, y_pos, text, fontsize=11, fontweight='bold', color=color)
            else:
                ax_info.text(0.05, y_pos, text, fontsize=11, color=color)
            y_pos -= 0.06
        
        return [robot_patch, robot_footprint, costmap_img, predicted_line] + obs_patches
    
    def on_key(event):
        nonlocal simulation_running, simulation_paused, true_state, total_distance
        nonlocal collision_count, success, simulation_time, last_position, dyn_obs, obs_patches
        nonlocal stuck_counter, last_positions
        
        if event.key == 's' or event.key == 'S':
            simulation_running = not simulation_running
            print(f"Simulation {'STARTED' if simulation_running else 'STOPPED'}")
        
        elif event.key == 'p' or event.key == 'P':
            simulation_paused = not simulation_paused
            print(f"Simulation {'PAUSED' if simulation_paused else 'RESUMED'}")
        
        elif event.key == 'r' or event.key == 'R':
            # Reset
            true_state = np.array([start[0], start[1], np.pi/4, 0.0, 0.0, 0.0])
            simulation_running = False
            simulation_paused = False
            total_distance = 0
            collision_count = 0
            success = False
            simulation_time = 0
            last_position = np.array(start)
            stuck_counter = 0
            last_positions = []
            
            # Reset robot
            cx, cy, th = true_state[:3]
            w, l = Config.ROBOT_WIDTH, Config.ROBOT_LENGTH
            corner_x = cx - (w/2)*np.cos(th) + (l/2)*np.sin(th)
            corner_y = cy - (w/2)*np.sin(th) - (l/2)*np.cos(th)
            robot_patch.set_xy((corner_x, corner_y))
            robot_patch.angle = np.degrees(th)
            robot_patch.set_color('green')
            
            # Reset obstacles
            for patch in obs_patches:
                patch.remove()
            obs_patches.clear()
            
            dyn_obs.clear()
            dyn_obs.extend(create_dynamic_obstacles())
            
            for obs in dyn_obs:
                patch = patches.Circle((obs.x, obs.y), Config.OBSTACLE_RADIUS, 
                                      color='orange', alpha=0.5, edgecolor='darkred')
                obs_patches.append(patch)
                ax_map.add_patch(patch)
            
            print("Simulation RESET")
        
        elif event.key == 'q' or event.key == 'Q':
            plt.close()
            print("Simulation terminated.")
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("\n🚀 READY TO START!")
    print("Controls: [S] Start/Stop  [P] Pause/Resume  [R] Reset  [Q] Quit")
    print("-" * 70)
    
    ani = FuncAnimation(fig, update, frames=Config.SIM_TIME, interval=50, blit=False)
    plt.tight_layout()
    plt.show()

# ==========================================
# 7. RUN THE SIMULATION
# ==========================================
if __name__ == "__main__":
    run_simulation()
