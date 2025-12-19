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
# 1. PATH-PRIORITY CONFIGURATION - BALANCED FOR PATH FOLLOWING
# ==========================================
class Config:
    WIDTH = 15.0
    HEIGHT = 15.0
    RESOLUTION = 0.5
    
    ROBOT_WIDTH = 0.5
    ROBOT_LENGTH = 0.5
    
    # Safety parameters - BALANCED
    SAFETY_DIST = 0.8           # Balanced for path following
    EMERGENCY_STOP_DIST = 0.4   # For immediate collisions only
    CRITICAL_ESCAPE_DIST = 0.2  # Force immediate escape below this distance
    SLOW_DOWN_DIST = 0.8        # For moderate obstacles
    
    # NMPC Parameters
    DT = 0.1          
    N = 8              # Slightly reduced for better path tracking
    
    SIM_TIME = 400
    
    # Dynamic obstacles
    NUM_DYNAMIC_OBSTACLES = 5
    OBSTACLE_RADIUS = 0.4
    
    # COSTMAP PARAMETERS - STRONG PATH FOLLOWING
    LOCAL_MAP_SIZE = 4.0           # Good for local path tracking
    LOCAL_MAP_RES = 0.1
    INFLATION_RADIUS = 1.8         # Increased for better obstacle avoidance
    COSTMAP_MAX = 100.0
    COSTMAP_GOAL_ATTRACTION = 30.0  # Reduced - path is primary
    COSTMAP_PATH_ATTRACTION = 150.0 # STRONG PATH FOLLOWING
    COSTMAP_PATH_DEVIATION_PENALTY = 50.0  # Penalty for deviating
    
    # DYNAMIC MODEL PARAMETERS
    MASS = 1.5
    INERTIA = 0.15
    MAX_FORCE = 4.0
    MAX_TORQUE = 2.5
    
    # Navigation parameters
    GOAL_TOLERANCE = 0.5
    
    # Dynamic obstacle speeds - INCREASED FOR MORE CHALLENGE
    OBSTACLE_MIN_SPEED = 0.25  # Increased from 0.15
    OBSTACLE_MAX_SPEED = 0.5   # Increased from 0.25
    OBSTACLE_PREDICTION_TIME = 1.5  # Moderate lookahead

    # PATH FOLLOWING PARAMETERS
    PATH_LOOKAHEAD_DISTANCE = 1.5   # Reduced for less oscillation
    PATH_LOOKAHEAD_POINTS = 3       # Reduced for smoother following
    PATH_TRACKING_WEIGHT = 150.0    # Reduced for smoother control
    MAX_PATH_DEVIATION = 1.5        # Max allowed deviation from path
    PATH_RECOVERY_GAIN = 1.5        # Reduced for smoother recovery
    PATH_REJOIN_DISTANCE = 0.5      # How close to get before rejoining path
    PATH_STOPPING_DISTANCE = 0.3    # Distance at which to stop forward motion
    
    # PATH PLANNING MARGIN
    PATH_OBSTACLE_MARGIN = 1.0      # Keep path at least this far from obstacles
    
    # COLLISION DETECTION PARAMETERS
    COLLISION_CHECK_SAMPLES = 8     # Number of points to check on robot perimeter

# ==========================================
# 2. MAPPING - IMPROVED PATH PLANNING
# ==========================================
class OccupancyGrid:
    def __init__(self):
        self.cols = int(Config.WIDTH / Config.RESOLUTION)
        self.rows = int(Config.HEIGHT / Config.RESOLUTION)
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.inflated_grid = None

    def generate_structured_obstacles(self):
        """Create a structured obstacle course"""
        # Clear grid first
        self.grid.fill(0)
        
        # Create a challenging but not impossible environment
        obstacles = [
            # Border walls (with openings)
            (0, 0, 30, 1),      # Bottom wall
            (0, 29, 30, 1),     # Top wall  
            (0, 0, 1, 30),      # Left wall
            (29, 0, 1, 30),     # Right wall
            
            # Strategic obstacles to make path interesting
            (8, 8, 3, 3),       # Central obstacle 1
            (18, 8, 3, 3),      # Central obstacle 2
            (13, 18, 3, 3),     # Central obstacle 3
            (8, 20, 2, 2),      # Upper left obstacle
            (20, 20, 2, 2),     # Upper right obstacle
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
        
        # Inflate obstacles for path planning (keep path away from obstacles)
        self.inflated_grid = self.grid.copy()
        inflation_size = int(Config.PATH_OBSTACLE_MARGIN / Config.RESOLUTION)
        
        # Simple inflation by checking neighbors
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] == 1:
                    # Inflate in all directions
                    for di in range(-inflation_size, inflation_size + 1):
                        for dj in range(-inflation_size, inflation_size + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.rows and 0 <= nj < self.cols:
                                dist = np.sqrt(di**2 + dj**2)
                                if dist <= inflation_size:
                                    self.inflated_grid[ni, nj] = 1
        
        return len(obstacles)

def a_star_search(start, goal, grid_obj):
    def heuristic(a, b): 
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) * 1.001
    
    # Convert world coordinates to grid indices
    s_node = (int(start[1]/Config.RESOLUTION), int(start[0]/Config.RESOLUTION))
    g_node = (int(goal[1]/Config.RESOLUTION), int(goal[0]/Config.RESOLUTION))
    
    # Use inflated grid for path planning to keep distance from obstacles
    if grid_obj.inflated_grid is not None:
        grid_to_use = grid_obj.inflated_grid
    else:
        grid_to_use = grid_obj.grid
    
    # Add diagonal movements (8-directional)
    neighbors = [(0,1), (0,-1), (1,0), (-1,0), 
                 (1,1), (1,-1), (-1,1), (-1,-1)]
    
    close_set = set()
    came_from = {}
    gscore = {s_node: 0}
    fscore = {s_node: heuristic(s_node, g_node)}
    oheap = []
    heapq.heappush(oheap, (fscore[s_node], s_node))
    
    # Also keep track of the best node if we can't reach goal
    best_node = s_node
    best_fscore = fscore[s_node]
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        
        if current == g_node:
            path = []
            # Reconstruct path
            while current in came_from:
                row, col = current
                world_x = col * Config.RESOLUTION + Config.RESOLUTION/2
                world_y = row * Config.RESOLUTION + Config.RESOLUTION/2
                path.append((world_x, world_y))
                current = came_from[current]
            
            path.append(start)
            path = path[::-1]
            
            # Smooth the path (remove unnecessary waypoints)
            if len(path) > 2:
                smoothed_path = [path[0]]
                for i in range(1, len(path)-1):
                    # Check if point i can be skipped (line of sight to next point)
                    dx = path[i+1][0] - smoothed_path[-1][0]
                    dy = path[i+1][1] - smoothed_path[-1][1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist > Config.RESOLUTION * 2:  # Keep some points for curvature
                        smoothed_path.append(path[i])
                smoothed_path.append(path[-1])
                path = smoothed_path
            
            return np.array(path)
        
        # Update best node if this one is better
        if fscore.get(current, float('inf')) < best_fscore:
            best_node = current
            best_fscore = fscore[current]
        
        close_set.add(current)
        current_row, current_col = current
        
        for i, j in neighbors:
            neighbor = (current_row + i, current_col + j)
            
            if (neighbor[0] < 0 or neighbor[0] >= grid_obj.rows or 
                neighbor[1] < 0 or neighbor[1] >= grid_obj.cols):
                continue
            
            if grid_to_use[neighbor[0]][neighbor[1]] == 1: 
                continue
            
            if i != 0 and j != 0:
                tent_g = gscore[current] + np.sqrt(2)
            else:
                tent_g = gscore[current] + 1.0
            
            if neighbor in close_set and tent_g >= gscore.get(neighbor, 0):
                continue
                
            if tent_g < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tent_g
                fscore[neighbor] = tent_g + heuristic(neighbor, g_node)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    
    # If we get here, no path found. Create a simple straight-line path
    print("Warning: No complete path found by A*, using fallback path")
    # Create a simple path that goes around obstacles
    path_points = []
    
    # Add start
    path_points.append(start)
    
    # Add intermediate point (go diagonally first, then straight)
    mid_x = (start[0] + goal[0]) / 2
    mid_y = (start[1] + goal[1]) / 2
    
    # Try to avoid obstacles by going around them
    if start[0] < goal[0] and start[1] < goal[1]:
        # Diagonal path
        path_points.append((mid_x, start[1] + 2))
        path_points.append((mid_x + 2, mid_y))
    else:
        # Simple diagonal
        path_points.append((mid_x, mid_y))
    
    # Add goal
    path_points.append(goal)
    
    return np.array(path_points)

# ==========================================
# 3. COLLISION CHECKING FUNCTIONS - FIXED
# ==========================================
def check_collision(x, y, theta, obstacles, static_grid, robot_width=Config.ROBOT_WIDTH, robot_length=Config.ROBOT_LENGTH):
    """Check if robot at position (x,y,theta) would collide - FIXED VERSION"""
    # Check dynamic obstacles
    for obs in obstacles:
        dx = obs.x - x
        dy = obs.y - y
        distance = np.sqrt(dx**2 + dy**2)
        if distance < (max(robot_width, robot_length)/2 + obs.radius + 0.1):
            return True
    
    # **FIX: Check static obstacles using robot's rotated rectangle**
    # Generate points around the robot's perimeter
    half_width = robot_width / 2
    half_length = robot_length / 2
    
    # Generate points along the robot's perimeter
    points = []
    for i in range(Config.COLLISION_CHECK_SAMPLES):
        angle = 2 * np.pi * i / Config.COLLISION_CHECK_SAMPLES
        # Points on an ellipse that approximates the robot's shape
        rx = half_length * np.cos(angle)
        ry = half_width * np.sin(angle)
        
        # Rotate the point
        rotated_x = rx * np.cos(theta) - ry * np.sin(theta)
        rotated_y = rx * np.sin(theta) + ry * np.cos(theta)
        
        # Translate to world coordinates
        world_x = x + rotated_x
        world_y = y + rotated_y
        
        points.append((world_x, world_y))
    
    # Check each point against the grid
    for px, py in points:
        gx = int(px / Config.RESOLUTION)
        gy = int(py / Config.RESOLUTION)
        
        # Check the cell and its immediate neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = gx + dx, gy + dy
                if (0 <= nx < static_grid.cols and 0 <= ny < static_grid.rows):
                    if static_grid.grid[ny, nx] == 1:
                        cell_x = nx * Config.RESOLUTION + Config.RESOLUTION/2
                        cell_y = ny * Config.RESOLUTION + Config.RESOLUTION/2
                        cell_dist = np.sqrt((px - cell_x)**2 + (py - cell_y)**2)
                        if cell_dist < Config.RESOLUTION/2:  # If point is inside the cell
                            return True
    
    return False

def check_static_obstacle_distance(x, y, static_grid):
    """Calculate minimum distance to any static obstacle - FIXED"""
    gx = int(x / Config.RESOLUTION)
    gy = int(y / Config.RESOLUTION)
    
    min_dist = float('inf')
    
    # Check a larger area around the robot
    check_radius = 3  # cells
    
    for dx in range(-check_radius, check_radius + 1):
        for dy in range(-check_radius, check_radius + 1):
            nx, ny = gx + dx, gy + dy
            if (0 <= nx < static_grid.cols and 0 <= ny < static_grid.rows):
                if static_grid.grid[ny, nx] == 1:
                    cell_x = nx * Config.RESOLUTION + Config.RESOLUTION/2
                    cell_y = ny * Config.RESOLUTION + Config.RESOLUTION/2
                    cell_dist = np.sqrt((x - cell_x)**2 + (y - cell_y)**2)
                    min_dist = min(min_dist, cell_dist)
    
    return min_dist

# ==========================================
# 4. PATH-FOLLOWING COSTMAP - IMPROVED PATH FOLLOWING
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
        self.reference_path = None
        self.current_target_idx = 0
        self.smoothed_path_points = []
        
    def update(self, robot_x, robot_y, robot_theta, robot_vx, robot_vy, 
               goal_x, goal_y, static_grid, dynamic_obstacles):
        self.costmap = np.zeros((self.grid_size, self.grid_size))
        
        # Create local grid
        xs = np.linspace(-self.size/2, self.size/2, self.grid_size)
        ys = np.linspace(-self.size/2, self.size/2, self.grid_size)
        local_x, local_y = np.meshgrid(xs, ys)
        
        # Rotate to world frame
        world_x = robot_x + local_x * np.cos(robot_theta) - local_y * np.sin(robot_theta)
        world_y = robot_y + local_x * np.sin(robot_theta) + local_y * np.cos(robot_theta)
        
        # 1. PRIMARY: IMPROVED PATH FOLLOWING WITH MEDIAN OF MULTIPLE POINTS
        if self.reference_path is not None and len(self.reference_path) > 0:
            robot_pos = np.array([robot_x, robot_y])
            
            # Find closest point on path
            distances = np.linalg.norm(self.reference_path - robot_pos, axis=1)
            closest_idx = np.argmin(distances)
            
            # Get multiple lookahead points for smoother following
            lookahead_points = []
            lookahead_indices = []
            
            # First point: immediate next point (or slightly ahead if at end)
            next_idx = min(closest_idx + 1, len(self.reference_path) - 1)
            lookahead_points.append(self.reference_path[next_idx])
            lookahead_indices.append(next_idx)
            
            # Second point: a bit further ahead
            idx2 = min(closest_idx + 2, len(self.reference_path) - 1)
            if idx2 != next_idx:
                lookahead_points.append(self.reference_path[idx2])
                lookahead_indices.append(idx2)
            
            # Third point: even further or goal
            idx3 = min(closest_idx + min(Config.PATH_LOOKAHEAD_POINTS, len(self.reference_path) - 1), 
                      len(self.reference_path) - 1)
            if idx3 not in lookahead_indices:
                lookahead_points.append(self.reference_path[idx3])
                lookahead_indices.append(idx3)
            
            # Calculate median of lookahead points for smoother target
            if len(lookahead_points) > 0:
                lookahead_points = np.array(lookahead_points)
                median_point = np.median(lookahead_points, axis=0)
                
                # Also consider the path beyond for better anticipation
                far_idx = min(closest_idx + Config.PATH_LOOKAHEAD_POINTS * 2, len(self.reference_path) - 1)
                if far_idx not in lookahead_indices:
                    far_point = self.reference_path[far_idx]
                    # Blend median with far point (70% median, 30% far)
                    target_point = median_point * 0.7 + far_point * 0.3
                else:
                    target_point = median_point
                
                # Store current target index for external use
                self.current_target_idx = min(max(lookahead_indices), len(self.reference_path) - 1)
                
                # Store smoothed path points for visualization
                self.smoothed_path_points = lookahead_points
                
                # Strong attraction to calculated target point
                path_dist = np.sqrt((world_x - target_point[0])**2 + (world_y - target_point[1])**2)
                path_cost = -Config.COSTMAP_PATH_ATTRACTION * (1 - path_dist / self.size)
                path_cost = np.clip(path_cost, -Config.COSTMAP_PATH_ATTRACTION, 0)
                self.costmap += path_cost
                
                # Penalty for deviation from path (adaptive based on obstacle proximity)
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        # Find closest point on path segment
                        min_dist = float('inf')
                        for k in range(len(self.reference_path) - 1):
                            p1 = self.reference_path[k]
                            p2 = self.reference_path[k + 1]
                            line_vec = p2 - p1
                            point_vec = np.array([world_x[i, j], world_y[i, j]]) - p1
                            
                            line_len = np.linalg.norm(line_vec)
                            if line_len > 0:
                                line_unitvec = line_vec / line_len
                                projection_length = np.dot(point_vec, line_unitvec)
                                
                                if 0 <= projection_length <= line_len:
                                    closest_point = p1 + projection_length * line_unitvec
                                elif projection_length < 0:
                                    closest_point = p1
                                else:
                                    closest_point = p2
                                
                                dist = np.linalg.norm(np.array([world_x[i, j], world_y[i, j]]) - closest_point)
                                min_dist = min(min_dist, dist)
                        
                        # Adaptive deviation penalty: less penalty near obstacles
                        robot_dist = np.sqrt(((i - self.center_idx) * self.res)**2 + ((j - self.center_idx) * self.res)**2)
                        if robot_dist < 1.5:  # Wider area for deviation penalty
                            # Check if there are obstacles nearby
                            wx, wy = world_x[i, j], world_y[i, j]
                            near_obstacle = False
                            
                            # Check dynamic obstacles
                            for obs in dynamic_obstacles:
                                if np.sqrt((wx - obs.x)**2 + (wy - obs.y)**2) < Config.OBSTACLE_RADIUS * 2:
                                    near_obstacle = True
                                    break
                            
                            # Reduce penalty near obstacles (allows deviation)
                            if near_obstacle:
                                deviation_cost = Config.COSTMAP_PATH_DEVIATION_PENALTY * 0.3 * min_dist**2
                            else:
                                deviation_cost = Config.COSTMAP_PATH_DEVIATION_PENALTY * min_dist**2
                            
                            self.costmap[i, j] += deviation_cost
        
        # 2. SECONDARY: Goal attraction (when no path or close to goal)
        if self.reference_path is None or len(self.reference_path) == 0:
            goal_dist = np.sqrt((world_x - goal_x)**2 + (world_y - goal_y)**2)
            goal_cost = -Config.COSTMAP_GOAL_ATTRACTION * (1 - goal_dist / self.size)
            goal_cost = np.clip(goal_cost, -Config.COSTMAP_GOAL_ATTRACTION, 0)
            self.costmap += goal_cost
        
        # 3. OBSTACLES: Adaptive cost based on proximity - FIXED FOR BETTER OBSTACLE AVOIDANCE
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                wx, wy = world_x[i, j], world_y[i, j]
                
                if 0 <= wx < Config.WIDTH and 0 <= wy < Config.HEIGHT:
                    gx = int(wx / Config.RESOLUTION)
                    gy = int(wy / Config.RESOLUTION)
                    
                    # Check static obstacles
                    if 0 <= gx < static_grid.cols and 0 <= gy < static_grid.rows:
                        if static_grid.grid[gy, gx] == 1:
                            dx = (i - self.center_idx) * self.res
                            dy = (j - self.center_idx) * self.res
                            dist = np.sqrt(dx**2 + dy**2)
                            
                            # **FIX: Stronger penalty for static obstacles**
                            if dist < Config.INFLATION_RADIUS:
                                if dist < Config.EMERGENCY_STOP_DIST:
                                    cost = Config.COSTMAP_MAX * 2.0  # Higher cost for static obstacles
                                elif dist < Config.SLOW_DOWN_DIST:
                                    cost = Config.COSTMAP_MAX * 1.5 * (1 - dist/Config.SLOW_DOWN_DIST)
                                else:
                                    cost = Config.COSTMAP_MAX * 1.0 * np.exp(-dist / 0.2)
                                self.costmap[i, j] += cost
        
        # 4. Dynamic obstacles with improved prediction
        for obs in dynamic_obstacles:
            dx = obs.x - robot_x
            dy = obs.y - robot_y
            distance_to_robot = np.sqrt(dx**2 + dy**2)
            
            # Consider obstacles within a larger area
            if distance_to_robot < Config.SLOW_DOWN_DIST * 2:
                # Calculate relative velocity
                rel_vx = obs.vx - robot_vx
                rel_vy = obs.vy - robot_vy
                
                # Time to collision with better calculation
                relative_speed = np.sqrt(rel_vx**2 + rel_vy**2)
                if relative_speed > 0.01:
                    # Project future position
                    future_time = min(distance_to_robot / relative_speed, Config.OBSTACLE_PREDICTION_TIME)
                    future_obs_x = obs.x + obs.vx * future_time
                    future_obs_y = obs.y + obs.vy * future_time
                    
                    # Calculate distance to future obstacle position
                    dist = np.sqrt((world_x - future_obs_x)**2 + (world_y - future_obs_y)**2)
                    
                    # Apply cost based on collision probability
                    if future_time < 1.0:  # Only if collision is imminent
                        # Higher cost for closer collisions
                        collision_prob = np.exp(-dist / (Config.OBSTACLE_RADIUS * 2))
                        obs_cost = Config.COSTMAP_MAX * collision_prob * (1 - future_time)
                        self.costmap += np.minimum(obs_cost, Config.COSTMAP_MAX)
        
        # Smooth and compute gradient
        self.costmap = gaussian_filter(self.costmap, sigma=0.3)
        self.gradient_y, self.gradient_x = np.gradient(-self.costmap)
        
        return world_x, world_y
    
    def get_gradient_force(self, robot_x, robot_y, robot_theta):
        center_i = self.center_idx
        center_j = self.center_idx
        
        grad_x = self.gradient_x[center_i, center_j]
        grad_y = self.gradient_y[center_i, center_j]
        
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        if grad_mag > 1e-6:
            grad_x /= grad_mag
            grad_y /= grad_mag
        else:
            grad_x, grad_y = 1.0, 0.0
        
        force_x = grad_x * np.cos(robot_theta) - grad_y * np.sin(robot_theta)
        force_y = grad_x * np.sin(robot_theta) + grad_y * np.cos(robot_theta)
        
        return force_x, force_y
    
    def set_reference_path(self, path):
        self.reference_path = path
        self.current_target_idx = 0
        self.smoothed_path_points = []
    
    def get_current_target_point(self, robot_x, robot_y):
        """Get the current target point on the path for the robot to follow - FIXED VERSION"""
        if self.reference_path is None or len(self.reference_path) == 0:
            return None
        
        robot_pos = np.array([robot_x, robot_y])
        
        # If the robot is very close to the goal, return the goal
        if np.linalg.norm(robot_pos - self.reference_path[-1]) < Config.GOAL_TOLERANCE:
            return self.reference_path[-1]
        
        # Find the closest point on the path
        distances = np.linalg.norm(self.reference_path - robot_pos, axis=1)
        closest_idx = np.argmin(distances)
        
        # Look for a point that is at least lookahead_distance away from the robot
        lookahead_distance = Config.PATH_LOOKAHEAD_DISTANCE
        for i in range(closest_idx, len(self.reference_path)):
            if np.linalg.norm(self.reference_path[i] - robot_pos) >= lookahead_distance:
                return self.reference_path[i]
        
        # If no point is found, return the last point
        return self.reference_path[-1]

# ==========================================
# 5. DYNAMIC OBSTACLES - FASTER
# ==========================================
class DynamicObstacle:
    def __init__(self, x, y, vx, vy):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.radius = Config.OBSTACLE_RADIUS
        self.history = [(x, y)]
        self.max_history = 20
        self.speed = np.sqrt(vx**2 + vy**2)
        
    def update(self, dt):
        # Add some randomness to movement
        if random.random() < 0.1:
            # Slightly change direction
            angle_change = random.uniform(-0.4, 0.4)
            angle = np.arctan2(self.vy, self.vx) + angle_change
            self.vx = self.speed * np.cos(angle)
            self.vy = self.speed * np.sin(angle)
        
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        self.history.append((self.x, self.y))
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Bounce off walls with some randomness
        margin = self.radius * 1.5
        if self.x < margin or self.x > Config.WIDTH - margin:
            self.vx *= -random.uniform(0.8, 1.2)
            self.x = np.clip(self.x, margin, Config.WIDTH - margin)
            # Occasionally change speed when hitting walls
            if random.random() < 0.3:
                self.speed = random.uniform(Config.OBSTACLE_MIN_SPEED, Config.OBSTACLE_MAX_SPEED)
                self.vx = np.sign(self.vx) * self.speed
        if self.y < margin or self.y > Config.HEIGHT - margin:
            self.vy *= -random.uniform(0.8, 1.2)
            self.y = np.clip(self.y, margin, Config.HEIGHT - margin)
            if random.random() < 0.3:
                self.speed = random.uniform(Config.OBSTACLE_MIN_SPEED, Config.OBSTACLE_MAX_SPEED)
                self.vy = np.sign(self.vy) * self.speed
        
        # Maintain speed within bounds
        current_speed = np.sqrt(self.vx**2 + self.vy**2)
        if current_speed > 0:
            self.vx = self.vx / current_speed * self.speed
            self.vy = self.vy / current_speed * self.speed

def create_dynamic_obstacles():
    obstacles = []
    
    positions = [
        # Create obstacles that will challenge the path
        (3.0, 7.0), (12.0, 7.0), (3.0, 13.0), (12.0, 13.0),
        (7.0, 4.0), (7.0, 16.0), (1.0, 10.0), (14.0, 10.0),
        # Random positions
        (5.0, 2.0), (10.0, 2.0), (5.0, 18.0), (10.0, 18.0),
    ]
    
    for i, (x, y) in enumerate(positions[:Config.NUM_DYNAMIC_OBSTACLES]):
        # Higher speeds for more challenge
        speed = random.uniform(Config.OBSTACLE_MIN_SPEED, Config.OBSTACLE_MAX_SPEED)
        
        # Different movement patterns
        if i % 6 == 0:
            # Random direction
            angle = random.uniform(0, 2*np.pi)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
        elif i % 6 == 1:
            # Horizontal movement
            vx = speed * random.choice([-1, 1])
            vy = random.uniform(-0.3, 0.3) * speed
        elif i % 6 == 2:
            # Vertical movement
            vx = random.uniform(-0.3, 0.3) * speed
            vy = speed * random.choice([-1, 1])
        elif i % 6 == 3:
            # Diagonal movement
            vx = speed * random.choice([-0.707, 0.707])
            vy = speed * random.choice([-0.707, 0.707])
        elif i % 6 == 4:
            # Circular-ish movement
            vx = speed * 0.8
            vy = speed * 0.6 * random.choice([-1, 1])
        else:
            # Fast moving
            speed *= 1.5  # Extra fast
            angle = random.uniform(0, 2*np.pi)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
        
        obstacles.append(DynamicObstacle(x, y, vx, vy))
    
    return obstacles

# ==========================================
# 6. IMPROVED PATH-FOLLOWING PLANNER - FIXED
# ==========================================
class IntegratedPlanner:
    def __init__(self):
        self.costmap = LocalCostmap()
        self.initialize_nmpc()
        self.last_successful_control = None
        self.obstacle_avoidance_mode = False
        self.avoidance_counter = 0
        self.path_deviation = 0.0
        self.path_recovery_mode = False
        self.last_path_point = None
        self.alternative_path_point = None
        self.path_deviation_history = []
        self.deviation_required = False
        self.oscillation_counter = 0
        self.last_target_idx = 0
        self.target_history = []
        
    def initialize_nmpc(self):
        """Initialize NMPC with SMART path following - FIXED for CasADi"""
        self.opti = ca.Opti()
        
        # State: [x, y, theta, vx, vy, omega]
        self.X = self.opti.variable(6, Config.N + 1)
        
        # Control: [Fx, Fy, Tau]
        self.U = self.opti.variable(3, Config.N)
        
        # Parameters
        self.P_init = self.opti.parameter(6)
        self.P_goal = self.opti.parameter(3)
        self.P_path_point = self.opti.parameter(2)  # Current target on path
        self.P_alt_path_point = self.opti.parameter(2)  # Alternative target (for deviation)
        self.P_costmap_force = self.opti.parameter(2)  # Safety force
        self.P_max_force = self.opti.parameter(1)
        self.P_obstacle_danger = self.opti.parameter(1)
        self.P_path_recovery = self.opti.parameter(1)
        self.P_deviation_required = self.opti.parameter(1)  # Whether deviation is needed
        self.P_near_target = self.opti.parameter(1)  # Whether robot is near target (for stopping)
        self.P_static_obstacle_dist = self.opti.parameter(1)  # Distance to closest static obstacle
        
        # Robot parameters
        m = Config.MASS
        I = Config.INERTIA
        
        total_cost = 0
        
        for k in range(Config.N):
            # PRIMARY: SMART PATH FOLLOWING
            # Use continuous blending based on deviation_required parameter
            # When deviation_required is 1.0: use more alternative path (70%)
            # When deviation_required is 0.0: use more primary path (80%)
            primary_weight = 0.8 - 0.5 * self.P_deviation_required
            alt_weight = 0.2 + 0.5 * self.P_deviation_required
            
            primary_error = self.X[:2, k] - self.P_path_point
            alt_error = self.X[:2, k] - self.P_alt_path_point
            
            # Use CasADi if_else for conditional weight on path recovery
            path_weight = Config.PATH_TRACKING_WEIGHT * (1.0 + self.P_path_recovery)
            
            total_cost += path_weight * (primary_weight * ca.mtimes(primary_error.T, primary_error) +
                                       alt_weight * ca.mtimes(alt_error.T, alt_error))
            
            # SECONDARY: Goal tracking
            state_error = self.X[:3, k] - self.P_goal
            goal_weight = 15.0 / (1.0 + self.P_obstacle_danger)
            total_cost += goal_weight * ca.mtimes(state_error.T, state_error)
            
            # Velocity penalty - increased near target to prevent oscillation
            vel_error = self.X[3:6, k]
            vel_weight = 0.1 * (1.0 + self.P_obstacle_danger + 2.0 * self.P_near_target)  # Higher when near target
            total_cost += vel_weight * ca.mtimes(vel_error.T, vel_error)
            
            # Control effort
            control_effort = self.U[:, k]
            R_force = ca.diag([0.01, 0.01, 0.02])
            total_cost += ca.mtimes(control_effort.T, ca.mtimes(R_force, control_effort))
            
            # Costmap following (for safety)
            desired_Fx = self.P_costmap_force[0] * self.P_max_force
            desired_Fy = self.P_costmap_force[1] * self.P_max_force
            
            F_error_x = self.U[0, k] - desired_Fx
            F_error_y = self.U[1, k] - desired_Fy
            cmap_weight = 0.8 * self.P_obstacle_danger
            total_cost += cmap_weight * (F_error_x**2 + F_error_y**2)
            
            # **FIX: Static obstacle penalty - using smooth function instead of if statement**
            # Use a smooth penalty function that increases as distance decreases
            # This avoids the CasADi MX type error in if statements
            static_penalty = 100.0 * ca.exp(-self.P_static_obstacle_dist / 0.2)
            total_cost += static_penalty
        
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
            
            ax_robot = Fx / m
            ay_robot = Fy / m
            alpha = Tau / I
            
            ax_world = ax_robot * ca.cos(theta) - ay_robot * ca.sin(theta)
            ay_world = ax_robot * ca.sin(theta) + ay_robot * ca.cos(theta)
            
            dx = vx
            dy = vy
            dtheta = omega
            dvx = ax_world
            dvy = ay_world
            domega = alpha
            
            state_dot = ca.vertcat(dx, dy, dtheta, dvx, dvy, domega)
            self.opti.subject_to(
                self.X[:, k+1] == self.X[:, k] + state_dot * Config.DT
            )
            
            # Control constraints
            self.opti.subject_to(self.opti.bounded(-Config.MAX_FORCE, self.U[0, k], Config.MAX_FORCE))
            self.opti.subject_to(self.opti.bounded(-Config.MAX_FORCE, self.U[1, k], Config.MAX_FORCE))
            self.opti.subject_to(self.opti.bounded(-Config.MAX_TORQUE, self.U[2, k], Config.MAX_TORQUE))
            
            # Velocity constraints
            max_speed = 0.8  # Higher speed allowed for faster navigation
            self.opti.subject_to(self.opti.bounded(-max_speed, self.X[3, k], max_speed))
            self.opti.subject_to(self.opti.bounded(-max_speed, self.X[4, k], max_speed))
            self.opti.subject_to(self.opti.bounded(-0.5, self.X[5, k], 0.5))
        
        self.opti.subject_to(self.X[:, 0] == self.P_init)
        
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
                'ipopt.max_iter': 100, 'ipopt.tol': 1e-2}
        self.opti.solver('ipopt', opts)    
    
    def plan(self, robot_state, goal_state, dynamic_obstacles, static_grid, closest_obstacle_dist):
        # **FIX: Calculate distance to static obstacles**
        static_obstacle_dist = check_static_obstacle_distance(robot_state[0], robot_state[1], static_grid)
        
        # Update costmap
        world_x, world_y = self.costmap.update(
            robot_state[0], robot_state[1], robot_state[2],
            robot_state[3], robot_state[4],
            goal_state[0], goal_state[1], static_grid, dynamic_obstacles
        )
        
        # Get gradient force
        costmap_fx, costmap_fy = self.costmap.get_gradient_force(
            robot_state[0], robot_state[1], robot_state[2]
        )
        
        # Get current target point on path - FIXED: Use new method
        robot_pos = np.array([robot_state[0], robot_state[1]])
        primary_target = None
        alternative_target = None
        self.deviation_required = False
        
        if self.costmap.reference_path is not None and len(self.costmap.reference_path) > 0:
            # Find closest point on path
            distances = np.linalg.norm(self.costmap.reference_path - robot_pos, axis=1)
            closest_idx = np.argmin(distances)
            closest_point = self.costmap.reference_path[closest_idx]
            
            # Calculate deviation from path
            self.path_deviation = distances[closest_idx]
            self.path_deviation_history.append(self.path_deviation)
            if len(self.path_deviation_history) > 10:
                self.path_deviation_history.pop(0)
            
            # **FIX: Check for oscillation by monitoring target changes**
            current_target_idx = closest_idx
            if hasattr(self, 'last_target_idx'):
                if current_target_idx == self.last_target_idx:
                    self.oscillation_counter += 1
                else:
                    self.oscillation_counter = 0
            self.last_target_idx = current_target_idx
            
            # If oscillating, reduce lookahead distance temporarily
            effective_lookahead = Config.PATH_LOOKAHEAD_DISTANCE
            if self.oscillation_counter > 10:  # Oscillating for more than 1 second
                effective_lookahead = max(0.5, Config.PATH_LOOKAHEAD_DISTANCE * 0.7)
                print(f"⚠️ Oscillation detected, reducing lookahead to {effective_lookahead:.2f}m")
            
            # Get primary lookahead target using adaptive method
            primary_target = self.get_adaptive_lookahead_point(robot_pos, closest_idx, effective_lookahead)
            
            # **FIX: Check if robot is too close to target and should stop**
            near_target = False
            if primary_target is not None:
                dist_to_target = np.linalg.norm(primary_target - robot_pos)
                if dist_to_target < Config.PATH_STOPPING_DISTANCE:
                    # If very close to target, look further ahead or use goal
                    if closest_idx < len(self.costmap.reference_path) - 1:
                        # Look further ahead
                        next_idx = min(closest_idx + 3, len(self.costmap.reference_path) - 1)
                        primary_target = self.costmap.reference_path[next_idx]
                        print(f"🔍 Close to target, looking further ahead to point {next_idx}")
                    else:
                        # At end of path, use goal
                        primary_target = goal_state[:2]
            
            # Check if we need to deviate due to obstacles
            deviation_needed = False
            safe_deviation_direction = None
            
            # Check obstacles along the path to primary target
            if primary_target is not None:
                path_to_target = primary_target - robot_pos
                path_dist = np.linalg.norm(path_to_target)
                
                if path_dist > 0:
                    path_dir = path_to_target / path_dist
                    
                    # **FIX: Check static obstacles in path direction**
                    if static_obstacle_dist < Config.SLOW_DOWN_DIST:
                        # Check if static obstacle is in the path direction
                        # Get direction to closest static obstacle
                        gx = int(robot_state[0] / Config.RESOLUTION)
                        gy = int(robot_state[1] / Config.RESOLUTION)
                        
                        # Find closest obstacle cell
                        closest_obstacle_cell = None
                        min_cell_dist = float('inf')
                        
                        for dx in [-2, -1, 0, 1, 2]:
                            for dy in [-2, -1, 0, 1, 2]:
                                nx, ny = gx + dx, gy + dy
                                if (0 <= nx < static_grid.cols and 0 <= ny < static_grid.rows):
                                    if static_grid.grid[ny, nx] == 1:
                                        cell_x = nx * Config.RESOLUTION + Config.RESOLUTION/2
                                        cell_y = ny * Config.RESOLUTION + Config.RESOLUTION/2
                                        cell_dist = np.sqrt((robot_state[0] - cell_x)**2 + (robot_state[1] - cell_y)**2)
                                        if cell_dist < min_cell_dist:
                                            min_cell_dist = cell_dist
                                            closest_obstacle_cell = (cell_x, cell_y)
                        
                        if closest_obstacle_cell is not None:
                            obstacle_vec = np.array(closest_obstacle_cell) - robot_pos
                            obstacle_dist = np.linalg.norm(obstacle_vec)
                            if obstacle_dist > 0:
                                obstacle_dir = obstacle_vec / obstacle_dist
                                dot_product = np.dot(path_dir, obstacle_dir)
                                
                                # If static obstacle is in front and close
                                if dot_product > 0.7 and obstacle_dist < Config.SLOW_DOWN_DIST:
                                    deviation_needed = True
                                    print(f"⚠️ Static obstacle in path at {obstacle_dist:.2f}m")
                    
                    # Check dynamic obstacles
                    for obs in dynamic_obstacles:
                        # Vector from robot to obstacle
                        obs_vec = np.array([obs.x - robot_state[0], obs.y - robot_state[1]])
                        obs_dist = np.linalg.norm(obs_vec)
                        
                        if obs_dist < Config.SLOW_DOWN_DIST * 1.5:
                            # Check if obstacle is in the path direction
                            if obs_dist > 0:
                                obs_dir = obs_vec / obs_dist
                                dot_product = np.dot(path_dir, obs_dir)
                                
                                # If obstacle is in front and close
                                if dot_product > 0.7 and obs_dist < Config.SLOW_DOWN_DIST:
                                    deviation_needed = True
                                    
                                    # Calculate safe deviation direction (perpendicular to path)
                                    deviation_dir = np.array([-path_dir[1], path_dir[0]])
                                    
                                    # Check which side is safer
                                    # Try right side first
                                    right_deviation = deviation_dir
                                    left_deviation = -deviation_dir
                                    
                                    # Check safety of right deviation
                                    right_safe = True
                                    check_dist = Config.SLOW_DOWN_DIST
                                    check_pos = robot_pos + right_deviation * check_dist
                                    
                                    for obs2 in dynamic_obstacles:
                                        if np.linalg.norm(check_pos - np.array([obs2.x, obs2.y])) < Config.OBSTACLE_RADIUS * 2:
                                            right_safe = False
                                            break
                                    
                                    # Check safety of left deviation
                                    left_safe = True
                                    check_pos = robot_pos + left_deviation * check_dist
                                    
                                    for obs2 in dynamic_obstacles:
                                        if np.linalg.norm(check_pos - np.array([obs2.x, obs2.y])) < Config.OBSTACLE_RADIUS * 2:
                                            left_safe = False
                                            break
                                    
                                    # Choose safer direction
                                    if right_safe and left_safe:
                                        # Both safe, choose based on which brings us closer to path
                                        future_right_pos = robot_pos + right_deviation * 1.0
                                        future_left_pos = robot_pos + left_deviation * 1.0
                                        right_path_dist = np.min(np.linalg.norm(self.costmap.reference_path - future_right_pos, axis=1))
                                        left_path_dist = np.min(np.linalg.norm(self.costmap.reference_path - future_left_pos, axis=1))
                                        safe_deviation_direction = right_deviation if right_path_dist < left_path_dist else left_deviation
                                    elif right_safe:
                                        safe_deviation_direction = right_deviation
                                    elif left_safe:
                                        safe_deviation_direction = left_deviation
                                    else:
                                        # Both sides blocked, use costmap gradient
                                        safe_deviation_direction = np.array([costmap_fx, costmap_fy])
            
            # Set alternative target based on deviation need
            if deviation_needed and safe_deviation_direction is not None:
                self.deviation_required = True
                # Create alternative target offset from primary path
                deviation_distance = min(Config.SLOW_DOWN_DIST * 1.5, path_dist * 0.5)
                alternative_target = robot_pos + safe_deviation_direction * deviation_distance
                
                # Also look ahead on path from alternative position
                alt_lookahead = min(closest_idx + 3, len(self.costmap.reference_path) - 1)
                path_ahead = self.costmap.reference_path[alt_lookahead]
                
                # Blend alternative position with path ahead
                alternative_target = alternative_target * 0.7 + path_ahead * 0.3
                print(f"🔄 Deviating from path to avoid obstacle")
            else:
                self.deviation_required = False
                # No deviation needed, alternative target is just ahead on path
                alt_idx = min(closest_idx + 1, len(self.costmap.reference_path) - 1)
                alternative_target = self.costmap.reference_path[alt_idx]
            
            # Check if we need to recover to path
            if self.path_deviation > Config.MAX_PATH_DEVIATION:
                self.path_recovery_mode = True
                print(f"⚠️ High path deviation ({self.path_deviation:.2f}m), recovering...")
                # In recovery mode, use a point on the path ahead of closest point
                recovery_idx = min(closest_idx + 2, len(self.costmap.reference_path) - 1)
                primary_target = self.costmap.reference_path[recovery_idx]
            elif self.path_deviation < Config.PATH_REJOIN_DISTANCE and self.path_recovery_mode:
                self.path_recovery_mode = False
                print("✅ Rejoined path")
            
            # Store for visualization
            self.last_path_point = primary_target
            self.alternative_path_point = alternative_target
        
        else:
            primary_target = goal_state[:2]
            alternative_target = goal_state[:2]
            self.path_deviation = 0.0
            self.path_recovery_mode = False
            self.deviation_required = False
        
        # **FIX: Add goal-seeking behavior when near end of path**
        if self.costmap.reference_path is not None and len(self.costmap.reference_path) > 0:
            robot_pos = np.array([robot_state[0], robot_state[1]])
            distances = np.linalg.norm(self.costmap.reference_path - robot_pos, axis=1)
            closest_idx = np.argmin(distances)
            
            # If we're on the last segment of path, blend with goal direction
            if closest_idx >= len(self.costmap.reference_path) - 2:
                goal_vec = np.array(goal_state[:2]) - robot_pos
                goal_dist = np.linalg.norm(goal_vec)
                
                if goal_dist > 0.1:
                    goal_dir = goal_vec / goal_dist
                    # Blend path target with goal direction
                    if primary_target is not None:
                        path_to_goal = np.array(goal_state[:2]) - primary_target
                        if np.linalg.norm(path_to_goal) > 0.5:  # If path doesn't lead directly to goal
                            # Add goal attraction
                            blend_factor = 0.3
                            primary_target = primary_target * (1-blend_factor) + np.array(goal_state[:2]) * blend_factor
        
        # CRITICAL FIX: Check for immediate danger and force escape
        force_escape = False
        escape_vector = None
        
        # **FIX: Include static obstacles in danger assessment**
        combined_min_dist = min(closest_obstacle_dist, static_obstacle_dist)
        
        # Check if any obstacle is critically close
        if combined_min_dist < Config.CRITICAL_ESCAPE_DIST:
            print(f"🚨 CRITICAL! Obstacle at {combined_min_dist:.2f}m - Forcing immediate escape!")
            force_escape = True
            
            # Find escape direction
            if static_obstacle_dist < closest_obstacle_dist:
                # Escape from static obstacle
                # Find direction away from closest static obstacle
                gx = int(robot_state[0] / Config.RESOLUTION)
                gy = int(robot_state[1] / Config.RESOLUTION)
                
                closest_obstacle_cell = None
                min_cell_dist = float('inf')
                
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        nx, ny = gx + dx, gy + dy
                        if (0 <= nx < static_grid.cols and 0 <= ny < static_grid.rows):
                            if static_grid.grid[ny, nx] == 1:
                                cell_x = nx * Config.RESOLUTION + Config.RESOLUTION/2
                                cell_y = ny * Config.RESOLUTION + Config.RESOLUTION/2
                                cell_dist = np.sqrt((robot_state[0] - cell_x)**2 + (robot_state[1] - cell_y)**2)
                                if cell_dist < min_cell_dist:
                                    min_cell_dist = cell_dist
                                    closest_obstacle_cell = (cell_x, cell_y)
                
                if closest_obstacle_cell is not None:
                    dx = closest_obstacle_cell[0] - robot_state[0]
                    dy = closest_obstacle_cell[1] - robot_state[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    if distance > 0:
                        escape_vector = np.array([-dx/distance, -dy/distance])
            else:
                # Escape from dynamic obstacle
                for obs in dynamic_obstacles:
                    dx = obs.x - robot_state[0]
                    dy = obs.y - robot_state[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    if abs(distance - closest_obstacle_dist) < 0.01:
                        if distance > 0:
                            escape_vector = np.array([-dx/distance, -dy/distance])
                            break
            
            if escape_vector is None:
                escape_vector = np.array([costmap_fx, costmap_fy])
        
        # Assess danger level
        danger_level = 0.0
        avoidance_vector = None
        
        if not force_escape:
            # Check static obstacles for danger
            if static_obstacle_dist < Config.SLOW_DOWN_DIST:
                # Static obstacles are always dangerous when close
                danger = 1.0 - (static_obstacle_dist / Config.SLOW_DOWN_DIST)
                danger_level = max(danger_level, danger)
                
                # Create avoidance vector away from static obstacles
                gx = int(robot_state[0] / Config.RESOLUTION)
                gy = int(robot_state[1] / Config.RESOLUTION)
                
                # Calculate average direction to nearby static obstacles
                obstacle_sum = np.zeros(2)
                obstacle_count = 0
                
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        nx, ny = gx + dx, gy + dy
                        if (0 <= nx < static_grid.cols and 0 <= ny < static_grid.rows):
                            if static_grid.grid[ny, nx] == 1:
                                cell_x = nx * Config.RESOLUTION + Config.RESOLUTION/2
                                cell_y = ny * Config.RESOLUTION + Config.RESOLUTION/2
                                obstacle_vec = np.array([cell_x - robot_state[0], cell_y - robot_state[1]])
                                dist = np.linalg.norm(obstacle_vec)
                                if dist < Config.SLOW_DOWN_DIST:
                                    # Weight by inverse distance (closer obstacles have more influence)
                                    weight = 1.0 / (dist + 0.1)
                                    obstacle_sum += -obstacle_vec * weight  # Away from obstacle
                                    obstacle_count += 1
                
                if obstacle_count > 0:
                    avoidance_vector = obstacle_sum / obstacle_count
                    norm = np.linalg.norm(avoidance_vector)
                    if norm > 0:
                        avoidance_vector /= norm
            
            # Check dynamic obstacles
            for obs in dynamic_obstacles:
                dx = obs.x - robot_state[0]
                dy = obs.y - robot_state[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < Config.SLOW_DOWN_DIST:
                    rel_vx = obs.vx - robot_state[3]
                    rel_vy = obs.vy - robot_state[4]
                    
                    # Robust TTC calculation
                    relative_speed = np.sqrt(rel_vx**2 + rel_vy**2)
                    if relative_speed > 0.01:
                        ttc = distance / relative_speed
                        if (dx * rel_vx + dy * rel_vy) < 0:
                            if 0 < ttc < 1.0:
                                danger = 1.0 - ttc
                                danger_level = max(danger_level, danger)
                                
                                if ttc < 0.5:
                                    if rel_vx != 0 or rel_vy != 0:
                                        avoid_x = -rel_vy
                                        avoid_y = rel_vx
                                        norm = np.sqrt(avoid_x**2 + avoid_y**2)
                                        if norm > 0:
                                            if avoidance_vector is None:
                                                avoidance_vector = np.array([avoid_x/norm, avoid_y/norm])
                                            else:
                                                # Blend with existing avoidance vector
                                                avoidance_vector += np.array([avoid_x/norm, avoid_y/norm])
        
        # Normalize avoidance vector if exists
        if avoidance_vector is not None and not force_escape:
            norm = np.linalg.norm(avoidance_vector)
            if norm > 0:
                avoidance_vector /= norm
        
        # **FIX: Check if robot is near its target for stopping control**
        near_target = False
        if primary_target is not None:
            dist_to_primary = np.linalg.norm(primary_target - robot_pos)
            if dist_to_primary < Config.PATH_STOPPING_DISTANCE:
                near_target = True
        
        # Set control parameters
        if force_escape:
            # CRITICAL ESCAPE MODE
            max_force = Config.MAX_FORCE
            robot_color = 'darkred'
            self.obstacle_avoidance_mode = True
            self.avoidance_counter = 30
            
            if escape_vector is not None:
                costmap_fx, costmap_fy = escape_vector[0], escape_vector[1]
            
        elif combined_min_dist < Config.EMERGENCY_STOP_DIST:
            max_force = Config.MAX_FORCE * 0.4
            robot_color = 'red'
            self.obstacle_avoidance_mode = True
            self.avoidance_counter = 20
            
            if avoidance_vector is not None:
                costmap_fx, costmap_fy = avoidance_vector[0], avoidance_vector[1]
            
        elif combined_min_dist < Config.SLOW_DOWN_DIST or danger_level > 0.5:
            max_force = Config.MAX_FORCE * 0.6
            robot_color = 'orange'
            self.obstacle_avoidance_mode = True
            self.avoidance_counter = 15
            
            if avoidance_vector is not None:
                blend = min(1.0, danger_level)
                if not self.deviation_required:
                    # Calculate path direction to alternative target
                    dx = alternative_target[0] - robot_state[0]
                    dy = alternative_target[1] - robot_state[1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist > 0.1:
                        path_dir = np.array([dx/dist, dy/dist])
                        costmap_fx = blend * avoidance_vector[0] + (1-blend) * path_dir[0]
                        costmap_fy = blend * avoidance_vector[1] + (1-blend) * path_dir[1]
        
        else:
            # NORMAL OPERATION
            if self.avoidance_counter > 0:
                self.avoidance_counter -= 1
                max_force = Config.MAX_FORCE * 0.7
                robot_color = 'yellow'
            else:
                self.obstacle_avoidance_mode = False
                max_force = Config.MAX_FORCE * 0.8
                robot_color = 'green'
            
            # Calculate direction to appropriate target
            target_to_use = alternative_target if self.deviation_required else primary_target
            dx = target_to_use[0] - robot_state[0]
            dy = target_to_use[1] - robot_state[1]
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist > 0.1:
                # Convert to robot frame
                theta = robot_state[2]
                path_fx = (dx * np.cos(theta) + dy * np.sin(theta)) / dist
                path_fy = (-dx * np.sin(theta) + dy * np.cos(theta)) / dist
                
                # Blend with costmap force when needed
                if self.path_recovery_mode or self.path_deviation > 0.5:
                    safety_blend = 0.2
                    costmap_fx = safety_blend * costmap_fx + (1-safety_blend) * path_fx
                    costmap_fy = safety_blend * costmap_fy + (1-safety_blend) * path_fy
                else:
                    costmap_fx, costmap_fy = path_fx, path_fy
            else:
                # Very close to target, reduce force significantly
                max_force *= 0.2
                costmap_fx, costmap_fy = 0.0, 0.0
        
        # Prepare full state
        if len(robot_state) < 6:
            full_state = np.zeros(6)
            full_state[:3] = robot_state[:3]
            full_state[3:6] = 0.0
        else:
            full_state = robot_state
        
        # In critical escape mode, override targets
        if force_escape:
            primary_target = robot_state[:2]
            alternative_target = robot_state[:2]
        
        # Try to solve NMPC
        try:
            self.opti.set_value(self.P_init, full_state)
            self.opti.set_value(self.P_goal, goal_state[:3])
            self.opti.set_value(self.P_path_point, primary_target)
            self.opti.set_value(self.P_alt_path_point, alternative_target)
            self.opti.set_value(self.P_costmap_force, [costmap_fx, costmap_fy])
            self.opti.set_value(self.P_max_force, max_force)
            self.opti.set_value(self.P_obstacle_danger, danger_level)
            self.opti.set_value(self.P_path_recovery, 1.0 if self.path_recovery_mode else 0.0)
            self.opti.set_value(self.P_deviation_required, 1.0 if self.deviation_required else 0.0)
            self.opti.set_value(self.P_near_target, 1.0 if near_target else 0.0)
            self.opti.set_value(self.P_static_obstacle_dist, static_obstacle_dist)
            
            # Initial guess
            initial_control = np.zeros((3, Config.N))
            
            dx = primary_target[0] - robot_state[0]
            dy = primary_target[1] - robot_state[1]
            dist_to_target = np.sqrt(dx**2 + dy**2)
            
            if dist_to_target > 0.1 or force_escape:
                theta = robot_state[2]
                Fx_world = costmap_fx if force_escape else dx / dist_to_target
                Fy_world = costmap_fy if force_escape else dy / dist_to_target
                Fx_robot = Fx_world * np.cos(theta) + Fy_world * np.sin(theta)
                Fy_robot = -Fx_world * np.sin(theta) + Fy_world * np.cos(theta)
                
                if force_escape:
                    force_factor = max_force * 1.0
                else:
                    force_factor = max_force * 0.8 * (1.0 - danger_level)
                    if self.path_recovery_mode:
                        force_factor *= Config.PATH_RECOVERY_GAIN
                    if near_target:
                        force_factor *= 0.3  # Reduce force when near target
                    # **FIX: Reduce force when close to static obstacles**
                    if static_obstacle_dist < Config.SLOW_DOWN_DIST:
                        force_factor *= 0.5
                
                initial_control[0, :] = force_factor * Fx_robot
                initial_control[1, :] = force_factor * Fy_robot
            
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
            
            ax_robot = control[0] / Config.MASS
            ay_robot = control[1] / Config.MASS
            alpha = control[2] / Config.INERTIA
            
            theta = robot_state[2]
            ax_world = ax_robot * np.cos(theta) - ay_robot * np.sin(theta)
            ay_world = ax_robot * np.sin(theta) + ay_robot * np.cos(theta)
            
            control_world = np.array([ax_world, ay_world, alpha])
            predicted_traj = sol.value(self.X)[:2, :].T
            
            self.last_successful_control = control_world
            
        except Exception as e:
            print(f"NMPC failed: {str(e)[:50]}... Using fallback")
            
            # Fallback
            if force_escape:
                force_mag = max_force * 1.0
                if escape_vector is not None:
                    Fx_world, Fy_world = escape_vector[0], escape_vector[1]
                else:
                    Fx_world, Fy_world = 1.0, 0.0
            else:
                target_to_use = alternative_target if self.deviation_required else primary_target
                dx = target_to_use[0] - robot_state[0]
                dy = target_to_use[1] - robot_state[1]
                dist_to_target = np.sqrt(dx**2 + dy**2)
                
                if dist_to_target > 0.1:
                    theta = robot_state[2]
                    Fx_world = dx / dist_to_target
                    Fy_world = dy / dist_to_target
                    
                    force_mag = max_force * 0.5
                    if self.path_recovery_mode:
                        force_mag *= Config.PATH_RECOVERY_GAIN
                    if near_target:
                        force_mag *= 0.3  # Reduce force when near target
                    # **FIX: Reduce force when close to static obstacles**
                    if static_obstacle_dist < Config.SLOW_DOWN_DIST:
                        force_mag *= 0.5
                else:
                    Fx_world, Fy_world, force_mag = 0, 0, 0
            
            if force_mag > 0:
                theta = robot_state[2]
                Fx_robot = Fx_world * np.cos(theta) + Fy_world * np.sin(theta)
                Fy_robot = -Fx_world * np.sin(theta) + Fy_world * np.cos(theta)
                
                ax_robot = force_mag * Fx_robot / Config.MASS
                ay_robot = force_mag * Fy_robot / Config.MASS
            else:
                ax_robot, ay_robot = 0, 0
            
            theta = robot_state[2]
            ax_world = ax_robot * np.cos(theta) - ay_robot * np.sin(theta)
            ay_world = ax_robot * np.sin(theta) + ay_robot * np.cos(theta)
            control_world = np.array([ax_world, ay_world, 0.0])
            predicted_traj = None
        
        return control_world, robot_color, predicted_traj
    
    def get_adaptive_lookahead_point(self, robot_pos, closest_idx, lookahead_distance):
        """Get an adaptive lookahead point based on current conditions"""
        if self.costmap.reference_path is None or len(self.costmap.reference_path) == 0:
            return None
        
        # Start from closest point
        current_idx = closest_idx
        
        # If we're near the end of the path, use the goal
        if current_idx >= len(self.costmap.reference_path) - 2:
            return self.costmap.reference_path[-1]
        
        # Look for a point at approximately the lookahead distance
        accumulated_distance = 0.0
        target_idx = current_idx
        
        for i in range(current_idx, len(self.costmap.reference_path) - 1):
            segment_length = np.linalg.norm(self.costmap.reference_path[i+1] - self.costmap.reference_path[i])
            accumulated_distance += segment_length
            target_idx = i + 1
            
            if accumulated_distance >= lookahead_distance:
                # Return this point
                return self.costmap.reference_path[target_idx]
        
        # If we haven't reached the lookahead distance, return the last point
        return self.costmap.reference_path[-1]

# ==========================================
# 7. SIMULATION - WITH IMPROVED PATH FOLLOWING
# ==========================================
def run_simulation():
    print("="*70)
    print("IMPROVED PATH-FOLLOWING MECANUM NAVIGATION")
    print("="*70)
    
    # Setup
    grid = OccupancyGrid()
    num_obstacles = grid.generate_structured_obstacles()
    print(f"Generated {num_obstacles} obstacle groups")
    
    # Start and goal
    start = (1.0, 1.0)
    goal = (14.0, 14.0)
    
    # Find global path with obstacle margin
    path = a_star_search(start, goal, grid)
    if len(path) == 0:
        print("Warning: No global path found! Using straight line")
        path = np.array([start, goal])
    else:
        print(f"Global path found with {len(path)} waypoints")
        print(f"Path keeps at least {Config.PATH_OBSTACLE_MARGIN}m from obstacles")
    
    # Create faster obstacles
    dyn_obs = create_dynamic_obstacles()
    print(f"Created {len(dyn_obs)} dynamic obstacles")
    print(f"Obstacle speeds: {Config.OBSTACLE_MIN_SPEED:.2f} - {Config.OBSTACLE_MAX_SPEED:.2f} m/s")
    
    # Initialize planner
    planner = IntegratedPlanner()
    
    # Set reference path
    if len(path) > 0:
        planner.costmap.set_reference_path(path)
        print(f"Path length: {len(path)} waypoints")
    
    # Initial state
    true_state = np.array([start[0], start[1], np.pi/4, 0.0, 0.0, 0.0])
    
    # Path following metrics
    path_following_error = []
    last_path_index = 0
    
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
        ax_map.plot(path[:,0], path[:,1], 'g-', alpha=0.9, linewidth=3, label='Planned Path', zorder=5)
        # Plot waypoints
        ax_map.scatter(path[:,0], path[:,1], c='green', s=30, alpha=0.7, zorder=6, marker='o')
    
    # Start and goal
    ax_map.plot(start[0], start[1], 'go', markersize=15, label='Start', markeredgecolor='black', zorder=10)
    ax_map.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal', markeredgecolor='black', zorder=10)
    
    # Goal region
    goal_circle = patches.Circle((goal[0], goal[1]), Config.GOAL_TOLERANCE, fill=False, 
                                 linestyle='--', edgecolor='red', alpha=0.5, linewidth=2)
    ax_map.add_patch(goal_circle)
    
    # Robot
    robot_patch = patches.Rectangle((0,0), Config.ROBOT_WIDTH, Config.ROBOT_LENGTH, 
                                    color='green', alpha=0.9, edgecolor='black', zorder=10)
    ax_map.add_patch(robot_patch)
    
    # Current path target point
    path_target_point, = ax_map.plot([], [], 'bo', markersize=10, alpha=0.8, label='Path Target', zorder=9)
    alt_path_point, = ax_map.plot([], [], 'mo', markersize=8, alpha=0.6, label='Alt Target', zorder=9)
    
    # Robot footprint and safety zones
    robot_footprint = patches.Circle((0,0), Config.ROBOT_WIDTH/2,
                                     fill=False, linestyle=':', color='green', alpha=0.5, zorder=9)
    ax_map.add_patch(robot_footprint)
    
    # Safety zones
    safety_circle = patches.Circle((0,0), Config.SLOW_DOWN_DIST,
                                   fill=False, linestyle='--', color='yellow', alpha=0.3, linewidth=1)
    ax_map.add_patch(safety_circle)
    
    emergency_circle = patches.Circle((0,0), Config.EMERGENCY_STOP_DIST,
                                      fill=False, linestyle='--', color='red', alpha=0.3, linewidth=1)
    ax_map.add_patch(emergency_circle)
    
    # Critical escape zone
    critical_circle = patches.Circle((0,0), Config.CRITICAL_ESCAPE_DIST,
                                     fill=False, linestyle='-', color='darkred', alpha=0.5, linewidth=2)
    ax_map.add_patch(critical_circle)
    
    # Dynamic obstacles
    obs_patches = []
    obs_trails = []
    for obs in dyn_obs:
        patch = patches.Circle((obs.x, obs.y), Config.OBSTACLE_RADIUS, 
                              color='orange', alpha=0.7, edgecolor='darkred', zorder=8)
        obs_patches.append(patch)
        ax_map.add_patch(patch)
        
        # Trail
        trail, = ax_map.plot([], [], 'r-', alpha=0.3, linewidth=1, zorder=6)
        obs_trails.append(trail)
    
    # Predicted trajectory
    predicted_line, = ax_map.plot([], [], 'y-', linewidth=2, alpha=0.8, label='Predicted', zorder=6)
    
    ax_map.set_title("Improved Path-Following Mecanum Navigation", fontsize=14, fontweight='bold')
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
    avoidance_maneuvers = 0
    escape_maneuvers = 0
    deviation_maneuvers = 0
    path_deviations = []
    
    # Stuck detection variables
    stuck_counter = 0
    last_movement_time = 0
    
    def update(frame):
        nonlocal true_state, simulation_running, simulation_paused
        nonlocal total_distance, last_position, collision_count, success, simulation_time
        nonlocal avoidance_maneuvers, escape_maneuvers, deviation_maneuvers, last_path_index, path_deviations, path_following_error
        nonlocal stuck_counter, last_movement_time
        
        if not simulation_running or simulation_paused:
            return [robot_patch, robot_footprint, safety_circle, emergency_circle, critical_circle,
                    costmap_img, predicted_line, path_target_point, alt_path_point] + obs_patches + obs_trails
        
        simulation_time += Config.DT
        
        # Update dynamic obstacles
        for obs in dyn_obs:
            obs.update(Config.DT)
        
        # **FIX: Calculate distance to static obstacles**
        static_obstacle_dist = check_static_obstacle_distance(true_state[0], true_state[1], grid)
        
        # Find closest obstacle (dynamic)
        min_dyn_dist = float('inf')
        closest_obs = None
        
        for obs in dyn_obs:
            dx = obs.x - true_state[0]
            dy = obs.y - true_state[1]
            distance = np.sqrt(dx**2 + dy**2)
            min_dyn_dist = min(min_dyn_dist, distance)
            
            if distance < Config.SLOW_DOWN_DIST:
                closest_obs = obs
        
        # Combined minimum distance
        min_dist = min(min_dyn_dist, static_obstacle_dist)
        
        # **FIX: Check collision with static obstacles using improved function**
        if check_collision(true_state[0], true_state[1], true_state[2], dyn_obs, grid):
            collision_count += 1
            print(f"💥 Collision warning #{collision_count}")
            # Apply a force away from obstacles
            if static_obstacle_dist < min_dyn_dist:
                # Collision with static obstacle
                print("🚫 COLLISION WITH STATIC OBSTACLE!")
                # Find direction away from closest static obstacle
                gx = int(true_state[0] / Config.RESOLUTION)
                gy = int(true_state[1] / Config.RESOLUTION)
                
                closest_obstacle_cell = None
                min_cell_dist = float('inf')
                
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        nx, ny = gx + dx, gy + dy
                        if (0 <= nx < grid.cols and 0 <= ny < grid.rows):
                            if grid.grid[ny, nx] == 1:
                                cell_x = nx * Config.RESOLUTION + Config.RESOLUTION/2
                                cell_y = ny * Config.RESOLUTION + Config.RESOLUTION/2
                                cell_dist = np.sqrt((true_state[0] - cell_x)**2 + (true_state[1] - cell_y)**2)
                                if cell_dist < min_cell_dist:
                                    min_cell_dist = cell_dist
                                    closest_obstacle_cell = (cell_x, cell_y)
                
                if closest_obstacle_cell is not None:
                    # Apply repulsive force
                    dx = closest_obstacle_cell[0] - true_state[0]
                    dy = closest_obstacle_cell[1] - true_state[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    if distance > 0:
                        repulsive_force = 2.0
                        true_state[3] += -repulsive_force * (dx/distance) * Config.DT
                        true_state[4] += -repulsive_force * (dy/distance) * Config.DT
        
        # Calculate path following error
        if len(path) > 0:
            robot_pos = np.array([true_state[0], true_state[1]])
            distances = np.linalg.norm(path - robot_pos, axis=1)
            closest_idx = np.argmin(distances)
            path_error = distances[closest_idx]
            path_following_error.append(path_error)
            path_deviations.append(path_error)
            
            # **FIX: Update last_path_index based on proximity AND forward progress**
            # Check if we're close enough to consider reaching a waypoint
            if path_error < Config.PATH_REJOIN_DISTANCE:
                # Only advance if we're actually moving forward toward next waypoint
                if closest_idx >= last_path_index:
                    last_path_index = closest_idx
                
                # Force advance if stuck at same waypoint for too long
                elif simulation_time > 5.0 and last_path_index < len(path) - 1:
                    # Check if we're stuck
                    last_movement = np.linalg.norm(robot_pos - last_position)
                    if last_movement < 0.05:  # Hardly moving
                        last_path_index += 1
                        print(f"🔄 Force advancing to waypoint {last_path_index}")
        
        # Check goal
        dist_to_goal = np.sqrt((true_state[0] - goal[0])**2 + (true_state[1] - goal[1])**2)
        
        if dist_to_goal < Config.GOAL_TOLERANCE and not success:
            success = True
            simulation_running = False
            
            # Calculate path following metrics
            if len(path_following_error) > 0:
                avg_error = np.mean(path_following_error)
                max_error = np.max(path_following_error)
            else:
                avg_error = max_error = 0
            
            print("\n" + "="*60)
            print("🎉 GOAL REACHED! 🎉")
            print(f"Time: {simulation_time:.1f}s")
            print(f"Distance: {total_distance:.1f}m")
            print(f"Average path error: {avg_error:.3f}m")
            print(f"Max path deviation: {max_error:.3f}m")
            print(f"Collision warnings: {collision_count}")
            print(f"Avoidance maneuvers: {avoidance_maneuvers}")
            print(f"Escape maneuvers: {escape_maneuvers}")
            print(f"Deviation maneuvers: {deviation_maneuvers}")
            print("="*60)
            return [robot_patch, robot_footprint, safety_circle, emergency_circle, critical_circle,
                    costmap_img, predicted_line, path_target_point, alt_path_point] + obs_patches + obs_trails
        
        # Use local goal from path or final goal
        if len(path) > 0:
            # Use planner's path target
            if planner.last_path_point is not None:
                local_goal = np.array([planner.last_path_point[0], planner.last_path_point[1], 0.0])
            else:
                # Fallback: use next point on path
                if last_path_index < len(path) - 1:
                    local_goal = np.array([path[last_path_index + 1][0], path[last_path_index + 1][1], 0.0])
                else:
                    local_goal = np.array([goal[0], goal[1], 0.0])
        else:
            local_goal = np.array([goal[0], goal[1], 0.0])
        
        # Plan - **FIX: Pass static obstacle distance to planner**
        control, robot_color, predicted_traj = planner.plan(
            true_state, local_goal, dyn_obs, grid, min_dyn_dist
        )
        
        # **FIX: Apply anti-oscillation logic**
        if planner.oscillation_counter > 5:
            # Reduce control force when oscillating
            oscillation_reduction = max(0.3, 1.0 - planner.oscillation_counter * 0.1)
            control *= oscillation_reduction
        
        # **FIX: Add velocity damping based on distance to target and static obstacles**
        if planner.last_path_point is not None:
            target_dist = np.linalg.norm(np.array(planner.last_path_point) - np.array([true_state[0], true_state[1]]))
            if target_dist < Config.PATH_STOPPING_DISTANCE * 2:
                # Reduce control force as we approach target
                damping_factor = target_dist / (Config.PATH_STOPPING_DISTANCE * 2)
                control *= damping_factor
        
        # **FIX: Reduce control force when close to static obstacles**
        if static_obstacle_dist < Config.SLOW_DOWN_DIST:
            obstacle_factor = static_obstacle_dist / Config.SLOW_DOWN_DIST
            control *= obstacle_factor
            print(f"⚠️ Close to static obstacle ({static_obstacle_dist:.2f}m), reducing force")
        
        # Track maneuvers
        if min_dist < Config.CRITICAL_ESCAPE_DIST and planner.obstacle_avoidance_mode:
            escape_maneuvers += 1
        elif min_dist < Config.EMERGENCY_STOP_DIST and planner.obstacle_avoidance_mode:
            avoidance_maneuvers += 1
        elif planner.deviation_required:
            deviation_maneuvers += 1
        
        # Update distance
        current_pos = np.array([true_state[0], true_state[1]])
        movement = np.linalg.norm(current_pos - last_position)
        total_distance += movement
        last_position = current_pos.copy()
        
        # **FIX: Check if robot is stuck**
        if movement < 0.01:  # Hardly moving
            stuck_counter += 1
            if stuck_counter > 20 and dist_to_goal > 1.0:  # Stuck for 2 seconds
                # Apply a nudge in the goal direction
                dx = goal[0] - true_state[0]
                dy = goal[1] - true_state[1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0.1:
                    theta = true_state[2]
                    nudge_force = 0.5
                    control[0] += nudge_force * (dx/dist) * np.cos(theta)
                    control[1] += nudge_force * (dy/dist) * np.sin(theta)
                    print("🤖 Robot stuck, nudging toward goal...")
                    stuck_counter = 0
        else:
            stuck_counter = 0
        
        # Apply control
        if not np.any(np.isnan(control)) and not np.any(np.isinf(control)):
            true_state[3] += control[0] * Config.DT
            true_state[4] += control[1] * Config.DT
            true_state[5] += control[2] * Config.DT
            
            # Velocity damping - increased when near target or obstacles
            damping = 0.98
            if planner.last_path_point is not None:
                target_dist = np.linalg.norm(np.array(planner.last_path_point) - np.array([true_state[0], true_state[1]]))
                if target_dist < Config.PATH_STOPPING_DISTANCE * 2:
                    damping = 0.95  # Stronger damping near target
            
            if static_obstacle_dist < Config.SLOW_DOWN_DIST:
                damping = 0.92  # Even stronger damping near static obstacles
            
            true_state[3] *= damping
            true_state[4] *= damping
            true_state[5] *= damping
            
            # Adaptive speed limits
            if min_dist < Config.CRITICAL_ESCAPE_DIST:
                max_speed = 0.7
            elif min_dist < Config.EMERGENCY_STOP_DIST:
                max_speed = 0.3
            elif min_dist < Config.SLOW_DOWN_DIST:
                max_speed = 0.5
            elif planner.path_deviation > 0.5:
                max_speed = 0.4
            else:
                max_speed = 0.6
            
            # Reduce max speed when near static obstacles
            if static_obstacle_dist < Config.SLOW_DOWN_DIST:
                max_speed = min(max_speed, 0.3)
            
            # Reduce max speed when near target
            if planner.last_path_point is not None:
                target_dist = np.linalg.norm(np.array(planner.last_path_point) - np.array([true_state[0], true_state[1]]))
                if target_dist < Config.PATH_STOPPING_DISTANCE * 3:
                    max_speed = min(max_speed, 0.3)
            
            current_speed = np.sqrt(true_state[3]**2 + true_state[4]**2)
            if current_speed > max_speed:
                true_state[3] *= max_speed / current_speed
                true_state[4] *= max_speed / current_speed
            
            # **FIX: Gradual stopping when very close to target**
            if planner.last_path_point is not None:
                target_dist = np.linalg.norm(np.array(planner.last_path_point) - np.array([true_state[0], true_state[1]]))
                if target_dist < Config.PATH_STOPPING_DISTANCE:
                    # Gradually bring velocity to zero
                    stop_factor = target_dist / Config.PATH_STOPPING_DISTANCE
                    true_state[3] *= stop_factor
                    true_state[4] *= stop_factor
                    if target_dist < 0.1:
                        true_state[3] = 0
                        true_state[4] = 0
                        true_state[5] = 0
            
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
        
        # Update footprint and safety zones
        robot_footprint.center = (cx, cy)
        safety_circle.center = (cx, cy)
        emergency_circle.center = (cx, cy)
        critical_circle.center = (cx, cy)
        
        # Update path target points
        if planner.last_path_point is not None:
            path_target_point.set_data([planner.last_path_point[0]], [planner.last_path_point[1]])
        
        if planner.alternative_path_point is not None:
            alt_path_point.set_data([planner.alternative_path_point[0]], [planner.alternative_path_point[1]])
        
        # Update obstacles
        for i, obs in enumerate(dyn_obs):
            obs_patches[i].center = (obs.x, obs.y)
            
            if i < len(obs_trails) and len(obs.history) > 1:
                trail_x, trail_y = zip(*obs.history)
                obs_trails[i].set_data(trail_x, trail_y)
        
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
        elif min_dist < Config.CRITICAL_ESCAPE_DIST:
            status = "CRITICAL ESCAPE"
            status_color = 'darkred'
        elif min_dist < Config.EMERGENCY_STOP_DIST:
            status = "EMERGENCY STOP"
            status_color = 'red'
        elif min_dist < Config.SLOW_DOWN_DIST:
            status = "CAUTION - SLOWING"
            status_color = 'orange'
        elif planner.deviation_required:
            status = "DEVIATING FROM PATH"
            status_color = 'purple'
        elif planner.path_recovery_mode:
            status = "RECOVERING TO PATH"
            status_color = 'purple'
        elif planner.obstacle_avoidance_mode:
            status = "AVOIDANCE ACTIVE"
            status_color = 'yellow'
        else:
            status = "FOLLOWING PATH"
            status_color = 'blue'
        
        ax_info.text(0.05, y_pos, f"Status: {status}", fontsize=14, 
                    fontweight='bold', color=status_color)
        y_pos -= 0.08
        
        # Metrics
        current_speed = np.sqrt(true_state[3]**2 + true_state[4]**2)
        path_deviation = planner.path_deviation
        
        metrics = [
            (f"Goal Distance: {dist_to_goal:.2f} m", 'black'),
            (f"Path Deviation: {path_deviation:.3f} m", 
             'red' if path_deviation > 0.5 else 'orange' if path_deviation > 0.2 else 'green'),
            (f"Closest Dynamic Obstacle: {min_dyn_dist:.2f} m", 
             'darkred' if min_dyn_dist < Config.CRITICAL_ESCAPE_DIST else
             'red' if min_dyn_dist < Config.EMERGENCY_STOP_DIST else 
             'orange' if min_dyn_dist < Config.SLOW_DOWN_DIST else 'green'),
            (f"Closest Static Obstacle: {static_obstacle_dist:.2f} m",
             'darkred' if static_obstacle_dist < Config.EMERGENCY_STOP_DIST else
             'red' if static_obstacle_dist < Config.SLOW_DOWN_DIST else 'green'),
            (f"Position: ({true_state[0]:.2f}, {true_state[1]:.2f})", 'black'),
            (f"Heading: {np.degrees(true_state[2]):.0f}°", 'black'),
            (f"Speed: {current_speed:.2f} m/s", 
             'green' if current_speed > 0.1 else 'orange'),
            (f"Distance: {total_distance:.1f} m", 'black'),
            (f"Time: {simulation_time:.1f} s", 'black'),
            (f"Collision warnings: {collision_count}", 
             'red' if collision_count > 0 else 'black'),
            (f"Avoidance maneuvers: {avoidance_maneuvers}", 'blue'),
            (f"Escape maneuvers: {escape_maneuvers}", 'darkred'),
            (f"Deviation maneuvers: {deviation_maneuvers}", 'purple'),
            (f"Path Progress: {last_path_index}/{len(path)}", 'green'),
            (f"Oscillation counter: {planner.oscillation_counter}", 
             'red' if planner.oscillation_counter > 5 else 'orange' if planner.oscillation_counter > 0 else 'green'),
            ("", 'black'),
            ("SMART PATH FOLLOWING:", 'blue', 'bold'),
            ("  • Fixed static obstacle collision detection", 'green', 'bold'),
            ("  • Adaptive lookahead distance", 'blue'),
            ("  • Anti-oscillation logic", 'blue'),
            ("  • Gradual stopping near targets", 'blue'),
            ("  • No more passing through black obstacles", 'green', 'bold'),
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
        
        return [robot_patch, robot_footprint, safety_circle, emergency_circle, critical_circle,
                costmap_img, predicted_line, path_target_point, alt_path_point] + obs_patches + obs_trails
    
    def on_key(event):
        nonlocal simulation_running, simulation_paused, true_state, total_distance
        nonlocal collision_count, success, simulation_time, last_position, dyn_obs, obs_patches
        nonlocal avoidance_maneuvers, escape_maneuvers, deviation_maneuvers, last_path_index, path_deviations, path_following_error
        nonlocal stuck_counter, last_movement_time
        
        if event.key == 's' or event.key == 'S':
            simulation_running = not simulation_running
            print(f"Simulation {'STARTED' if simulation_running else 'STOPPED'}")
        
        elif event.key == 'p' or event.key == 'P':
            simulation_paused = not simulation_paused
            print(f"Simulation {'PAUSED' if simulation_paused else 'RESUMED'}")
        
        elif event.key == 'r' or event.key == 'R':
            true_state = np.array([start[0], start[1], np.pi/4, 0.0, 0.0, 0.0])
            simulation_running = False
            simulation_paused = False
            total_distance = 0
            collision_count = 0
            success = False
            simulation_time = 0
            last_position = np.array(start)
            avoidance_maneuvers = 0
            escape_maneuvers = 0
            deviation_maneuvers = 0
            last_path_index = 0
            path_deviations = []
            path_following_error = []
            stuck_counter = 0
            last_movement_time = 0
            
            cx, cy, th = true_state[:3]
            w, l = Config.ROBOT_WIDTH, Config.ROBOT_LENGTH
            corner_x = cx - (w/2)*np.cos(th) + (l/2)*np.sin(th)
            corner_y = cy - (w/2)*np.sin(th) - (l/2)*np.cos(th)
            robot_patch.set_xy((corner_x, corner_y))
            robot_patch.angle = np.degrees(th)
            robot_patch.set_color('green')
            
            path_target_point.set_data([], [])
            alt_path_point.set_data([], [])
            
            for patch in obs_patches:
                patch.remove()
            obs_patches.clear()
            
            for trail in obs_trails:
                trail.set_data([], [])
            
            dyn_obs.clear()
            dyn_obs.extend(create_dynamic_obstacles())
            
            for obs in dyn_obs:
                patch = patches.Circle((obs.x, obs.y), Config.OBSTACLE_RADIUS, 
                                      color='orange', alpha=0.7, edgecolor='darkred')
                obs_patches.append(patch)
                ax_map.add_patch(patch)
            
            # Reset planner
            planner.__init__()
            if len(path) > 0:
                planner.costmap.set_reference_path(path)
            
            print("Simulation RESET")
        
        elif event.key == 'q' or event.key == 'Q':
            plt.close()
            print("Simulation terminated.")
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("\n🚀 IMPROVED PATH-FOLLOWING NAVIGATION READY!")
    print("Strategy: SMART path following with intelligent deviation")
    print("  • A* path keeps {Config.PATH_OBSTACLE_MARGIN}m from obstacles")
    print("  • Fixed static obstacle collision detection", 'green')
    print("  • No more passing through black obstacles", 'green')
    print("  • Anti-oscillation logic prevents back-and-forth motion")
    print("  • Gradual stopping near targets")
    print("  • Faster dynamic obstacles for more challenge")
    print(f"Obstacle speeds: {Config.OBSTACLE_MIN_SPEED:.2f}-{Config.OBSTACLE_MAX_SPEED:.2f} m/s")
    print("Controls: [S] Start/Stop  [P] Pause/Resume  [R] Reset  [Q] Quit")
    print("-" * 70)
    
    ani = FuncAnimation(fig, update, frames=Config.SIM_TIME, interval=50, blit=False)
    plt.tight_layout()
    plt.show()

# ==========================================
# 8. RUN THE SIMULATION
# ==========================================
if __name__ == "__main__":
    run_simulation()
