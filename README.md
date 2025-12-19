# Robust NMPC-Based Obstacle Avoidance for Mobile Robots

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![CasADi](https://img.shields.io/badge/Solver-CasADi-orange)

Abstract
This project implements a robust integrated navigation stack for a mobile robot in a dynamic environment. It combines "A* Search" for global path planning, a "Gradient-based Local Costmap" for immediate environmental perception, integrated with " Adaptive Nonlinear Model Predictive Control (NMPC)" for optimal trajectory generation. The system is designed to navigate a 15x15m grid with static maze-like structures and moving dynamic obstacles.

Key Features
Global Planning: A* algorithm with Euclidean heuristics for optimal global pathfinding.
Local Perception:Dynamic Costmap generation using Gaussian inflation and gradient calculation.
NMPC Controller:
    Utilizes "CasADi" for symbolic optimization.
    Optimizes control inputs ($v, \omega$) over a receding horizon ($N=8$).
    Includes hard constraints for safety distance and kinematic limits.
Dynamic Obstacles:Handles moving agents with predictive repulsion costs.
Fallback Mechanism:Automatic switching to reactive gradient-following if the NMPC solver fails to converge.

Overview
This is a complete robotics simulation system that shows a robot navigating through a challenging environment with both static obstacles (black blocks) and dynamic obstacles (moving orange circles) to reach a goal. The robot intelligently follows a pre-planned path while avoiding collisions in real-time.

What This Program Does
Main Scenario
Imagine a delivery robot in a warehouse:

Starting point: Green circle at (1,1)

Goal: Red star at (14,14)

Path: Green line that the robot tries to follow

Obstacles:

Black blocks: Fixed obstacles like shelves, walls

Orange circles: Moving obstacles like other robots or people

Robot: Green rectangle that can move in any direction (omnidirectional)

Key Features
Path Planning: Finds the best route from start to goal while avoiding obstacles

Obstacle Avoidance: Detects and avoids both static and moving obstacles

Path Following: Tries to stay on the green path but can deviate when needed

Real-time Control: Adjusts movement 10 times per second (100ms intervals)

Visualization: Shows everything happening in real-time

Technology Stack Used
1. Core Python Libraries
python
- **NumPy**: Mathematics and matrix operations (the "brain" for calculations)
- **Matplotlib**: Visualization (drawing everything you see)
- **CasADi**: Advanced mathematical optimization (makes smart decisions)
- **SciPy**: Scientific computing (smoothing and filtering)
- **Heapq**: Pathfinding algorithm (finding the best route)
2. Key Algorithms Implemented
A. Path Planning (A Algorithm)*
Purpose: Find the shortest safe path from start to goal
How it works:

Divides the map into a grid (like chess board)

Checks each cell: is it occupied or free?

Finds the shortest route while staying away from obstacles

Creates waypoints (green dots) for the robot to follow

B. Model Predictive Control (MPC)
Purpose: Make smart movement decisions in real-time
How it works:

Looks ahead 8 steps (0.8 seconds into the future)

Considers:

Where the robot wants to go (path/waypoints)

Where obstacles are moving

Physical limits (max speed, turning ability)

Calculates optimal forces to apply

Adjusts every 0.1 seconds

C. Local Costmap System
Purpose: Create a "heat map" of safe/unsafe areas around the robot
How it works:

Creates a 4m x 4m map centered on the robot

Colors areas:

Green: Safe to go (attractive)

Red: Dangerous (repulsive)

Yellow: Caution areas

The robot follows the green gradients

D. Collision Detection
Purpose: Prevent the robot from hitting anything
How it works:

Checks 8 points around the robot's perimeter

For static obstacles: Checks if any point is inside black areas

For dynamic obstacles: Checks distance to orange circles

Has 3 safety zones:

Dark red (0.2m): Emergency! Force escape

Red (0.4m): Emergency stop

Yellow (0.8m): Slow down and be careful

How Everything Works Together
Step-by-Step Process
1. Setup Phase
python
# Creates the environment
- Draws walls with openings
- Places black obstacles in strategic locations
- Creates 12 moving obstacles with different patterns
- Finds the initial path (green line)
2. Simulation Loop (100 times per second)
python
# Each 0.1-second cycle:
1. UPDATE OBSTACLES:
   - Move orange circles
   - Make them bounce off walls realistically

2. CHECK SAFETY:
   - Measure distances to nearest obstacles
   - Detect potential collisions

3. PLAN NEXT MOVES:
   - Look at the green path ahead
   - Check if obstacles are blocking the way
   - Decide: follow path or deviate around obstacle
   - Calculate optimal acceleration forces

4. APPLY CONTROLS:
   - Apply calculated forces to robot
   - Update position and orientation
   - Ensure speed limits aren't exceeded

5. UPDATE DISPLAY:
   - Move robot visualization
   - Update obstacle positions
   - Show predicted trajectory (yellow line)
   - Update costmap colors
3. Intelligent Behaviors
A. Path Following Strategy

Looks 1.5 meters ahead on the path

Takes the average of 3 upcoming waypoints

This prevents zig-zagging

Can recover if pushed off path

B. Obstacle Avoidance Logic

text
When approaching an obstacle:
1. Check if it's on collision course
2. Calculate which side is safer (left or right)
3. Create a temporary target point away from obstacle
4. Blend between path target and avoidance target
5. Return to path when safe
C. Emergency Responses

text
DISTANCE TO OBSTACLE | RESPONSE
< 0.2m              | PANIC! Full force escape
< 0.4m              | Emergency stop, very slow
< 0.8m              | Slow down, prepare to avoid
> 0.8m              | Normal operation
The Robot's "Brain" (Control System)
Mathematical Model
The robot is treated as a physical object with mass and inertia:

Mass: 1.5 kg (how heavy it is)

Inertia: 0.15 kg·m² (how hard it is to turn)

Max force: 4.0 N (how hard it can push)

Max torque: 2.5 N·m (how hard it can turn)

Optimization Problem (MPC)
At each step, the computer solves:

text
Minimize:
1. Distance to target point (path following)
2. Distance to goal (ultimate objective)
3. Control effort (energy efficiency)
4. Proximity to obstacles (safety)

Subject to:
1. Physics laws (F=ma)
2. Speed limits (0.8 m/s max)
3. Force limits (4.0 N max)
4. Turning limits
Why This is Sophisticated
The robot doesn't just "go toward goal." It:

Predicts where obstacles will be

Balances multiple objectives (safety vs. speed)

Considers physical limitations

Adapts to changing situations


Costmap (Top Right)
A heat map showing:

Red: Dangerous (near obstacles)

Green: Good direction to go

The robot follows green gradients downhill

Info Panel (Bottom Right)
Shows real-time statistics:

Distance to goal

Path deviation

Closest obstacle

Speed

Time elapsed

Collision warnings

Current status

Key Innovations in This Code
1. Adaptive Path Following
Instead of blindly following waypoints, the robot:

Looks ahead multiple points

Takes their median (reduces zig-zag)

Can temporarily leave path to avoid obstacles

Intelligently returns to path

2. Multi-Layer Safety System
text
Layer 1: Costmap (preventative)
  - Makes obstacles "repulsive" in calculations
  
Layer 2: Reactive avoidance
  - Checks for obstacles in path
  - Creates deviation paths
  
Layer 3: Emergency escape
  - When too close, force away from danger
  
Layer 4: Collision response
  - If collision detected, apply repulsive force
3. Smooth Control Transitions
The robot doesn't jerk or stop suddenly. It:

Gradually slows when approaching targets

Smoothly transitions between behaviors

Maintains stability through damping
