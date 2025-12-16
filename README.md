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

