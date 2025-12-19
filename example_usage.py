#!/usr/bin/env python3
"""
Generate minimal example figures for MARL Games documentation.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import os
from marl_games import MatrixGame, DiGrid, QL

def setup():
    os.makedirs('figures', exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white'})

def fig1_summary():
    """1. Epsilon-Greedy Summary Plot"""
    print("Generating Summary Plot...")
    # User-specified settings
    payoff_matrix = np.array([
        [[2, 3], [4, 1]],
        [[3, 1], [2, 4]]
    ], dtype=np.float64)
    num_iterations = 400
    numExperiments = 100
    init_q_values = np.array([[0, 1], [2, 3]], np.float64)

    qLogList = QL.QLogList()
    for i in range(numExperiments):
        qLog = QL.epsilon_greedy_q_learning(
            payoff=payoff_matrix, 
            num_iterations=num_iterations,
            init_q_values=init_q_values.copy(), # Ensure copy to avoid mutation
            alpha=0.1,
            epsilon=0.1
        )
        qLogList.append(qLog)
    qLog = qLogList.median()
    
    QL.QPlot.summary(qLog)
    plt.tight_layout()
    plt.savefig('figures/1_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

def fig2_field():
    """2. Vector Field Plot"""
    print("Generating Vector Field Plot...")
    payoff = MatrixGame.STAG_HUNT
    X, Y, DX, DY = DiGrid.replicator_dynamics(payoff, grid_shape=(12, 12))
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(X, Y, DX, DY, color='steelblue')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Replicator Dynamics (Stag Hunt)")
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('figures/2_field.png', dpi=150, bbox_inches='tight')
    plt.close()

def fig3_trace():
    """3. Learning Trace Comparison"""
    print("Generating Trace Plot...")
    payoff = MatrixGame.STAG_HUNT
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    start1 = MatrixGame.generate_policy(0.1, 0.9)
    init_q1 = QL.generate_mean_q_values(payoff, 0.1, start1)
    qLog1 = QL.boltzmann_q_learning(
        payoff=payoff, 
        num_iterations=1000, 
        alpha=0.001, 
        temperature=0.1,
        init_q_values=init_q1
    )
    QL.QPlot.trace(ax, qLog1) # Default color
    
    start2 = MatrixGame.generate_policy(0.9, 0.1)
    init_q2 = QL.generate_mean_q_values(payoff, 0.1, start2)
    
    logList = QL.QLogList()
    for _ in range(32):
        log = QL.boltzmann_q_learning(
            payoff=payoff,
            num_iterations=1000,
            alpha=0.001, 
            temperature=0.1, 
            init_q_values=init_q2.copy()
        )
        logList.append(log)
    
    meanLog = logList.mean()
    QL.QPlot.trace(ax, meanLog)
    
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Trace Comparison: Speed vs Smoothness")
    ax.set_aspect('equal')
    ax.set_xlabel('P1 (Prob Action 0)')
    ax.set_ylabel('P2 (Prob Action 0)')
    
    plt.tight_layout()
    plt.savefig('figures/3_trace.png', dpi=150, bbox_inches='tight')
    plt.close()

def fig4_combined():
    """4. Vector Field + Traces (9 Grid Points)"""
    print("Generating Combined Plot...")
    payoff = MatrixGame.STAG_HUNT
    
    # Background: Dynamics (Boltzmann to match Q-learning)
    X, Y, DX, DY = DiGrid.boltzmann_replicator_dynamics(
        payoff, grid_shape=(12, 12), temperature=0.1
    )
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(X, Y, DX, DY, color='lightgray', alpha=0.5)
    
    # Foreground: 9 Traces using generate_policy_grid
    # Grid shape (3, 3) gives 9 points
    policy_grid = MatrixGame.generate_policy_grid((3, 3))
    
    for init_policy in policy_grid:
        init_q = QL.generate_mean_q_values(payoff, 0.1, init_policy)
        
        qLog = QL.boltzmann_q_learning(
            payoff=payoff,
            num_iterations=1000,
            alpha=0.001,
            temperature=0.1,
            init_q_values=init_q
        )
        QL.QPlot.trace(ax, qLog)
        
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Dynamics + 9 Learning Traces")
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('figures/4_combined.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    setup()
    fig1_summary()
    fig2_field()
    fig3_trace()
    fig4_combined()
    print("Done.")
