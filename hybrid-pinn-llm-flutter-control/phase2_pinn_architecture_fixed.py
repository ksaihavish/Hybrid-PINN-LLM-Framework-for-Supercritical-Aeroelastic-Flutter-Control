"""
Phase 2: Physics-Informed Neural Network (PINN) for Aeroelastic Cont VERSION - Corrected automatic differentiation and physics loss
Author: Research Project - PINN-LLM Hybrid Control
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
import json
import os

class AeroelasticPINN(nn.Module):
    """
    Physics-Informed Neural Network for 2-DOF Aeroelastic System
    
    Network learns control law u(t, x) while respecting physics:
    Mẍ + Cẋ + Kx + Q_aero = u + d
    """
    
    def __init__(self, hidden_layers=[64, 64, 64], input_dim=5):
        """
        Args:
            hidden_layers: List of neurons in each hidden layer
            input_dim: Input dimension (t, h, α, ḣ, α̇)
        """
        super(AeroelasticPINN, self).__init__()
        
        # Network architecture
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Tanh())  # Smooth activation for physics
            prev_dim = h_dim
        
        # Output layer: control forces [F_h, M_α]
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
        
        # Physics parameters (will be set externally)
        self.physics_params = {}
        
        # Loss history
        self.loss_history = {
            'total': [],
            'physics': [],
            'stability': [],
            'control_penalty': []
        }
        
    def forward(self, t, state):
        """
        Forward pass: predict control input
        
        Args:
            t: Time (N, 1)
            state: State vector (N, 4) - [h, α, ḣ, α̇]
        
        Returns:
            control: Control input (N, 2) - [F_h, M_α]
        """
        # Concatenate inputs
        x = torch.cat([t, state], dim=1)
        control = self.network(x)
        # ---- Actuator saturation (very important) ----
        max_force = 50.0   # adjust if needed
        control = torch.clamp(control, -max_force, max_force)
        return control
    
    def set_physics_params(self, wing):
        """Set physical parameters from wing object"""
        self.physics_params = {
            'M': torch.tensor(wing.get_mass_matrix(), dtype=torch.float32),
            'C': torch.tensor(wing.get_damping_matrix(), dtype=torch.float32),
            'K': torch.tensor(wing.get_stiffness_matrix(), dtype=torch.float32),
            'rho': wing.rho,
            'V': wing.V,
            'b': wing.b,
            'a': wing.a,
            'm': wing.m,
            'I_alpha': wing.I_alpha
        }
        
    def compute_aerodynamic_forces(self, state):
        """
        Compute aerodynamic forces Q_aero using quasi-steady approximation
        """
        h, alpha, h_dot, alpha_dot = state[:, 0:1], state[:, 1:2], state[:, 2:3], state[:, 3:4]
        
        rho = self.physics_params['rho']
        V = self.physics_params['V']
        b = self.physics_params['b']
        a = self.physics_params['a']
        
        # Quasi-steady lift and moment (simplified Theodorsen)
        # Lift force (plunge)
        L = -2 * torch.pi * rho * b * V * (h_dot + V * alpha)
        
        # Moment (pitch)
        M_aero = -2 * torch.pi * rho * b**2 * V * (
            V * alpha * (0.5 - a) + h_dot * (0.5 - a) - b * alpha_dot * (0.5 + a)
        )
        
        Q_aero = torch.cat([L, M_aero], dim=1)
        return Q_aero
    
    def compute_physics_residual(self, state, control):
        """
        Compute physics residual using the state-space dynamics
        
        For state x = [h, α, ḣ, α̇], we have:
        ẋ = Ax + Bu
        
        The physics residual checks if the controlled system satisfies the dynamics
        """
        M = self.physics_params['M']
        C = self.physics_params['C']
        K = self.physics_params['K']
        
        # Extract position and velocity
        pos = state[:, :2]  # [h, α]
        vel = state[:, 2:]  # [ḣ, α̇]
        
        # Compute aerodynamic forces
        Q_aero = self.compute_aerodynamic_forces(state)
        
        # Compute acceleration from dynamics: ẍ = M^(-1)(u - Cẋ - Kx - Q_aero)
        M_inv = torch.inverse(M)
        forces = control - Q_aero - (C @ vel.T).T - (K @ pos.T).T
        predicted_acc = (M_inv @ forces.T).T
        
        # For physics loss, we check energy dissipation and system stability
        # The control should work against the motion to stabilize
        return predicted_acc, Q_aero
    
    def loss_function(self, t, state, control,true_acc, 
                     weights={'physics': 1.0, 'stability': 5.0, 'control': 0.1}):
        """
        Comprehensive PINN loss function
        
        Components:
        1. Physics loss: Ensure dynamics are physically reasonable
        2. Stability loss: Drive system toward equilibrium
        3. Control penalty: Regularize control magnitude
        """
        
        # 1. Physics-informed loss
        # Compute what the acceleration should be given current control
        predicted_acc, _ = self.compute_physics_residual(state, control)
        loss_physics = torch.mean((predicted_acc - true_acc)**2)
        
        # 2. Stability loss: Control should oppose motion (negative damping)
        # -----------------------------
        # New Stability Objective
        # -----------------------------

        h = state[:, 0:1]
        alpha = state[:, 1:2]
        h_dot = state[:, 2:3]
        alpha_dot = state[:, 3:4]

        F_h = control[:, 0:1]
        M_alpha = control[:, 1:2]

        # 1️⃣ Directly minimize state magnitude
        # 2️⃣ Enforce true damping: force must oppose velocity
        M = self.physics_params['M']
        K = self.physics_params['K']

        pos = state[:, :2]
        vel = state[:, 2:]

        energy_pos = 0.5 * torch.sum(pos * (K @ pos.T).T, dim=1)
        energy_vel = 0.5 * torch.sum(vel * (M @ vel.T).T, dim=1)

        loss_stability = torch.mean(energy_pos + energy_vel)

        
        # 3. Control penalty (avoid excessive control)
        loss_control = 0.001*torch.mean(control**2)
        
        # Total loss
        loss_total = (weights['physics'] * loss_physics + 
                     weights['stability'] * loss_stability +
                     weights['control'] * loss_control)
        
        # Store losses
        losses = {
            'total': loss_total.item(),
            'physics': loss_physics.item(),
            'stability': loss_stability.item(),
            'control_penalty': loss_control.item()
        }
        
        return loss_total, losses
    
    def train_step(self, optimizer, t, state, true_acc, weights=None):
        """Single training step"""
        optimizer.zero_grad()
        
        if weights is None:
            weights = {'physics': 0.0, 'stability': 10.0, 'control': 1.0}
        
        # Forward pass
        control = self.forward(t, state)
        
        # Compute loss
        loss, losses = self.loss_function(t, state, control, true_acc, weights)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update history
        for key, value in losses.items():
            self.loss_history[key].append(value)
        
        return losses
    
    def save_checkpoint(self, path, optimizer=None, metadata=None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'loss_history': self.loss_history,
            'physics_params': {k: v.tolist() if isinstance(v, torch.Tensor) else v 
                             for k, v in self.physics_params.items()}
        }
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if metadata:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.loss_history = checkpoint['loss_history']
        
        # Convert back to tensors
        if 'physics_params' in checkpoint:
            for key, value in checkpoint['physics_params'].items():
                if isinstance(value, list):
                    self.physics_params[key] = torch.tensor(value, dtype=torch.float32)
                else:
                    self.physics_params[key] = value
        
        return checkpoint


def generate_training_data(wing, V_ratio=0.8, n_samples=1000, noise_level=0.005):
    """
    Generate training data from physics simulation
    
    Args:
        wing: TwoDOFWing object
        V_ratio: Velocity as fraction of flutter velocity
        n_samples: Number of training samples
        noise_level: Noise to add to data
    """
    import sys
    sys.path.append('.')
    from phase1_physics_environment import TwoDOFWing
    
    # Find flutter velocity
    result = wing.find_flutter_velocity(plot=False)
    V_flutter = result[0]  # Extract flutter velocity (first element)
    # Ignore other elements (e.g., frequencies, damping_ratios, V_range)
    V_test = V_flutter * V_ratio
    wing.set_velocity(V_test)
    
    # Simulate with multiple initial conditions
    t_max = 15.0
    all_t = []
    all_states = []
    
    # Various initial conditions
    initial_conditions = [
        np.array([0.02, 0.1, 0, 0]),
        np.array([-0.02, -0.1, 0, 0]),
        np.array([0.01, -0.05, 0.1, 0]),
        np.array([-0.01, 0.08, -0.1, 0.2]),
        np.array([0.015, 0.0, 0, 0.15])
    ]
    
    for i, x0 in enumerate(initial_conditions):
        t = np.linspace(0, t_max, n_samples // len(initial_conditions))
        states = wing.simulate(t, x0)
        
        all_t.append(t)
        all_states.append(states)
    
    # Combine and add noise
    t = np.concatenate(all_t)
    states = np.concatenate(all_states)
    states += np.random.randn(*states.shape) * noise_level
    
    return t, states, V_test


def train_pinn(wing, epochs=5000, lr=5e-4, plot_interval=500):
    """
    Train PINN on aeroelastic system
    """
    print("\n" + "="*60)
    print("Training PINN for Aeroelastic Control")
    print("="*60)
    
    # Generate training data
    print("\n1. Generating training data...")
    t, states, V_test = generate_training_data(wing, V_ratio=0.98, n_samples=1000)
    dt = t[1] - t[0]
      # [h_dot, alpha_dot]
      # numerical acceleration
    
    print(f"   [OK] Generated {len(t)} samples at V = {V_test:.2f} m/s")
    
    # Convert to tensors
    t_tensor = torch.tensor(t.reshape(-1, 1), dtype=torch.float32)
    state_tensor = torch.tensor(states, dtype=torch.float32)
    
    # Initialize PINN
    print("\n2. Initializing PINN...")
    pinn = AeroelasticPINN(hidden_layers=[64, 64, 64])
    pinn.set_physics_params(wing)
    print(f"   [OK] Network: {sum(p.numel() for p in pinn.parameters())} parameters")
    
    # Optimizer with learning rate schedule
    optimizer = Adam(pinn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5)
    
    # Training loop
    print(f"\n3. Training for {epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        # --- Closed-loop rollout training ---

        dt_tensor = torch.tensor(dt, dtype=torch.float32)
        # Start from first initial state
        state = state_tensor[0].unsqueeze(0).clone()

        rollout_loss = 0.0
        rollout_steps = 200

        for step in range(rollout_steps):

            t_current = t_tensor[step].unsqueeze(0)

            # Predict control
            control = pinn.forward(t_current, state)
 
            # Get acceleration from physics
            predicted_acc, _ = pinn.compute_physics_residual(state, control)

            pos = state[:, :2]
            vel = state[:, 2:]

            # Euler integration
            new_vel = vel + dt_tensor * predicted_acc
            new_pos = pos + dt_tensor * vel

            state = torch.cat([new_pos, new_vel], dim=1)

            # Energy-based stability loss
            M = pinn.physics_params['M']
            K = pinn.physics_params['K']

            energy_pos = 0.5 * torch.sum(new_pos * (K @ new_pos.T).T, dim=1)
            energy_vel = 0.5 * torch.sum(new_vel * (M @ new_vel.T).T, dim=1)
 
            rollout_loss += torch.mean(energy_pos + energy_vel)

        loss = rollout_loss / rollout_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
        optimizer.step()

        scheduler.step(loss.item())

        losses = {
            'total': loss.item(),
            'physics': 0.0,
            'stability': loss.item(),
            'control_penalty': 0.0
        }

        # Learning rate scheduling
        scheduler.step(losses['total'])
        
        if (epoch + 1) % plot_interval == 0:
            print(f"   Epoch {epoch+1}/{epochs}: Loss = {losses['total']:.6f} "
                  f"(Physics: {losses['physics']:.6f}, Stability: {losses['stability']:.6f}, "
                  f"Control: {losses['control_penalty']:.6f})")  # ← Correct key
            
            # Save best model
            if losses['total'] < best_loss:
                best_loss = losses['total']
                pinn.save_checkpoint(
                    'pinn_model_best.pt',
                    optimizer=optimizer,
                    metadata={'epochs': epoch+1, 'velocity': V_test, 'loss': best_loss}
                )
    
    # Save final model
    pinn.save_checkpoint(
        'pinn_model.pt',
        optimizer=optimizer,
        metadata={'epochs': epochs, 'velocity': V_test}
    )
    print(f"\n[OK] Model saved to: pinn_model.pt")
    
    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(pinn.loss_history['total'], 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(pinn.loss_history['physics'], 'r-', linewidth=1.5)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Physics Loss')
    axes[0, 1].set_title('Physics Constraints')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(pinn.loss_history['stability'], 'g-', linewidth=1.5)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Stability Loss')
    axes[1, 0].set_title('Stability (Energy Dissipation)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(pinn.loss_history['control_penalty'], 'm-', linewidth=1.5)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Control Penalty')
    axes[1, 1].set_title('Control Magnitude')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pinn_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Training history saved to: pinn_training_history.png")
    
    return pinn


def main():
    """Test PINN training"""
    import sys
    sys.path.append('.')
    from phase1_physics_environment import TwoDOFWing
    
    # Load wing
    wing = TwoDOFWing()
    
    # Train PINN
    pinn = train_pinn(wing, epochs=3000, lr=1e-4)
    
    print("\n" + "="*60)
    print("Phase 2 Complete! Ready for Phase 3 (LLM Integration)")
    print("="*60)


if __name__ == "__main__":
    main()
