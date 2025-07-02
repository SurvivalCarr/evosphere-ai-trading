"""
Deep Q-Network Agent implementation using scikit-learn
Compatible alternative to TensorFlow-based DQN
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from collections import deque
import pickle
import logging
from typing import Tuple, Optional, List, Any
import random

# Suppress sklearn convergence warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.neural_network")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DQNAgentSklearn:
    """Deep Q-Network Agent using scikit-learn's MLPRegressor."""
    
    def __init__(self, state_shape: Tuple[int, ...], action_size: int, 
                 learning_rate: float = 0.001, batch_size: int = 32,
                 memory_size: int = 10000, epsilon: float = 1.0, 
                 epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                 gamma: float = 0.95, target_update_freq: int = 100,
                 trade_size_options: List[float] = [1.0],
                 volatility_threshold: float = 0.005, 
                 pause_probability_high_vol: float = 0.3):
        """
        Initialize the DQN Agent.
        
        Args:
            state_shape: Shape of the input state
            action_size: Number of possible actions
            learning_rate: Learning rate for the neural network
            batch_size: Size of training batches
            memory_size: Size of experience replay memory
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            gamma: Discount factor for future rewards
            target_update_freq: Frequency to update target network
            trade_size_options: Available trade sizes
            volatility_threshold: Threshold for high volatility detection
            pause_probability_high_vol: Probability to pause during high volatility
        """
        self.state_size = np.prod(state_shape)
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.trade_size_options = trade_size_options
        self.volatility_threshold = volatility_threshold
        self.pause_probability_high_vol = pause_probability_high_vol
        
        # Training statistics
        self.training_step = 0
        self.total_episodes = 0
        self.losses = []
        self.episode_rewards = []
        
        # Initialize networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.scaler = StandardScaler()
        
        # Flag to track if scaler is fitted
        self.scaler_fitted = False
        
        logger.info(f"DQN Agent initialized with state_size={self.state_size}, action_size={self.action_size}")
    
    def _build_network(self) -> MLPRegressor:
        """Build the neural network for Q-learning."""
        return MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate_init=self.learning_rate,
            max_iter=10,  # Increased from 1 to prevent convergence warnings
            warm_start=True,
            random_state=42,
            alpha=0.001  # L2 regularization
        )
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, current_volatility: Optional[float] = None) -> int:
        """Choose action using epsilon-greedy policy with volatility consideration."""
        try:
            # Handle high volatility by potentially pausing
            if (current_volatility is not None and 
                current_volatility > self.volatility_threshold and
                np.random.random() < self.pause_probability_high_vol):
                return 0  # Hold action
            
            # Epsilon-greedy action selection
            if np.random.random() <= self.epsilon:
                return random.randrange(self.action_size)
            
            # Predict Q-values
            state_flat = state.flatten().reshape(1, -1)
            
            # Fit scaler if not already fitted
            if not self.scaler_fitted:
                self.scaler.fit(state_flat)
                self.scaler_fitted = True
            
            state_scaled = self.scaler.transform(state_flat)
            
            try:
                q_values = self.q_network.predict(state_scaled)
                return np.argmax(q_values[0])
            except Exception:
                # If prediction fails (network not trained yet), return random action
                return random.randrange(self.action_size)
                
        except Exception as e:
            logger.warning(f"Error in act method: {e}, returning random action")
            return random.randrange(self.action_size)
    
    def _get_trade_size_from_action(self, action: int) -> float:
        """Convert action to trade size percentage."""
        if hasattr(self, 'trade_size_options') and self.trade_size_options:
            # Map actions to trade sizes
            size_index = action % len(self.trade_size_options)
            return self.trade_size_options[size_index]
        return 1.0
    
    def _convert_to_simple_action(self, action: int) -> int:
        """Convert extended action space to simple action space."""
        # Assuming action space: 0=Hold, 1=Buy, 2=Sell (with size variations)
        if action == 0:
            return 0  # Hold
        elif action % 2 == 1:
            return 1  # Buy
        else:
            return 2  # Sell
    
    def replay(self) -> float:
        """Train the model on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        try:
            # Sample batch from memory
            batch = random.sample(self.memory, self.batch_size)
            states = np.array([e[0].flatten() for e in batch])
            actions = np.array([e[1] for e in batch])
            rewards = np.array([e[2] for e in batch])
            next_states = np.array([e[3].flatten() for e in batch])
            dones = np.array([e[4] for e in batch])
            
            # Fit scaler if not already fitted
            if not self.scaler_fitted:
                self.scaler.fit(states)
                self.scaler_fitted = True
            
            # Scale states
            states_scaled = self.scaler.transform(states)
            next_states_scaled = self.scaler.transform(next_states)
            
            # Predict current Q-values
            try:
                current_q_values = self.q_network.predict(states_scaled)
            except Exception:
                # Initialize with zeros if network not trained yet
                current_q_values = np.zeros((len(states), self.action_size))
            
            # Predict next Q-values using target network
            try:
                next_q_values = self.target_network.predict(next_states_scaled)
            except Exception:
                # Use main network if target network not ready
                try:
                    next_q_values = self.q_network.predict(next_states_scaled)
                except Exception:
                    next_q_values = np.zeros((len(next_states), self.action_size))
            
            # Calculate target Q-values
            target_q_values = current_q_values.copy()
            
            for i in range(len(batch)):
                if dones[i]:
                    target_q_values[i][actions[i]] = rewards[i]
                else:
                    target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
            
            # Train the network
            try:
                self.q_network.fit(states_scaled, target_q_values)
                
                # Calculate loss (approximate)
                predicted = self.q_network.predict(states_scaled)
                loss = np.mean((target_q_values - predicted) ** 2)
                self.losses.append(loss)
                
            except Exception as e:
                logger.warning(f"Training failed: {e}")
                loss = 0.0
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            self.training_step += 1
            
            # Update target network
            if self.training_step % self.target_update_freq == 0:
                self.update_target_model()
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in replay: {e}")
            return 0.0
    
    def update_target_model(self):
        """Update the target network with weights from the main network."""
        try:
            # For scikit-learn, we copy the entire model
            self.target_network = self._build_network()
            
            # If the main network has been trained, copy its parameters
            if hasattr(self.q_network, 'coefs_'):
                # Create a new model with same architecture
                self.target_network.fit(
                    np.random.randn(self.batch_size, self.state_size),
                    np.random.randn(self.batch_size, self.action_size)
                )
                # Copy the trained parameters
                self.target_network.coefs_ = [coef.copy() for coef in self.q_network.coefs_]
                self.target_network.intercepts_ = [intercept.copy() for intercept in self.q_network.intercepts_]
                
            logger.debug("Target network updated")
            
        except Exception as e:
            logger.warning(f"Error updating target network: {e}")
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        try:
            model_data = {
                'q_network': self.q_network,
                'target_network': self.target_network,
                'scaler': self.scaler,
                'scaler_fitted': self.scaler_fitted,
                'epsilon': self.epsilon,
                'training_step': self.training_step,
                'losses': self.losses,
                'episode_rewards': self.episode_rewards,
                'state_size': self.state_size,
                'action_size': self.action_size
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_network = model_data['q_network']
            self.target_network = model_data['target_network']
            self.scaler = model_data['scaler']
            self.scaler_fitted = model_data['scaler_fitted']
            self.epsilon = model_data['epsilon']
            self.training_step = model_data['training_step']
            self.losses = model_data['losses']
            self.episode_rewards = model_data['episode_rewards']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def get_training_stats(self) -> dict:
        """Get training statistics."""
        return {
            'training_step': self.training_step,
            'total_episodes': self.total_episodes,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            'recent_losses': self.losses[-20:] if self.losses else [],
            'recent_rewards': self.episode_rewards[-20:] if self.episode_rewards else []
        }
    
    def reset_epsilon(self, epsilon: Optional[float] = None):
        """Reset epsilon for training."""
        if epsilon is not None:
            self.epsilon = epsilon
        else:
            self.epsilon = 1.0
        logger.info(f"Epsilon reset to {self.epsilon}")
    
    def add_episode_reward(self, reward: float):
        """Add episode reward to statistics."""
        self.episode_rewards.append(reward)
        self.total_episodes += 1