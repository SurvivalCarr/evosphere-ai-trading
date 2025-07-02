import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DQNAgent:
    """Deep Q-Network Agent for Forex Trading."""
    
    def __init__(self, state_shape, action_size, learning_rate=0.001, batch_size=32,
                 memory_size=10000, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 gamma=0.95, target_update_freq=100, trade_size_options=[1.0],
                 volatility_threshold=0.005, pause_probability_high_vol=0.3):
        
        self.state_shape = state_shape
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
        
        # Build neural networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_model()
        
        # Training statistics
        self.training_step = 0
        self.total_rewards = []
        self.losses = []
        
        logger.info(f"DQN Agent initialized with state_shape={state_shape}, action_size={action_size}")
    
    def _build_network(self):
        """Build the neural network for Q-learning."""
        try:
            model = Sequential([
                # Input layer - reshape for LSTM if needed
                Reshape((self.state_shape[0], self.state_shape[1]), input_shape=self.state_shape),
                
                # LSTM layer to capture temporal patterns
                LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
                LSTM(64, dropout=0.2, recurrent_dropout=0.2),
                
                # Dense layers with regularization
                Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
                BatchNormalization(),
                Dropout(0.2),
                
                Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
                Dropout(0.2),
                
                # Output layer
                Dense(self.action_size, activation='linear')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0),
                loss='mse',
                metrics=['mae']
            )
            
            logger.info(f"Neural network built with {model.count_params()} parameters")
            return model
            
        except Exception as e:
            logger.error(f"Error building network: {e}")
            raise
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        try:
            self.memory.append((state, action, reward, next_state, done))
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
    
    def act(self, state, current_volatility=None):
        """Choose action using epsilon-greedy policy with volatility consideration."""
        try:
            # Handle volatility-based pausing
            if current_volatility is not None and current_volatility > self.volatility_threshold:
                if random.random() < self.pause_probability_high_vol:
                    return 0, 1.0  # HOLD action with full position size
            
            # Epsilon-greedy action selection
            if random.random() <= self.epsilon:
                # Random action
                if self.action_size == 3:
                    # Simple action space: [HOLD, BUY, SELL]
                    action = random.randint(0, 2)
                    trade_size = random.choice(self.trade_size_options)
                else:
                    # Extended action space with different trade sizes
                    action = random.randint(0, self.action_size - 1)
                    trade_size = self._get_trade_size_from_action(action)
            else:
                # Greedy action selection
                if len(state.shape) == 2:
                    state = np.expand_dims(state, axis=0)
                
                q_values = self.q_network.predict(state, verbose=0)
                action = np.argmax(q_values[0])
                trade_size = self._get_trade_size_from_action(action)
            
            # Convert extended action space to simple action
            simple_action = self._convert_to_simple_action(action)
            
            return simple_action, trade_size
            
        except Exception as e:
            logger.error(f"Error in act method: {e}")
            return 0, 1.0  # Default to HOLD
    
    def _get_trade_size_from_action(self, action):
        """Convert action to trade size percentage."""
        try:
            if self.action_size == 3:
                # Simple action space
                return 1.0
            else:
                # Extended action space: HOLD + BUY_options + SELL_options
                if action == 0:  # HOLD
                    return 1.0
                else:
                    # BUY or SELL with different sizes
                    size_index = (action - 1) % len(self.trade_size_options)
                    return self.trade_size_options[size_index]
        except Exception as e:
            logger.error(f"Error getting trade size: {e}")
            return 1.0
    
    def _convert_to_simple_action(self, action):
        """Convert extended action space to simple action space."""
        try:
            if self.action_size == 3:
                return action
            else:
                if action == 0:
                    return 0  # HOLD
                elif action <= len(self.trade_size_options):
                    return 1  # BUY
                else:
                    return 2  # SELL
        except Exception as e:
            logger.error(f"Error converting action: {e}")
            return 0
    
    def replay(self):
        """Train the model on a batch of experiences."""
        try:
            if len(self.memory) < self.batch_size:
                return
            
            # Sample batch from memory
            batch = random.sample(self.memory, self.batch_size)
            states = np.array([e[0] for e in batch])
            actions = np.array([e[1] for e in batch])
            rewards = np.array([e[2] for e in batch])
            next_states = np.array([e[3] for e in batch])
            dones = np.array([e[4] for e in batch])
            
            # Convert actions to extended action space if needed
            if self.action_size > 3:
                actions = self._convert_simple_to_extended_actions(actions)
            
            # Calculate target Q-values using Double DQN
            current_q_values = self.q_network.predict(states, verbose=0)
            next_q_values = self.q_network.predict(next_states, verbose=0)
            target_next_q_values = self.target_network.predict(next_states, verbose=0)
            
            targets = current_q_values.copy()
            
            for i in range(self.batch_size):
                if dones[i]:
                    targets[i][actions[i]] = rewards[i]
                else:
                    # Double DQN: use main network to select action, target network to evaluate
                    best_action = np.argmax(next_q_values[i])
                    targets[i][actions[i]] = rewards[i] + self.gamma * target_next_q_values[i][best_action]
            
            # Train the network
            history = self.q_network.fit(states, targets, batch_size=self.batch_size, 
                                       epochs=1, verbose=0)
            
            # Track loss
            if history.history['loss']:
                self.losses.append(history.history['loss'][0])
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            self.training_step += 1
            
            # Update target network periodically
            if self.training_step % self.target_update_freq == 0:
                self.update_target_model()
                logger.debug(f"Target network updated at step {self.training_step}")
                
        except Exception as e:
            logger.error(f"Error in replay: {e}")
    
    def _convert_simple_to_extended_actions(self, simple_actions):
        """Convert simple actions to extended action space."""
        try:
            extended_actions = []
            for action in simple_actions:
                if action == 0:  # HOLD
                    extended_actions.append(0)
                elif action == 1:  # BUY - map to first BUY option
                    extended_actions.append(1)
                elif action == 2:  # SELL - map to first SELL option
                    extended_actions.append(1 + len(self.trade_size_options))
                else:
                    extended_actions.append(0)  # Default to HOLD
            return np.array(extended_actions)
        except Exception as e:
            logger.error(f"Error converting actions: {e}")
            return np.zeros_like(simple_actions)
    
    def update_target_model(self):
        """Update the target network with weights from the main network."""
        try:
            self.target_network.set_weights(self.q_network.get_weights())
            logger.debug("Target network weights updated")
        except Exception as e:
            logger.error(f"Error updating target model: {e}")
    
    def save_model(self, filepath):
        """Save the trained model."""
        try:
            self.q_network.save(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath):
        """Load a pre-trained model."""
        try:
            self.q_network = tf.keras.models.load_model(filepath)
            self.target_network = clone_model(self.q_network)
            self.update_target_model()
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def get_training_stats(self):
        """Get training statistics."""
        return {
            'total_steps': self.training_step,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'total_rewards': self.total_rewards
        }
    
    def reset_epsilon(self, epsilon=None):
        """Reset epsilon for training."""
        if epsilon is not None:
            self.epsilon = epsilon
        else:
            self.epsilon = 1.0
        logger.info(f"Epsilon reset to {self.epsilon}")
