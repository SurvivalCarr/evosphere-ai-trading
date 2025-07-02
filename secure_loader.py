
"""
EvoSphere Secure Algorithm Loader
Use this to safely access encrypted trading algorithms
"""
import os
import sys
from typing import Dict, Any

class SecureAlgorithmLoader:
    """Secure loader for encrypted EvoSphere algorithms"""
    
    def __init__(self):
        self.algorithms = {}
        self.unlocked = False
        
    def unlock(self, master_key: str = None) -> bool:
        """Unlock all encrypted algorithms"""
        try:
            # Set master key
            if master_key:
                os.environ['ALGORITHM_MASTER_KEY'] = master_key
                
            print("🔐 Unlocking EvoSphere Algorithms...")
            
            # Import protected modules
            from protected_dqn_agent import unlock_algorithm as unlock_dqn
            from protected_evolutionary_feature_selector import unlock_algorithm as unlock_evo
            from protected_main_ea_drl_forex_trader import unlock_algorithm as unlock_main
            from protected_technical_indicator_calculator_pdta import unlock_algorithm as unlock_tech
            from protected_forex_simulation_env import unlock_algorithm as unlock_env
            
            # Unlock each algorithm
            algorithms_status = {
                'DQN Agent': unlock_dqn(master_key),
                'Evolutionary Selector': unlock_evo(master_key), 
                'Main Trading System': unlock_main(master_key),
                'Technical Indicators': unlock_tech(master_key),
                'Trading Environment': unlock_env(master_key)
            }
            
            # Check results
            success_count = sum(algorithms_status.values())
            total_count = len(algorithms_status)
            
            if success_count == total_count:
                self.unlocked = True
                print(f"✅ Successfully unlocked {success_count}/{total_count} algorithms")
                return True
            else:
                print(f"⚠️ Partially unlocked {success_count}/{total_count} algorithms")
                for name, status in algorithms_status.items():
                    status_icon = "✅" if status else "❌"
                    print(f"  {status_icon} {name}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to unlock algorithms: {e}")
            return False
    
    def get_dqn_agent(self):
        """Get DQN Agent class"""
        if not self.unlocked:
            raise RuntimeError("Algorithms not unlocked!")
        from protected_dqn_agent import get_algorithm
        return get_algorithm('DQNAgent')
    
    def get_evolutionary_selector(self):
        """Get Evolutionary Feature Selector"""
        if not self.unlocked:
            raise RuntimeError("Algorithms not unlocked!")
        from protected_evolutionary_feature_selector import get_algorithm
        return get_algorithm('EvolutionaryFeatureSelector')
    
    def get_trading_system(self):
        """Get Main Trading System"""
        if not self.unlocked:
            raise RuntimeError("Algorithms not unlocked!")
        from protected_main_ea_drl_forex_trader import get_algorithm
        return get_algorithm('ForexTradingSystem')
    
    def get_technical_calculator(self):
        """Get Technical Indicator Calculator"""
        if not self.unlocked:
            raise RuntimeError("Algorithms not unlocked!")
        from protected_technical_indicator_calculator_pdta import get_algorithm
        return get_algorithm('TechnicalIndicatorCalculatorPandasTa')
    
    def get_simulation_env(self):
        """Get Forex Simulation Environment"""
        if not self.unlocked:
            raise RuntimeError("Algorithms not unlocked!")
        from protected_forex_simulation_env import get_algorithm
        return get_algorithm('ForexSimulationEnv')

# Global secure loader instance
_loader = SecureAlgorithmLoader()

def unlock_algorithms(master_key: str = None) -> bool:
    """Unlock all encrypted algorithms"""
    return _loader.unlock(master_key)

def get_dqn_agent():
    """Get DQN Agent class"""
    return _loader.get_dqn_agent()

def get_evolutionary_selector():
    """Get Evolutionary Feature Selector class"""
    return _loader.get_evolutionary_selector()

def get_trading_system():
    """Get Main Trading System class"""
    return _loader.get_trading_system()

def get_technical_calculator():
    """Get Technical Indicator Calculator class"""
    return _loader.get_technical_calculator()

def get_simulation_env():
    """Get Forex Simulation Environment class"""
    return _loader.get_simulation_env()

# Example usage
if __name__ == "__main__":
    # Unlock algorithms
    success = unlock_algorithms()
    
    if success:
        print("🚀 Algorithms ready for use!")
        
        # Example: Get DQN Agent
        try:
            DQNAgent = get_dqn_agent()
            print(f"✅ DQN Agent available: {DQNAgent}")
        except Exception as e:
            print(f"❌ Error accessing DQN Agent: {e}")
    else:
        print("❌ Failed to unlock algorithms")
