"""
EvoSphere Algorithm Access
Use this to unlock and access protected trading algorithms
"""
import os
from simple_encryption import SimpleEncryption

class AlgorithmAccess:
    """Access protected algorithms"""
    
    def __init__(self, password: str = None):
        self.password = password or os.environ.get('ALGORITHM_KEY', 'EvoSphere2025Vietnam!')
        self.encryptor = SimpleEncryption(self.password)
        self.unlocked = {}
        
    def unlock(self, algorithm_name: str):
        """Unlock a specific algorithm"""
        protected_file = f"{algorithm_name}.py.protected"
        
        if not os.path.exists(protected_file):
            print(f"❌ Protected file not found: {protected_file}")
            return False
            
        try:
            # Decrypt to temporary file
            temp_file = f"temp_{algorithm_name}.py"
            self.encryptor.decrypt_file(protected_file, temp_file)
            
            # Import the module
            import importlib.util
            spec = importlib.util.spec_from_file_location(algorithm_name, temp_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Store in memory
            self.unlocked[algorithm_name] = module
            
            # Clean up temp file
            os.remove(temp_file)
            
            print(f"🔓 Unlocked: {algorithm_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to unlock {algorithm_name}: {e}")
            return False
    
    def get_algorithm(self, name: str):
        """Get unlocked algorithm"""
        if name not in self.unlocked:
            self.unlock(name)
            
        return self.unlocked.get(name)
    
    def unlock_all(self):
        """Unlock all protected algorithms"""
        algorithms = [
            'dqn_agent',
            'evolutionary_feature_selector', 
            'main_ea_drl_forex_trader',
            'technical_indicator_calculator_pdta',
            'forex_simulation_env'
        ]
        
        success_count = 0
        for alg in algorithms:
            if self.unlock(alg):
                success_count += 1
                
        print(f"✅ Unlocked {success_count}/{len(algorithms)} algorithms")
        return success_count == len(algorithms)

# Global access instance
_access = AlgorithmAccess()

def unlock_algorithms(password: str = None):
    """Unlock all algorithms"""
    global _access
    if password:
        _access = AlgorithmAccess(password)
    return _access.unlock_all()

def get_dqn_agent():
    """Get DQN Agent class"""
    module = _access.get_algorithm('dqn_agent')
    return getattr(module, 'DQNAgent', None)

def get_evolutionary_selector():
    """Get Evolutionary Feature Selector"""
    module = _access.get_algorithm('evolutionary_feature_selector')
    return getattr(module, 'EvolutionaryFeatureSelector', None)

def get_trading_system():
    """Get Trading System"""
    module = _access.get_algorithm('main_ea_drl_forex_trader')
    return getattr(module, 'ForexTradingSystem', None)

def get_technical_calculator():
    """Get Technical Calculator"""
    module = _access.get_algorithm('technical_indicator_calculator_pdta')
    return getattr(module, 'TechnicalIndicatorCalculatorPandasTa', None)

def get_simulation_env():
    """Get Simulation Environment"""
    module = _access.get_algorithm('forex_simulation_env')
    return getattr(module, 'ForexSimulationEnv', None)

# Example usage
if __name__ == "__main__":
    print("🔐 EvoSphere Algorithm Access")
    
    # Try to unlock algorithms
    if unlock_algorithms():
        print("🚀 All algorithms unlocked successfully!")
        
        # Test access
        DQNAgent = get_dqn_agent()
        if DQNAgent:
            print(f"✅ DQN Agent accessible: {DQNAgent}")
        else:
            print("❌ DQN Agent not accessible")
    else:
        print("❌ Failed to unlock algorithms")
        print("💡 Try setting ALGORITHM_KEY environment variable")
