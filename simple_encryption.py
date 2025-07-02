"""
Simple Algorithm Encryption for EvoSphere
Protects core trading algorithms from unauthorized access
"""
import os
import base64
import pickle
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import hashlib

class SimpleEncryption:
    """Simple but effective encryption for trading algorithms"""
    
    def __init__(self, password: str = "EvoSphere2025Vietnam!"):
        """Initialize with master password"""
        self.password = password
        self.salt = b'evosphere_vietnam_salt_2025'
        self.key = self._derive_key()
        self.cipher = Fernet(self.key)
        
    def _derive_key(self):
        """Create encryption key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password.encode()))
        return key
    
    def encrypt_file(self, input_file: str, output_file: str = None):
        """Encrypt a Python file"""
        if not output_file:
            output_file = input_file + '.protected'
            
        # Read original file
        with open(input_file, 'rb') as f:
            file_data = f.read()
            
        # Create package
        package = {
            'filename': os.path.basename(input_file),
            'data': file_data,
            'hash': hashlib.sha256(file_data).hexdigest(),
            'timestamp': time.time()
        }
        
        # Encrypt
        encrypted_data = self.cipher.encrypt(pickle.dumps(package))
        
        # Save
        with open(output_file, 'wb') as f:
            f.write(encrypted_data)
            
        print(f"✅ Encrypted: {input_file} -> {output_file}")
        return output_file
    
    def decrypt_file(self, encrypted_file: str, output_file: str = None):
        """Decrypt a file"""
        # Read encrypted data
        with open(encrypted_file, 'rb') as f:
            encrypted_data = f.read()
            
        # Decrypt
        package = pickle.loads(self.cipher.decrypt(encrypted_data))
        
        # Verify integrity
        data = package['data']
        if hashlib.sha256(data).hexdigest() != package['hash']:
            raise ValueError("File integrity check failed!")
            
        # Save decrypted file
        if not output_file:
            output_file = f"decrypted_{package['filename']}"
            
        with open(output_file, 'wb') as f:
            f.write(data)
            
        print(f"✅ Decrypted: {encrypted_file} -> {output_file}")
        return output_file

def protect_algorithms():
    """Encrypt all core EvoSphere algorithms"""
    encryptor = SimpleEncryption()
    
    # Files to protect
    files_to_encrypt = [
        'dqn_agent.py',
        'evolutionary_feature_selector.py',
        'main_ea_drl_forex_trader.py',
        'technical_indicator_calculator_pdta.py',
        'forex_simulation_env.py'
    ]
    
    protected_files = []
    
    print("🔐 Protecting EvoSphere Algorithms...")
    print("=" * 40)
    
    for file_path in files_to_encrypt:
        if os.path.exists(file_path):
            try:
                protected_file = encryptor.encrypt_file(file_path)
                protected_files.append(protected_file)
            except Exception as e:
                print(f"❌ Failed to encrypt {file_path}: {e}")
        else:
            print(f"⚠️ File not found: {file_path}")
    
    # Create access script
    create_access_script()
    
    print(f"\n🔒 Protection Complete!")
    print(f"✅ Protected {len(protected_files)} files")
    print("📋 Use access_algorithms.py to unlock")
    
    return protected_files

def create_access_script():
    """Create script to access protected algorithms"""
    access_code = '''"""
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
'''
    
    with open('access_algorithms.py', 'w') as f:
        f.write(access_code)
    
    print("✅ Created access_algorithms.py")

if __name__ == "__main__":
    # Run protection
    protect_algorithms()