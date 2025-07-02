"""
EvoSphere Algorithm Encryption System
Protects core trading algorithms and strategies from unauthorized access
"""
import os
import base64
import pickle
import json
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import importlib.util
import sys
from typing import Any, Dict, List, Optional
import hashlib

class AlgorithmEncryption:
    """Encrypts and protects core trading algorithms"""
    
    def __init__(self, master_password: str = None):
        """Initialize encryption with master password"""
        self.master_password = master_password or os.environ.get('ALGORITHM_MASTER_KEY', 'EvoSphere2025!')
        self.salt = b'evosphere_vietnam_trading_salt_2025'
        self.key = self._derive_key()
        self.cipher = Fernet(self.key)
        
    def _derive_key(self) -> bytes:
        """Derive encryption key from master password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_password.encode()))
        return key
    
    def encrypt_file(self, file_path: str, output_path: str = None) -> str:
        """Encrypt a Python file containing algorithms"""
        if not output_path:
            output_path = file_path + '.encrypted'
            
        with open(file_path, 'rb') as f:
            file_data = f.read()
            
        # Add file integrity hash
        file_hash = hashlib.sha256(file_data).hexdigest()
        
        # Create encrypted package
        package = {
            'data': file_data,
            'hash': file_hash,
            'filename': os.path.basename(file_path),
            'timestamp': os.path.getmtime(file_path)
        }
        
        # Encrypt the package
        encrypted_data = self.cipher.encrypt(pickle.dumps(package))
        
        # Save encrypted file
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
            
        print(f"✅ Encrypted {file_path} -> {output_path}")
        return output_path
    
    def decrypt_file(self, encrypted_path: str, output_path: str = None) -> str:
        """Decrypt an encrypted algorithm file"""
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()
            
        try:
            # Decrypt and unpack
            decrypted_package = pickle.loads(self.cipher.decrypt(encrypted_data))
            
            # Verify integrity
            data = decrypted_package['data']
            expected_hash = decrypted_package['hash']
            actual_hash = hashlib.sha256(data).hexdigest()
            
            if expected_hash != actual_hash:
                raise ValueError("⚠️ File integrity check failed - possible tampering!")
                
            # Write decrypted file
            if not output_path:
                original_name = decrypted_package['filename']
                output_path = f"decrypted_{original_name}"
                
            with open(output_path, 'wb') as f:
                f.write(data)
                
            print(f"✅ Decrypted {encrypted_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Decryption failed: {e}")
            raise
    
    def encrypt_algorithm_object(self, algorithm_obj: Any) -> str:
        """Encrypt a Python object (like a trained model or strategy)"""
        # Serialize the object
        serialized = pickle.dumps(algorithm_obj)
        
        # Create package with metadata
        package = {
            'data': serialized,
            'type': type(algorithm_obj).__name__,
            'hash': hashlib.sha256(serialized).hexdigest(),
            'timestamp': time.time()
        }
        
        # Encrypt
        encrypted = self.cipher.encrypt(pickle.dumps(package))
        
        # Return base64 encoded string for storage
        return base64.b64encode(encrypted).decode()
    
    def decrypt_algorithm_object(self, encrypted_string: str) -> Any:
        """Decrypt and return a Python object"""
        try:
            # Decode from base64
            encrypted_data = base64.b64decode(encrypted_string.encode())
            
            # Decrypt
            package = pickle.loads(self.cipher.decrypt(encrypted_data))
            
            # Verify integrity
            data = package['data']
            expected_hash = package['hash']
            actual_hash = hashlib.sha256(data).hexdigest()
            
            if expected_hash != actual_hash:
                raise ValueError("Object integrity check failed!")
                
            # Deserialize and return
            return pickle.loads(data)
            
        except Exception as e:
            print(f"❌ Object decryption failed: {e}")
            raise
    
    def create_encrypted_module(self, module_path: str, encrypted_output: str):
        """Create an encrypted Python module that can be imported"""
        # Read the module
        with open(module_path, 'r') as f:
            source_code = f.read()
            
        # Create encrypted wrapper
        wrapper_code = f'''
"""
Encrypted EvoSphere Trading Algorithm Module
Protected by military-grade encryption
"""
import base64
import pickle
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import hashlib

# Encrypted algorithm data
ENCRYPTED_DATA = """{self.encrypt_algorithm_object(source_code)}"""

class ProtectedAlgorithm:
    """Protected algorithm loader"""
    
    def __init__(self):
        self.master_key = os.environ.get('ALGORITHM_MASTER_KEY', 'EvoSphere2025!')
        self._algorithm = None
        
    def unlock(self, password: str = None) -> bool:
        """Unlock the algorithm with password"""
        try:
            if password:
                self.master_key = password
                
            # Initialize decryption
            from algorithm_encryption import AlgorithmEncryption
            encryptor = AlgorithmEncryption(self.master_key)
            
            # Decrypt source code
            source_code = encryptor.decrypt_algorithm_object(ENCRYPTED_DATA)
            
            # Execute in protected namespace
            namespace = {{"__name__": __name__}}
            exec(source_code, namespace)
            
            # Extract algorithm components
            self._algorithm = namespace
            print("🔓 Algorithm unlocked successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to unlock algorithm: {{e}}")
            return False
    
    def get_algorithm(self, component: str = None):
        """Get algorithm component"""
        if not self._algorithm:
            raise RuntimeError("Algorithm not unlocked! Call unlock() first.")
            
        if component:
            return self._algorithm.get(component)
        return self._algorithm

# Global protected instance
_protected = ProtectedAlgorithm()

def unlock_algorithm(password: str = None):
    """Unlock the protected algorithm"""
    return _protected.unlock(password)

def get_algorithm(component: str = None):
    """Get algorithm component after unlocking"""
    return _protected.get_algorithm(component)
'''
        
        # Save encrypted module
        with open(encrypted_output, 'w') as f:
            f.write(wrapper_code)
            
        print(f"✅ Created encrypted module: {encrypted_output}")

def encrypt_evosphere_algorithms():
    """Encrypt all core EvoSphere algorithms"""
    encryptor = AlgorithmEncryption()
    
    # Core files to encrypt
    core_files = [
        'dqn_agent.py',
        'evolutionary_feature_selector.py', 
        'main_ea_drl_forex_trader.py',
        'technical_indicator_calculator_pdta.py',
        'forex_simulation_env.py'
    ]
    
    encrypted_files = []
    
    print("🔐 Encrypting EvoSphere Core Algorithms...")
    print("=" * 50)
    
    for file_path in core_files:
        if os.path.exists(file_path):
            try:
                # Encrypt the file
                encrypted_path = encryptor.encrypt_file(file_path)
                encrypted_files.append(encrypted_path)
                
                # Create protected module version
                protected_path = f"protected_{file_path}"
                encryptor.create_encrypted_module(file_path, protected_path)
                encrypted_files.append(protected_path)
                
            except Exception as e:
                print(f"❌ Failed to encrypt {file_path}: {e}")
        else:
            print(f"⚠️ File not found: {file_path}")
    
    # Create secure loader script
    create_secure_loader(encryptor)
    
    print("\n🔒 Encryption Complete!")
    print(f"✅ Protected {len(encrypted_files)} algorithm files")
    print("\n📋 Usage Instructions:")
    print("1. Use secure_loader.py to access algorithms")
    print("2. Set ALGORITHM_MASTER_KEY environment variable")
    print("3. Call unlock_algorithms() before using")
    print("\n⚠️ IMPORTANT: Keep your master key secure!")
    
    return encrypted_files

def create_secure_loader(encryptor: AlgorithmEncryption):
    """Create secure algorithm loader"""
    loader_code = '''
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
'''
    
    with open('secure_loader.py', 'w') as f:
        f.write(loader_code)
    
    print("✅ Created secure_loader.py")

if __name__ == "__main__":
    # Encrypt all algorithms
    encrypt_evosphere_algorithms()