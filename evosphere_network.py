"""
EvoSphere Decentralized Neural Network
Blockchain-connected trading spheres forming a global collective intelligence
"""
import hashlib
import time
import json
import uuid
from typing import Dict, List, Optional, Any
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import threading
import socket
import requests
from dataclasses import dataclass
from datetime import datetime, timezone

@dataclass
class TradeSignal:
    """Trading signal from an EvoSphere device"""
    sphere_id: str
    timestamp: float
    market_pair: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    technical_data: Dict[str, float]
    fitness_score: float
    generation: int
    signature: bytes

@dataclass
class SphereNode:
    """EvoSphere node in the network"""
    sphere_id: str
    public_key: bytes
    location: str  # Country/region
    online_since: float
    total_trades: int
    success_rate: float
    contribution_score: float
    last_seen: float

class EvoSphereBlockchain:
    """Blockchain for EvoSphere network consensus"""
    
    def __init__(self):
        self.chain: List[Dict] = []
        self.pending_signals: List[TradeSignal] = []
        self.sphere_nodes: Dict[str, SphereNode] = {}
        self.difficulty = 4  # Mining difficulty
        
        # Create genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """Create the first block in the chain"""
        genesis_block = {
            'index': 0,
            'timestamp': time.time(),
            'signals': [],
            'previous_hash': '0',
            'nonce': 0,
            'hash': self.calculate_hash('0', [], 0, time.time())
        }
        self.chain.append(genesis_block)
    
    def calculate_hash(self, previous_hash: str, signals: List[Dict], nonce: int, timestamp: float) -> str:
        """Calculate block hash"""
        block_string = f"{previous_hash}{json.dumps(signals, sort_keys=True)}{nonce}{timestamp}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def add_signal(self, signal: TradeSignal):
        """Add trading signal to pending pool"""
        # Verify signal signature and fitness threshold
        if self.verify_signal(signal) and signal.fitness_score > 0.7:
            self.pending_signals.append(signal)
            
            # Auto-mine block when we have enough signals
            if len(self.pending_signals) >= 10:
                self.mine_block()
    
    def mine_block(self) -> Dict:
        """Mine a new block with pending signals"""
        previous_block = self.chain[-1]
        new_index = previous_block['index'] + 1
        timestamp = time.time()
        
        # Convert signals to dict format
        signal_data = [self.signal_to_dict(signal) for signal in self.pending_signals]
        
        # Mine for correct nonce
        nonce = 0
        while True:
            hash_attempt = self.calculate_hash(
                previous_block['hash'], 
                signal_data, 
                nonce, 
                timestamp
            )
            
            if hash_attempt.startswith('0' * self.difficulty):
                new_block = {
                    'index': new_index,
                    'timestamp': timestamp,
                    'signals': signal_data,
                    'previous_hash': previous_block['hash'],
                    'nonce': nonce,
                    'hash': hash_attempt
                }
                
                self.chain.append(new_block)
                self.pending_signals = []  # Clear pending signals
                
                print(f"🔗 Block {new_index} mined! Hash: {hash_attempt[:16]}...")
                return new_block
            
            nonce += 1
    
    def verify_signal(self, signal: TradeSignal) -> bool:
        """Verify signal authenticity and quality"""
        # Check if sphere is registered
        if signal.sphere_id not in self.sphere_nodes:
            return False
        
        # Verify fitness score is reasonable
        if not (0.0 <= signal.confidence <= 1.0):
            return False
        
        # Check timestamp is recent (within 5 minutes)
        if abs(time.time() - signal.timestamp) > 300:
            return False
            
        return True
    
    def signal_to_dict(self, signal: TradeSignal) -> Dict:
        """Convert signal to dictionary for blockchain storage"""
        return {
            'sphere_id': signal.sphere_id,
            'timestamp': signal.timestamp,
            'market_pair': signal.market_pair,
            'signal_type': signal.signal_type,
            'confidence': signal.confidence,
            'technical_data': signal.technical_data,
            'fitness_score': signal.fitness_score,
            'generation': signal.generation
        }
    
    def get_consensus_signal(self, market_pair: str) -> Optional[Dict]:
        """Get consensus trading signal for a market pair"""
        recent_signals = []
        current_time = time.time()
        
        # Collect recent signals from last 3 blocks
        for block in self.chain[-3:]:
            for signal in block['signals']:
                if (signal['market_pair'] == market_pair and 
                    current_time - signal['timestamp'] < 600):  # Last 10 minutes
                    recent_signals.append(signal)
        
        if not recent_signals:
            return None
        
        # Calculate weighted consensus
        buy_weight = sum(s['confidence'] for s in recent_signals if s['signal_type'] == 'buy')
        sell_weight = sum(s['confidence'] for s in recent_signals if s['signal_type'] == 'sell')
        hold_weight = sum(s['confidence'] for s in recent_signals if s['signal_type'] == 'hold')
        
        total_weight = buy_weight + sell_weight + hold_weight
        
        if total_weight == 0:
            return None
        
        # Determine consensus
        if buy_weight > sell_weight and buy_weight > hold_weight:
            consensus_type = 'buy'
            consensus_confidence = buy_weight / total_weight
        elif sell_weight > buy_weight and sell_weight > hold_weight:
            consensus_type = 'sell'
            consensus_confidence = sell_weight / total_weight
        else:
            consensus_type = 'hold'
            consensus_confidence = hold_weight / total_weight
        
        return {
            'signal_type': consensus_type,
            'confidence': consensus_confidence,
            'contributing_spheres': len(recent_signals),
            'timestamp': current_time
        }

class EvoSphereNetworking:
    """Peer-to-peer networking for EvoSphere devices"""
    
    def __init__(self, sphere_id: str, port: int = 8333):
        self.sphere_id = sphere_id
        self.port = port
        self.peers: List[str] = []  # List of peer IP addresses
        self.running = False
        
        # Known seed nodes (bootstrap servers)
        self.seed_nodes = [
            'evosphere-seed1.tradingpro.com:8333',
            'evosphere-seed2.tradingpro.com:8333',
            'evosphere-vietnam.tradingpro.com:8333'
        ]
    
    def start_networking(self):
        """Start P2P networking"""
        self.running = True
        
        # Start server thread
        server_thread = threading.Thread(target=self.start_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Connect to seed nodes
        self.connect_to_seeds()
        
        print(f"🌐 EvoSphere {self.sphere_id} networking started on port {self.port}")
    
    def start_server(self):
        """Start listening server for incoming connections"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind(('0.0.0.0', self.port))
            server_socket.listen(10)
            
            while self.running:
                try:
                    client_socket, address = server_socket.accept()
                    # Handle peer connection
                    peer_thread = threading.Thread(
                        target=self.handle_peer_connection,
                        args=(client_socket, address)
                    )
                    peer_thread.daemon = True
                    peer_thread.start()
                except:
                    continue
                    
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            server_socket.close()
    
    def handle_peer_connection(self, client_socket, address):
        """Handle incoming peer connection"""
        try:
            # Simple protocol: exchange sphere IDs and sync data
            message = client_socket.recv(1024).decode()
            
            if message.startswith('HELLO'):
                peer_id = message.split(':')[1]
                response = f"HELLO:{self.sphere_id}"
                client_socket.send(response.encode())
                
                # Add to peers if not already present
                peer_address = f"{address[0]}:{self.port}"
                if peer_address not in self.peers:
                    self.peers.append(peer_address)
                    print(f"🤝 Connected to peer {peer_id} at {peer_address}")
                    
        except:
            pass
        finally:
            client_socket.close()
    
    def connect_to_seeds(self):
        """Connect to seed nodes to join network"""
        for seed in self.seed_nodes:
            try:
                host, port = seed.split(':')
                self.connect_to_peer(host, int(port))
            except:
                continue
    
    def connect_to_peer(self, host: str, port: int):
        """Connect to a specific peer"""
        try:
            peer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            peer_socket.settimeout(5)
            peer_socket.connect((host, port))
            
            # Send hello message
            hello_msg = f"HELLO:{self.sphere_id}"
            peer_socket.send(hello_msg.encode())
            
            # Receive response
            response = peer_socket.recv(1024).decode()
            if response.startswith('HELLO'):
                peer_id = response.split(':')[1]
                peer_address = f"{host}:{port}"
                
                if peer_address not in self.peers:
                    self.peers.append(peer_address)
                    print(f"🤝 Connected to peer {peer_id}")
                    
            peer_socket.close()
            
        except Exception as e:
            print(f"Failed to connect to {host}:{port} - {e}")
    
    def broadcast_signal(self, signal: TradeSignal):
        """Broadcast trading signal to all peers"""
        signal_data = {
            'type': 'TRADE_SIGNAL',
            'signal': {
                'sphere_id': signal.sphere_id,
                'timestamp': signal.timestamp,
                'market_pair': signal.market_pair,
                'signal_type': signal.signal_type,
                'confidence': signal.confidence,
                'technical_data': signal.technical_data,
                'fitness_score': signal.fitness_score,
                'generation': signal.generation
            }
        }
        
        message = json.dumps(signal_data)
        
        # Send to all connected peers
        for peer in self.peers:
            try:
                self.send_to_peer(peer, message)
            except:
                continue
    
    def send_to_peer(self, peer_address: str, message: str):
        """Send message to specific peer"""
        host, port = peer_address.split(':')
        
        try:
            peer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            peer_socket.settimeout(3)
            peer_socket.connect((host, int(port)))
            peer_socket.send(message.encode())
            peer_socket.close()
        except:
            # Remove failed peer
            if peer_address in self.peers:
                self.peers.remove(peer_address)

class EvoSphereCollectiveIntelligence:
    """Collective intelligence system for connected EvoSpheres"""
    
    def __init__(self, sphere_id: str, location: str = "Unknown"):
        self.sphere_id = sphere_id
        self.location = location
        
        # Initialize components
        self.blockchain = EvoSphereBlockchain()
        self.networking = EvoSphereNetworking(sphere_id)
        
        # AI components
        self.local_fitness = 0.0
        self.generation = 0
        self.knowledge_base = {}
        
        # Network intelligence
        self.collective_signals = {}
        self.trust_scores = {}
        
    def start_collective_intelligence(self):
        """Start the collective intelligence system"""
        # Start networking
        self.networking.start_networking()
        
        # Register this sphere
        self.register_sphere()
        
        print(f"🧠 EvoSphere Collective Intelligence started")
        print(f"   Sphere ID: {self.sphere_id}")
        print(f"   Location: {self.location}")
        print(f"   Connected peers: {len(self.networking.peers)}")
    
    def register_sphere(self):
        """Register this sphere in the network"""
        sphere_node = SphereNode(
            sphere_id=self.sphere_id,
            public_key=b"placeholder_key",  # Would use real RSA key
            location=self.location,
            online_since=time.time(),
            total_trades=0,
            success_rate=0.0,
            contribution_score=0.0,
            last_seen=time.time()
        )
        
        self.blockchain.sphere_nodes[self.sphere_id] = sphere_node
    
    def contribute_signal(self, market_pair: str, signal_type: str, confidence: float, 
                         technical_data: Dict[str, float]):
        """Contribute a trading signal to the collective"""
        
        signal = TradeSignal(
            sphere_id=self.sphere_id,
            timestamp=time.time(),
            market_pair=market_pair,
            signal_type=signal_type,
            confidence=confidence,
            technical_data=technical_data,
            fitness_score=self.local_fitness,
            generation=self.generation,
            signature=b"placeholder_signature"  # Would use real cryptographic signature
        )
        
        # Add to local blockchain
        self.blockchain.add_signal(signal)
        
        # Broadcast to network
        self.networking.broadcast_signal(signal)
        
        print(f"📡 Broadcasted {signal_type} signal for {market_pair} (confidence: {confidence:.2f})")
    
    def get_collective_decision(self, market_pair: str) -> Optional[Dict]:
        """Get collective trading decision from network"""
        return self.blockchain.get_consensus_signal(market_pair)
    
    def update_local_fitness(self, new_fitness: float, generation: int):
        """Update local AI fitness score"""
        self.local_fitness = new_fitness
        self.generation = generation
        
        # Update sphere node info
        if self.sphere_id in self.blockchain.sphere_nodes:
            self.blockchain.sphere_nodes[self.sphere_id].last_seen = time.time()
    
    def get_network_stats(self) -> Dict:
        """Get network statistics"""
        return {
            'total_spheres': len(self.blockchain.sphere_nodes),
            'connected_peers': len(self.networking.peers),
            'blocks_mined': len(self.blockchain.chain),
            'pending_signals': len(self.blockchain.pending_signals),
            'local_fitness': self.local_fitness,
            'local_generation': self.generation
        }

# Global network instance
_collective_intelligence = None

def initialize_evosphere_network(sphere_id: str = None, location: str = "Vietnam") -> EvoSphereCollectiveIntelligence:
    """Initialize EvoSphere network connection"""
    global _collective_intelligence
    
    if not sphere_id:
        sphere_id = f"evosphere_{uuid.uuid4().hex[:8]}"
    
    _collective_intelligence = EvoSphereCollectiveIntelligence(sphere_id, location)
    _collective_intelligence.start_collective_intelligence()
    
    return _collective_intelligence

def contribute_to_network(market_pair: str, signal_type: str, confidence: float, 
                         technical_data: Dict[str, float]):
    """Contribute trading signal to global network"""
    if _collective_intelligence:
        _collective_intelligence.contribute_signal(market_pair, signal_type, confidence, technical_data)

def get_network_consensus(market_pair: str) -> Optional[Dict]:
    """Get consensus decision from global network"""
    if _collective_intelligence:
        return _collective_intelligence.get_collective_decision(market_pair)
    return None

def update_fitness(fitness: float, generation: int):
    """Update local fitness in network"""
    if _collective_intelligence:
        _collective_intelligence.update_local_fitness(fitness, generation)

def get_network_status() -> Dict:
    """Get current network status"""
    if _collective_intelligence:
        return _collective_intelligence.get_network_stats()
    return {'status': 'not_connected'}

# Example usage
if __name__ == "__main__":
    # Initialize network
    network = initialize_evosphere_network("vietnam_sphere_001", "Ho Chi Minh City")
    
    # Simulate contributing signals
    time.sleep(2)
    
    # Contribute a buy signal
    contribute_to_network(
        market_pair="BTC/USD",
        signal_type="buy", 
        confidence=0.85,
        technical_data={
            'rsi': 25.5,
            'momentum': 0.045,
            'volatility': 0.032
        }
    )
    
    # Get network consensus
    time.sleep(1)
    consensus = get_network_consensus("BTC/USD")
    if consensus:
        print(f"🌐 Network consensus: {consensus}")
    
    # Show network stats
    stats = get_network_status()
    print(f"📊 Network stats: {stats}")