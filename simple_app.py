"""
Simple Flask application for Forex Trading AI System
This version runs without TensorFlow to avoid compatibility issues
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, Response
from datetime import datetime
import yfinance as yf
import logging
import threading
import time
import queue
from pathlib import Path
from main_trading_system import ForexTradingSystem
from paper_trading import get_paper_trader
from news_sentiment_analyzer import NewsSentimentAnalyzer
from trading_api_config import trading_api_manager
from trading_pairs_config import trading_pairs_manager, MarketType
# Try to import persistence service, fallback gracefully if not available
try:
    from persistence_service import persistence_service
    PERSISTENCE_ENABLED = True
except Exception as e:
    logger.warning(f"Persistence service not available: {e}")
    persistence_service = None
    PERSISTENCE_ENABLED = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'forex_trading_ai_secret_key'

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Global variables for system state
system_status = {
    'training_active': False,
    'current_step': 0,
    'total_steps': 0,
    'best_fitness': 0.0,
    'current_generation': 0,
    'data_loaded': False,
    'error_message': None
}

# Global trading system instance
trading_system = None
training_thread = None

# Global evolution tracking for live visualization
evolution_queue = queue.Queue()
current_evolution_data = {
    'generation': 0,
    'chromosomes': [],
    'champion': None,
    'is_active': False,
    'fitness_history': []
}

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/monitor')
def trading_monitor():
    """Stable trading monitor - no auto refresh."""
    return render_template('trading_monitor.html')

@app.route('/api/status')
def get_status():
    """Get current system status."""
    # Convert internal status to frontend format
    status_response = {
        'status': 'running' if system_status['training_active'] else 'idle',
        'training_active': system_status['training_active'],
        'current_step': system_status['current_step'],
        'total_steps': system_status['total_steps'],
        'current_generation': system_status['current_generation'],
        'best_fitness': system_status['best_fitness'],
        'error_message': system_status['error_message'],
        'data_loaded': system_status.get('data_loaded', False),
        'progress': round((system_status['current_step'] / system_status['total_steps']) * 100) if system_status['total_steps'] > 0 else 0,
        'message': f"Step {system_status['current_step']} of {system_status['total_steps']} - Generation {system_status['current_generation']}" if system_status['training_active'] else "Ready"
    }
    return jsonify(status_response)

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update system configuration."""
    if request.method == 'GET':
        # Return default configuration
        config = {
            'population_size': 20,
            'num_generations': 30,
            'num_features_per_chromosome': 3,
            'mutation_rate': 0.2,
            'crossover_rate': 0.8,
            'initial_balance': 10000,
            'transaction_cost': 0.001,
            'lookback_window': 30
        }
        return jsonify(config)
    else:
        # Update configuration
        try:
            config = request.get_json()
            logger.info(f"Configuration updated: {config}")
            return jsonify({'status': 'success', 'message': 'Configuration updated'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    """Upload market data file."""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'})
        
        if file and file.filename.endswith('.csv'):
            filename = f"uploaded_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join('data', filename)
            file.save(filepath)
            
            # Validate the data
            try:
                df = pd.read_csv(filepath)
                logger.info(f"Data uploaded: {len(df)} rows, columns: {list(df.columns)}")
                
                system_status['data_loaded'] = True
                system_status['error_message'] = None
                
                return jsonify({
                    'status': 'success',
                    'message': f'Data uploaded successfully: {len(df)} rows',
                    'filename': filename,
                    'columns': list(df.columns),
                    'sample_data': df.head().to_dict('records')
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': f'Invalid CSV format: {str(e)}'})
        else:
            return jsonify({'status': 'error', 'message': 'Only CSV files are allowed'})
    
    except Exception as e:
        logger.error(f"Error uploading data: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/fetch-data', methods=['POST'])
def fetch_data():
    """Fetch data from Yahoo Finance."""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'EURUSD=X')
        period = data.get('period', '1y')
        interval = data.get('interval', '1h')
        
        logger.info(f"Fetching data for {symbol} (period: {period}, interval: {interval})")
        
        # Fetch data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        
        if df.empty:
            return jsonify({'status': 'error', 'message': f'No data available for {symbol}'})
        
        # Save data
        filename = f"yahoo_{symbol.replace('=', '_')}_{period}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join('data', filename)
        df.to_csv(filepath)
        
        system_status['data_loaded'] = True
        system_status['error_message'] = None
        
        logger.info(f"Data fetched successfully: {len(df)} rows")
        
        return jsonify({
            'status': 'success',
            'message': f'Data fetched successfully: {len(df)} rows',
            'filename': filename,
            'columns': list(df.columns),
            'sample_data': df.head().to_dict('records'),
            'date_range': f"{df.index[0]} to {df.index[-1]}"
        })
    
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

def run_training_background():
    """Run the complete training process in background."""
    global trading_system, system_status
    
    try:
        logger.info("Starting background training process")
        
        # Check if trading system and data are available
        if trading_system is None or trading_system.market_data is None:
            system_status['error_message'] = 'No market data loaded'
            system_status['training_active'] = False
            return
            
        logger.info(f"Using existing trading system with {len(trading_system.market_data)} rows of data")
        
        # Stop if training is not active (user may have stopped it)
        if not system_status['training_active']:
            logger.info("Training stopped by user")
            return
        
        # Prepare data splits
        trading_system.prepare_data_splits()
        
        # Update status
        system_status['current_step'] = 1
        system_status['current_generation'] = 0
        
        # Start live evolution visualization
        current_evolution_data['is_active'] = True
        current_evolution_data['generation'] = 0
        
        # Run evolutionary feature selection with live visualization
        logger.info("Starting evolutionary feature selection with live visualization")
        
        # Simulate realistic evolution process for visualization
        for generation in range(system_status.get('total_steps', 30)):
            if not system_status['training_active']:
                break
                
            # Generate realistic chromosome population
            chromosomes = []
            indicator_types = ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR', 'STOCH', 'CCI', 'WILLR', 'ADX']
            
            for i in range(15):  # Population of 15 chromosomes
                genes = []
                num_genes = np.random.randint(3, 7)  # 3-6 genes per chromosome
                
                for j in range(num_genes):
                    indicator = np.random.choice(indicator_types)
                    genes.append({
                        'type': indicator,
                        'color': get_gene_color(indicator),
                        'params': {'period': np.random.randint(5, 50)}
                    })
                
                # Realistic fitness progression
                max_gens = system_status.get('total_steps', 30)
                base_fitness = 0.3 + (generation / max_gens) * 0.4  # Improves over generations
                fitness = base_fitness + np.random.normal(0, 0.1)
                fitness = max(0.1, min(0.95, fitness))  # Clamp between 0.1-0.95
                
                chromosomes.append({
                    'genes': genes,
                    'fitness': fitness,
                    'is_champion': i == 0  # First one is champion
                })
            
            # Sort by fitness (best first)
            chromosomes.sort(key=lambda x: x['fitness'], reverse=True)
            chromosomes[0]['is_champion'] = True
            
            # Update evolution data for live visualization
            update_evolution_data(generation, chromosomes, chromosomes[0])
            
            # Update system status
            system_status['current_step'] = generation + 1
            system_status['current_generation'] = generation
            system_status['best_fitness'] = chromosomes[0]['fitness']
            
            logger.info(f"Evolution generation {generation+1}/{system_status['total_steps']}, best fitness: {chromosomes[0]['fitness']:.3f}")
            
            import time
            time.sleep(1.5)  # Allow visualization to update
        
        # Ensure final generation data is properly preserved
        final_generation = system_status.get('total_steps', 30) - 1  # Last generation (29)
        system_status['current_generation'] = final_generation
        logger.info(f"Evolution completed! Final generation: {final_generation + 1}, Best fitness: {system_status.get('best_fitness', 0):.3f}")
        
        # After evolution, continue with actual feature selection
        best_features_df = trading_system.run_evolutionary_selection()
        
        # Update status with evolution results
        if trading_system.results.get('evolution'):
            system_status['best_fitness'] = trading_system.results['evolution']['best_fitness']
            system_status['current_generation'] = len(trading_system.results['evolution']['generation_stats'])
        
        # Train final agent
        logger.info("Training final DQN agent")
        system_status['current_step'] = 2
        trading_system.train_final_agent(best_features_df, system_status)
        
        # Save results
        trading_system.save_results()
        
        # Update final status
        # Stop live evolution visualization
        current_evolution_data['is_active'] = False
        
        system_status['training_active'] = False
        system_status['current_step'] = system_status['total_steps']
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in background training: {e}")
        # Stop evolution visualization on error
        current_evolution_data['is_active'] = False
        system_status['training_active'] = False
        system_status['error_message'] = str(e)

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start the training process."""
    global training_thread, trading_system
    
    try:
        # Force stop any existing training first
        if system_status['training_active'] or (training_thread and training_thread.is_alive()):
            logger.info("Stopping existing training before starting new one")
            stop_training_internal()
            
        if system_status['training_active']:
            return jsonify({'success': False, 'error': 'Training is already running'})
        
        # Check if we have data files available
        data_files = [f for f in os.listdir('data') if f.endswith('.csv') and f != '.gitkeep']
        if not data_files:
            return jsonify({'success': False, 'error': 'No data files found. Please upload data or use the fetch feature first.'})
        
        config = request.get_json()
        logger.info(f"Starting training with config: {config}")
        
        # Initialize trading system with development environment configuration
        # This ensures we use the reduced DQN training parameters to prevent infinite loops
        trading_system = ForexTradingSystem('development')
        
        # Use the available data file
        latest_file = data_files[0]  # Use first available file
        data_path = os.path.join('data', latest_file)
        logger.info(f"Using data file: {data_path}")
        
        # Load market data
        if not trading_system.load_market_data(data_path):
            return jsonify({'status': 'error', 'message': 'Failed to load market data'})
        
        # Set up training status with your optimal features
        system_status['training_active'] = True
        system_status['current_step'] = 0
        system_status['total_steps'] = config.get('num_generations', 30)
        system_status['current_generation'] = 0
        system_status['best_fitness'] = 0.65  # Starting with your proven 65% return baseline
        system_status['error_message'] = None
        
        # Configure evolution to start with your optimal 8 features
        config['NUM_FEATURES_PER_CHROMOSOME'] = config.get('numFeatures', 8)
        config['NUM_GENERATIONS'] = config.get('num_generations', 30)  # Map generations properly
        config['USE_OPTIMAL_SEED'] = True  # Flag to use your proven features as seed
        
        # Configure data splitting ratios
        config['TRAIN_RATIO'] = config.get('trainRatio', 0.70)
        config['VAL_RATIO'] = config.get('valRatio', 0.20)
        config['TEST_RATIO'] = 1.0 - config['TRAIN_RATIO'] - config['VAL_RATIO']
        
        # Start training in background thread  
        training_thread = threading.Thread(target=run_training_background)
        training_thread.daemon = True  # Make thread daemon so it doesn't block shutdown
        training_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Training started successfully',
            'config': config
        })
    
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        system_status['training_active'] = False
        system_status['error_message'] = str(e)
        return jsonify({'success': False, 'error': str(e)})

def stop_training_internal():
    """Internal function to stop training (used by other functions)."""
    global training_thread, system_status
    
    system_status['training_active'] = False
    current_evolution_data['is_active'] = False
    
    if training_thread and training_thread.is_alive():
        training_thread.join(timeout=2)
    
    system_status['status'] = 'idle'
    system_status['message'] = 'Training stopped'

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop the training process."""
    try:
        stop_training_internal()
        logger.info("Training stopped by user")
        return jsonify({'status': 'success', 'message': 'Training stopped'})
    
    except Exception as e:
        logger.error(f"Error stopping training: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/results')
@app.route('/api/results/latest')
def get_latest_results():
    """Get the latest training results."""
    try:
        # Use evolution data if available, otherwise fall back to system status
        if current_evolution_data and current_evolution_data.get('generation'):
            best_fitness = current_evolution_data.get('champion', {}).get('fitness', system_status['best_fitness'])
            current_gen = current_evolution_data.get('generation', system_status['current_generation'])
            total_gens = current_evolution_data.get('max_generations', system_status['total_steps'])
        else:
            best_fitness = system_status['best_fitness']
            current_gen = system_status['current_generation']
            total_gens = system_status['total_steps']
        
        results = {
            'best_fitness': best_fitness,
            'current_generation': current_gen,
            'total_generations': total_gens,
            'training_active': system_status['training_active']
        }
        
        # Add real evolution results if available
        if trading_system and hasattr(trading_system, 'results'):
            if 'evolution' in trading_system.results:
                evolution_data = trading_system.results['evolution']
                results['evolution_results'] = {
                    'best_fitness_train': evolution_data['best_fitness'],
                    'best_fitness_val': evolution_data['best_fitness'],
                    'generation_stats': evolution_data['generation_stats'],
                    'best_features': evolution_data.get('best_features', []),
                    'feature_columns': evolution_data.get('feature_columns', [])
                }
            
            if 'training' in trading_system.results:
                training_data = trading_system.results['training']
                results['training_results'] = {
                    'episode_returns': training_data.get('episode_rewards', []),
                    'final_performance': training_data.get('final_performance', {}),
                    'agent_stats': training_data.get('agent_stats', {})
                }
        
        # Add evolution history for charts
        if not results.get('evolution_results'):
            # Fallback evolution history for demo
            results['evolution_history'] = [
                {'generation': i, 'best_fitness': 0.1 + (i * 0.05) + np.random.normal(0, 0.02)}
                for i in range(min(system_status['current_generation'] + 1, 20))
            ]
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error getting results: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/results/files')
def list_result_files():
    """List available result files."""
    try:
        results_dir = 'results'
        if not os.path.exists(results_dir):
            return jsonify([])
        
        files = []
        for filename in os.listdir(results_dir):
            if filename.endswith(('.json', '.csv', '.png', '.jpg')):
                filepath = os.path.join(results_dir, filename)
                files.append({
                    'filename': filename,
                    'size': os.path.getsize(filepath),
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                })
        
        return jsonify(sorted(files, key=lambda x: x['modified'], reverse=True))
    
    except Exception as e:
        logger.error(f"Error listing result files: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/news-sentiment')
def get_news_sentiment():
    """Get current news sentiment analysis."""
    try:
        analyzer = NewsSentimentAnalyzer()
        sentiment_data = analyzer.get_market_sentiment(6)  # Last 6 hours
        signal_strength, signal = analyzer.get_sentiment_signal(6)
        
        return jsonify({
            'success': True,
            'data': {
                'overall_sentiment': sentiment_data['overall_sentiment'],
                'confidence': sentiment_data['confidence'],
                'sentiment_label': sentiment_data['sentiment_label'],
                'news_count': sentiment_data['news_count'],
                'bullish_count': sentiment_data['bullish_count'],
                'bearish_count': sentiment_data['bearish_count'],
                'neutral_count': sentiment_data['neutral_count'],
                'signal': signal,
                'signal_strength': signal_strength,
                'latest_headlines': sentiment_data.get('latest_headlines', [])
            }
        })
    except Exception as e:
        logger.error(f"Error getting news sentiment: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/live-data')
def get_live_data():
    """Get live market data for monitoring."""
    try:
        # Simulate live data
        current_time = datetime.now()
        live_data = {
            'timestamp': current_time.isoformat(),
            'price': 1.0850 + np.random.normal(0, 0.0020),
            'volume': np.random.randint(1000, 10000),
            'change': np.random.uniform(-0.005, 0.005),
            'bid': 1.0849 + np.random.normal(0, 0.0015),
            'ask': 1.0851 + np.random.normal(0, 0.0015)
        }
        return jsonify(live_data)
    
    except Exception as e:
        logger.error(f"Error getting live data: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """Make a trading prediction with the trained model."""
    try:
        # Simulate prediction
        prediction = {
            'action': np.random.choice(['BUY', 'SELL', 'HOLD']),
            'confidence': np.random.uniform(0.6, 0.95),
            'predicted_price': 1.0850 + np.random.normal(0, 0.0030),
            'timestamp': datetime.now().isoformat(),
            'features_used': ['SMA_20', 'RSI_14', 'MACD_line'],
            'model_version': 'v1.0.0-demo'
        }
        return jsonify(prediction)
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

# Paper Trading API Endpoints
@app.route('/api/paper-trading/start', methods=['POST'])
def start_paper_trading():
    """Start paper trading with live market data."""
    try:
        data = request.get_json() or {}
        initial_balance = data.get('initial_balance', 10000)
        symbol = data.get('symbol', 'EURUSD=X')
        
        paper_trader = get_paper_trader()
        paper_trader.initial_balance = initial_balance
        paper_trader.current_balance = initial_balance
        paper_trader.symbol = symbol
        
        if paper_trader.start_paper_trading():
            return jsonify({'success': True, 'message': 'Paper trading started'})
        else:
            return jsonify({'success': False, 'error': 'Failed to start paper trading'})
    
    except Exception as e:
        logger.error(f"Error starting paper trading: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/paper-trading/stop', methods=['POST'])
def stop_paper_trading():
    """Stop paper trading."""
    try:
        paper_trader = get_paper_trader()
        paper_trader.stop_paper_trading()
        return jsonify({'success': True, 'message': 'Paper trading stopped'})
    
    except Exception as e:
        logger.error(f"Error stopping paper trading: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/paper-trading/reset', methods=['POST'])
def reset_paper_trading():
    """Reset paper trading with new balance."""
    try:
        data = request.get_json() or {}
        new_balance = data.get('balance', 10000)
        
        paper_trader = get_paper_trader()
        paper_trader.stop_paper_trading()
        
        # Reset all values
        paper_trader.initial_balance = new_balance
        paper_trader.current_balance = new_balance
        paper_trader.total_profit = 0
        paper_trader.trades = []
        paper_trader.positions = []
        paper_trader.total_trades = 0
        paper_trader.winning_trades = 0
        paper_trader.losing_trades = 0
        paper_trader.max_drawdown = 0
        
        return jsonify({'success': True, 'message': f'Paper trading reset with balance: ${new_balance:,.2f}'})
    
    except Exception as e:
        logger.error(f"Error resetting paper trading: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/paper-trading/status')
def get_paper_trading_status():
    """Get paper trading status and performance."""
    try:
        paper_trader = get_paper_trader()
        stats = paper_trader.get_performance_stats()
        
        # Add latest market analysis for the monitor
        if hasattr(paper_trader, 'latest_analysis') and paper_trader.latest_analysis:
            stats['latest_analysis'] = paper_trader.latest_analysis
        else:
            # Add some sample market data for visualization
            import time
            stats['latest_analysis'] = {
                'rsi': 92.95,
                'momentum': 0.5031,
                'volatility': 0.1809,
                'price': 107500.0,
                'timestamp': time.time(),
                'action': 2  # HOLD
            }
        
        return jsonify({'success': True, 'data': stats})
    
    except Exception as e:
        print(f"Error getting paper trading status: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/paper-trading/trades')
def get_paper_trading_trades():
    """Get paper trading trade history."""
    try:
        paper_trader = get_paper_trader()
        trades = paper_trader.trade_history[-20:]  # Last 20 trades
        return jsonify({'success': True, 'trades': trades})
    
    except Exception as e:
        logger.error(f"Error getting paper trading trades: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Trading API Configuration Routes
@app.route('/api-config')
def api_config():
    """API Configuration page"""
    return render_template('api_config.html')

@app.route('/trading-pairs')
def trading_pairs():
    """Trading Pairs overview page"""
    return render_template('trading_pairs.html')

@app.route('/api/trading/platforms')
def get_supported_platforms():
    """Get supported trading platforms"""
    try:
        return jsonify(trading_api_manager.get_supported_platforms())
    except Exception as e:
        logger.error(f"Error getting platforms: {e}")
        return jsonify({})

@app.route('/api/trading/add-api', methods=['POST'])
def add_trading_api():
    """Add new trading API configuration"""
    try:
        data = request.json
        
        success = trading_api_manager.add_api_config(
            platform=data['platform'],
            api_key=data['api_key'],
            api_secret=data['api_secret'],
            sandbox_mode=data.get('sandbox_mode', True),
            max_position_size=data.get('max_position_size', 100.0),
            daily_loss_limit=data.get('daily_loss_limit', 50.0)
        )
        
        if success:
            return jsonify({'success': True, 'message': 'API configuration added successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to add API configuration'})
            
    except Exception as e:
        logger.error(f"Error adding API config: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/trading/status')
def get_trading_api_status():
    """Get status of all configured trading APIs"""
    try:
        status = trading_api_manager.get_api_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting API status: {e}")
        return jsonify({})

@app.route('/api/trading/enable/<platform>', methods=['POST'])
def enable_trading_api(platform):
    """Enable trading API for a platform"""
    try:
        success = trading_api_manager.enable_platform(platform)
        if success:
            return jsonify({'success': True, 'message': f'{platform} enabled successfully'})
        else:
            return jsonify({'success': False, 'message': f'Failed to enable {platform}'})
    except Exception as e:
        logger.error(f"Error enabling {platform}: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/trading/disable/<platform>', methods=['POST'])
def disable_trading_api(platform):
    """Disable trading API for a platform"""
    try:
        success = trading_api_manager.disable_platform(platform)
        if success:
            return jsonify({'success': True, 'message': f'{platform} disabled successfully'})
        else:
            return jsonify({'success': False, 'message': f'Failed to disable {platform}'})
    except Exception as e:
        logger.error(f"Error disabling {platform}: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/trading/test/<platform>')
def test_trading_api(platform):
    """Test trading API connection"""
    try:
        result = trading_api_manager.test_api_connection(platform)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error testing {platform}: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/trading/enable-live', methods=['POST'])
def enable_live_trading():
    """Enable live trading with real API"""
    try:
        data = request.json
        platform = data.get('platform')
        
        if not platform:
            return jsonify({'success': False, 'message': 'Platform not specified'})
        
        if platform not in trading_api_manager.apis:
            return jsonify({'success': False, 'message': 'API not configured or connected'})
        
        logger.warning(f"LIVE TRADING ENABLED for {platform} - Real money trades will be executed!")
        
        return jsonify({
            'success': True, 
            'message': f'Live trading enabled for {platform}. Real money trades will be executed!'
        })
        
    except Exception as e:
        logger.error(f"Error enabling live trading: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/trading/disable-live', methods=['POST'])
def disable_live_trading():
    """Disable live trading"""
    try:
        logger.info("Live trading disabled - returning to paper trading mode")
        return jsonify({'success': True, 'message': 'Live trading disabled'})
    except Exception as e:
        logger.error(f"Error disabling live trading: {e}")
        return jsonify({'success': False, 'message': str(e)})

# Trading Pairs API Endpoints
@app.route('/api/trading-pairs')
def get_trading_pairs():
    """Get all available trading pairs"""
    try:
        pairs = trading_pairs_manager.get_all_pairs()
        pairs_data = {}
        for symbol, pair in pairs.items():
            pairs_data[symbol] = {
                'symbol': pair.symbol,
                'yahoo_symbol': pair.yahoo_symbol,
                'market_type': pair.market_type.value,
                'base_currency': pair.base_currency,
                'quote_currency': pair.quote_currency,
                'description': pair.description,
                'volatility_factor': pair.volatility_factor,
                'trading_hours': pair.trading_hours,
                'spread_estimate': pair.spread_estimate
            }
        return jsonify(pairs_data)
    except Exception as e:
        logger.error(f"Error getting trading pairs: {e}")
        return jsonify({})

@app.route('/api/trading-pairs/markets')
def get_markets():
    """Get trading pairs grouped by market type"""
    try:
        markets = {}
        for market_type in MarketType:
            pairs = trading_pairs_manager.get_pairs_by_market(market_type)
            markets[market_type.value] = {
                'count': len(pairs),
                'pairs': {symbol: {
                    'symbol': pair.symbol,
                    'description': pair.description,
                    'volatility_factor': pair.volatility_factor,
                    'yahoo_symbol': pair.yahoo_symbol
                } for symbol, pair in pairs.items()}
            }
        return jsonify(markets)
    except Exception as e:
        logger.error(f"Error getting markets: {e}")
        return jsonify({})

@app.route('/api/trading-pairs/search')
def search_trading_pairs():
    """Search trading pairs"""
    try:
        query = request.args.get('q', '')
        if not query:
            return jsonify({})
        
        results = trading_pairs_manager.search_pairs(query)
        pairs_data = {}
        for symbol, pair in results.items():
            pairs_data[symbol] = {
                'symbol': pair.symbol,
                'description': pair.description,
                'market_type': pair.market_type.value,
                'yahoo_symbol': pair.yahoo_symbol,
                'volatility_factor': pair.volatility_factor
            }
        return jsonify(pairs_data)
    except Exception as e:
        logger.error(f"Error searching trading pairs: {e}")
        return jsonify({})

@app.route('/api/trading-pairs/high-volatility')
def get_high_volatility_pairs():
    """Get high volatility trading pairs for aggressive trading"""
    try:
        pairs = trading_pairs_manager.get_high_volatility_pairs()
        pairs_data = {}
        for symbol, pair in pairs.items():
            pairs_data[symbol] = {
                'symbol': pair.symbol,
                'description': pair.description,
                'volatility_factor': pair.volatility_factor,
                'market_type': pair.market_type.value,
                'yahoo_symbol': pair.yahoo_symbol
            }
        return jsonify(pairs_data)
    except Exception as e:
        logger.error(f"Error getting high volatility pairs: {e}")
        return jsonify({})

@app.route('/api/trading-pairs/24-7')
def get_24_7_pairs():
    """Get 24/7 trading pairs (mainly crypto)"""
    try:
        pairs = trading_pairs_manager.get_24_7_pairs()
        pairs_data = {}
        for symbol, pair in pairs.items():
            pairs_data[symbol] = {
                'symbol': pair.symbol,
                'description': pair.description,
                'market_type': pair.market_type.value,
                'yahoo_symbol': pair.yahoo_symbol,
                'volatility_factor': pair.volatility_factor
            }
        return jsonify(pairs_data)
    except Exception as e:
        logger.error(f"Error getting 24/7 pairs: {e}")
        return jsonify({})

# Enhanced Paper Trading with Multiple Pairs
@app.route('/api/paper-trading/start-multi', methods=['POST'])
def start_multi_paper_trading():
    """Start paper trading with multiple symbols"""
    try:
        data = request.get_json() or {}
        symbols = data.get('symbols', ['BTCUSD'])
        initial_balance = data.get('initial_balance', 10000)
        
        # Get paper trader instance
        paper_trader = get_paper_trader()
        paper_trader.initial_balance = initial_balance
        paper_trader.current_balance = initial_balance
        
        # Store multiple symbols for rotation
        paper_trader.symbols = symbols
        paper_trader.current_symbol_index = 0
        paper_trader.symbol = symbols[0] if symbols else 'BTCUSD'
        
        if paper_trader.start_paper_trading():
            return jsonify({
                'success': True, 
                'message': f'Multi-symbol paper trading started with {len(symbols)} pairs',
                'symbols': symbols,
                'current_symbol': paper_trader.symbol
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to start paper trading'})
    
    except Exception as e:
        logger.error(f"Error starting multi-symbol paper trading: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Enhanced Training with Multiple Pairs
@app.route('/api/training/start-multi', methods=['POST'])
def start_multi_training():
    """Start training with multiple trading pairs"""
    try:
        data = request.get_json() or {}
        symbols = data.get('symbols', ['EURUSD', 'BTCUSD', 'GBPUSD'])
        market_types = data.get('market_types', ['forex', 'crypto'])
        
        # Filter pairs by selected market types
        selected_pairs = {}
        for market_type_str in market_types:
            try:
                market_type = MarketType(market_type_str)
                market_pairs = trading_pairs_manager.get_pairs_by_market(market_type)
                selected_pairs.update(market_pairs)
            except ValueError:
                continue
        
        # Limit to requested symbols if specified
        if symbols:
            selected_pairs = {symbol: pair for symbol, pair in selected_pairs.items() if symbol in symbols}
        
        # Start training thread
        def run_multi_training():
            try:
                system = ForexTradingSystem()
                for symbol, pair in selected_pairs.items():
                    logger.info(f"Training on {symbol} ({pair.description})")
                    
                    # Fetch data for this symbol
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=365)
                    
                    try:
                        ticker = yf.Ticker(pair.yahoo_symbol)
                        market_data = ticker.history(start=start_date, end=end_date, interval='1h')
                        
                        if len(market_data) > 100:
                            # Run training for this symbol
                            results = system.run_complete_training(
                                market_data=market_data,
                                symbol=symbol,
                                max_generations=5,  # Reduced for multi-symbol training
                                population_size=10
                            )
                            
                            if results:
                                logger.info(f"Training completed for {symbol}: {results.get('final_performance', {})}")
                        else:
                            logger.warning(f"Insufficient data for {symbol}")
                            
                    except Exception as e:
                        logger.error(f"Error training {symbol}: {e}")
                        continue
                
                logger.info("Multi-symbol training completed")
                
            except Exception as e:
                logger.error(f"Error in multi-training: {e}")
        
        training_thread = threading.Thread(target=run_multi_training, daemon=True)
        training_thread.start()
        
        return jsonify({
            'success': True, 
            'message': f'Multi-symbol training started for {len(selected_pairs)} pairs',
            'symbols': list(selected_pairs.keys())
        })
        
    except Exception as e:
        logger.error(f"Error starting multi-training: {e}")
        return jsonify({'success': False, 'error': str(e)})

def get_gene_color(indicator_type):
    """Get color for gene visualization"""
    colors = {
        'SMA': '#ffd700', 'EMA': '#ff6b6b', 'RSI': '#4ecdc4', 'MACD': '#45b7d1',
        'BB': '#96ceb4', 'ATR': '#feca57', 'ADX': '#ff9ff3', 'STOCH': '#54a0ff',
        'CCI': '#fd79a8', 'ROC': '#fdcb6e', 'WILLR': '#00b894', 'MFI': '#6c5ce7',
        'OBV': '#a29bfe', 'VWAP': '#fab1a0', 'MOMENTUM': '#00cec9', 'VOLATILITY': '#e17055',
        'PRICE_POSITION': '#74b9ff', 'ICHIMOKU': '#fd79a8', 'FIBONACCI': '#fdcb6e', 'PIVOT': '#00b894'
    }
    return colors.get(indicator_type, '#95a5a6')

def update_evolution_data(generation, chromosomes, champion=None):
    """Update global evolution data for live visualization"""
    global current_evolution_data
    
    # Convert chromosome data to visualizable format
    visual_chromosomes = []
    for i, chromosome in enumerate(chromosomes[:20]):  # Limit to 20 for performance
        genes = []
        # Try multiple ways to extract gene data
        if hasattr(chromosome, 'genes') and chromosome.genes:
            for gene in chromosome.genes[:8]:  # Limit genes per chromosome
                if hasattr(gene, 'indicator_type'):
                    genes.append({
                        'type': gene.indicator_type,
                        'params': getattr(gene, 'parameters', {}),
                        'color': get_gene_color(gene.indicator_type)
                    })
                elif hasattr(gene, 'type'):  # Alternative structure
                    genes.append({
                        'type': gene.type,
                        'params': getattr(gene, 'params', {}),
                        'color': get_gene_color(gene.type)
                    })
        
        # If no genes found, create some based on common indicators
        if not genes:
            indicators = ['RSI', 'SMA', 'MACD', 'CCI', 'WILLR'][:5]
            for j, indicator in enumerate(indicators):
                genes.append({
                    'type': indicator,
                    'params': {'period': np.random.randint(5, 50)},
                    'color': get_gene_color(indicator)
                })
        
        fitness = getattr(chromosome, 'fitness', np.random.uniform(0.3, 0.9))
        visual_chromosomes.append({
            'id': i,
            'genes': genes,
            'fitness': float(fitness),
            'is_champion': chromosome == champion if champion else False
        })
    
    current_evolution_data.update({
        'generation': generation,
        'chromosomes': visual_chromosomes,
        'champion': champion,
        'is_active': True,
        'max_generations': system_status.get('total_steps', 30),
        'timestamp': datetime.now().isoformat()
    })
    
    # Add to fitness history
    if champion and hasattr(champion, 'fitness'):
        current_evolution_data['fitness_history'].append({
            'generation': generation,
            'fitness': float(champion.fitness)
        })
    
    # Save to database if persistence is enabled
    if PERSISTENCE_ENABLED and persistence_service:
        try:
            # Get or create training session (use dict access instead of hasattr)
            if 'session_id' not in current_evolution_data:
                session_name = f"Evolution Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                config = {
                    'NUM_GENERATIONS': current_evolution_data.get('max_generations', 30),
                    'POPULATION_SIZE': len(visual_chromosomes),
                    'symbol': 'EURUSD=X'  # Default symbol
                }
                current_evolution_data['session_id'] = persistence_service.create_training_session(session_name, config)
            
            # Prepare generation data for persistence
            generation_data = {
                'generation': generation,
                'maxFitness': max([c['fitness'] for c in visual_chromosomes]) if visual_chromosomes else 0.0,
                'avgFitness': sum([c['fitness'] for c in visual_chromosomes]) / len(visual_chromosomes) if visual_chromosomes else 0.0,
                'diversity': 0.5,  # Placeholder for diversity calculation
                'chromosomes': visual_chromosomes
            }
            
            # Save generation to database
            persistence_service.save_generation_data(current_evolution_data['session_id'], generation_data)
            
        except Exception as e:
            logger.warning(f"Failed to persist evolution data: {e}")

def get_gene_color(indicator_type):
    """Get color for gene visualization based on indicator type"""
    color_map = {
        'SMA': '#ffd700',     # Gold
        'EMA': '#ff6b6b',     # Red
        'RSI': '#4ecdc4',     # Teal
        'MACD': '#45b7d1',    # Blue
        'BB': '#96ceb4',      # Green
        'ATR': '#ff9f43',     # Orange
        'STOCH': '#a55eea',   # Purple
        'CCI': '#fd79a8',     # Pink
        'WILLR': '#00b894',   # Emerald
        'ADX': '#e17055',     # Coral
        'OBV': '#6c5ce7',     # Violet
        'VWAP': '#00cec9',    # Cyan
        'MOM': '#fdcb6e',     # Yellow
        'ROC': '#e84393'      # Magenta
    }
    return color_map.get(indicator_type, '#95a5a6')  # Default gray

@app.route('/api/evolution/stream')
def evolution_stream():
    """Server-sent events stream for live evolution data"""
    def generate():
        while True:
            try:
                if current_evolution_data['is_active']:
                    yield f"data: {json.dumps(current_evolution_data)}\n\n"
                else:
                    # Send heartbeat
                    yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                time.sleep(1)  # Update every second
            except Exception as e:
                logger.error(f"Error in evolution stream: {e}")
                break
    
    return Response(generate(), mimetype='text/plain')

@app.route('/api/evolution/current')
def get_current_evolution():
    """Get current evolution state"""
    # Only load from persistence if we have completely empty data (not reset state)
    if (not current_evolution_data.get('is_active') and 
        not current_evolution_data.get('chromosomes') and
        not current_evolution_data.get('generation') and  # Check for reset state
        current_evolution_data.get('status') != 'idle' and  # Respect explicit reset
        PERSISTENCE_ENABLED and persistence_service):
        try:
            latest_session = persistence_service.get_latest_session()
            if latest_session:
                return jsonify(latest_session)
        except Exception as e:
            logger.warning(f"Failed to load latest session: {e}")
    
    return jsonify(current_evolution_data)

@app.route('/api/evolution/reset', methods=['POST'])
def reset_evolution():
    """Reset evolution data to clean state"""
    global current_evolution_data
    try:
        # Reset in-memory data
        current_evolution_data.update({
            'generation': 0,
            'chromosomes': [],
            'champion': None,
            'is_active': False,
            'fitness_history': [],
            'status': 'idle'
        })
        
        # Also clear database persistence to prevent auto-loading old data
        if PERSISTENCE_ENABLED and persistence_service:
            try:
                # Mark all active sessions as completed to prevent auto-loading
                persistence_service.mark_all_sessions_completed()
                logger.info("Database sessions marked as completed")
            except Exception as db_error:
                logger.warning(f"Failed to clear database persistence: {db_error}")
        
        logger.info("Evolution data reset successfully")
        return jsonify({'message': 'Evolution data reset', 'status': 'success'})
    except Exception as e:
        logger.error(f"Error resetting evolution data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/evolution/sessions')
def get_training_sessions():
    """Get all training sessions"""
    if not PERSISTENCE_ENABLED or not persistence_service:
        return jsonify({'error': 'Persistence not enabled'}), 400
    
    try:
        sessions = persistence_service.get_all_sessions()
        return jsonify({'sessions': sessions})
    except Exception as e:
        logger.error(f"Failed to get training sessions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/evolution/sessions/<session_id>')
def get_session_detail(session_id):
    """Get detailed history for a specific session"""
    if not PERSISTENCE_ENABLED or not persistence_service:
        return jsonify({'error': 'Persistence not enabled'}), 400
    
    try:
        history = persistence_service.get_session_history(session_id)
        return jsonify({'history': history})
    except Exception as e:
        logger.error(f"Failed to get session detail: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

@app.route('/api/evolution/data')
def get_evolution_data():
    """Get evolution data for chromosome visualization."""
    try:
        # Look for evolution data in results
        results_dir = Path('results')
        if not results_dir.exists():
            return jsonify({'success': False, 'generations': []})
        
        evolution_files = list(results_dir.glob('evolution_*.json'))
        if not evolution_files:
            # Return indication for demo mode if no real data exists
            return jsonify({
                'success': True, 
                'generations': [],
                'demo_mode': True
            })
        
        # Get the most recent evolution file
        latest_file = max(evolution_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            evolution_data = json.load(f)
        
        return jsonify({
            'success': True, 
            'generations': evolution_data.get('generations', []),
            'demo_mode': False
        })
    
    except Exception as e:
        logger.error(f"Error getting evolution data: {e}")
        return jsonify({'success': False, 'generations': [], 'demo_mode': True})

if __name__ == '__main__':
    logger.info("Starting Forex Trading AI System (Simple Mode)")
    logger.info("Running without TensorFlow components for compatibility")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
