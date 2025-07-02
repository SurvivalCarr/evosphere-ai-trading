from flask import Flask, render_template, jsonify, request, send_file
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import traceback
import threading
import time

from main_ea_drl_forex_trader import ForexTradingSystem
from utils.data_loader import DataLoader
from config.settings import DEFAULT_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'forex_trading_ai_secret_key'

# Global variables for system state
trading_system = None
system_status = {
    'status': 'idle',
    'progress': 0,
    'message': 'System ready',
    'results': None,
    'error': None
}

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get current system status."""
    return jsonify(system_status)

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update system configuration."""
    if request.method == 'GET':
        return jsonify(DEFAULT_CONFIG)
    
    elif request.method == 'POST':
        try:
            new_config = request.json
            # Validate and update configuration
            updated_config = DEFAULT_CONFIG.copy()
            updated_config.update(new_config)
            
            # Save to file
            config_file = 'config/current_config.json'
            os.makedirs('config', exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(updated_config, f, indent=2)
            
            return jsonify({'success': True, 'config': updated_config})
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/data/upload', methods=['POST'])
def upload_data():
    """Upload market data file."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = f"uploaded_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join('data', filename)
        os.makedirs('data', exist_ok=True)
        file.save(filepath)
        
        # Validate data
        try:
            df = pd.read_csv(filepath)
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return jsonify({
                    'success': False, 
                    'error': f'Missing required columns: {missing_cols}'
                }), 400
            
            data_info = {
                'filename': filename,
                'filepath': filepath,
                'rows': len(df),
                'columns': list(df.columns),
                'date_range': {
                    'start': str(df.index[0]) if hasattr(df.index, '__getitem__') else 'N/A',
                    'end': str(df.index[-1]) if hasattr(df.index, '__getitem__') else 'N/A'
                }
            }
            
            return jsonify({'success': True, 'data_info': data_info})
            
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid data format: {e}'}), 400
            
    except Exception as e:
        logger.error(f"Error uploading data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/data/fetch')
def fetch_data():
    """Fetch data from Yahoo Finance."""
    try:
        symbol = request.args.get('symbol', 'EURUSD=X')
        period = request.args.get('period', '1y')
        interval = request.args.get('interval', '1h')
        
        data_loader = DataLoader()
        df = data_loader.fetch_yahoo_data(symbol, period=period, interval=interval)
        
        if df is not None and not df.empty:
            # Save fetched data
            filename = f"fetched_{symbol.replace('=', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join('data', filename)
            os.makedirs('data', exist_ok=True)
            df.to_csv(filepath)
            
            data_info = {
                'filename': filename,
                'filepath': filepath,
                'rows': len(df),
                'columns': list(df.columns),
                'symbol': symbol,
                'period': period,
                'interval': interval
            }
            
            return jsonify({'success': True, 'data_info': data_info})
        else:
            return jsonify({'success': False, 'error': 'No data fetched'}), 400
            
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start the training process."""
    global trading_system, system_status
    
    try:
        params = request.json or {}
        data_file = params.get('data_file')
        symbol = params.get('symbol', 'EURUSD=X')
        config_updates = params.get('config', {})
        
        # Update system status
        system_status.update({
            'status': 'running',
            'progress': 0,
            'message': 'Initializing training...',
            'error': None
        })
        
        # Start training in background thread
        def run_training():
            try:
                # Load configuration
                config = DEFAULT_CONFIG.copy()
                config.update(config_updates)
                
                # Initialize trading system
                global trading_system
                trading_system = ForexTradingSystem(config)
                
                # Update status
                system_status['message'] = 'Loading market data...'
                system_status['progress'] = 10
                
                # Load market data
                market_data = trading_system.load_market_data(data_file, symbol)
                train_data, val_data = trading_system.prepare_data_splits(market_data)
                
                # Update status
                system_status['message'] = 'Running evolutionary selection...'
                system_status['progress'] = 20
                
                # Run evolutionary selection
                best_chromosome, best_features = trading_system.run_evolutionary_selection(train_data, val_data)
                
                # Update status
                system_status['message'] = 'Training final agent...'
                system_status['progress'] = 60
                
                # Train final agent
                final_agent = trading_system.train_final_agent(best_features, market_data)
                
                # Update status
                system_status['message'] = 'Evaluating system...'
                system_status['progress'] = 80
                
                # Evaluate system
                evaluation_results = trading_system.evaluate_system()
                
                # Update status
                system_status['message'] = 'Saving results...'
                system_status['progress'] = 90
                
                # Save results
                trading_system.save_results('results')
                trading_system.create_performance_plots('results')
                
                # Complete
                system_status.update({
                    'status': 'completed',
                    'progress': 100,
                    'message': 'Training completed successfully',
                    'results': {
                        'evolution_results': trading_system.evolution_results,
                        'training_results': trading_system.training_results,
                        'evaluation_results': evaluation_results,
                        'best_features': [str(gene) for gene in best_chromosome.genes] if best_chromosome else []
                    }
                })
                
            except Exception as e:
                logger.error(f"Training error: {e}")
                logger.debug(traceback.format_exc())
                system_status.update({
                    'status': 'error',
                    'progress': 0,
                    'message': f'Training failed: {str(e)}',
                    'error': str(e)
                })
        
        # Start training thread
        training_thread = threading.Thread(target=run_training)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({'success': True, 'message': 'Training started'})
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        system_status.update({
            'status': 'error',
            'message': f'Failed to start training: {str(e)}',
            'error': str(e)
        })
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop the training process."""
    global system_status
    
    try:
        system_status.update({
            'status': 'idle',
            'progress': 0,
            'message': 'Training stopped by user',
            'error': None
        })
        
        return jsonify({'success': True, 'message': 'Training stopped'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/results/latest')
def get_latest_results():
    """Get the latest training results."""
    try:
        if system_status.get('results'):
            return jsonify({'success': True, 'results': system_status['results']})
        else:
            return jsonify({'success': False, 'error': 'No results available'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/results/files')
def list_result_files():
    """List available result files."""
    try:
        results_dir = 'results'
        if not os.path.exists(results_dir):
            return jsonify({'success': True, 'files': []})
        
        files = []
        for filename in os.listdir(results_dir):
            filepath = os.path.join(results_dir, filename)
            if os.path.isfile(filepath):
                files.append({
                    'filename': filename,
                    'size': os.path.getsize(filepath),
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                })
        
        files.sort(key=lambda x: x['modified'], reverse=True)
        return jsonify({'success': True, 'files': files})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/results/download/<filename>')
def download_result_file(filename):
    """Download a result file."""
    try:
        filepath = os.path.join('results', filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(filepath, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/live/data')
def get_live_data():
    """Get live market data for monitoring."""
    try:
        symbol = request.args.get('symbol', 'EURUSD=X')
        
        # Fetch recent data
        data_loader = DataLoader()
        df = data_loader.fetch_yahoo_data(symbol, period='1d', interval='1m')
        
        if df is not None and not df.empty:
            # Get last few data points
            recent_data = df.tail(60)  # Last 60 minutes
            
            data = {
                'timestamps': recent_data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'close_prices': recent_data['Close'].tolist(),
                'volumes': recent_data['Volume'].tolist(),
                'current_price': float(recent_data['Close'].iloc[-1]),
                'price_change': float(recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[-2]) if len(recent_data) > 1 else 0,
                'symbol': symbol
            }
            
            return jsonify({'success': True, 'data': data})
        else:
            return jsonify({'success': False, 'error': 'No live data available'}), 404
            
    except Exception as e:
        logger.error(f"Error fetching live data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """Make a trading prediction with the trained model."""
    try:
        if trading_system is None or trading_system.dqn_agent is None:
            return jsonify({'success': False, 'error': 'No trained model available'}), 400
        
        # Get current market data
        symbol = request.json.get('symbol', 'EURUSD=X')
        data_loader = DataLoader()
        current_data = data_loader.fetch_yahoo_data(symbol, period='1d', interval='1h')
        
        if current_data is None or current_data.empty:
            return jsonify({'success': False, 'error': 'No current data available'}), 400
        
        # Generate features for current data
        if trading_system.best_chromosome:
            features = trading_system.best_chromosome.generate_features_df(current_data)
            
            # Prepare state for prediction
            if not features.empty:
                # Use latest data point for prediction
                latest_features = features.tail(trading_system.config['LOOKBACK_WINDOW'])
                
                # Pad if necessary
                if len(latest_features) < trading_system.config['LOOKBACK_WINDOW']:
                    padding_rows = trading_system.config['LOOKBACK_WINDOW'] - len(latest_features)
                    first_row = latest_features.iloc[0:1]
                    padding = pd.concat([first_row] * padding_rows, ignore_index=True)
                    latest_features = pd.concat([padding, latest_features], ignore_index=True)
                
                # Convert to numpy array
                state = latest_features.values
                state = np.reshape(state, [1, *state.shape])
                
                # Make prediction
                trading_system.dqn_agent.epsilon = 0  # No exploration for prediction
                action, trade_size = trading_system.dqn_agent.act(state)
                
                # Get Q-values for confidence
                q_values = trading_system.dqn_agent.q_network.predict(state, verbose=0)[0]
                confidence = float(np.max(q_values))
                
                # Map action to recommendation
                action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
                recommendation = action_map.get(action, 'HOLD')
                
                prediction = {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'trade_size': float(trade_size),
                    'current_price': float(current_data['Close'].iloc[-1]),
                    'q_values': q_values.tolist(),
                    'features_used': list(features.columns)
                }
                
                return jsonify({'success': True, 'prediction': prediction})
            else:
                return jsonify({'success': False, 'error': 'No features generated'}), 400
        else:
            return jsonify({'success': False, 'error': 'No trained chromosome available'}), 400
            
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
