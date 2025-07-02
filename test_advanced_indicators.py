#!/usr/bin/env python3
"""
Test script to verify advanced technical indicators (Ichimoku and Fibonacci)
"""

import pandas as pd
import numpy as np
from technical_indicators_native import TechnicalIndicatorCalculatorNative
from evolutionary_feature_selector import EvolutionaryFeatureSelector, FeatureGene
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n_periods = 200
    
    # Generate sample price data with realistic patterns
    close_prices = []
    high_prices = []
    low_prices = []
    open_prices = []
    volumes = []
    
    base_price = 100.0
    for i in range(n_periods):
        # Random walk with slight upward trend
        change = np.random.normal(0, 0.02) + 0.0005  # 0.05% daily upward drift
        base_price *= (1 + change)
        
        # Create OHLC from close price
        close = base_price
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = low + (high - low) * np.random.random()
        volume = np.random.randint(1000, 10000)
        
        open_prices.append(open_price)
        high_prices.append(high)
        low_prices.append(low)
        close_prices.append(close)
        volumes.append(volume)
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    # Add dates
    data.index = pd.date_range(start='2024-01-01', periods=n_periods, freq='D')
    
    logger.info(f"Created sample data with {len(data)} periods")
    logger.info(f"Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    return data

def test_ichimoku_indicators():
    """Test Ichimoku Cloud indicators."""
    logger.info("=" * 50)
    logger.info("TESTING ICHIMOKU CLOUD INDICATORS")
    logger.info("=" * 50)
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize calculator
    calculator = TechnicalIndicatorCalculatorNative(data)
    
    # Test Ichimoku calculation
    ichimoku_data = calculator.ichimoku()
    
    if ichimoku_data is not None:
        logger.info(f"✅ Ichimoku calculation successful")
        logger.info(f"Ichimoku columns: {list(ichimoku_data.columns)}")
        logger.info(f"Ichimoku data shape: {ichimoku_data.shape}")
        
        # Display some sample values
        recent_data = ichimoku_data.tail(5)
        logger.info("\nRecent Ichimoku values:")
        for col in ichimoku_data.columns:
            if not col.startswith('ICHIMOKU_PRICE_'):  # Skip binary columns for readability
                latest_val = recent_data[col].iloc[-1]
                if pd.notna(latest_val):
                    logger.info(f"  {col}: {latest_val:.4f}")
        
        return True
    else:
        logger.error("❌ Ichimoku calculation failed")
        return False

def test_fibonacci_indicators():
    """Test Fibonacci retracement and extension indicators."""
    logger.info("=" * 50)
    logger.info("TESTING FIBONACCI INDICATORS")
    logger.info("=" * 50)
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize calculator
    calculator = TechnicalIndicatorCalculatorNative(data)
    
    # Test Fibonacci retracement
    fib_retracement = calculator.fibonacci_retracement()
    
    if fib_retracement is not None:
        logger.info(f"✅ Fibonacci retracement calculation successful")
        logger.info(f"Fibonacci retracement columns: {list(fib_retracement.columns)}")
        logger.info(f"Fibonacci retracement shape: {fib_retracement.shape}")
        
        # Display some sample values
        recent_data = fib_retracement.tail(5)
        key_columns = ['FIB_382_UP', 'FIB_618_UP', 'FIB_PCT_RETRACEMENT']
        logger.info("\nRecent Fibonacci retracement values:")
        for col in key_columns:
            if col in recent_data.columns:
                latest_val = recent_data[col].iloc[-1]
                if pd.notna(latest_val):
                    logger.info(f"  {col}: {latest_val:.4f}")
    else:
        logger.error("❌ Fibonacci retracement calculation failed")
        return False
    
    # Test Fibonacci extensions
    fib_extensions = calculator.fibonacci_extensions()
    
    if fib_extensions is not None:
        logger.info(f"✅ Fibonacci extensions calculation successful")
        logger.info(f"Fibonacci extensions columns: {list(fib_extensions.columns)}")
        logger.info(f"Fibonacci extensions shape: {fib_extensions.shape}")
        
        return True
    else:
        logger.error("❌ Fibonacci extensions calculation failed")
        return False

def test_pivot_points():
    """Test Pivot Points indicators."""
    logger.info("=" * 50)
    logger.info("TESTING PIVOT POINTS")
    logger.info("=" * 50)
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize calculator
    calculator = TechnicalIndicatorCalculatorNative(data)
    
    # Test different pivot point methods
    methods = ['standard', 'fibonacci', 'camarilla']
    
    for method in methods:
        pivot_data = calculator.pivot_points(method=method)
        
        if pivot_data is not None:
            logger.info(f"✅ {method.title()} pivot points calculation successful")
            logger.info(f"{method.title()} pivot columns: {list(pivot_data.columns)}")
            
            # Display some sample values
            recent_data = pivot_data.tail(1)
            key_columns = [col for col in pivot_data.columns if any(x in col for x in ['_PP', '_R1', '_S1'])]
            logger.info(f"Recent {method} pivot values:")
            for col in key_columns[:3]:  # Show first 3 key levels
                if col in recent_data.columns:
                    latest_val = recent_data[col].iloc[-1]
                    if pd.notna(latest_val):
                        logger.info(f"  {col}: {latest_val:.4f}")
        else:
            logger.error(f"❌ {method.title()} pivot points calculation failed")
            return False
    
    return True

def test_evolutionary_integration():
    """Test that advanced indicators work with evolutionary feature selector."""
    logger.info("=" * 50)
    logger.info("TESTING EVOLUTIONARY INTEGRATION")
    logger.info("=" * 50)
    
    # Create sample data
    data = create_sample_data()
    
    # Test creating advanced indicator genes
    advanced_genes = [
        FeatureGene('ICHIMOKU_TENKAN', {'conversion_periods': 9, 'base_periods': 26, 'displacement': 26}),
        FeatureGene('FIB_618_UP', {'lookback_periods': 100}),
        FeatureGene('PIVOT_STANDARD_PP', {}),
        FeatureGene('FIB_EXT_1618_UP', {'lookback_periods': 100}),
    ]
    
    success_count = 0
    for gene in advanced_genes:
        try:
            logger.info(f"Testing gene: {gene}")
            
            # Create calculator
            calculator = TechnicalIndicatorCalculatorNative(data)
            
            # Test feature calculation
            if gene.indicator_type.startswith("ICHIMOKU_"):
                ichimoku_data = calculator.ichimoku(
                    conversion_periods=gene.parameters.get('conversion_periods', 9),
                    base_periods=gene.parameters.get('base_periods', 26),
                    displacement=gene.parameters.get('displacement', 26)
                )
                if ichimoku_data is not None and gene.indicator_type in ichimoku_data.columns:
                    feature_series = ichimoku_data[gene.indicator_type]
                    if len(feature_series.dropna()) > 0:
                        logger.info(f"  ✅ {gene.indicator_type} - {len(feature_series.dropna())} valid values")
                        success_count += 1
                    else:
                        logger.warning(f"  ⚠️ {gene.indicator_type} - No valid values")
                else:
                    logger.error(f"  ❌ {gene.indicator_type} - Calculation failed")
                    
            elif gene.indicator_type.startswith("FIB_"):
                if gene.indicator_type.startswith("FIB_EXT_"):
                    fib_data = calculator.fibonacci_extensions(
                        lookback_periods=gene.parameters.get('lookback_periods', 100)
                    )
                else:
                    fib_data = calculator.fibonacci_retracement(
                        lookback_periods=gene.parameters.get('lookback_periods', 100)
                    )
                    
                if fib_data is not None and gene.indicator_type in fib_data.columns:
                    feature_series = fib_data[gene.indicator_type]
                    if len(feature_series.dropna()) > 0:
                        logger.info(f"  ✅ {gene.indicator_type} - {len(feature_series.dropna())} valid values")
                        success_count += 1
                    else:
                        logger.warning(f"  ⚠️ {gene.indicator_type} - No valid values")
                else:
                    logger.error(f"  ❌ {gene.indicator_type} - Calculation failed")
                    
            elif gene.indicator_type.startswith("PIVOT_"):
                method = gene.indicator_type.split('_')[1].lower()
                pivot_data = calculator.pivot_points(method=method)
                
                if pivot_data is not None and gene.indicator_type in pivot_data.columns:
                    feature_series = pivot_data[gene.indicator_type]
                    if len(feature_series.dropna()) > 0:
                        logger.info(f"  ✅ {gene.indicator_type} - {len(feature_series.dropna())} valid values")
                        success_count += 1
                    else:
                        logger.warning(f"  ⚠️ {gene.indicator_type} - No valid values")
                else:
                    logger.error(f"  ❌ {gene.indicator_type} - Calculation failed")
                    
        except Exception as e:
            logger.error(f"  ❌ {gene.indicator_type} - Exception: {e}")
    
    logger.info(f"\nEvolutionary integration test: {success_count}/{len(advanced_genes)} genes successful")
    return success_count == len(advanced_genes)

def main():
    """Run all tests for advanced indicators."""
    logger.info("🚀 STARTING ADVANCED INDICATORS TEST SUITE")
    logger.info("=" * 60)
    
    test_results = []
    
    # Run individual tests
    test_results.append(("Ichimoku Cloud", test_ichimoku_indicators()))
    test_results.append(("Fibonacci Indicators", test_fibonacci_indicators()))
    test_results.append(("Pivot Points", test_pivot_points()))
    test_results.append(("Evolutionary Integration", test_evolutionary_integration()))
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        logger.info("🎉 ALL ADVANCED INDICATORS WORKING SUCCESSFULLY!")
        logger.info("💡 The genetic algorithm can now discover Ichimoku, Fibonacci, and Pivot Point strategies!")
    else:
        logger.warning("⚠️ Some tests failed. Please check the logs above.")
    
    return passed == len(test_results)

if __name__ == "__main__":
    main()