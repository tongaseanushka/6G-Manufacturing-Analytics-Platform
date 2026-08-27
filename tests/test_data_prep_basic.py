"""
Basic unit tests for data preparation module.

Tests the Network Quality Band classification logic and basic data preparation functionality.
"""

import pytest
import sys
import os

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_prep import classify_network_quality


class TestNetworkQualityClassification:
    """Test cases for classify_network_quality function."""
    
    def test_high_quality_both_thresholds_met(self):
        """Test High quality when both latency < 20ms AND packet_loss < 1%."""
        assert classify_network_quality(15.0, 0.5) == 'High'
        assert classify_network_quality(10.0, 0.1) == 'High'
        assert classify_network_quality(19.9, 0.9) == 'High'
    
    def test_medium_quality_both_thresholds_met(self):
        """Test Medium quality when latency < 50ms AND packet_loss < 5%."""
        assert classify_network_quality(35.0, 3.0) == 'Medium'
        assert classify_network_quality(25.0, 2.0) == 'Medium'
        assert classify_network_quality(49.9, 4.9) == 'Medium'
    
    def test_low_quality_high_latency(self):
        """Test Low quality when latency >= 50ms."""
        assert classify_network_quality(60.0, 2.0) == 'Low'
        assert classify_network_quality(50.0, 0.5) == 'Low'
        assert classify_network_quality(100.0, 1.0) == 'Low'
    
    def test_low_quality_high_packet_loss(self):
        """Test Low quality when packet_loss >= 5%."""
        assert classify_network_quality(25.0, 6.0) == 'Low'
        assert classify_network_quality(15.0, 5.0) == 'Low'
        assert classify_network_quality(10.0, 10.0) == 'Low'
    
    def test_low_quality_both_high(self):
        """Test Low quality when both latency and packet_loss are high."""
        assert classify_network_quality(60.0, 8.0) == 'Low'
        assert classify_network_quality(100.0, 10.0) == 'Low'
    
    def test_boundary_cases_high_to_medium(self):
        """Test boundary between High and Medium quality (20ms, 1%)."""
        # Exactly at boundary should NOT be High
        assert classify_network_quality(20.0, 0.5) == 'Medium'
        assert classify_network_quality(15.0, 1.0) == 'Medium'
        assert classify_network_quality(20.0, 1.0) == 'Medium'
        
        # Just below boundary should be High
        assert classify_network_quality(19.999, 0.999) == 'High'
    
    def test_boundary_cases_medium_to_low(self):
        """Test boundary between Medium and Low quality (50ms, 5%)."""
        # Exactly at boundary should be Low
        assert classify_network_quality(50.0, 3.0) == 'Low'
        assert classify_network_quality(30.0, 5.0) == 'Low'
        assert classify_network_quality(50.0, 5.0) == 'Low'
        
        # Just below boundary should be Medium
        assert classify_network_quality(49.999, 4.999) == 'Medium'
    
    def test_extreme_values(self):
        """Test classification with extreme values."""
        # Very low values - should be High
        assert classify_network_quality(0.0, 0.0) == 'High'
        assert classify_network_quality(5.0, 0.1) == 'High'
        
        # Very high values - should be Low
        assert classify_network_quality(200.0, 50.0) == 'Low'
        assert classify_network_quality(1000.0, 100.0) == 'Low'
    
    def test_classification_is_deterministic(self):
        """Test that classification produces consistent results."""
        # Same inputs should always produce same output
        for _ in range(5):
            assert classify_network_quality(15.0, 0.5) == 'High'
            assert classify_network_quality(35.0, 3.0) == 'Medium'
            assert classify_network_quality(60.0, 2.0) == 'Low'
    
    def test_all_inputs_produce_valid_output(self):
        """Test that all valid inputs produce one of the three categories."""
        valid_categories = {'Low', 'Medium', 'High'}
        
        test_cases = [
            (0, 0), (5, 0.5), (15, 0.9), (20, 1),
            (25, 2), (35, 3), (49, 4.9), (50, 5),
            (60, 6), (100, 10), (200, 20)
        ]
        
        for latency, packet_loss in test_cases:
            result = classify_network_quality(latency, packet_loss)
            assert result in valid_categories, \
                f"classify_network_quality({latency}, {packet_loss}) returned '{result}', " \
                f"expected one of {valid_categories}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
