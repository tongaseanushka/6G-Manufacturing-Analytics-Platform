"""
Unit tests for data preparation module.

Tests the data loading, validation, and error handling functionality
of the data_prep module.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_prep import load_and_prepare_dataset, DataValidationError


class TestDataValidationError:
    """Test the custom DataValidationError exception."""
    
    def test_exception_can_be_raised(self):
        """Test that DataValidationError can be instantiated and raised."""
        with pytest.raises(DataValidationError, match="Test error message"):
            raise DataValidationError("Test error message")


class TestLoadAndPrepareDataset:
    """Test the load_and_prepare_dataset function."""
    
    def test_file_not_found_error(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_and_prepare_dataset("nonexistent_file.csv")
    
    def test_row_count_validation(self):
        """Test that DataValidationError is raised when row count != 100,000."""
        # Create a temporary CSV with wrong row count
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("Timestamp,Machine_ID,Network_Latency_ms,Packet_Loss_%,Efficiency_Status,")
            f.write("Production_Speed,Error_Rate,Quality_Control_Defect_Rate,Operation_Mode,")
            f.write("Col10,Col11,Col12,Col13,Col14\n")
            
            # Write only 100 rows instead of 100,000
            for i in range(100):
                f.write(f"01-01-2025,M{i},10.0,0.5,Low,100,0.01,0.02,Active,1,2,3,4,5\n")
            
            temp_path = f.name
        
        try:
            with pytest.raises(DataValidationError, match="Row count validation failed"):
                load_and_prepare_dataset(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_column_count_validation(self):
        """Test that DataValidationError is raised when column count != 14."""
        # Create a temporary CSV with wrong column count
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            # Only 10 columns instead of 14
            f.write("Timestamp,Machine_ID,Network_Latency_ms,Packet_Loss_%,Efficiency_Status,")
            f.write("Production_Speed,Error_Rate,Quality_Control_Defect_Rate,Operation_Mode,Col10\n")
            
            for i in range(100000):
                f.write(f"01-01-2025,M{i},10.0,0.5,Low,100,0.01,0.02,Active,1\n")
            
            temp_path = f.name
        
        try:
            with pytest.raises(DataValidationError, match="Column count validation failed"):
                load_and_prepare_dataset(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_missing_required_column(self):
        """Test that DataValidationError is raised for missing required columns."""
        # Create a temporary CSV missing Network_Latency_ms but with correct column count
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            # 14 columns but Network_Latency_ms is replaced with DummyCol
            f.write("Timestamp,Machine_ID,DummyCol,Packet_Loss_%,Efficiency_Status,")
            f.write("Production_Speed,Error_Rate,Quality_Control_Defect_Rate,Operation_Mode,")
            f.write("Col10,Col11,Col12,Col13,Col14\n")
            
            for i in range(100000):
                f.write(f"01-01-2025,M{i},999,0.5,Low,100,0.01,0.02,Active,1,2,3,4,5\n")
            
            temp_path = f.name
        
        try:
            with pytest.raises(DataValidationError, match="Missing required column.*Network_Latency_ms"):
                load_and_prepare_dataset(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_invalid_efficiency_status(self):
        """Test that DataValidationError is raised for invalid Efficiency_Status values."""
        # Create a temporary CSV with invalid efficiency status
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("Timestamp,Machine_ID,Network_Latency_ms,Packet_Loss_%,Efficiency_Status,")
            f.write("Production_Speed,Error_Rate,Quality_Control_Defect_Rate,Operation_Mode,")
            f.write("Col10,Col11,Col12,Col13,Col14\n")
            
            for i in range(100000):
                # Use "Invalid" instead of Low/Medium/High
                status = "Invalid" if i % 10 == 0 else "Low"
                f.write(f"01-01-2025,M{i},10.0,0.5,{status},100,0.01,0.02,Active,1,2,3,4,5\n")
            
            temp_path = f.name
        
        try:
            with pytest.raises(DataValidationError, match="Invalid Efficiency_Status value"):
                load_and_prepare_dataset(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_invalid_operation_mode(self):
        """Test that DataValidationError is raised for invalid Operation_Mode values."""
        # Create a temporary CSV with invalid operation mode
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("Timestamp,Machine_ID,Network_Latency_ms,Packet_Loss_%,Efficiency_Status,")
            f.write("Production_Speed,Error_Rate,Quality_Control_Defect_Rate,Operation_Mode,")
            f.write("Col10,Col11,Col12,Col13,Col14\n")
            
            for i in range(100000):
                # Use "Invalid" instead of Active/Idle/Maintenance
                mode = "Invalid" if i % 10 == 0 else "Active"
                f.write(f"01-01-2025,M{i},10.0,0.5,Low,100,0.01,0.02,{mode},1,2,3,4,5\n")
            
            temp_path = f.name
        
        try:
            with pytest.raises(DataValidationError, match="Invalid Operation_Mode value"):
                load_and_prepare_dataset(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_negative_network_latency(self):
        """Test that DataValidationError is raised for negative Network_Latency_ms."""
        # Create a temporary CSV with negative latency
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("Timestamp,Machine_ID,Network_Latency_ms,Packet_Loss_%,Efficiency_Status,")
            f.write("Production_Speed,Error_Rate,Quality_Control_Defect_Rate,Operation_Mode,")
            f.write("Col10,Col11,Col12,Col13,Col14\n")
            
            for i in range(100000):
                latency = -5.0 if i % 10 == 0 else 10.0
                f.write(f"01-01-2025,M{i},{latency},0.5,Low,100,0.01,0.02,Active,1,2,3,4,5\n")
            
            temp_path = f.name
        
        try:
            with pytest.raises(DataValidationError, match="Network_Latency_ms must be non-negative"):
                load_and_prepare_dataset(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_packet_loss_out_of_range(self):
        """Test that DataValidationError is raised for Packet_Loss_% outside [0, 100]."""
        # Create a temporary CSV with packet loss > 100
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("Timestamp,Machine_ID,Network_Latency_ms,Packet_Loss_%,Efficiency_Status,")
            f.write("Production_Speed,Error_Rate,Quality_Control_Defect_Rate,Operation_Mode,")
            f.write("Col10,Col11,Col12,Col13,Col14\n")
            
            for i in range(100000):
                loss = 150.0 if i % 10 == 0 else 0.5
                f.write(f"01-01-2025,M{i},10.0,{loss},Low,100,0.01,0.02,Active,1,2,3,4,5\n")
            
            temp_path = f.name
        
        try:
            with pytest.raises(DataValidationError, match="Packet_Loss_% must be in range"):
                load_and_prepare_dataset(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_date_parsing_format(self):
        """Test that dates are parsed correctly with DD-MM-YYYY format."""
        # Create a valid temporary CSV with ambiguous date
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("Timestamp,Machine_ID,Network_Latency_ms,Packet_Loss_%,Efficiency_Status,")
            f.write("Production_Speed,Error_Rate,Quality_Control_Defect_Rate,Operation_Mode,")
            f.write("Col10,Col11,Col12,Col13,Col14\n")
            
            for i in range(100000):
                # Use date like 15-03-2025 which should be March 15, not month 15
                f.write(f"15-03-2025,M{i},10.0,0.5,Low,100,0.01,0.02,Active,1,2,3,4,5\n")
            
            temp_path = f.name
        
        try:
            df = load_and_prepare_dataset(temp_path)
            
            # Verify date was parsed correctly as March 15, 2025
            # Check that it's a datetime type (pandas may use ns or us resolution)
            assert pd.api.types.is_datetime64_any_dtype(df['Timestamp'])
            first_date = df['Timestamp'].iloc[0]
            assert first_date.day == 15
            assert first_date.month == 3
            assert first_date.year == 2025
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
