"""
Performance tests for data preparation module.

Tests that the data loading completes within performance requirements.
"""

import pytest
import pandas as pd
import tempfile
import os
import time
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_prep import load_and_prepare_dataset


class TestPerformance:
    """Test performance requirements for data loading."""
    
    def test_loading_completes_within_5_seconds(self):
        """
        Test that dataset loading completes within 5 seconds.
        
        Requirement 1.10: THE Data_Prep_Module SHALL complete initial Dataset 
        loading within 5 seconds.
        """
        # Create a temporary CSV with 100,000 rows
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("Timestamp,Machine_ID,Network_Latency_ms,Packet_Loss_%,Efficiency_Status,")
            f.write("Production_Speed,Error_Rate,Quality_Control_Defect_Rate,Operation_Mode,")
            f.write("Col10,Col11,Col12,Col13,Col14\n")
            
            # Write 100,000 rows
            for i in range(100000):
                machine_id = f"M{i % 50}"  # 50 machines
                latency = 10.0 + (i % 100)
                packet_loss = 0.5 + (i % 10) * 0.1
                efficiency = ['Low', 'Medium', 'High'][i % 3]
                operation_mode = ['Active', 'Idle', 'Maintenance'][i % 3]
                day = (i % 28) + 1  # Days 1-28
                date = f"{day:02d}-01-2025"  # Format as DD-MM-YYYY
                
                f.write(f"{date},{machine_id},{latency},{packet_loss},{efficiency},")
                f.write(f"100.0,0.01,0.02,{operation_mode},1,2,3,4,5\n")
            
            temp_path = f.name
        
        try:
            # Measure loading time
            start_time = time.time()
            df = load_and_prepare_dataset(temp_path)
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            
            # Assert loading completed within 5 seconds
            assert elapsed_time < 5.0, f"Loading took {elapsed_time:.2f}s, exceeds 5s requirement"
            
            # Verify data was loaded correctly
            # The function returns 15 columns: 14 original + 1 derived (Network_Quality_Band)
            assert len(df) == 100000
            assert len(df.columns) == 15
            assert 'Network_Quality_Band' in df.columns
            
            print(f"\nPerformance test passed: Loading completed in {elapsed_time:.2f} seconds")
        
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
