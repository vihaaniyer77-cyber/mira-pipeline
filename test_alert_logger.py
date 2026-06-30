import os
import shutil
import numpy as np
from alert_logger import AlertLogger

def test_alert_logger():
    test_dir = "test_discoveries"
    
    # Clean up from any previous test
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    logger = AlertLogger(output_dir=test_dir)
    
    # Create a mock image
    mock_img = np.random.normal(10, 2, (100, 100))
    mock_img[50, 50] = 500 # The transient
    
    # Log an alert
    logger.log_alert("Engine A (New)", 50, 50, mock_img, crop_size=20)
    
    # Check that CSV was created
    csv_file = os.path.join(test_dir, "discoveries.csv")
    assert os.path.exists(csv_file)
    
    # Check that the CSV has content
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 2 # Header + 1 data row
        assert "Engine A (New)" in lines[1]
        assert "50,50" in lines[1]
        
    # Check that a PNG was saved
    png_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
    assert len(png_files) == 1
    
    # Clean up
    shutil.rmtree(test_dir)
