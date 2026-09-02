import os
import re

files_to_fix = [
    "baselines/admm_optimizer.py",
    "baselines/alternating_optimization.py",
    "baselines/centralized_learning.py",
    "baselines/sca_optimizer.py",
    "baselines/sdr_optimizer.py",
    "utils/report_generator.py"
]

for filepath in files_to_fix:
    if not os.path.exists(filepath): continue
    
    with open(filepath, "r") as f:
        content = f.read()
        
    if "from utils.logger import logger" not in content:
        content = re.sub(r'(import .*?\n)', r'\1from utils.logger import logger\n', content, count=1)
        
    content = re.sub(r'print\(', r'logger.info(', content)
    
    with open(filepath, "w") as f:
        f.write(content)
        
