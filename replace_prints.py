import re
import glob

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # If already using logger, skip
    if 'import logger' in content or 'from utils.logger' in content:
        # Check if we need to continue
        pass

    # Add logger import if there are print statements
    if 'print(' in content:
        if 'from utils.logger import logger' not in content:
            # find first import
            content = re.sub(r'^(import .*?\n|from .*? import .*?\n)', r'\1from utils.logger import logger\n', content, count=1)
        
        # We will replace some specific print blocks manually or using regex
        # But wait, python's regex for print is hard because of nested brackets.
        # I will replace specific patterns.

