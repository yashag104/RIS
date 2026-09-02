import re

def replace_in_file(filepath, replacements, import_stmt):
    with open(filepath, "r") as f:
        content = f.read()
    
    if import_stmt and import_stmt not in content:
        content = re.sub(r'(import .*?\n)', r'\1' + import_stmt + '\n', content, count=1)
        
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(filepath, "w") as f:
        f.write(content)

replace_in_file(
    "src/dataset_utils.py", 
    [('print(f"Datasets saved to {save_path}")', 'logger.info(f"Datasets saved to {save_path}")')], 
    "from utils.logger import logger"
)

replace_in_file(
    "src/channel_model.py", 
    [
        ('print("WARNING: DeepMIMOv3 not found. Place the DeepMIMOv3 folder in the project root.")', 'logger.warning("DeepMIMOv3 not found. Place the DeepMIMOv3 folder in the project root.")'),
        ('print(f"DeepMIMO generation failed: {e}")', 'logger.error(f"DeepMIMO generation failed: {e}")'),
        ('print("Falling back to synthetic Rician channel model.")', 'logger.info("Falling back to synthetic Rician channel model.")')
    ], 
    "from utils.logger import logger"
)
