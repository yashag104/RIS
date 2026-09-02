import re

with open('test_experiments.py', 'r') as f:
    content = f.read()

# Replace `return True` at the end of try blocks with `pass` and remove the except blocks that return False.
# But it's easier to just append an assert for the returned value.

content = re.sub(r'return True', r'assert True; return True', content)
content = re.sub(r'return False', r'assert False; return False', content)
content = re.sub(r'return all\(checks\)', r'assert all(checks); return all(checks)', content)

with open('test_experiments.py', 'w') as f:
    f.write(content)
