with open("src/client.py", "r") as f:
    lines = f.readlines()

if "from utils.logger import logger\n" not in lines:
    lines.insert(6, "from utils.logger import logger\n")

for i in range(len(lines)):
    if "print(f\"  Client {self.client_id}" in lines[i]:
        lines[i] = lines[i].replace("print(", "logger.debug(")
    elif "print(f\"  WARNING:" in lines[i]:
        lines[i] = lines[i].replace("print(", "logger.warning(")

with open("src/client.py", "w") as f:
    f.writelines(lines)
