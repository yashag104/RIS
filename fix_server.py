with open("src/server.py", "r") as f:
    content = f.read()

import re
content = re.sub(r'import copy\n', 'import copy\nfrom utils.logger import logger\n', content, count=1)

content = content.replace('print(f"  [Server] Broadcasted model to {len(clients)} clients "', 'logger.info(f"  [Server] Broadcasted model to {len(clients)} clients "')
content = content.replace('print(f"\\n{\'=\'*60}")', 'logger.info(f"\\n{\'=\'*60}")')
content = content.replace('print(f"Round {round_num + 1}/{self.config.FL_ROUNDS} [{self.aggregation_method}]")', 'logger.info(f"Round {round_num + 1}/{self.config.FL_ROUNDS} [{self.aggregation_method}]")')
content = content.replace('print(f"{\'=\'*60}")', 'logger.info(f"{\'=\'*60}")')
content = content.replace('print(f"\\n[Client {client.client_id}] Starting local training...")', 'logger.debug(f"\\n[Client {client.client_id}] Starting local training...")')
content = content.replace('print(f"\\n[Server] Aggregating weights from {len(clients)} clients...")', 'logger.info(f"\\n[Server] Aggregating weights from {len(clients)} clients...")')
content = content.replace('print(f"\\n[Round {round_num + 1} Summary]")', 'logger.info(f"\\n[Round {round_num + 1} Summary]")')
content = content.replace('print(f"  Avg Loss: {round_metric[\'avg_client_loss\']:.6f}")', 'logger.info(f"  Avg Loss: {round_metric[\'avg_client_loss\']:.6f}")')
content = content.replace('print(f"  Total Energy: {round_metric[\'total_energy\']:.6f} J")', 'logger.info(f"  Total Energy: {round_metric[\'total_energy\']:.6f} J")')
content = content.replace('print(f"  Communication: {round_metric[\'total_bytes\'] / 1024:.2f} KB")', 'logger.info(f"  Communication: {round_metric[\'total_bytes\'] / 1024:.2f} KB")')

with open("src/server.py", "w") as f:
    f.write(content)
