import re
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
# Regular expression to match the required patterns
pattern = re.compile(r"Epoch\s+(\d+).*?Loss:\s+(\d+\.\d+)")

# Dictionary to store the loss values for each epoch
epoch_losses = defaultdict(list)

# Read the log file
result_dir = Path('./result')
for d in result_dir.glob('*_train'):
    with open(d / 'log.txt', 'r') as file:
        step = 0
        for line in tqdm(file, desc='Parsing log file'):
            match = pattern.search(line)
            if match:
                step += 1
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epoch_losses[epoch].append(loss)

# Calculate mean loss for each epoch
mean_losses = {epoch: sum(losses) / len(losses) for epoch, losses in epoch_losses.items()}

# Create a DataFrame
df = pd.DataFrame(list(mean_losses.items()), columns=['Epoch', 'Loss'])
df.to_csv('loss.csv', index=False)
print(df)
