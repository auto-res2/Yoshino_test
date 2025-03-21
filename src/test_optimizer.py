import torch
import torch.nn as nn
import torch.nn.functional as F
from optimizer import AdamWAlignOptimizer

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def test_optimizer():
    # Test the optimizer
    model = SimpleModel()
    optimizer = AdamWAlignOptimizer(model.parameters(), lr=0.01)
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))

    # Run a few optimization steps
    for i in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        print(f'Step {i+1}, Loss: {loss.item():.4f}')

    print('Optimizer test completed successfully!')

if __name__ == "__main__":
    test_optimizer()
