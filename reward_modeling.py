import torch
import numpy as np  
import torch.nn as nn
import torch.nn.functional as F
import argparse

def argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--same_model', type=bool, default=True)
    parser.add_argument('--data_path1', type=str, default='data')
    parser.add_argument('--data_path2', type=str, default='data')
    return parser.parse_args()

class RMDataset(torch.utils.data.Dataset):
    def __init__(self, df1,df2, transform=None):
        self.d1 = df1
        self.d2 = df2
        self.transform = transform
    def __len__(self):
        return len(self.d1)
    def __getitem__(self, index):
        trajectory1 = self.d1.iloc[index]
        trajectory2 = self.d2.iloc[index]
        return trajectory1, trajectory2

class reward_model_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 1)
    def forward(IL, action):
        output = torch.cat((IL, action), dim=1)
        output = self.linear(torch.flatten(output))
        return output

if __name__ == __main__:
    args = argparse()
    
    data1 = pd.read_csv(args.data_path1)
    data2 = pd.read_csv(args.data_path2)

    custom_dataset = RMDataset(data1, data2)
    dataloader = DataLoader(custom_dataset, batch_size=args.batch_size, shuffle=True)

    if args.same_model:
        model = reward_model_A()
        model.load_state_dict(torch.load('reward_model_A.pth'))
    else:
        model = reward_model_A()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(args.epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            trag1, trag2 = batch
            output1 = model(trag1['memories'], trag1['logprobs'])
            output2 = model(trag2['memories'], trag2['logprobs'])
            loss = trag1["human1"]*((torch.exp(torch.sum(trag1['rewards'])))/(torch.exp(torch.sum(trag1['rewards'])) + torch.exp(torch.sum(trag2['rewards'])))) + trag2["human2"]*((torch.exp(torch.sum(trag2['rewards'])))/(torch.exp(torch.sum(trag1['rewards'])) + torch.exp(torch.sum(trag2['rewards']))))
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), 'reward_model_A.pth')