import torch
import torch.optim as optim
from torch.distributions.categorical import Categorical
#from torch.utils.tensorboard import SummaryWriter
from distutils.util import strtobool
import time
import os
import gym
from gym.spaces import Dict,Box, Text, Sequence
import numpy as np  
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from htm_pytorch import HTMBlock
import MatterSim
import csv
import numpy as np
import math
import base64
import json
import random
import argparse
from sentence_transformers import SentenceTransformer
import pandas as pd
#from reward_modeling import reward_model_A

def create_df(obs, actions, logprobs, rewards, dones, values):
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="VLNRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-steps", type=int, default=100,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=2,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument('--new_reward_m', type=bool, default=False)
    parser.add_argument('--same_agent', type=bool, default=False)
    parser.add_argument('--init', type=bool, default=False)
    args = parser.parse_args()
    args.env_batch_size = 10
    args.batch_size = int(args.env_batch_size*args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

def load_datasets(splits):
    data = []
    for split in splits:
        assert split in ['train', 'val_seen', 'val_unseen', 'test']
        with open('/root/mount/Matterport3DSimulator/data/splits/R2R_%s.json' % split) as f:
            data += json.load(f)
    return data
    
# env
class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments '''

    def __init__(self, batch_size=10):
        self.image_w = 640
        self.image_h = 480
        self.vfov = 60
        self.batch_size = batch_size
        self.sim = MatterSim.Simulator()
        self.sim.setRenderingEnabled(False)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setBatchSize(self.batch_size)
        self.sim.setCameraResolution(self.image_w, self.image_h)
        self.sim.setCameraVFOV(math.radians(self.vfov))
        self.sim.initialize()
        self.images = np.array([10, 480, 640, 3])

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        self.sim.newEpisode(scanIds, viewpointIds, headings, [0]*self.batch_size)

    def getStates(self):
        ''' Get list of states augmented with precomputed image features. rgb field will be empty. '''
        feature_states = []
        images = [] 
        with h5py.File('/root/mount/Matterport3DSimulator/feat/feat_vit', "r") as f:
                for state in self.sim.getState():
                    long_id = self._make_id(state.scanId, state.location.viewpointId)
                    feature = f[long_id][state.viewIndex]
                    feature_states.append((feature, state))
                    rgb = np.array(state.rgb, copy=False)
                    image = rgb[:, :, ::-1]
                    self.images[i] = image
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        ix = []
        heading = []
        elevation = []
        for i,h,e in actions:
            ix.append(int(i))
            heading.append(float(h))
            elevation.append(float(e))
        self.sim.makeAction(ix, heading, elevation)

class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, batch_size=10, seed=10, splits=['train'], tokenizer=None):
        self.env = EnvBatch(batch_size=batch_size)
        self.data = []
        self.scans = []
        for item in load_datasets(splits):
            # Split multiple instructions into separate entries
            for j,instr in enumerate(item['instructions']):
                self.scans.append(item['scan'])
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instructions'] = instr
                self.data.append(new_item)
        self.scans = set(self.scans)
        self.splits = splits
        self.seed = seed
        self.actions = [
            (0,-1, 0), # left
            (0, 1, 0), # right
            (0, 0, 1), # up
            (0, 0,-1), # down
            (1, 0, 0), # forward
            (0, 0, 0)] # <end>
        random.seed(self.seed)
        random.shuffle(self.data)
        self.ix = 0
        self.batch_size = batch_size
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def _next_minibatch(self):
        batch = self.data[self.ix:self.ix+self.batch_size]
        if len(batch) < self.batch_size:
            random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        self.batch = batch

    def reset_epoch(self):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        self.ix = 0

    def _get_obs(self):
        obs = []
        for i,(fs,state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : fs,
                'step' : state.step,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions']
            })
        return obs

    def reset(self):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch()
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        dones = []
        act = []
        for action in actions:
            act.append(self.actions[action])
            if action[0] == 5:
                dones.append(True)
            else:
                dones.append(False)
        self.env.makeActions(act) 
        return self._get_obs(), dones


class R2RGym(gym.Env):
    def __init__(self, batch_size=10):
        self.batch_size = batch_size
        self.r2r_batch = R2RBatch(batch_size=self.batch_size)
        navigation_space = Dict({
            'viewpointId': Box(low=0, high=float('inf'), shape=(1,), dtype=int),
            'ix': Box(low=0, high=float('inf'), shape=(1,), dtype=int),
            'x': Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=float),
            'y': Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=float),
            'z': Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=float),
            'rel_heading': Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=float),
            'rel_elevation': Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=float),
            'rel_distance': Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=float)
        })
        observation_space = Dict({
            'instr_id': Text(100),
            'scan': Text(100),
            'viewpoint': Text(100),
            'viewIndex': Box(low=0, high=float('inf'), shape=(1,), dtype=int),
            'heading': Box(low=-3.141592, high=3.141592, shape=(1,), dtype=float),
            'elevation': Box(low=-3.141592, high=3.141592, shape=(1,), dtype=float),
            'feature': Box(low=-float('inf'), high=float('inf'), shape=(512,), dtype=float),
            'step': Box(low=0, high=float('inf'), shape=(1,), dtype=int),
            'instructions': Text(1000),
            'navigableLocations': Sequence(navigation_space)
        })
        self.action_space =  self.action_space = gym.spaces.Discrete(6)
        self.current_step = 0
        self.episode_images = {}
        for i in range(self.batch_size):
            self.episode_images[i] = []

    def reset(self):
        obs = self.r2r_batch.reset()
        self.current_step = 0
        self.render()
        return obs

    def step(self, action):
        self.current_step += 1
        obs, done = self.r2r_batch.step(action)
        self.render()
        return obs, done, {}

    def render(self, mode='rgb_array'):
        
        frames = self.capture_current_frame(obs)
        for i,frame in enumerate(iterable=frames):
            self.episode_images[i].append(frame)

    def capture_current_frame(self):
        frames = self.r2r_batch.images
        return frames
    
    def save_videos(self, output_dir):
        def save_video(output_dir, i):
            if len(self.episode_images) == 0:
                return
            
            height, width, _ = self.episode_images[0].shape
            # Create a video writer object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for video output
            video_writer = cv2.VideoWriter(output_dir, fourcc, 0.75, (width, height))
            
            # Write each frame to the video
            for frame in self.episode_images[i]:
                video_writer.write(frame)
            
            # Release the video writer
            video_writer.release()
            
            print(f"Video saved to: {output_path}")
        for i in range(self.batch_size):
            save_video(os.path.join(output_dir, f"episode_{i}.mp4"), i)

# Rl agent
def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = F.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)

class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )
class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))

class attention_Layer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, q: Tensor,k:Tensor,v:Tensor) -> Tensor:
        src = self.attention(q, k, v)
        return src

class language_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer("johngiorgi/declutr-small")
    
    def forward(self, texts):
        with torch.no_grad():
            embeddings = self.model.encode(texts)
        return embeddings

class RLHF_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_space = 6
        self.language_encoder = language_encoder()
        #self.image_encoder = image_encoder()
        self.cross_attention_1 = attention_Layer()
        self.HCAM_1 = HTMBlock(
                    dim = 512,
                    heads = 4, 
                    topk_mems = 4,
                    mem_chunk_size = 32,
                    add_pos_enc = False
                    )
        self.text_transoform = nn.Linear(768, 512)
        self.cross_attention_2 = attention_Layer()
        self.HCAM_2 =  HTMBlock(
                    dim = 512,
                    heads = 4, 
                    topk_mems = 4,
                    mem_chunk_size = 32,
                    add_pos_enc = False
                    )
        # Define a custom output layer for the action distribution.
        # The code below is just an example, please modify as needed.
        self.logits = layer_init(nn.Linear(512, self.action_space), std=0.01)
        self.value_f = layer_init(nn.Linear(512, 1), std=1)
    
    def forward(self, image, text, memories) -> Tensor:
        image_features = image
        image_features = image_features.unsqueeze(1)
        text_features = self.language_encoder(text)
        text_features = torch.from_numpy(text_features)
        text_features = text_features.unsqueeze(1)
        transform_text_features = self.text_transoform(text_features)
        Memory = self.cross_attention_1(text_features, image_features, image_features)
        x = self.HCAM_1(Memory, memories)
        x = self.cross_attention_2(x, text_features, text_features)
        x = self.HCAM_2(x, memories)
        logits = self.logits(x)
        value = self.value_f(x)
        return logits, value, Memory.detach()

class reward_model_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 1)
    def forward(I,L, action):
        output = torch.cat((I,L, action), dim=0)
        output = self.linear(torch.flatten(output))
        return output

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.model = RLHF_net()
    def get_value(self, image, text, memories):
        _, values, memory = self.model(image, text, memories)
        return values

    def get_action_and_value(self, image, text, memories, action=None):
        logits,values,memories = self.model(image, text, memories)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), values, memories

def RWM_dfs(env, agent,args):
    def init(env, agent, outdir):
            obs = torch.zeros((args.num_steps, args.env_batch_size) + (512,768)).to(device)
            actions = torch.zeros((args.num_steps, args.env_batch_size) + (1,)).to(device)
            logprobs = torch.zeros((args.num_steps, args.env_batch_size)).to(device)
            rewards = torch.zeros((args.num_steps, args.env_batch_size)).to(device)
            dones = torch.zeros((args.num_steps, args.env_batch_size)).to(device)
            values = torch.zeros((args.num_steps, args.env_batch_size)).to(device)
            memories = torch.zeros((args.num_steps, args.env_batch_size, 512)).to(device)
            text_fea = language_encoder()
            # TRY NOT TO MODIFY: start the game
            start_time = time.time()
            obs_i = env.reset()
            next_obs_f = torch.Tensor(obs_i["features"]).to(device)
            next_done = torch.zeros(args.env_batch_size).to(device)

            for step in range(0, args.num_steps):
                xnext_obs_f = torch.Tensor(next_obs["features"]).to(device)
                obs_i = next_obs
                with torch.no_grad():
                    text = text_fea(next_obs['text'])
                obs[step] = (next_obs['features'],text)
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value, memory = agent.get_action_and_value(next_obs_f, obs_i['text'].to(device), memories)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                memories[step] = memory
            
                reward = reward_model_A(next_obs['features'],text, logprob)
                wandb.log({"reward" : reward})
                next_obs, done, info = env.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            env.save_video(outdir)
            return create_df(obs, actions, logprobs, rewards, dones, values)
    df1 = init(env, agent, '/root/mount/Matterport3DSimulator/videos/1')
    df2 = init(env, agent, '/root/mount/Matterport3DSimulator/videos/2')
    return df1, df2

def rollout_update(env, agent, args, reward_model,outdir, **kwargs):
        obs = torch.zeros((args.num_steps, env_batch_size) + (512,768)).to(device)
        actions = torch.zeros((args.num_steps, env_batch_size) + (1,)).to(device)
        logprobs = torch.zeros((args.num_steps, env_batch_size)).to(device)
        rewards = torch.zeros((args.num_steps, env_batch_size)).to(device)
        dones = torch.zeros((args.num_steps, env_batch_size)).to(device)
        values = torch.zeros((args.num_steps, env_batch_size)).to(device)
        memories = torch.zeros((args.num_steps, env_batch_size, 512)).to(device)

        global_step = 0
        start_time = time.time()
        obs_i = env.reset()
        next_obs_f = torch.Tensor(obs_i["features"]).to(device)
        next_done = torch.zeros(10).to(device)

        # if args.anneal_lr:
        #     frac = 1.0 - (update - 1.0) / num_updates
        #     lrnow = frac * args.learning_rate
        #     optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            with torch.no_grad():
                    text = text_fea(obs_i['text'])
            obs[step] = (next_obs_f,torch.Tensor(text))
            dones[step] = next_done
            global_step += 1
            
            with torch.no_grad():
                action, logprob, _, value, memory = agent.get_action_and_value(next_obs_f, obs_i['text'].to(device), memories[step])
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            memories[step] = memory
            
            reward = reward_model_A(next_obs_f,torch.Tensor(text), logprob)
            wandb.log({"reward" : reward})
            next_obs, done, info = env.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            next_obs_f = torch.Tensor(next_obs["features"]).to(device)
            obs_i = next_obs
        env.save_video(outdir)
        df = create_df(obs, actions, logprobs, rewards, dones, values)
        
        with torch.no_grad():
            next_value = agent.get_value(next_obs_f, obs_i['text'].to(device)).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        
        b_obs = obs.reshape((-1,) + (512,768))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + (1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_memories = memories.reshape((-1, 512))
        
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(b_obs[mb_inds][0],b_obs[mb_inds][1], memories[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print("SPS:", int(global_step / (time.time() - start_time)))
        wandb.log({"losses/value_loss": v_loss.item(), "losses/policy_loss": pg_loss.item(), "losses/entropy": entropy_loss.item(), "losses/old_approx_kl": old_approx_kl.item(), "losses/approx_kl": approx_kl.item(),"losses/explained_variance": explained_var })

        return df

def main(device, args):
    Env = R2RGym()
    if args.new_reward_m:
        reward_model = reward_model_A()
        reward_model.eval()
    else:
        reward_model = reward_model_A()
        reward_model.load_state_dict(torch.load('reward_model.pth'))
        #reward_model.to(device)
        reward_model.eval()
    if args.same_agent:
        agent = Agent()
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        agent.load_state_dict(torch.load('agent.pth', map_location=device))
        optimizer.load_state_dict(torch.load('optimizer.pth', map_location=device))
        agent.to(device)
    else: 
        agent = Agent().to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    if args.init:
        RWM_dfs(Env, agent,args)
    else:
        df1 = rollout_update(Env, agent,args,reward_model,'/root/mount/Matterport3DSimulator/videos/1')
        df2 = rollout_update(Env, agent,args,reward_model,'/root/mount/Matterport3DSimulator/videos/2')
        df1.to_csv(f'df{num}.csv')
        df2.to_csv(f'df{num}.csv')

    torch.save(agent.state_dict(), 'agent.pth')
    torch.save(optimizer.state_dict(), 'optimizer.pth') 

if __name__ == "__main__":
    args = parse_args()

    if args.track:
        import wandb

        wandb.init(
            project='VLNRL',
            sync_tensorboard=False,
            config=vars(args),
            monitor_gym=True,
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    main(device, args)




