[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/luca037/Snake-RL)

<p align="center">
<img width="200" height="200" alt="snake_icon" src="https://github.com/user-attachments/assets/63f262e8-7556-406e-9913-e58d77085565" />
</p>

# Snake-RL

## Table of Contents

- [Overview](#overview)
- [About AtariAgent](#atariagent)
- [About CerberusAgent](#cerberusagent)
- [How to run the code](#how-to-run-the-code)
- [Class Diagram](#class-diagram)
- [Train agent from scratch](#train-agent-from-scratch)
- [Repository Structure](#repository-structure)
- [References](#references)

## Overview

In this project we've trained 3 agents, namely: **BlindAgent**, **LidarAgent**, **AtariAgent**.
Each of them has its own state-space representation and
all of them are trained using the DQN algoritm (more info at this [link](https://www.nature.com/articles/nature14236)).
After identifying the best state-space representation, the **CerberusAgent** was built by ensembling three pre-trained agents
(3 AtariAgents). You can find more information in the `report.pdf`.

In the following we show the record achieved by each agent.

---

<div align="center">

<table>
  <tr>
    <th align="center">CerberusAgent</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/e770b9af-a43c-4857-95ce-0a957af7beac">
    </td>
  </tr>
</table>
  
<table>
  <tr>
    <th align="center">BlindAgent</th>
    <th align="center">LidarAgent</th>
    <th align="center">AtariAgent</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/9a6f0f67-dcc4-4b08-8ff4-f094c3772954">
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/fcd08280-450c-4e6e-a022-6b024721e5e6">
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/747d96d0-5176-4562-b8cf-2bd987cca479">
    </td>
  </tr>
</table>
  
</div>

---

## AtariAgent

I will not introduce all the agents since they're explained in the `report.pdf`, in this readme i will present only the
**CerberusAgent**. However, to introduce it, I need to present the **AtariAgent**.

The AtariAgent takes its name from the famous [DeepMind paper](https://www.nature.com/articles/nature14236).
In the paper they use a CNN and the input is built by stacking the last 4 frames of the game, the
AtariAgent use the same concept. Its Q-network has the following structure:

<div align="center">
  <img width="2856" align="center" height="432" alt="image" src="https://github.com/user-attachments/assets/386ab609-56c2-4b84-a0ce-781e2f788d68" />
</div>

## CerberusAgent

### Why using it?

Is AtariAgent not enough to win Snake? No. Well, at least I was't able to train it to do so. By looking at the record achieved by
the AtariAgent you will see that the policy learned can be splitted in two: at the beginning the snake rushes towards the
food (like BlindAgent does), then it switches to a more survival policy and it starts to follow a circular path.

So we can just train 2 different AtariAgent: the first one is the one we have already trained (aka the one playing in
the gif); the second one is trained using a longer snake (like 50 for example) from the beginning. In this way the second one
must learn a survival strategy and by combining the two agents we should win the game.

### Why 3 heads and not just 2?

The CerberusAgent comes from that observation. This agent is composed by 3 AtariAgents, we call them **heads**. **Why 3 and not just 2?** The short answer is: *because it
works better in this way*. The idea behind the introduction of the second head, is related to the input distribution of the third head. However note that, by removing it, you
will get similar results.

### How does it play?

At each step of the game, a single head gives the next action to perform. The head that decides the action is selected based on the current score.
The simple logic is reported in the following picture:

<div align="center">
<img width="500" height="260" alt="image" src="https://github.com/user-attachments/assets/b3859a48-3c33-4bb5-9caa-1693fd2c21e8" />
</div>

### Does it always win the game?

Hell no. The overall strategy *can* win the game but it is far from being a perfect strategy.

<div align="center">
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/d061bb84-6581-4ff7-8700-7050b6fd8b2e" />
</div>

In the picture above we can see the score distribution. The bars are colored according to the head that was responsible for the snake’s death.
Note that: when game starts, the lenght of the snake is 3 and the maximum is 100, so the maximum score is 97 (the score is the number of apples eaten). 
We observe that the agent reaches the end-game very frequently, although it is less likely to fully win the game.

## How to run the code

First install the requirements:

```shell
pip install -r requirements.txt
```

Print help message with `python main.py --help`,  the output:

```
usage: main.py [-h] [--agent AGENT] [--train] [--loadbuf] [--record] [--nogui]

optional arguments:
  -h, --help     show this help message and exit
  --agent AGENT  Specify agent type. Values: atari, blind, lidar or baseline.
  --train        If you want to train the agent.
  --loadbuf      If you want to load the buffer to 'device'.
  --record       If you want to watch the record of the agent.
  --nogui        If you want to deactivate the gui.
```

Below I report all the commands you may want to try:

```shell
# Run baseline algorithm
python main.py --agent baseline

# See replay of the record
python main.py --agent cerberus --record
python main.py --agent blind --record
python main.py --agent lidar --record

# Train agent
python main.py --agent cerberus --train
python main.py --agent blind --train
python main.py --agent lidar --train

# Create plots with stats (stored in ./output/plots/)
python plot.py
```

**Note that**: with `--train` you're not training a model from scratch. 
It will automatically load a pre-trained model stored in `./output/models/`. 

**Note that**: the training stats are stored in `./output/csv/stats.csv`.
If you stop training and then restart, the file will not be deleted, new
information will be added at the bottom. Moreover, the output model will be
stored to `./output/models/model.pth` and by default a snapshot is created
only when the agent achives a new record.

When you use `--train` you can also add option `--nogui` to disable the gui (you just get the terminal output
with the info). Also you can add option `--loadbuf`: this will load a replay buffer stored in `./output/models/`.

---

You can also plot the stats of the pre-trained models.
For example, to plot the stats of the BlindAgent:

```bash
# Copy stats
cp ./output/csv/stats_BlindAgent.csv ./output/csv/stats.csv"
python plot.py
```

## Class Diagram

<div align="center">

  <img width="2008" height="1251" alt="class_diag" src="https://github.com/user-attachments/assets/7f7a33bb-2f6c-41cd-b0f4-ed1ef0da45da" />

</div>
*(Generated by Gemini)*

## Train agent from scratch

If you want to train an agent from scratch, you need to create a new file.
The minimal code you need is:

```python
from Agent import *
from Game import ReplaySnakeGame, Point, SnakeGame

# Define the device variable.
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("INFO: CUDA is available. Running on GPU.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print ("INFO: MPS device found. Running on GPU")
else:
    device = torch.device("cpu")
    print("INFO: CUDA and MPS not available. Running on CPU.")

agent = BlindAgent( # or LidarAgent or AtariAgent
    max_dataset_size = 100_000,
    batch_size       = 32,
    lr               = 0.00025,
    epsilon          = 1,     # Starting value
    decaying_epsilon = 0.995, # Only for BlindAgent
    min_epsilon      = 0.0001,
    gamma            = 0.99,
    target_sync      = 2000,
    out_model_path   = "./output/models/model.pth",
    memory_path      = "./output/models/memory.h5",
    out_csv_path     = "./output/csv/stats.csv",
    device           = device,
    gui              = False,
    checkpoint_path  = None,
    load_buffer      = False
)
agent.train()
```

Training a `BlindAgent` is very fast (~200 games is enough), the other two need more time.

## Repository Structure

```
.
├── Agent.py        # Implementations of BlindAgent, LidarAgent, AtariAgent, Cerberus
├── Baseline.py     # Implemetation of the Baseline Algorithm
├── Game.py         # Implementation of Snake Game
├── main.py         # ... main file
├── Model.py        # Neural Nets used to implement the Q-Net and DQN trainer.
├── plot.py         # Used to plot stats inside './output/plots/'
├── ReplayBuffer.py # Implementation of the Replay Buffer.
└── output
    ├── csv         # Directory where stats are stored (csv format)
    ├── models      # Some pre-trained models
    ├── plots       # Outputs of 'plot.py'
    └── uml         # Class diagram uml
```

## References

Here a list some useful resources:

- The starting point for Snake and Reinforcement learning -> [link](https://towardsdatascience.com/snake-played-by-a-deep-reinforcement-learning-agent-53f2c4331d36-2/)
- BlindAgent was inspired by this project -> [link](https://github.com/patrickloeber/snake-ai-pytorch)
- LidarAgent was inspired by this project. This use a Genetic Algorithm however -> [link](https://github.com/greerviau/SnakeAI)
- Another Genetic Algorithm approach -> [link](https://github.com/Chrispresso/SnakeAI)
# Snake-RL-The-Return
