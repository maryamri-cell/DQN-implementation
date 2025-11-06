import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import time
import matplotlib.colors as mcolors

# Définir le device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomGoalGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=8, render_mode=None):
        super().__init__()
        self.size = size
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=size - 1, shape=(4,), dtype=np.int32)
        self.agent_pos = None
        self.goal_pos = None
        self.start_pos = (0, 0)
        
        self.fig, self.ax = None, None

    def _select_new_goal(self):
        while True:
            goal_x = random.randint(0, self.size - 1)
            goal_y = random.randint(0, self.size - 1)
            if (goal_x, goal_y) != self.start_pos:
                return np.array([goal_x, goal_y], dtype=np.int32)

    def _get_obs(self):
        return np.concatenate((self.agent_pos, self.goal_pos))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array(self.start_pos, dtype=np.int32)
        self.goal_pos = self._select_new_goal()
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        x, y = self.agent_pos
        if action == 0: new_pos = np.array([x - 1, y])
        elif action == 1: new_pos = np.array([x + 1, y])
        elif action == 2: new_pos = np.array([x, y - 1])
        elif action == 3: new_pos = np.array([x, y + 1])
        else: raise ValueError(f"Action invalide: {action}")

        new_pos[0] = np.clip(new_pos[0], 0, self.size - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.size - 1)
        
        agent_did_move = (new_pos != self.agent_pos).any()
        self.agent_pos = new_pos

        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        
        if terminated:
            reward = 10.0
        elif not agent_did_move:
            reward = -1.0
        else:
            reward = -0.1

        truncated = False 
        observation = self._get_obs()
        return observation, reward, terminated, truncated, {}

    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
            self.ax.set_xticks(np.arange(-0.5, self.size, 1), minor=True)
            self.ax.set_yticks(np.arange(-0.5, self.size, 1), minor=True)
            self.ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
            self.ax.tick_params(which='minor', size=0)
            self.ax.set_xticks([]); self.ax.set_yticks([])
            self.ax.set_xlim(-0.5, self.size - 0.5)
            self.ax.set_ylim(-0.5, self.size - 0.5)
            self.ax.invert_yaxis()
            self.fig.canvas.manager.set_window_title('GridWorld DQN Demo')

        self.ax.clear()
        self.ax.set_xticks(np.arange(-0.5, self.size, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.size, 1), minor=True)
        self.ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        self.ax.tick_params(which='minor', size=0)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_xlim(-0.5, self.size - 0.5)
        self.ax.set_ylim(-0.5, self.size - 0.5)
        self.ax.invert_yaxis()
        
        goal_x, goal_y = self.goal_pos
        self.ax.add_patch(plt.Rectangle((goal_y - 0.5, goal_x - 0.5), 1, 1, facecolor='lightcoral', alpha=0.5))
        self.ax.text(goal_y, goal_x, 'G', ha='center', va='center', fontsize=20, color='red', weight='bold')

        agent_x, agent_y = self.agent_pos
        self.ax.plot(agent_y, agent_x, marker='o', markersize=30, color='blue', markeredgecolor='black', alpha=0.7)
        self.ax.text(agent_y, agent_x, 'A', ha='center', va='center', fontsize=16, color='white', weight='bold')

        plt.draw()
        plt.pause(1e-6)
        
    def close(self):
        if self.fig:
            plt.close(self.fig)



class DuelingDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DuelingDQN, self).__init__()
        
        # Le tronc du réseau (commun aux deux têtes)
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # 1. Tête de la Valeur (V) - Le Baseline
        self.value_stream = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Sortie : V(S), la valeur de l'état (le Baseline)
        )
        
        # 2. Tête de l'Avantage (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_size) # Sortie : A(S, a), l'avantage pour chaque action
        )

    def forward(self, x):
        x = x.float()
        features = self.feature_layer(x) # L'état passe par le tronc commun
        
        # Calcul de V(S) et A(S, a)
        V = self.value_stream(features)
        A = self.advantage_stream(features)
        
        # Combinaison : Q(S, a) = V(S) + [A(S, a) - Moyenne(A(S, a))]
        # La soustraction de la moyenne stabilise la combinaison (évite l'indentifiabilité)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        
        return Q


class DQNAgentWithTarget:
    def __init__(self, state_size, action_size, learning_rate=1e-3, gamma=0.95, epsilon=1.0, update_target_freq=200):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        
        # --- ATTENTION : Utilise DuelingDQN ici ! ---
        self.model = DuelingDQN(state_size, action_size).to(DEVICE) 
        self.target_model = DuelingDQN(state_size, action_size).to(DEVICE) # Et ici !
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() 
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_freq = update_target_freq
        self.train_step_count = 0 


    def remember(self, state, action, reward, next_state, terminated):
        state = torch.from_numpy(state).to(DEVICE)
        next_state = torch.from_numpy(next_state).to(DEVICE)
        self.memory.append((state, action, reward, next_state, terminated))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            q_values = self.model(state_tensor) 
            return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size: return
        
        self.train_step_count += 1
        
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, terminateds = zip(*minibatch)
        states = torch.stack(states).to(DEVICE)
        actions = torch.tensor(actions).to(DEVICE).unsqueeze(1)
        rewards = torch.tensor(rewards).float().to(DEVICE).unsqueeze(1)
        next_states = torch.stack(next_states).to(DEVICE)
        terminateds = torch.tensor(terminateds).to(DEVICE).unsqueeze(1)
        
        # 1. Calcul du Q actuel (utilisant le Réseau Principal Dueling)
        current_q_values = self.model(states).gather(1, actions)
        
        # 2. Calcul du Q max futur (utilisant le Réseau Cible Dueling)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0].detach().unsqueeze(1) 
        
        # 3. Calcul de la Cible Y
        target_q_values = rewards + (self.gamma * next_q_values * (~terminateds).float())
        
        # 4. Optimisation
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 5. Décroissance Epsilon et Mise à jour Target
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        if self.train_step_count % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())


def plot_results(reward_history, epsilon_history, window=100):
    episodes = range(1, len(reward_history) + 1)
    avg_rewards = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episodes, reward_history, alpha=0.3, label='Récompense par Épisode')
    plt.plot(episodes[window-1:], avg_rewards, color='red', label=f'Moyenne Glissante ({window} épisodes)')
    plt.title('Historique des Récompenses d\'Entraînement (Dueling DQN)')
    plt.xlabel('Épisode')
    plt.ylabel('Récompense Totale')
    plt.grid(True, alpha=0.5)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(episodes, epsilon_history, color='green')
    plt.title('Décroissance du Taux d\'Exploration (Epsilon)')
    plt.xlabel('Épisode')
    plt.ylabel('Epsilon (ε)')
    plt.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    # --- Hyperparamètres ---
    GRID_SIZE = 8
    N_EPISODES = 1000
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 200
    
    # --- Paramètres de Visualisation ---
    RENDER_INTERVAL = 100 
    RENDER_DELAY = 0.05
    
    # Initialisation
    env = CustomGoalGridEnv(size=GRID_SIZE)
    state_size = env.observation_space.shape[0] 
    action_size = env.action_space.n
    
    # --- Utilisation du NOUVEL agent avec DuelingDQN ---
    agent = DQNAgentWithTarget(state_size, action_size, update_target_freq=TARGET_UPDATE_FREQ)

    reward_history = []
    epsilon_history = []

    print(f"🚀 Démarrage de l'entraînement Dueling DQN sur une grille {GRID_SIZE}x{GRID_SIZE} ({DEVICE})...")

    try:
        for e in range(1, N_EPISODES + 1):
            state, _ = env.reset() 
            state = np.array(state)
            total_reward = 0
            
            terminated = False
            truncated = False
            time_step = 0
            
            is_demo_episode = (e % RENDER_INTERVAL == 0)

            while not terminated and not truncated:
                
                if is_demo_episode:
                    env.render() 
                    time.sleep(RENDER_DELAY)
                
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = np.array(next_state)
                
                agent.remember(state, action, reward, next_state, terminated)
                
                state = next_state
                total_reward += reward
                time_step += 1
                
                if time_step > GRID_SIZE * GRID_SIZE * 5:
                    truncated = True
            
            agent.replay(BATCH_SIZE)
            reward_history.append(total_reward)
            epsilon_history.append(agent.epsilon)
            
            if e % 100 == 0 or e == N_EPISODES:
                avg_reward = np.mean(reward_history[-100:])
                print(f"Épisode: {e}/{N_EPISODES} | Étapes: {time_step:3d} | Récompense: {total_reward:6.2f} | Avg (100 Ep): {avg_reward:6.2f} | Epsilon: {agent.epsilon:.3f}")

            if is_demo_episode:
                env.render()
                print(f"--- ÉPISODE DÉMO {e} TERMINÉ --- (Récompense: {total_reward:.2f}, Étapes: {time_step})")
                time.sleep(1)

    finally:
        env.close()
        print("\n\n✅ Entraînement terminé. Génération des graphiques...")

        plot_results(reward_history, epsilon_history)