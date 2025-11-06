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
        
        # Initialisation des objets Matplotlib pour la visualisation
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
        """Affiche la grille dans une fenêtre Matplotlib."""
        if self.fig is None:
            # 1. Initialisation de la figure
            self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
            self.ax.set_xticks(np.arange(-0.5, self.size, 1), minor=True)
            self.ax.set_yticks(np.arange(-0.5, self.size, 1), minor=True)
            self.ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
            self.ax.tick_params(which='minor', size=0)
            self.ax.set_xticks([]); self.ax.set_yticks([]) # Masquer les ticks majeurs
            self.ax.set_xlim(-0.5, self.size - 0.5)
            self.ax.set_ylim(-0.5, self.size - 0.5)
            self.ax.invert_yaxis() # Convention de grille (0,0) en haut à gauche
            self.fig.canvas.manager.set_window_title('GridWorld DQN Demo')

        # 2. Nettoyage de l'affichage précédent
        self.ax.clear()
        self.ax.set_xticks(np.arange(-0.5, self.size, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.size, 1), minor=True)
        self.ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        self.ax.tick_params(which='minor', size=0)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_xlim(-0.5, self.size - 0.5)
        self.ax.set_ylim(-0.5, self.size - 0.5)
        self.ax.invert_yaxis()
        
        # 3. Dessiner le but (Goal: Cercle rouge)
        # Goal: (X, Y) -> (Y + 0.5, X + 0.5) pour le centre de la cellule
        goal_x, goal_y = self.goal_pos
        self.ax.add_patch(plt.Rectangle((goal_y - 0.5, goal_x - 0.5), 1, 1, facecolor='lightcoral', alpha=0.5))
        self.ax.text(goal_y, goal_x, 'G', ha='center', va='center', fontsize=20, color='red', weight='bold')

        # 4. Dessiner l'agent (Agent: Cercle bleu)
        agent_x, agent_y = self.agent_pos
        self.ax.plot(agent_y, agent_x, marker='o', markersize=30, color='blue', markeredgecolor='black', alpha=0.7)
        self.ax.text(agent_y, agent_x, 'A', ha='center', va='center', fontsize=16, color='white', weight='bold')

        # 5. Mettre à jour la fenêtre
        plt.draw()
        plt.pause(1e-6) # Petite pause pour le rafraîchissement
        
    def close(self):
        plt.close(self.fig)



class DQN(nn.Module):
    # ... (code inchangé pour DQN)
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_size, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 32)
        self.layer4 = nn.Linear(32, 32)
        self.layer5 = nn.Linear(32, 32)
        
        self.output_layer = nn.Linear(32, output_size)

    def forward(self, x):
        x = x.float()
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output_layer(x)
    
# 3. AGENT DQN ET LOGIQUE D'APPRENTISSAGE
class DQNAgent:
    # ... (code inchangé pour DQNAgent)
    def __init__(self, state_size, action_size, learning_rate=1e-3, gamma=0.95, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.model = DQN(state_size, action_size).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

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
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, terminateds = zip(*minibatch)
        states = torch.stack(states).to(DEVICE)
        actions = torch.tensor(actions).to(DEVICE).unsqueeze(1)
        rewards = torch.tensor(rewards).float().to(DEVICE).unsqueeze(1)
        next_states = torch.stack(next_states).to(DEVICE)
        terminateds = torch.tensor(terminateds).to(DEVICE).unsqueeze(1)
        current_q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states).max(1)[0].detach().unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (~terminateds).float())
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
def plot_results(reward_history, epsilon_history, window=100):
    # ... (code inchangé pour les graphiques de résultats)
    episodes = range(1, len(reward_history) + 1)
    avg_rewards = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episodes, reward_history, alpha=0.3, label='Récompense par Épisode')
    plt.plot(episodes[window-1:], avg_rewards, color='red', label=f'Moyenne Glissante ({window} épisodes)')
    plt.title('Historique des Récompenses d\'Entraînement')
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
    
    # --- Paramètres de Visualisation ---
    RENDER_INTERVAL = 100 # Affiche l'épisode de démo tous les 100 épisodes
    RENDER_DELAY = 0.1    # Délai en secondes entre chaque pas
    
    # Initialisation
    env = CustomGoalGridEnv(size=GRID_SIZE)
    state_size = env.observation_space.shape[0] 
    action_size = env.action_space.n           
    agent = DQNAgent(state_size, action_size)

    reward_history = []
    epsilon_history = []

    print(f"🚀 Démarrage de l'entraînement sur une grille {GRID_SIZE}x{GRID_SIZE} avec DQN/PyTorch ({DEVICE})...")

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
                
                # Visualisation dans la fenêtre Matplotlib
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
            
            if is_demo_episode:
                env.render() # Afficher la position finale (but atteint ou fin de temps)
                print(f"--- ÉPISODE DÉMO {e} TERMINÉ --- (Récompense: {total_reward:.2f}, Étapes: {time_step})")
                time.sleep(1)
            
            agent.replay(BATCH_SIZE)
            reward_history.append(total_reward)
            epsilon_history.append(agent.epsilon)
            
            if e % 100 == 0 or e == N_EPISODES:
                avg_reward = np.mean(reward_history[-100:])
                print(f"Épisode: {e}/{N_EPISODES} | Étapes: {time_step:3d} | Récompense: {total_reward:6.2f} | Avg (100 Ep): {avg_reward:6.2f} | Epsilon: {agent.epsilon:.3f}")

    finally:
        env.close()
        print("\n\n✅ Entraînement terminé. Génération des graphiques...")

        # Générer la visualisation finale des résultats d'apprentissage
        plot_results(reward_history, epsilon_history)
        