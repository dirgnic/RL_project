#  CHEAT SHEET RL – Concepte de Bază
**Pentru echipa de proiect** | Scris în română, fără LLM

---

##  1. Ce este Reinforcement Learning?

**Idee centrală:** Un **agent** învață să ia **decizii bune** într-un **mediu incert**, primind **feedback** (reward) după fiecare acțiune.

- **Nu avem date etichetate** (ca în supervised learning)
- **Nu căutăm pattern-uri** (ca în unsupervised learning)
- **Învățăm prin încercare și eroare** (trial and error)

### Diferența față de alte metode ML:
| Tip | Date | Scop |
|-----|------|------|
| **Supervised** | Input + Label corect | Prezicere |
| **Unsupervised** | Input neetichetat | Găsire pattern-uri |
| **Reinforcement** | Interacțiune + Reward | Optimizare decizii |

---

##  2. Componente de bază RL

```
         action (a)        
  AGENT   > ENVIRONMENT 
                                                 
         <             
  state (s) + reward (r)  
```

### Componentele principale:
- **State (s)** = starea curentă a mediului
- **Action (a)** = ce poate face agentul
- **Reward (r)** = feedback numeric (cât de bună a fost acțiunea)
- **Policy (π)** = strategia agentului (ce acțiune alege în fiecare stare)
- **Value Function (V/Q)** = cât de "bună" e o stare sau o acțiune

---

##  3. Procese Markov (MDP - Markov Decision Process)

### Proprietatea Markov:
> **"Viitorul depinde doar de prezent, nu de întreg trecutul"**

Matematic:
```
P(s_t | s_{t-1}, s_{t-2}, ..., s_0) = P(s_t | s_{t-1})
```

### Un MDP conține:
- **S** = mulțimea de stări
- **A** = mulțimea de acțiuni
- **P(s'|s,a)** = probabilitatea de tranziție (unde ajung dacă fac acțiunea a în starea s)
- **R(s,a,s')** = reward-ul primit
- **γ (gamma)** = factor de discount (0-1, cât de importantă e recompensa viitoare)

---

##  4. Ecuația Bellman (intuitiv)

**Ideea:** Valoarea unei stări = reward imediat + valoarea stării viitoare (cu discount)

```
V(s) = R(s,a) + γ * V(s')
```

- **R(s,a)** = ce primesc acum
- **γ * V(s')** = ce voi primi în viitor (redus cu γ)

Această ecuație stă la baza tuturor algoritmilor value-based!

---

##  5. Policy (Politica) π

**Ce este?** O regulă care spune ce acțiune să alegi în fiecare stare.

### Tipuri:
- **Deterministă:** π(s) = a (mereu aceeași acțiune pentru starea s)
- **Stochastică:** π(a|s) = probabilitate (aleg diferite acțiuni cu anumite probabilități)

**Scop final RL:** Să găsim politica optimă π* care maximizează recompensa totală!

---

##  6. Value Function vs Q-Function

### Value Function V(s):
> "Cât de bună este starea s dacă urmez politica π de aici încolo?"

```
V^π(s) = E[r_t + γ*r_{t+1} + γ²*r_{t+2} + ... | s_t = s]
```

### Q-Function Q(s,a):
> "Cât de bună este acțiunea a în starea s?"

```
Q^π(s,a) = E[r_t + γ*Q(s',a') | s_t=s, a_t=a]
```

**Diferența:**
- V(s) = evaluează **stări**
- Q(s,a) = evaluează **perechi stare-acțiune**

---

##  7. Explorare vs Exploatare

**Dilema fundamentală în RL:**

- **Exploatare (Exploit):** Fac ce știu că merge (cea mai bună acțiune cunoscută)
- **Explorare (Explore):** Încerc ceva nou (poate găsesc ceva mai bun)

### Soluția: ε-greedy
```
Cu probabilitate 1-ε: aleg cea mai bună acțiune (greedy)
Cu probabilitate ε: aleg o acțiune random
```

În practică: ε = 0.05 - 0.2, sau cu **decay** (scade în timp)

---

##  8. Categorii de Algoritmi RL

### A. Value-Based (bazați pe învățarea valorilor)
**Idee:** Învață V(s) sau Q(s,a), apoi extrage politica din ele

**Algoritmi:**
- Value Iteration (model-based)
- Q-Learning (model-free, off-policy)
- SARSA (model-free, on-policy)
- DQN (Deep Q-Network - cu rețele neuronale)

**Când se folosesc:** Spații discrete, probleme mici-medii

---

### B. Policy-Based (bazați pe învățarea directă a politicii)
**Idee:** Învață direct politica π, fără să calculezi V sau Q

**Algoritmi:**
- REINFORCE
- Actor-Critic (hibrid)
- PPO (Proximal Policy Optimization)
- A3C (Asynchronous Actor-Critic)

**Când se folosesc:** Spații continue de acțiuni, probleme complexe

---

### C. Hybrid (Actor-Critic)
**Idee:** Combină cele două abordări
- **Actor** = politica (ce acțiuni să fac)
- **Critic** = value function (cât de bune sunt)

**Avantaje:** Mai stabil și eficient decât metodele pure

---

##  9. Model-Based vs Model-Free

### Model-Based:
- **Cunoaștem modelul mediului** (P și R)
- Putem planifica (ex: Value Iteration)
- Mai eficient, dar rar avem modelul complet

### Model-Free:
- **Nu știm cum funcționează mediul**
- Învățăm doar din experiență
- Mai des folosit în practică (Q-Learning, DQN, PPO)

---

##  10. Algoritmi Principali (rezumat rapid)

### Value Iteration
```
Repeat until convergence:
    Pentru fiecare stare s:
        V(s) = max_a [R(s,a) + γ * Σ P(s'|s,a) * V(s')]
```
- **Tip:** Model-based, value-based
- **Când:** Probleme mici, cunoaștem modelul

---

### Q-Learning (tabular)
```
Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
```
- **Tip:** Model-free, off-policy, value-based
- **Când:** Spații discrete mici (CartPole, FrozenLake)
- **Avantaj:** Simplu, învață din orice politică
- **Dezavantaj:** Nu scalează (tabel poate exploda)

---

### SARSA
```
Q(s,a) ← Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
```
- **Diferență față de Q-Learning:** Folosește acțiunea efectiv aleasă (a'), nu max
- **Tip:** On-policy (învață din propria politică)
- **Comportament:** Mai conservator, evită riscuri

---

### DQN (Deep Q-Network)
```
Neural Network: Q(s,a) = NN(s, a)
Loss = (r + γ * max_a' Q_target(s',a') - Q(s,a))²
```
- **Tip:** Model-free, value-based, cu rețele neuronale
- **Când:** Spații continue de stări (LunarLander, Atari)
- **Trucuri:** Replay Buffer + Target Network (pentru stabilitate)

---

### Monte Carlo (MC)
```
După un episod complet:
V(s) ← V(s) + α * (G - V(s))
G = suma tuturor rewardurilor din episod
```
- **Tip:** Model-free, învață din episoade complete
- **Avantaj:** Simplu, bias zero
- **Dezavantaj:** Varianță mare, trebuie să aștepți sfârșitul episodului

---

### Temporal Difference (TD)
```
V(s_t) ← V(s_t) + α * (r_{t+1} + γ*V(s_{t+1}) - V(s_t))
```
- **Tip:** Combină MC + Dynamic Programming
- **Avantaj:** Învață online (nu așteaptă sfârșitul episodului)
- **Când:** Medii fără stare terminală, învățare continuă

---

### PPO (Proximal Policy Optimization)
- **Tip:** Policy-based, hibrid
- **Când:** Probleme complexe, spații continue
- **Avantaj:** Foarte stabil, performanță bună
- **Used in:** Robotică, jocuri complexe

---

##  11. Environments (Gymnasium)

### Classic Control (simple, bune pentru început):
- **CartPole-v1:** Balansare băț pe cărucior (discrete)
- **MountainCar-v0:** Urcare pe deal cu momentum (discrete)
- **Pendulum-v1:** Balansare pendul (continuous)

### Box2D (mediu, mai complexe):
- **LunarLander-v2:** Aterizare navă (discrete, 4 acțiuni)
- **BipedalWalker-v3:** Robot cu 2 picioare (continuous)

### Atari (complexe, deep RL):
- Pong, Breakout, Space Invaders (pixel input)

---

##  12. Checklist pentru proiect

### Ce trebuie să înțeleagă toată echipa:
- [ ] Ce e un agent, mediu, reward, policy
- [ ] Diferența RL vs supervised vs unsupervised
- [ ] Proces Markov (viitorul = doar prezent)
- [ ] MDP (S, A, P, R, γ)
- [ ] V(s) și Q(s,a)
- [ ] Ecuația Bellman (reward acum + viitor)
- [ ] Value-based vs policy-based vs hybrid
- [ ] Explorare vs exploatare (ε-greedy)
- [ ] MC vs TD (episoade complete vs online)

### Ce trebuie să facem:
1. **Alegem environment** (ex: CartPole, LunarLander, MountainCar)
2. **Modificăm reward-ul** (reward shaping pentru experimente)
3. **Implementăm 3 algoritmi** din categorii diferite:
   - Ex: Q-Learning (tabular, value-based)
   - Ex: DQN (deep, value-based)
   - Ex: PPO (policy-based)
4. **Rulăm experimente** (seed-uri, hiperparametri)
5. **Comparăm rezultate** (grafice, tabele)
6. **Documentăm** (ppt/pdf, fără LLM!)
7. **Prezentăm** în săptămâna 14

---

##  Termeni cheie (vocabular)

| Termen | Explicație |
|--------|-----------|
| **Agent** | „Creierul" care ia decizii |
| **Environment** | Lumea în care trăiește agentul |
| **State (s)** | Situația curentă |
| **Action (a)** | Ce poate face agentul |
| **Reward (r)** | Feedback numeric (bun/rău) |
| **Episode** | O rundă completă (start → final) |
| **Policy (π)** | Strategia (ce acțiune în fiecare stare) |
| **Value V(s)** | Cât de bună e starea s |
| **Q-value Q(s,a)** | Cât de bună e acțiunea a în starea s |
| **Discount (γ)** | Factor 0-1, importanța viitorului |
| **Learning rate (α)** | Cât de repede învață |
| **Epsilon (ε)** | Probabilitatea de explorare |
| **Greedy** | Aleg mereu cea mai bună acțiune |
| **Bootstrap** | Învăț din estimări (nu aștept final) |
| **On-policy** | Învață din propria politică |
| **Off-policy** | Învață din orice politică |
| **Replay Buffer** | Memorie pentru experiențe trecute |
| **Target Network** | Rețea auxiliară pentru stabilitate |

---

##  Resurse rapide

- **Sutton & Barto** - "Reinforcement Learning: An Introduction" (biblia RL)
- **Gymnasium Docs** - https://gymnasium.farama.org/
- **Stable Baselines3** - https://stable-baselines3.readthedocs.io/
- **CS234 Stanford** - Emma Brunskill (video lectures)
- **CS285 Berkeley** - Sergey Levine (deep RL)

---

**Scris de echipă, pentru echipă** 
_Versiunea 1.0 - Noiembrie 2025_
