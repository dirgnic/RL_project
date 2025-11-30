#  INGRID - TASKURI PERSONALE

**Rol în proiect:** Algorithms + Environment Research Lead  
**Focus:** Q-Learning (algoritm #1) + Environment analysis + Cheat sheet  
**Last update:** 29 noiembrie 2025

---

##  CE AI FĂCUT DEJA (SUPER JOB!)

- [x] Ai citit primele cursuri RL
- [x] Ai făcut research pe context proiect
- [x] Ai creat întrebările principale pentru plan
- [x] Ai înțeles структура проекта

---

##  OBIECTIVE TALE ÎN PROIECT

### Obiectiv principal:
**Să fii expertă pe Q-Learning și să ajuți echipa să înțeleagă algoritmii clasici**

### Ce înseamnă asta concret:
1. **Înțelegi perfect Q-Learning** (update rule, Q-table, exploration, convergence)
2. **Implementezi primul agent** (Q-Learning pe environment ales)
3. **Explici celorlalți** cum funcționează (prezentare + cod)
4. **Ajuți la environment research** (analiză reward shaping)
5. **Coordonezi documentația teoretică** (cheat sheet + mini-rezumate)

---

##  TASKURI PE ETAPE

###  ETAPA 1: Research & Teorie (29 nov - 5 dec)
**Obiectiv:** Să vii la meeting #1 pregătită cu propuneri concrete

#### Task 1.1: Citit teorie (DONE )
- [x] Citit cursurile 1-3 RL
- [x] Făcut cheat sheet pentru echipă
- [x] Înțeles concepte de bază (MDP, Bellman, policy, value)

#### Task 1.2: Environment Research (IN PROGRESS )
**Deadline:** 5 decembrie (înainte de meeting)

**Ce trebuie să faci:**
- [ ] **Studiază CartPole-v1 în detaliu:**
  - [ ] Citește documentația Gymnasium: https://gymnasium.farama.org/environments/classic_control/cart_pole/
  - [ ] Rulează un agent random pe CartPole (test local)
  - [ ] Notează: state space, action space, reward default
  - [ ] Gândește-te la 3 idei de reward shaping (am pus deja în doc, dar tu adaugă detalii)
  
- [ ] **Studiază LunarLander-v2:**
  - [ ] Citește documentația: https://gymnasium.farama.org/environments/box2d/lunar_lander/
  - [ ] Uită-te la Lab 4 (avem cod acolo)
  - [ ] Notează complexitatea (8D state, 4 actions)
  - [ ] Idei reward shaping (fuel, safety, speed)
  
- [ ] **Studiază MountainCar-v0:**
  - [ ] Documentație: https://gymnasium.farama.org/environments/classic_control/mountain_car/
  - [ ] Înțelege de ce e greu (sparse reward)
  - [ ] Idei reward shaping (momentum, height, potential)

**Output:**
- [ ] Completează secțiunea ta în `02_ENVIRONMENT_RESEARCH.md`
- [ ] Pregătește argumentele PRO/CONTRA pentru fiecare
- [ ] Recomandare personală: care environment alegi și de ce?

**Timp estimat:** 3-4 ore

---

#### Task 1.3: Mini-tutorial Q-Learning pentru tine
**Deadline:** 5 decembrie

**Ce trebuie să faci:**
- [ ] Citește Lab 4 - secțiunea Q-Learning
- [ ] Înțelege formula:
  ```
  Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
  ```
- [ ] Explică în cuvinte proprii (scrie într-un doc):
  - Ce înseamnă fiecare termen?
  - De ce max_a'? (off-policy)
  - Cum inițializăm Q-table?
  - Ce e ε-greedy și de ce?
  
**Output:**
- [ ] Document personal: `Ingrid_Notes_QLearning.md` (pentru tine, să reții)
- [ ] Poți explica la meeting cum funcționează Q-Learning

**Timp estimat:** 2 ore

---

#### Task 1.4: Setup Python environment
**Deadline:** 5 decembrie

**Ce trebuie să faci:**
- [ ] Instalează Python 3.10+ (dacă nu ai)
- [ ] Creează environment conda sau venv:
  ```bash
  conda create -n rl_project python=3.10
  conda activate rl_project
  ```
- [ ] Instalează librării de bază:
  ```bash
  pip install gymnasium
  pip install numpy matplotlib
  pip install torch  # optional, pentru DQN mai târziu
  ```
- [ ] Testează că merge:
  ```python
  import gymnasium as gym
  env = gym.make("CartPole-v1", render_mode="human")
  env.reset()
  for _ in range(100):
      env.step(env.action_space.sample())
  env.close()
  ```

**Output:**
- [ ] Environment funcțional
- [ ] Screenshot că merge (pune în chat echipa)

**Timp estimat:** 30 min - 1 oră

---

###  ETAPA 2: Setup Proiect (5 dec - 12 dec)
**Obiectiv:** Repo GitHub + structură + environment wrapper

#### Task 2.1: Participă la setup repo (cu echipa)
- [ ] Creați împreună repo GitHub
- [ ] Clone local
- [ ] Structură de foldere (vezi `01_PLAN_PROIECT.md`)

#### Task 2.2: Înțelege environment-ul ales
- [ ] După confirmare prof, studiază în profunzime env-ul ales
- [ ] Rulează agent random
- [ ] Notează statistici baseline (avg reward, episoade)

#### Task 2.3: Ajută la wrapper reward modificat (cu Irina)
- [ ] Înțelege cum se face un Gymnasium wrapper
- [ ] Contribuie la ideile de reward shaping
- [ ] Testează că wrapper-ul funcționează

**Timp estimat:** 4-5 ore (împărțit cu echipa)

---

###  ETAPA 3: Implementare Q-Learning (12 dec - 30 dec)
**Obiectiv:** Agent Q-Learning funcțional + rezultate

#### Task 3.1: Studiază implementare Q-Learning
**Deadline:** 15 decembrie

**Ce trebuie să faci:**
- [ ] Citește exemplu Q-Learning din Lab 4
- [ ] Studiază template-uri online (CleanRL, Spinning Up)
- [ ] Înțelege structura:
  ```
  1. Inițializare Q-table (sau Q-dict pentru spații mari)
  2. Loop episoade:
      a. Reset environment
      b. Loop timesteps:
          - Alege acțiune (ε-greedy)
          - Execute acțiune
          - Primește (s', r, done)
          - Update Q(s,a)
          - s ← s'
      c. Decay epsilon
      d. Log reward
  3. Plot rezultate
  ```

**Output:**
- [ ] Pseudocod propriu (scris de tine)
- [ ] Lista de întrebări (dacă ai nelămuriri)

**Timp estimat:** 3-4 ore

---

#### Task 3.2: Implementează Q-Learning - Versiunea 1 (default reward)
**Deadline:** 20 decembrie

**Ce trebuie să faci:**
- [ ] Creează `agents/q_learning/train.py`
- [ ] Implementează:
  - [ ] Q-table (dict sau numpy array, depinde de env)
  - [ ] Funcție `choose_action(state, epsilon)`
  - [ ] Funcție `update_q(s, a, r, s_next, done)`
  - [ ] Loop training
  - [ ] Logging (reward per episod)
  - [ ] Salvare Q-table final
  
- [ ] Config file: `experiments/q_learning_config.json`
  ```json
  {
    "algorithm": "Q-Learning",
    "environment": "CartPole-v1",
    "reward_type": "default",
    "hyperparameters": {
      "alpha": 0.1,
      "gamma": 0.99,
      "epsilon_start": 1.0,
      "epsilon_end": 0.01,
      "epsilon_decay": 0.995,
      "num_episodes": 5000
    },
    "seed": 42
  }
  ```

**Output:**
- [ ] Script `train.py` funcțional
- [ ] Q-table salvată (`q_table_default.pkl`)
- [ ] Log file cu reward-uri (`results/q_learning/default_rewards.csv`)
- [ ] Grafic: Reward vs Episode (`results/q_learning/default_learning_curve.png`)

**Timp estimat:** 8-10 ore (împărțit în mai multe zile)

---

#### Task 3.3: Discretizare state space (dacă e nevoie)
**Doar dacă environment-ul are state continuous și nu merge altfel**

Exemplu pentru CartPole:
```python
def discretize_state(state, bins):
    # state = [cart_pos, cart_vel, pole_angle, pole_vel]
    # bins = [10, 10, 10, 10] (de ex.)
    state_discrete = []
    for i, val in enumerate(state):
        state_discrete.append(np.digitize(val, bins[i]))
    return tuple(state_discrete)
```

- [ ] Implementează discretizare
- [ ] Testează cu diferite număr de bins
- [ ] Documentează alegerea (de ce X bins?)

---

#### Task 3.4: Implementează Q-Learning - Versiunea 2 (reward modificat)
**Deadline:** 25 decembrie

**Ce trebuie să faci:**
- [ ] Copy script de la 3.2
- [ ] Schimbă environment la wrapper cu reward modificat
- [ ] Păstrează aceiași hiperparametri (fair comparison)
- [ ] Rulează training
- [ ] Salvează rezultate separate

**Output:**
- [ ] Script `train_modified.py` (sau flag în config)
- [ ] Q-table nou (`q_table_modified.pkl`)
- [ ] Log file (`results/q_learning/modified_rewards.csv`)
- [ ] Grafic learning curve

**Timp estimat:** 2-3 ore (code reuse)

---

#### Task 3.5: Experimente & Tuning
**Deadline:** 28 decembrie

**Ce trebuie să faci:**
- [ ] **Experiment 1:** Variație learning rate
  - α = [0.05, 0.1, 0.2]
  - Păstrezi restul constant
  - 3 seed-uri per configurație
  
- [ ] **Experiment 2:** Variație epsilon decay
  - epsilon_decay = [0.99, 0.995, 0.999]
  - Observi: cât de repede converge?
  
- [ ] **Experiment 3:** Variație gamma (discount factor)
  - γ = [0.95, 0.99, 0.999]
  - Observi: cât de "farsighted" e agentul?

**Output:**
- [ ] Toate rezultatele în `results/q_learning/experiments/`
- [ ] Tabel comparativ (Excel/CSV):
  ```
  Config | Alpha | Gamma | Epsilon_decay | Avg_reward | Std_dev | Episodes_to_solve
  ```
- [ ] Grafice comparative (overlayed learning curves)

**Timp estimat:** 4-5 ore

---

#### Task 3.6: Documentare Q-Learning
**Deadline:** 30 decembrie

**Ce trebuie să faci:**
- [ ] Scrie `agents/q_learning/README.md` cu:
  - Explicație algoritm (intuitiv, fără formule complicate)
  - Hiperparametri aleși + motivație
  - Rezultate (reward default vs modified)
  - Probleme întâmpinate + soluții
  - Instructiuni rulare: `python train.py --config config.json`
  
- [ ] Comentează codul (docstrings pentru funcții)

- [ ] Pregătește 2-3 slide-uri pentru prezentare:
  - Slide 1: Ce e Q-Learning? (schema + update rule)
  - Slide 2: Rezultate (grafice mari!)
  - Slide 3: Insights (ce ai învățat?)

**Output:**
- [ ] README complet
- [ ] Cod comentat
- [ ] Draft slide-uri (ppt sau markdown)

**Timp estimat:** 3-4 ore

---

###  ETAPA 4: Comparație & Analiză (30 dec - 8 ian)
**Obiectiv:** Comparație cu DQN & PPO (agenții făcuți de Matei & Iustin)

#### Task 4.1: Rulează baseline comparison
- [ ] Asigură-te că toți algoritmii au rulat cu același seed
- [ ] Colectează metrici:
  - Reward mediu final
  - Episoade până la convergență
  - Timp de training
  - Sample efficiency

#### Task 4.2: Analiză Q-Learning vs DQN vs PPO
- [ ] De ce Q-Learning e mai rapid/mai lent?
- [ ] Care algoritm beneficiază mai mult de reward shaping?
- [ ] Stabilitate: varianța reward-ului

**Output:**
- [ ] Tabel comparativ mare (toți algoritmii)
- [ ] Grafic: learning curves overlayed (3 culori)
- [ ] Mini-raport: 1-2 pagini cu insights

**Timp estimat:** 3-4 ore

---

###  ETAPA 5: Prezentare (8 ian - 15 ian)
**Obiectiv:** Prezentare clară, sigură, profesională

#### Task 5.1: Pregătește partea ta din prezentare
- [ ] 2-3 slide-uri Q-Learning
- [ ] Grafice mari și clare
- [ ] Explicații simple (nu hardcore math)
- [ ] Demo opțional: video agent învățat

#### Task 5.2: Rehearsal
- [ ] Exersează singură (3-5 min)
- [ ] Rehearsal cu echipa (2-3 zile înainte)
- [ ] Pregătește răspunsuri la întrebări posibile:
  - De ce Q-Learning?
  - Cum ai ales hiperparametrii?
  - De ce discretizare? (dacă e cazul)
  - Ce probleme ai întâmpinat?

#### Task 5.3: Review final documentație
- [ ] Verifică că tot textul e scris de voi (fără LLM)
- [ ] Verifică că README e clar
- [ ] Verifică că toți înțeleg codul

**Timp estimat:** 4-5 ore

---

##  RESURSE SPECIFICE PENTRU TINE

### Documentație Q-Learning:
- **Sutton & Barto - Cap 6** (Temporal Difference Learning)
- **Lab 4** din materiale noastre
- **CleanRL Q-Learning:** https://github.com/vwxyzjn/cleanrl

### Tutoriale video:
- **David Silver Lecture 4 & 5** (Model-Free Prediction & Control)
- **Deeplizard - Q-Learning series** (YouTube)

### Gymnasium:
- **Docs:** https://gymnasium.farama.org/
- **Examples:** https://gymnasium.farama.org/content/basic_usage/

### Python tools:
- **Matplotlib** pentru grafice
- **Pandas** pentru logging
- **Pickle** pentru salvare Q-table

---

##  TIPS & TRICKS PENTRU TINE

### Pentru implementare:
 **Începe simplu:** Fă-l să meargă pe reward default întâi  
 **Debug cu print-uri:** Verifică că Q-values cresc  
 **Seed-uri fixe:** `np.random.seed(42)`, `env.reset(seed=42)`  
 **Logging frecvent:** La fiecare 100 episoade, printează avg reward  
 **Salvează Q-table:** Să poți reîncepe dacă cade  

### Pentru învățare:
 **Înțelege, nu memora:** Explică cu voce tare algoritmul  
 **Desenează:** Schema MDP, update rule vizual  
 **Experimentează:** Schimbă alpha, vezi ce se întâmplă  
 **Compară cu DQN:** Când ajungeți acolo, vezi diferențele  

### Pentru prezentare:
 **Povestește:** "Am început cu reward default, nu mergea, am adăugat shaping..."  
 **Vizual:** Grafice mari, culori clare  
 **Demo:** Un gif/video cu agentul învățat e gold  
 **Onestitate:** E ok să spui "am avut probleme cu X, am rezolvat prin Y"  

---

##  ÎNTREBĂRI DE VERIFICARE (pentru tine)

**Înainte de meeting #1 (5 dec), poți răspunde la:**
- [ ] Ce e un Q-value?
- [ ] De ce folosim max în update rule?
- [ ] Ce e diferența dintre on-policy și off-policy?
- [ ] De ce avem nevoie de ε-greedy?
- [ ] Cum aleg numărul de bins pentru discretizare?

**Înainte de implementare (15 dec), poți răspunde la:**
- [ ] Cum inițializez Q-table?
- [ ] Ce fac dacă state space e prea mare?
- [ ] Cum știu că algoritmul converge?
- [ ] Ce e TD error?
- [ ] De ce scad epsilon în timp?

**Înainte de prezentare (15 ian), poți răspunde la:**
- [ ] De ce Q-Learning e mai simplu decât DQN?
- [ ] Care sunt limitările Q-Learning?
- [ ] Când folosesc Q-Learning vs DQN?
- [ ] Ce am învățat din experimente?

---

##  SUCCESS CRITERIA

### Știi că ai reușit când:
 Ai un agent Q-Learning care învață (reward crește în timp)  
 Poți explica fiecare linie de cod  
 Ai grafice clare cu rezultate  
 Ai comparat reward default vs modified  
 Ai făcut experimente cu hiperparametri  
 Poți prezenta în 3-5 min fără să te blochezi  
 Poți răspunde la întrebări despre cod  
 Echipa ta înțelege ce ai făcut  

---

##  CÂND CERI AJUTOR

**Cere ajutor dacă:**
- Nu înțelegi update rule după 30 min
- Codul nu rulează după 1 oră de debug
- Q-values nu cresc deloc după multe episoade
- Nu știi cum să discretizezi state space
- Ești blocată la un bug ciudat

**Cui ceri ajutor:**
- **Irina:** Întrebări conceptuale, clarificări proiect
- **Matei/Iustin:** Probleme de cod, debug
- **Prof:** Întrebări teoretice, clarificări algoritm
- **Online:** Stack Overflow, Reddit r/reinforcementlearning

---

## ⏱ TIME MANAGEMENT

**Ore totale estimate:** ~40-50 ore (pe 6-7 săptămâni)

**Breakdown:**
- Teorie & research: 8-10 ore
- Setup & learning: 5-6 ore
- Implementare: 15-20 ore
- Experimente: 6-8 ore
- Documentare: 6-8 ore

**Weekly:**
- Săptămâna 1-2: 3-4 ore (research + teorie)
- Săptămâna 3-5: 6-8 ore (implementare + experimente)
- Săptămâna 6-7: 3-4 ore (documentare + prezentare)

---

##  MOTIVAȚIE

**De ce e important ce faci:**
- Q-Learning e FUNDAMENTUL RL → dacă înțelegi asta, restul vine mai ușor
- E primul algoritm → setezi standardul pentru echipă
- E simplu și elegant → beauty of simplicity!
- Poți să-l explici oricui → perfect pentru prezentare
- O să-l folosești toată viața în RL

**Mindset:**
- "Eu sunt experta echipei pe value-based methods"
- "Orice problemă are soluție, doar trebuie să caut"
- "Dacă nu merge, e ocazia să înțeleg mai bine"
- "Codul meu o să fie exemplu pentru ceilalți"

---

**TU POȚI! **

**Remember:**
- Focus pe înțelegere, nu pe memorare
- Ask questions (stupid questions don't exist!)
- Celebrate small wins (rulează codul? YAY!)
- E ok să faci greșeli (de aia e research!)

---

**Last updated:** 29 noiembrie 2025  
**Next check-in:** 5 decembrie (Meeting #1)  
**Tu ești:**  Algorithms Lead & Q-Learning Expert 
