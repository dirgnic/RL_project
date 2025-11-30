#  PLAN PROIECT RL - Echipa Irina, Ingrid, Matei, Iustin

**Deadline prezentare:** Săptămâna 14 (ianuarie 2026)  
**Data ultimului update:** 30 noiembrie 2025

---

## 🎯 ENVIRONMENT ALES: **TAXI-V3 (Extended)**

### Task Split (Official):
- **Person A (Iustin):** Taxi world designer - Add extensions (extra passengers/fuel/obstacles)
- **Person B (Irina):** Tabular RL engineer - Q-learning & SARSA implementation
- **Person C (Ingrid):** Deep value-based engineer - **DQN implementation**
- **Person D (Matei):** Policy-based engineer - REINFORCE implementation

---

##  CERINȚE PROIECT (din documentație)

### Ce trebuie să livrăm:
 **Environment ales/modificat**
- ✅ **TAXI-V3** with extensions (extra passengers, fuel, traffic penalties)
- Iustin implementează wrapper-ul custom peste Taxi-v3

 **Minim 3 algoritmi RL** din categorii diferite
- ✅ **Tabular:** Q-learning & SARSA (Irina)
- ✅ **Deep Value-Based:** DQN (Ingrid)
- ✅ **Policy-Based:** REINFORCE (Matei)

 **Experimente & tuning hiperparametri**
- Comparare: Original Taxi-v3 vs Extended Taxi
- Tuning pentru fiecare algoritm
- Ablation studies pentru extensii

 **Rezultate vizualizate**
- Training curves (reward, loss, convergență)
- Comparison plots: Tabular vs DQN vs REINFORCE
- State space analysis

 **Documentație**
- PowerPoint / PDF / LaTeX
- **FĂRĂ LLM pentru text!** (proful verifică)
- Structură: Intro → Taxi Extensions → Algoritmi → Experimente → Concluzii

 **Prezentare în echipă** (săptămâna 14)
- Fiecare prezintă partea sa
- Explicăm de ce Taxi-v3 + ce extensii am adăugat

---

##  ETAPE PROIECT

###  ETAPA 1: Teorie + Alegere Environment (29 nov - 5 dec)
**Status:**  În progres

#### Taskuri individuale:
- [x] **Toți:** Citiți primele 3 cursuri RL  (Ingrid a făcut rezumat)
- [ ] **Toți:** Citiți primele pagini din pdf-ul profului
- [ ] **Fiecare:** Research 1-2 environment-uri Gymnasium + idei reward shaping

#### Taskuri echipă:
- [ ] **Meeting #1** (până pe 5 dec):
  - Fiecare prezintă 1-2 env propuse
  - Alegem 1-2 finaliste
  - Stabilim întrebări pentru prof
- [ ] **Irina:** Scrie profului pe Teams cu propunerea

**Deliverables:**
- Document "02_ENVIRONMENT_RESEARCH.md" (completat de toți)
- 1-2 environment-uri finaliste
- Lista de întrebări pentru prof

---

###  ETAPA 2: Setup Proiect (5 dec - 12 dec)
**Status:**  Nu a început

**Depinde de:** Confirmarea profului pentru environment

#### Taskuri:
- [ ] **Toți:** Creăm repo GitHub
- [ ] **Matei + Iustin:** Setup Python environment
  - Python 3.10+
  - Gymnasium
  - PyTorch/TensorFlow
  - Stable Baselines3 (optional)
  - Matplotlib, wandb (logging)
  
- [ ] **Structură proiect:**
  ```
  RL_PROJECT/
   env/                 # Environment + reward modificat
   agents/              # Script-uri algoritmi
      q_learning/
      dqn/
      ppo/
   experiments/         # Configuri + seed-uri
   results/             # Grafice + logs
   docs/                # Documentație
   notebooks/           # Jupyter pentru explorare
  ```

- [ ] **Implementare environment:**
  - [ ] Load environment default
  - [ ] Testăm că merge (random agent)
  - [ ] Implementăm wrapper pentru reward modificat
  - [ ] Documentăm reward original vs modificat

**Deliverables:**
- Repo GitHub funcțional
- Environment rulează + wrapper reward

---

###  ETAPA 3: Implementare Agenți (12 dec - 30 dec)
**Status:**  Nu a început

#### Împărțire roluri (sugestii):

#####  **Ingrid: Agent #1 - Q-Learning (Tabular)**
**De ce:** Simplu, bun pentru învățare, Ingrid își dorește să stăpânească algoritmii clasici

**Taskuri:**
- [ ] Studiază Q-Learning (lab 4)
- [ ] Implementează Q-table
- [ ] ε-greedy exploration
- [ ] Update rule: `Q(s,a) ← Q(s,a) + α[r + γ*max Q(s',a') - Q(s,a)]`
- [ ] Rulează pe reward default
- [ ] Rulează pe reward modificat
- [ ] Logging: reward/episod, Q-values
- [ ] Mini-raport: ce parametri, ce rezultate

**Output:**
- Script `agents/q_learning/train.py`
- Config `experiments/q_learning_config.json`
- Grafice în `results/q_learning/`

---

#####  **Matei: Agent #2 - DQN (Deep Q-Network)**
**De ce:** Deep RL, rețele neuronale, scalabil

**Taskuri:**
- [ ] Studiază DQN (lab 5)
- [ ] Implementează neural network (PyTorch)
- [ ] Replay Buffer
- [ ] Target Network (actualizare periodică)
- [ ] Loss function: MSE între Q-predicted și Q-target
- [ ] Rulează pe reward default
- [ ] Rulează pe reward modificat
- [ ] Logging: reward, loss, epsilon decay
- [ ] Mini-raport: arhitectură rețea, hiperparametri

**Output:**
- Script `agents/dqn/train.py`
- Model salvat `agents/dqn/model.pth`
- Config `experiments/dqn_config.json`
- Grafice în `results/dqn/`

---

#####  **Iustin: Agent #3 - PPO (Policy-Based)**
**De ce:** State-of-the-art, policy gradient, stabilitate

**Taskuri:**
- [ ] Studiază PPO (lab 7 + Stable Baselines3 docs)
- [ ] Opțiune A: Implementare de la zero (advanced)
- [ ] Opțiune B: Folosește Stable Baselines3 (recomandat)
- [ ] Config: learning rate, clip ratio, GAE lambda
- [ ] Rulează pe reward default
- [ ] Rulează pe reward modificat
- [ ] Logging: reward, policy loss, value loss
- [ ] Mini-raport: arhitectură, clip ratio, rezultate

**Output:**
- Script `agents/ppo/train.py` (sau wrapper SB3)
- Config `experiments/ppo_config.json`
- Grafice în `results/ppo/`

---

#####  **Irina: Coordonare + Environment + Setup Experimente**
**De ce:** Rol de coordonare, design environment, setup infrastructură

**Taskuri:**
- [ ] Design reward shaping (2-3 variante)
- [ ] Wrapper environment custom
- [ ] Setup logging comun (wandb sau matplotlib)
- [ ] Template config pentru experimente
- [ ] Documentare reward original vs modificat
- [ ] Verifică că toți respectă structura proiectului
- [ ] Comunicare cu proful (întrebări, clarificări)
- [ ] Review code (ajută la debug)

**Output:**
- `env/custom_env.py` (wrapper)
- `env/reward_analysis.md`
- `experiments/template_config.json`
- Documentație în `docs/environment.md`

---

###  ETAPA 4: Experimente & Tuning (30 dec - 8 ian)
**Status:**  Nu a început

#### Scenarii de experimente:

**Experiment 1: Baseline (reward default)**
- Rulăm Q-Learning, DQN, PPO
- Seed fix: 42
- Hiperparametri default
- Logging: reward/episod, timp de training

**Experiment 2: Reward modificat**
- Aceiași algoritmi
- Reward shaping aplicat
- Comparație cu baseline

**Experiment 3: Tuning hiperparametri**
- Variații learning rate: [0.001, 0.0001]
- Variații epsilon decay
- Variații arhitectură rețea (DQN)
- Grid search sau random search

**Experiment 4: Seed-uri multiple**
- 3-5 seed-uri diferite pentru fiecare algoritm
- Calculăm medie + deviere standard
- Verificăm stabilitatea

#### Metrici de comparat:
- Reward mediu per episod
- Timp de convergență (câte episoade până la threshold)
- Stabilitate (varianța reward-ului)
- Sample efficiency (câte samples până la rezultat ok)

**Deliverables:**
- Toate rezultatele în `results/`
- Tabele comparative în Excel/CSV
- Grafice pentru fiecare experiment

---

###  ETAPA 5: Documentare & Prezentare (8 ian - 15 ian)
**Status:**  Nu a început

#### Structură documentație:

**1. Introducere & Motivație** (1-2 slide-uri)
- De ce am ales tema asta?
- Ce probleme rezolvă?
- Context real (opțional)

**2. Environment** (2-3 slide-uri)
- Descriere mediu (stări, acțiuni, terminare)
- Reward original
- Reward modificat (motivație, formulă)
- Visualizări (screenshot, schema)

**3. Algoritmi** (3-4 slide-uri, 1 per algoritm)
- Explicație intuitivă (nu hardcore math)
- Pseudocod / schema
- Hiperparametri folosiți
- Arhitectură (pentru DQN/PPO)

**4. Experimente** (3-4 slide-uri)
- Setup experimente (seed-uri, config)
- Rezultate: GRAFICE MARI ȘI CLARE
- Tabele comparative
- Analiză: ce a mers, ce nu, de ce

**5. Probleme & Soluții** (1-2 slide-uri)
- Ce a fost greu?
- Cum am rezolvat?
- Instabilități, bug-uri interesante

**6. Concluzii & Lecții** (1 slide)
- Ce am învățat?
- Ce am face diferit?
- Direcții viitoare

**7. Demo (opțional)** (1 slide + video scurt)
- Agent antrenat în acțiune
- Comparație înainte/după

#### Taskuri documentare:
- [ ] **Irina:** Coordonare, introducere, environment
- [ ] **Ingrid:** Secțiunea Q-Learning, explicații intuitive
- [ ] **Matei:** Secțiunea DQN, arhitecturi
- [ ] **Iustin:** Secțiunea PPO, experimente
- [ ] **Toți:** Review final, verificare text (fără LLM!)

#### Pregătire prezentare:
- [ ] Fiecare își prezintă partea (3-5 min/persoană)
- [ ] Rehearsal complet (2-3 zile înainte)
- [ ] Pregătim răspunsuri la întrebări posibile
- [ ] Verificăm că toți putem explica codul

**Deliverables:**
- PowerPoint final
- README.md pentru GitHub
- Cod comentat și curat
- Video demo (optional, dar nice to have)

---

##  TRACKING PROGRES

### Week by week:

| Săptămână | Dată | Milestone | Status |
|-----------|------|-----------|--------|
| S1 | 29 nov - 5 dec | Research env + meeting #1 |  În progres |
| S2 | 5 dec - 12 dec | Confirmare env + setup proiect |  Planned |
| S3 | 12 dec - 19 dec | Implementare agent #1 + #2 |  Planned |
| S4 | 19 dec - 26 dec | Implementare agent #3 |  Planned |
| S5 | 26 dec - 2 ian | Finalizare agenți + teste |  Planned |
| S6 | 2 ian - 8 ian | Experimente & tuning |  Planned |
| S7 | 8 ian - 15 ian | Documentare + rehearsal |  Planned |
| S8 | 15 ian - 22 ian | PREZENTARE (săpt 14) |  Planned |

---

##  ÎNTREBĂRI PENTRU PROF

**De întrebat la următorul contact:**

1. Environment-ul nostru ales e ok? (după ce alegem)
2. Reward shaping înseamnă doar modificarea formulei de reward, sau și alte chestii?
3. E ok să folosim Stable Baselines3 pentru unul din algoritmi, sau trebuie totul de la zero?
4. Cât de complexe trebuie să fie experimentele? (câte seed-uri, câte combinații hiperparametri?)
5. Lungimea prezentării? (15-20 min?)
6. Ce fel de întrebări pune de obicei la prezentare?

---

##  NEXT STEPS IMEDIATE (TOATĂ ECHIPA)

### Până pe 5 decembrie (Meeting #1):

1. **Toți citesc:**
   - [ ] `00_CHEAT_SHEET_RL.md` (făcut de Ingrid)
   - [ ] Primele 3 cursuri RL
   - [ ] `02_ENVIRONMENT_RESEARCH.md` (creăm următorul pas)

2. **Fiecare completează în `02_ENVIRONMENT_RESEARCH.md`:**
   - 1-2 environment-uri propuse
   - Pro/contra pentru fiecare
   - Idei de reward shaping

3. **Ingrid (task special):**
   - Aprofundează CartPole, MountainCar, LunarLander
   - Pregătește comparație detaliată
   - Idei concrete de reward modifications

4. **Irina:**
   - Verifică disponibilitate pentru meeting
   - Pregătește întrebările pentru prof
   - Verifică că toți au înțeles cerințele

5. **Meeting #1:**
   - Prezentăm toți ce am găsit
   - Alegem 1-2 env finaliste
   - Stabilim următorii pași

---

##  RESURSE UTILE

- **Lab materials:** `/Laboratoare/` (avem deja!)
- **Cheat sheet:** `00_CHEAT_SHEET_RL.md`
- **Gymnasium docs:** https://gymnasium.farama.org/
- **Stable Baselines3:** https://stable-baselines3.readthedocs.io/
- **ChatGPT/Copilot:** OK pentru cod (dar să înțelegem!)

---

##  TIPS & TRICKS

### Pentru cod:
-  Folosiți seed-uri fixe (reproducibilitate)
-  Logging consistent (wandb sau csv)
-  Comentați codul (ajută la prezentare)
-  Git commits regulate (nu tot într-un commit)
-  Testați pe reward default întâi (baseline)

### Pentru documentație:
-  Scrieți în română simplă (fără LLM!)
-  Grafice MARI și clare
-  Explicații intuitive (nu formule complicate)
-  Focus pe rezultate și insights
-  Includeți "ce nu a mers" (e ok!)

### Pentru prezentare:
-  Fiecare știe ce a făcut
-  Putem explica codul dacă întreabă
-  Rehearsal obligatoriu (2-3 zile înainte)
-  Pregătim răspunsuri la întrebări frecvente
-  Demo vizual (video/gif) e super util

---

**Last updated:** 29 noiembrie 2025  
**Echipă:** Irina , Ingrid ^^, Matei, Iustin  
**Vers: 1.0**

 **LET'S GO!!!** 
