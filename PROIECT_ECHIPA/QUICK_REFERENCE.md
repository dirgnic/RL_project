#  QUICK REFERENCE CARD - Proiect RL

**Print this or keep it open while working!**

---

##  DOCUMENTELE (în ordine de citire)

```
1.  README.md                    → Start here!
2.  05_TEAM_SUMMARY.md          → Ce face toată echipa
3.  06_VISUAL_OVERVIEW.md       → Overview vizual
4.  00_CHEAT_SHEET_RL.md        → Concepte RL
5.  01_PLAN_PROIECT.md          → Plan detaliat
6.  02_ENVIRONMENT_RESEARCH.md  → Environments (COMPLETAȚI!)
7.  03_INGRID_TASKURI.md        → Taskuri Ingrid
8.  04_QUICK_START.md           → Setup & code templates
```

---

##  CHECKLIST IMEDIAT (săptămâna 1)

### Astăzi (29 nov):
- [ ] Citit README.md
- [ ] Citit 05_TEAM_SUMMARY.md
- [ ] Citit 06_VISUAL_OVERVIEW.md
- [ ] Început setup Python

### Mâine (30 nov):
- [ ] Citit curs 1 + 2 RL
- [ ] Research CartPole/LunarLander
- [ ] Test random agent local

### Weekend (30 nov - 1 dec):
- [ ] Citit curs 3 RL
- [ ] Research aprofundat environments
- [ ] Completat 02_ENVIRONMENT_RESEARCH.md

### Până pe 5 dec:
- [ ] Citit 00_CHEAT_SHEET_RL.md
- [ ] Setup Python complet
- [ ] **MEETING #1** 

---

##  CINE FACE CE

```
IRINA       → Coordonare + Environment + Experiments
INGRID ^^     → Q-Learning + Env Research + Docs
MATEI         → DQN + Neural Networks
IUSTIN        → PPO + Experiments + Grafice
```

---

##  MILESTONES

```
5 dec   → Meeting #1: alegem environment
12 dec  → Setup proiect + repo GitHub
2 ian   → Toți algoritmii funcționali
8 ian   → Experimente done
15 ian  → Documentare + rehearsal
Săpt 14 → PREZENTARE! 
```

---

##  CE LIVRĂM

```
 1 Environment (Gymnasium + reward shaping)
 3 Algoritmi (Q-Learning, DQN, PPO)
 Experimente (default vs modified reward)
 Rezultate (grafice + tabele)
 Documentație (PPT, fără LLM!)
 Prezentare (săpt 14, în echipă)
```

---

##  ALGORITMI (ce învață fiecare)

```
Q-LEARNING (Ingrid)
 Type: Value-based, tabular
 Formula: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
 Best for: Discrete spaces
 Time: ~40-50 ore

DQN (Matei)
 Type: Value-based, deep
 Network: Q(s,a) ≈ NN(s,a)
 Tricks: Replay Buffer + Target Network
 Time: ~40-50 ore

PPO (Iustin)
 Type: Policy-based, actor-critic
 Idea: Direct policy learning with clip
 Library: Stable Baselines3 (sau manual)
 Time: ~40-50 ore
```

---

##  ENVIRONMENTS (candidați)

```
CARTPOLE-v1
 Difficulty: 
 Pro: Simplu, rapid, bine documentat
 Contra: Poate prea simplu
 Recomandat: Safe choice (nota 8-9)

LUNARLANDER-v2
 Difficulty: 
 Pro: Foarte vizual, complex, challenge
 Contra: Mai lent, Q-Learning dificil
 Recomandat: Ambitious choice (nota 9-10)

MOUNTAINCAR-v0
 Difficulty:  (tricky!)
 Pro: Demonstrează importanța shaping
 Contra: Sparse reward, greu inițial
 Recomandat: Didactic choice (nota 8-9)
```

---

##  REWARD SHAPING IDEAS

```
CartPole:
  • Penalizare oscilații (angular velocity)
  • Bonus pentru poziție centrală
  • Penalizare progresivă la margini

LunarLander:
  • Fuel efficiency (mai puțin combustibil)
  • Safety first (evită crash-uri)
  • Speed landing (aterizare rapidă)

MountainCar:
  • Height-based (bonus pentru înălțime)
  • Momentum reward (bonus pentru viteză)
  • Potential-based shaping
```

---

##  SETUP RAPID

```bash
# Python environment
conda create -n rl_project python=3.10
conda activate rl_project

# Install libraries
pip install gymnasium numpy matplotlib torch

# Test
python -c "import gymnasium; print(' Works!')"
```

---

##  METRICI IMPORTANTE

```
Pentru fiecare algoritm:
  • Average Reward (ultimele 100 ep)
  • Convergence Time (ep până la threshold)
  • Sample Efficiency (samples → target)
  • Stability (std deviation)
  • Training Time (minutes)

Comparații:
  • Q-Learning vs DQN vs PPO
  • Default reward vs Modified
  • Impact hiperparametri
  • Robustness (multiple seeds)
```

---

##  DON'Ts (NU FACEȚI)

```
 NU folosiți LLM pentru text documentație
 NU așteptați ultima clipă
 NU copiați cod fără să înțelegeți
 NU ignorați seed-urile (reproducibilitate!)
 NU faceți un singur commit la final
 NU veniți la prezentare fără rehearsal
```

---

##  DO's (FACEȚI)

```
 Citiți documentele în ordine
 Întrebați când nu înțelegeți
 Testați frecvent (nu așteptați)
 Seed-uri fixe (42, 123, 456)
 Git commits regulate
 Comunicare în echipă
 Rehearsal minim 2x
```

---

##  RESURSE QUICK

```
Docs:
  • Gymnasium: gymnasium.farama.org
  • Stable-Baselines3: stable-baselines3.rtfd.io
  • PyTorch: pytorch.org/tutorials

Book:
  • Sutton & Barto (free PDF)
    - Cap 6 → Q-Learning
    - Cap 9 → DQN
    - Cap 13 → PPO

Code:
  • CleanRL: github.com/vwxyzjn/cleanrl
  • Spinning Up: spinningup.openai.com
```

---

##  STUCK? (când ceri ajutor)

```
1. Blocat >1h pe cod → Ask în grup
2. Algoritm nu converge → Check hiperparametri + seed
3. Bug ciudat → Google + Stack Overflow
4. Nelămuriri teoretice → Ask prof
5. Nelămuriri cerințe → Ask Irina
```

---

##  SUCCESS FORMULA

```
Success = 
    Understanding RL concepts
  + Clean implementation
  + Good experiments
  + Clear documentation
  + Team collaboration
  + Confident presentation
```

---

##  CONTACTS

```
Echipă: [Grup WhatsApp/Telegram]
Irina:  [coordonare + prof communication]
Prof:   stefan.iordache10@s.unibuc.ro
Repo:   [GitHub link TBD]
```

---

##  MOTIVATION

```
"The only way to learn RL is to implement RL."

You're not just building algorithms,
you're building:
  • Problem-solving skills
  • Coding skills
  • Research skills
  • Presentation skills
  • Teamwork skills

AND a grade! 
```

---

##  FINAL DELIVERABLES

```
GitHub Repo with:
   Code (3 algorithms)
   Results (logs, plots, tables)
   Docs (README, PPT)
   Demo (video/gif optional)

Presentation (săpt 14):
   15-20 slides
   3-5 min/person
   Q&A ready
   Demo (if have)
```

---

##  THIS WEEK (29 nov - 5 dec)

```
Monday:    Read all docs, setup Python
Tuesday:   Read course 1-2, research env
Wednesday: Read course 3, test agents
Thursday:  Complete research, finalize
Friday:    MEETING #1 
```

---

**PRINT THIS & KEEP IT VISIBLE!** 

**Next action:** Read `05_TEAM_SUMMARY.md` 

**You got this!** 

---

_Created: 29 noiembrie 2025_  
_Team: Irina , Ingrid ^^, Matei, Iustin_  
_Version: 1.0_
