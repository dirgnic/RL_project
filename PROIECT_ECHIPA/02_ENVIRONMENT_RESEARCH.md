#  ENVIRONMENT RESEARCH - Proiect RL

**Echipă:** Irina, Ingrid, Matei, Iustin  
**Scop:** Alegerea unui environment Gymnasium pentru proiect  
**Deadline decizie:** 5 decembrie 2025

---

##  CRITERII DE SELECȚIE

### Ce vrea proful:
 Environment din **Gymnasium** (recomandat pentru complexitate potrivită)  
 **Modificare = reward shaping** (schimbăm formula reward-ului)  
 Nu prea simplu (FrozenLake), dar nici prea complex (Atari cu pixeli)  
 Trebuie să fie compatibil cu cei 3 algoritmi (Q-Learning, DQN, PPO)

### Ce vrem noi:
 Să înțelegem cum funcționează  
 Să fie interesant vizual (demo la prezentare)  
 Documentație bună (tutoriale existente)  
 Să permită modificări clare de reward  
 Să nu fie prea lent la training

---

##  TOP 3 CANDIDATI (recomandare)

###  #1: CartPole-v1
**Categorie:** Classic Control  
**Dificultate:**  (Ușor-Mediu)

#### Descriere:
Un băț vertical este atasat de un cărucior. Scopul este să balansezi bățul vertical cât mai mult timp, mișcând căruciorul stânga/dreapta.

#### Specificații tehnice:
- **State space:** Continuous (4 dimensiuni)
  - Poziție cărucior: [-4.8, 4.8]
  - Viteză cărucior: [-Inf, Inf]
  - Unghi băț: [-0.418 rad, 0.418 rad] (~24°)
  - Viteză unghiulară băț: [-Inf, Inf]
  
- **Action space:** Discrete (2 acțiuni)
  - 0 = împinge stânga
  - 1 = împinge dreapta
  
- **Reward default:**
  - +1 pentru fiecare timestep în care bățul e vertical
  - Episod se termină dacă:
    - Băț cade (unghi > 12°)
    - Cărucior iese din bounds (|x| > 2.4)
    - 500 de timesteps (success!)

####  PRO:
- Foarte bine documentat (tone de tutoriale)
- Vizual simplu și clar
- Rapid de antrenat (câteva minute)
- Perfect pentru Q-Learning (cu discretizare)
- Perfect pentru DQN (state continuous, action discrete)
- Merge și cu PPO
- Folosit în lab-uri deja (avem experiență)

####  CONTRA:
- Poate fi prea simplu (dar depinde de reward shaping!)
- State space mic (dar e ok pentru început)

####  IDEI REWARD SHAPING:

**Variantă 1: Penalizare pentru oscilații**
```python
# Reward default: +1 per step
# Reward modificat:
reward = 1.0
reward -= 0.1 * abs(angular_velocity)  # penalizăm mișcări bruște
reward -= 0.05 * abs(cart_velocity)    # preferăm stabilitate
```
**Efect așteptat:** Agent mai smooth, mai puține oscilații

---

**Variantă 2: Bonus pentru centru**
```python
reward = 1.0
reward += 0.5 * (1 - abs(cart_position) / 2.4)  # bonus dacă e la centru
reward += 0.5 * (1 - abs(pole_angle) / 0.418)   # bonus dacă e vertical
```
**Efect așteptat:** Agent învață mai repede, preferă poziții centrale

---

**Variantă 3: Penalizare progresivă**
```python
reward = 1.0
if abs(pole_angle) > 0.2:  # dacă începe să cadă
    reward -= 1.0 * abs(pole_angle)  # penalizare mare
if abs(cart_position) > 2.0:  # dacă se apropie de margine
    reward -= 0.5
```
**Efect așteptat:** Agent evită situații periculoase mai agresiv

---

**Experimente posibile:**
- Reward default vs. Variantă 1 vs. Variantă 2 vs. Variantă 3
- Comparare Q-Learning, DQN, PPO pe fiecare variantă
- Analiză: care algoritm beneficiază cel mai mult de shaping?

---

###  #2: LunarLander-v2
**Categorie:** Box2D  
**Dificultate:**  (Mediu)

#### Descriere:
Aterizare navă spațială între doi markeri. Trebuie să controlezi motoarele pentru a ateriza safe, fără să te prăbușești.

#### Specificații tehnice:
- **State space:** Continuous (8 dimensiuni)
  - Coordonate x, y
  - Viteze vx, vy
  - Unghi și viteză unghiulară
  - Contact picioare stânga/dreapta (boolean)
  
- **Action space:** Discrete (4 acțiuni)
  - 0 = nimic
  - 1 = motor principal (sus)
  - 2 = motor stânga
  - 3 = motor dreapta
  
- **Reward default:**
  - +100 până la +140 pentru aterizare corectă
  - -100 dacă se prăbușește
  - +10 pentru fiecare picior care atinge solul
  - -0.3 pentru fiecare frame de motor principal
  - -0.03 pentru motoare laterale
  - Bonus pentru a fi aproape de landing pad
  - Penalizare pentru viteză/unghi mari

####  PRO:
- Foarte vizual (super pentru demo!)
- Reward complex → multe oportunități de shaping
- Challenge interesant
- Folosit în lab 4 (avem cod de început)
- State space suficient de complex
- Perfect pentru DQN

####  CONTRA:
- Mai lent de antrenat decât CartPole
- Q-Learning tabular necesită discretizare agresivă (multe stări)
- Poate fi frustrant (agenții mor mult la început)

####  IDEI REWARD SHAPING:

**Variantă 1: Fuel efficiency (economisire combustibil)**
```python
# Penalizăm mai mult folosirea combustibilului
reward_fuel = -0.5 * (main_engine + 0.1 * side_engines)  # default: -0.3, -0.03
# Bonus pentru aterizare soft
if landed_safely:
    reward += 50 * (1 - abs(velocity))  # bonus pentru viteză mică
```
**Efect așteptat:** Agent învață să folosească mai puțin combustibil

---

**Variantă 2: Safety first (evită crash-uri)**
```python
# Penalizare mare pentru viteză periculoasă
if abs(velocity_y) > 0.5 and altitude < 0.5:
    reward -= 50  # risc mare de crash
# Bonus pentru apropiere lentă
reward += 10 * (1 / (abs(velocity) + 0.1))
```
**Efect așteptat:** Agent mai conservator, evită riscuri

---

**Variantă 3: Speed landing (aterizare rapidă)**
```python
# Bonus pentru aterizare rapidă
reward += 0.1  # mic bonus per timestep
if landed_safely:
    reward += 100 / timesteps  # mai repede = mai bine
```
**Efect așteptat:** Agent învață să aterizeze repede, nu să planeze

---

**Experimente posibile:**
- Reward default vs. 3 variante
- Comparare DQN vs PPO (Q-Learning doar dacă discretizăm)
- Analiza traiectoriilor (plot trajectory pentru fiecare variant)
- Statistici: fuel usage, landing speed, success rate

---

###  #3: MountainCar-v0
**Categorie:** Classic Control  
**Dificultate:**  (Mediu - dar tricky!)

#### Descriere:
O mașină în vale între doi munți. Motorul e prea slab să urce direct, trebuie să folosești momentum (mergi înapoi, apoi accelerezi).

#### Specificații tehnice:
- **State space:** Continuous (2 dimensiuni)
  - Poziție: [-1.2, 0.6]
  - Viteză: [-0.07, 0.07]
  
- **Action space:** Discrete (3 acțiuni)
  - 0 = accelerează stânga
  - 1 = nu face nimic
  - 2 = accelerează dreapta
  
- **Reward default:**
  - -1 pentru fiecare timestep (penalizare pentru timp)
  - 0 când ajunge la flag (poziție >= 0.5)
  - Max 200 timesteps

####  PRO:
- Problemă interesantă (necesită strategie, nu doar reacție)
- State space foarte mic (2D) → ușor de vizualizat
- Bun pentru Q-Learning (discretizare simplă)
- Challenge: sparse reward (greu de învățat inițial)
- Demonstrează importanța reward shaping!

####  CONTRA:
- Foarte greu cu reward default (agentul nu învață aproape niciodată)
- Poate fi frustrant
- Mai puțin vizual decât LunarLander
- Necesită OBLIGATORIU reward shaping (altfel nu merge)

####  IDEI REWARD SHAPING:

**Variantă 1: Reward pentru înălțime (height-based)**
```python
# Reward default: -1 per step
# Reward modificat:
reward = -1
reward += 10 * (position - (-0.5))  # bonus pentru a ajunge mai sus
# Position e între -1.2 și 0.6, deci normalizăm
```
**Efect așteptat:** Agent învață să urce, nu să stea în vale

---

**Variantă 2: Reward pentru momentum**
```python
reward = -1
reward += abs(velocity) * 10  # recompensăm viteză mare (momentum)
if velocity > 0 and position > 0:  # merge în direcția bună
    reward += 20
```
**Efect așteptat:** Agent învață să construiască momentum

---

**Variantă 3: Potential-based shaping**
```python
# Folosim "potential function" (teoretic corect!)
potential = position  # sau position + abs(velocity)
reward = -1 + gamma * potential_next - potential_current
```
**Efect așteptat:** Converge la aceeași politică optimă, dar mai repede

---

**Experimente posibile:**
- Demonstrează că reward default NU merge (baseline fail)
- Compară cele 3 variante de shaping
- Vizualizează traiectorii (momentum learning)
- Foarte bun pentru a arăta IMPORTANȚA reward shaping

---

##  COMPARAȚIE CANDIDAȚI

| Criteriu | CartPole | LunarLander | MountainCar |
|----------|----------|-------------|-------------|
| **Complexitate** |  |  |  |
| **Vizual** |  |  |  |
| **Q-Learning** |  |  (needs discretization) |  |
| **DQN** |  |  |  |
| **PPO** |  |  |  |
| **Training speed** |  |  |  |
| **Documentație** |  |  |  |
| **Reward shaping oportunități** |  |  |  |
| **Demo potential** |  |  |  |

---

##  RECOMANDĂRI ECHIPĂ

### Opțiunea SAFE (nota 8-9): **CartPole-v1**
**De ce:**
- Simplu, rapid, bine documentat
- Toți algoritmii merg perfect
- Putem face multe experimente repede
- Sigur terminăm la timp
- Focus pe **calitatea implementării** și **experimentelor**

**Riscuri:**
- Poate părea "prea simplu" (dar depinde de shaping!)
- Trebuie să compensăm cu experimente solide

---

### Opțiunea AMBIȚIOASĂ (nota 9-10): **LunarLander-v2**
**De ce:**
- Foarte vizual (WOW factor la prezentare)
- Complex, multe oportunități de shaping
- Challenge interesant
- Demonstrează înțelegere profundă

**Riscuri:**
- Mai lent de antrenat
- Q-Learning mai complicat (discretizare)
- Poate să ne ia mai mult timp

---

### Opțiunea DIDACTICĂ (nota 8-9): **MountainCar-v0**
**De ce:**
- Demonstrează PERFECT importanța reward shaping
- Story bun pentru prezentare (fail → shaping → success)
- State space mic (vizualizări frumoase)

**Riscuri:**
- Mai puțin vizual
- Trebuie să facem shaping obligatoriu (nu e opțional)

---

##  SECȚIUNE COMPLETARE INDIVIDUALĂ

###  Ingrid:
**Environment-uri propuse:**
1. **CartPole-v1**
   - **Pro:** Simplu, rapid, perfect pentru Q-Learning, bine documentat
   - **Contra:** Poate prea simplu, dar cu reward shaping devine interesant
   - **Idei reward:** Penalizare oscilații + bonus centru + penalizare progresivă
   - **De ce îmi place:** Pot să-l stăpânesc bine, focus pe algoritmi

2. **LunarLander-v2**
   - **Pro:** Super vizual, complex, multe oportunități shaping
   - **Contra:** Mai lent, Q-Learning necesită mult preprocesing
   - **Idei reward:** Fuel efficiency, safety first, speed landing
   - **De ce îmi place:** Challenge + demo spectaculos

**Preferință:** CartPole pentru siguranță, dar sunt deschisă la LunarLander dacă echipa vrea!

---

###  Irina:
**Environment-uri propuse:**
(Completează tu aici)

---

###  Matei:
**Environment-uri propuse:**
(Completează tu aici)

---

###  Iustin:
**Environment-uri propuse:**
(Completează tu aici)

---

##  VOTING (completați la meeting)

După ce toți și-au prezentat argumentele:

| Environment | Voturi | Notițe |
|-------------|--------|--------|
| CartPole-v1 | | |
| LunarLander-v2 | | |
| MountainCar-v0 | | |
| Altul: _______ | | |

**Decizie finală:** ________________

**Motivație:** ________________

**Întrebări pentru prof:** ________________

---

##  RESURSE UTILE

### Documentație Gymnasium:
- CartPole: https://gymnasium.farama.org/environments/classic_control/cart_pole/
- LunarLander: https://gymnasium.farama.org/environments/box2d/lunar_lander/
- MountainCar: https://gymnasium.farama.org/environments/classic_control/mountain_car/

### Tutoriale:
- CartPole + Q-Learning: Multiple online (CleanRL, Spinning Up)
- LunarLander + DQN: Lab 5 + Stable Baselines3 docs
- MountainCar + Shaping: Cunoscut în literatură RL

### Lab-uri noastre:
- Lab 4: Q-Learning + SARSA (LunarLander)
- Lab 5: DQN (CartPole + LunarLander)
- Lab 7: Policy Gradients

---

**Next step:** Meeting #1 - prezentare + voting + întrebări prof!

**Deadline:** 5 decembrie 2025

 **GOOD LUCK!** 
