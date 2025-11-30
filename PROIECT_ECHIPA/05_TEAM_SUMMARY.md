#  SUMMARY - Ce trebuie să facă TOATĂ ECHIPA

**Deadline Meeting #1:** 5 decembrie 2025  
**Prezentare finală:** Săptămâna 14 (ianuarie 2026)

---

##  CE AM FĂCUT PÂNĂ ACUM

 **Ingrid** a citit primele cursuri RL  
 **Ingrid** a creat cheat sheet RL pentru echipă  
 **Echipa** știe cerințele proiectului (3 algoritmi + environment + reward shaping)  
 **Irina** a vorbit cu proful (environment Gymnasium + reward shaping)

---

##  CE TREBUIE SĂ FACĂ FIECARE ACUM

###  TOATĂ ECHIPA (până pe 5 decembrie):

1. **Citiți documentele create:**
   - [ ] `00_CHEAT_SHEET_RL.md` - Concepte de bază (30-40 min)
   - [ ] `01_PLAN_PROIECT.md` - Plan complet (20 min)
   - [ ] `02_ENVIRONMENT_RESEARCH.md` - Research environments (20 min)

2. **Citiți teorie RL:**
   - [ ] Primele 3 cursuri RL (curs 1, 2, 3) - (2-3 ore)
   - [ ] Primele pagini din pdf-ul profului

3. **Research individual environment:**
   - [ ] Fiecare alege 1-2 environment-uri Gymnasium
   - [ ] Completați secțiunea voastră în `02_ENVIRONMENT_RESEARCH.md`
   - [ ] Pregătiți argumente PRO/CONTRA

4. **Setup Python environment:**
   - [ ] Python 3.10+
   - [ ] Instalați: `gymnasium`, `numpy`, `matplotlib`
   - [ ] Testați că merge (rulați un agent random)

---

##  TASKURI PERSONALE (detaliate)

### **Ingrid** (Algorithms Lead + Q-Learning):
 **Document:** `03_INGRID_TASKURI.md` (citește în detaliu!)

**Prioritate 1 (până pe 5 dec):**
- [ ] Research aprofundat: CartPole, LunarLander, MountainCar
- [ ] Completează secțiunea ta în `02_ENVIRONMENT_RESEARCH.md`
- [ ] Studiază Q-Learning (lab 4, formula update)
- [ ] Setup Python environment local

**Responsabilități proiect:**
- Agent #1: Q-Learning (tabular, value-based)
- Environment analysis + idei reward shaping
- Cheat sheet & documentație teoretică
- Explicații algoritmi clasici pentru echipă

---

### **Irina** (Coordonare + Environment + Experiments):

**Prioritate 1 (până pe 5 dec):**
- [ ] Citește toate documentele create
- [ ] Research 1-2 environments (completează în `02_ENVIRONMENT_RESEARCH.md`)
- [ ] Studiază primele 3 cursuri RL
- [ ] Organizează Meeting #1 (stabilește dată/oră)

**Responsabilități proiect:**
- Coordonare echipă + comunicare cu proful
- Design reward shaping (2-3 variante)
- Implementare wrapper environment custom
- Setup logging comun + template config experimente
- Review code + documentare finală

**Întrebări de pregătit pentru prof:**
1. Environment-ul nostru ales e ok?
2. Reward shaping = doar modificare formulă reward?
3. E ok Stable Baselines3 pentru PPO?
4. Câte seed-uri / experimente trebuie?
5. Lungimea prezentării?

---

### **Matei** (DQN + Deep RL):

**Prioritate 1 (până pe 5 dec):**
- [ ] Citește documentele create (cheat sheet + plan + env research)
- [ ] Studiază primele 3 cursuri RL (focus pe Bellman, Q-values)
- [ ] Research 1-2 environments (completează în `02_ENVIRONMENT_RESEARCH.md`)
- [ ] Studiază Lab 5 (DQN) + PyTorch basics

**Responsabilități proiect:**
- Agent #2: DQN (Deep Q-Network)
- Implementare neural network (PyTorch)
- Replay Buffer + Target Network
- Logging: reward, loss, epsilon decay
- Setup experimente DQN

**Ce trebuie să înveți:**
- PyTorch basics (tensors, nn.Module, optimizer)
- Neural networks pentru RL
- Experience Replay (de ce e necesar?)
- Target Network (pentru stabilitate)

---

### **Iustin** (PPO + Policy-Based + Experiments):

**Prioritate 1 (până pe 5 dec):**
- [ ] Citește documentele create
- [ ] Studiază primele 3 cursuri RL (focus pe policy)
- [ ] Research 1-2 environments (completează în `02_ENVIRONMENT_RESEARCH.md`)
- [ ] Setup Python environment + instalează Stable Baselines3

**Responsabilități proiect:**
- Agent #3: PPO (Proximal Policy Optimization)
- Implementare Actor-Critic (sau folosește SB3)
- Setup experimente automate (scripts)
- Grafice comparative + analiză rezultate
- Logging advanced (wandb optional)

**Ce trebuie să înveți:**
- Policy gradient methods (intuitiv)
- PPO algorithm (clip ratio, GAE)
- Stable Baselines3 (dacă folosești)
- Plotting + data analysis (matplotlib, pandas)

---

##  TIMELINE MARE (toată echipa)

| Dată | Milestone | Taskuri |
|------|-----------|---------|
| **29 nov - 5 dec** | Research env | Citit teorie + research env + meeting #1 |
| **5 dec - 12 dec** | Setup proiect | Repo GitHub + confirmare env + wrapper reward |
| **12 dec - 19 dec** | Implementare #1 | Q-Learning + DQN start |
| **19 dec - 26 dec** | Implementare #2 | DQN + PPO start |
| **26 dec - 2 ian** | Finalizare agenți | Toți algoritmii funcționali |
| **2 ian - 8 ian** | Experimente | Rulare + tuning + logging |
| **8 ian - 15 ian** | Documentare | PPT + README + rehearsal |
| **15 ian - 22 ian** | **PREZENTARE** | Săptămâna 14 |

---

##  MEETING #1 (până pe 5 decembrie)

**Agenda:**

1. **Check-in (5 min):**
   - Toți au citit documentele?
   - Toți au făcut research?

2. **Prezentare environments (15-20 min):**
   - Fiecare prezintă 1-2 environments propuse
   - PRO/CONTRA pentru fiecare
   - Idei reward shaping

3. **Voting & Decizie (10 min):**
   - Alegem 1-2 environment-uri finaliste
   - Sau decidem direct unul

4. **Împărțire roluri (10 min):**
   - Confirmăm rolurile (Ingrid=Q-Learning, Matei=DQN, Iustin=PPO)
   - Clarificăm ce face fiecare

5. **Next steps (5 min):**
   - Irina scrie profului cu propunerea
   - Toți instalează Python environment
   - Stabilim când e Meeting #2 (după confirmare prof)

6. **Q&A (5 min):**
   - Întrebări, nelămuriri

**Total:** 45-60 min

**Pregătire pentru meeting:**
- Toți vin cu 1-2 environments + argumente
- Toți au citit documentele
- Toți au setup Python (măcar început)

---

##  ÎMPĂRȚIRE RESPONSABILITĂȚI (rezumat)

| Persoană | Algoritm | Rol secundar | Ore estimate |
|----------|----------|--------------|---------------|
| **Irina** | - | Coordonare + Environment + Experiments | 40-50 ore |
| **Ingrid** | Q-Learning | Environment research + Docs teoretice | 40-50 ore |
| **Matei** | DQN | Neural networks + Setup tehnic | 40-50 ore |
| **Iustin** | PPO | Experiments + Grafice + Analysis | 40-50 ore |

**Total echipă:** ~160-200 ore (40-50 ore/persoană pe 7-8 săptămâni)

---

##  CHECKLIST PROGRES

### Etapa 1: Research (29 nov - 5 dec)
- [ ] **Toți:** Citit teorie RL (cursuri 1-3)
- [ ] **Toți:** Research 1-2 environments
- [ ] **Toți:** Completat `02_ENVIRONMENT_RESEARCH.md`
- [ ] **Toți:** Setup Python environment
- [ ] **Echipa:** Meeting #1 done
- [ ] **Irina:** Scris profului cu propunere

### Etapa 2: Setup (5 dec - 12 dec)
- [ ] **Echipa:** Repo GitHub creat
- [ ] **Echipa:** Structură proiect done
- [ ] **Irina + Ingrid:** Wrapper reward implementat
- [ ] **Toți:** Testing environment funcționează

### Etapa 3: Implementare (12 dec - 2 ian)
- [ ] **Ingrid:** Q-Learning funcțional (default + modified)
- [ ] **Matei:** DQN funcțional (default + modified)
- [ ] **Iustin:** PPO funcțional (default + modified)
- [ ] **Toți:** Logging + rezultate salvate

### Etapa 4: Experimente (2 ian - 8 ian)
- [ ] **Toți:** Experimente cu hiperparametri
- [ ] **Toți:** Multiple seed-uri
- [ ] **Iustin:** Grafice comparative
- [ ] **Echipa:** Tabel comparativ metrici

### Etapa 5: Documentare (8 ian - 15 ian)
- [ ] **Toți:** PowerPoint finalizat (fără LLM!)
- [ ] **Toți:** README.md complet
- [ ] **Toți:** Cod comentat
- [ ] **Echipa:** Rehearsal (minim 2x)
- [ ] **Toți:** Pregătire întrebări posibile

---

##  TIPS GENERALE

### Pentru toată echipa:

**Setup:**
 Seed-uri fixe pentru reproducibilitate  
 Git commits regulate (nu tot într-un commit!)  
 Comunicare constantă (grup WhatsApp/Telegram)  
 Checkpoint-uri săptămânale (quick sync 15 min)

**Dezvoltare:**
 Start simplu, crește complexitatea treptat  
 Test frecvent (nu așteptați până la final)  
 Documentare pe măsură ce scrieți cod  
 Code review mutual (ajutați-vă reciproc)

**Documentație:**
 Scrieți în română simplă (fără LLM!)  
 Grafice MARI și clare  
 Focus pe insights, nu doar rezultate  
 E OK să includeți "ce nu a mers"

**Prezentare:**
 Fiecare prezintă partea la care a lucrat  
 3-5 min/persoană  
 Trebuie să puteți explica codul  
 Demo vizual = gold (video/gif)  
 Rehearsal OBLIGATORIU

---

##  RESURSE PENTRU TOATĂ ECHIPA

### Documentație:
- **Gymnasium:** https://gymnasium.farama.org/
- **Stable Baselines3:** https://stable-baselines3.readthedocs.io/
- **PyTorch tutorials:** https://pytorch.org/tutorials/

### Cursuri video (optional, dar utile):
- **David Silver RL Course** (YouTube) - foundational
- **Stanford CS234** (Emma Brunskill) - modern
- **Berkeley CS285** (Sergey Levine) - deep RL

### Cărți:
- **Sutton & Barto** - "Reinforcement Learning: An Introduction" (free PDF online)
- Cap 6 (TD Learning) - pentru Q-Learning
- Cap 9 (Function Approximation) - pentru DQN
- Cap 13 (Policy Gradient) - pentru PPO

### Codebases (inspirație):
- **CleanRL:** https://github.com/vwxyzjn/cleanrl (single-file implementations)
- **Spinning Up:** https://spinningup.openai.com/ (tutoriale + cod)

---

##  ÎNTREBĂRI FRECVENTE

**Q: Cât de complex trebuie să fie codul?**  
A: Simplu și clar > complex și confuz. Proful verifică înțelegerea, nu complexitatea.

**Q: Putem folosi ChatGPT/Copilot pentru cod?**  
A: DA, dar trebuie să înțelegeți ce face codul! Proful poate întreba.

**Q: Putem folosi LLM pentru documentație?**  
A: NU pentru text (proful a zis clar!). Doar pentru structurare/outline, apoi scrieți voi.

**Q: Cât de multe experimente?**  
A: Minim: baseline (default reward) + 1 reward modificat, pe toți algoritmii. Ideal: 2-3 variante reward + tuning hiperparametri.

**Q: Ce facem dacă un algoritm nu converge?**  
A: E OK! Documentați problema, ce ați încercat, de ce credeți că nu merge. E parte din research.

**Q: Cât de long PowerPoint-ul?**  
A: 15-20 slide-uri, prezentare ~20-25 min total (4-5 min/persoană + Q&A).

**Q: Trebuie demo live?**  
A: Nu e obligatoriu, dar un video/gif cu agent învățat e super impressive!

---

##  CÂND CERI AJUTOR

**Cereți ajutor dacă:**
- Blocați pe cod >1-2 ore
- Algoritm nu converge după multe încercări
- Bug ciudat care nu are sens
- Nelămuriri teoretice după research

**Cui cereți ajutor:**
- **În echipă:** Problema X cu cod/teorie
- **Proful:** Întrebări teoretice, clarificări cerințe
- **Online:** Stack Overflow, Reddit r/reinforcementlearning, GitHub issues

**NU așteptați până în ultima clipă!**

---

##  MOTIVAȚIE

**De ce e important:**
- RL e skill valoros (industrie: robotică, gaming, finance, AI)
- Experiență practică cu algoritmi foundational
- Învățare în echipă (colaborare, code review, prezentare)
- Satisfaction când vezi agentul învățând! 

**Mindset:**
- "Nu e despre nota perfectă, e despre ce învățăm"
- "Orice problemă are soluție, doar trebuie să căutăm"
- "Dacă e prea ușor, nu e challenge!"
- "Echipa > individual"

---

##  CONTACT

**Grup echipă:** [WhatsApp/Telegram]  
**GitHub repo:** [TBD după creare]  
**Profesor:** stefan.iordache10@s.unibuc.ro

---

##  NEXT IMMEDIATE ACTIONS (TOATĂ ECHIPA)

### Astăzi (29 nov):
- [ ] Citiți acest document
- [ ] Citiți `00_CHEAT_SHEET_RL.md`
- [ ] Setup Python environment (măcar început)

### Până mâine (30 nov):
- [ ] Citiți curs 1 + 2 RL
- [ ] Începeți research environment (CartPole/LunarLander)

### Până luni (2 dec):
- [ ] Citiți curs 3 RL
- [ ] Completați research environment
- [ ] Completați secțiunea voastră în `02_ENVIRONMENT_RESEARCH.md`

### Până joi (5 dec):
- [ ] Finalizați research
- [ ] Setup Python complet
- [ ] **MEETING #1**

---

**TU POȚI! NOI PUTEM! **

**Remember:**
- Communication is key 
- Ask questions early 
- Small steps every day 
- Celebrate small wins 
- We're a team! 

---

**Last updated:** 29 noiembrie 2025  
**Next update:** După Meeting #1 (5 dec)  
**Versiune:** 1.0

 **LET'S GO!!!** 
