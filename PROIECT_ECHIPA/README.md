#  PROIECT RL - DOCUMENTAȚIE ECHIPĂ

**Universitate:** București, Facultatea Matematică-Informatică  
**Curs:** Introducere în Reinforcement Learning (2025-2026)  
**Echipă:** Irina , Ingrid ^^, Matei, Iustin  
**Profesor:** Drd. Ștefan Iordache & Conf. Dr. Ciprian Păduraru

---

##  SCOP

Această mapă conține **toate resursele necesare** pentru realizarea proiectului de Reinforcement Learning:
- Teorie & concepte de bază
- Plan detaliat pe etape
- Research environments
- Taskuri personalizate
- Quick start guide
- Templates & structură

---

##  STRUCTURA DOCUMENTELOR

###  START HERE - Citiți în această ordine:

1. ** 05_TEAM_SUMMARY.md**  **CITEȘTE PRIMUL!**
   - Ce trebuie să facă TOATĂ ECHIPA
   - Taskuri imediate
   - Timeline mare
   - Next actions

2. ** 06_VISUAL_OVERVIEW.md**  **APOI ĂSTA!**
   - Overview vizual al proiectului
   - Responsabilități echipă
   - Timeline săptămânal
   - Quick reference

3. ** 00_CHEAT_SHEET_RL.md**
   - Concepte fundamentale RL
   - Algoritmi explicați simplu
   - Vocabular & termeni
   - Resurse utile

---

###  PLANNING & ORGANIZATION:

4. ** 01_PLAN_PROIECT.md**
   - Plan detaliat pe etape (8 săptămâni)
   - Cerințe proiect
   - Împărțire roluri
   - Tracking progres
   - Întrebări pentru prof

5. ** 02_ENVIRONMENT_RESEARCH.md**
   - Research CartPole / LunarLander / MountainCar
   - PRO/CONTRA pentru fiecare
   - Idei reward shaping
   - ** TOȚI COMPLETAȚI SECȚIUNEA VOASTRĂ!**

---

###  IMPLEMENTARE:

6. ** 03_INGRID_TASKURI.md** (personalizat pentru Ingrid)
   - Taskuri detaliate pentru Q-Learning
   - Timeline personal
   - Resurse specifice
   - Success criteria

7. ** 04_QUICK_START.md**
   - Structură proiect
   - Setup Python environment
   - Code templates (wrapper, logging, plotting)
   - Config files
   - Workflow tipic

---

##  QUICK START (TOATĂ ECHIPA)

### Pasul 1: Citiți documentele
```
 05_TEAM_SUMMARY.md      (20 min)  PRIORITATE
 06_VISUAL_OVERVIEW.md   (15 min)  PRIORITATE
 00_CHEAT_SHEET_RL.md    (40 min)
 02_ENVIRONMENT_RESEARCH.md (20 min)
```

### Pasul 2: Setup Python environment
```bash
# Conda (recomandat)
conda create -n rl_project python=3.10
conda activate rl_project

# Instalare librării
pip install gymnasium numpy matplotlib torch
```

### Pasul 3: Test quick
```python
import gymnasium as gym
env = gym.make("CartPole-v1")
print(" Environment works!")
env.close()
```

### Pasul 4: Research individual
```
 Citit cursuri 1-3 RL (3-4 ore)
 Research 1-2 environments (2-3 ore)
 Completat secțiunea în 02_ENVIRONMENT_RESEARCH.md
```

### Pasul 5: Meeting #1 (5 decembrie)
```
 Toți pregătiți cu propuneri
 Alegem environment
 Irina scrie profului
```

---

##  TIMELINE SCURT

| Data | Milestone |
|------|-----------|
| **29 nov - 5 dec** | Research env + Meeting #1 |
| **5 dec - 12 dec** | Setup proiect + repo |
| **12 dec - 2 ian** | Implementare algoritmi |
| **2 ian - 8 ian** | Experimente & tuning |
| **8 ian - 15 ian** | Documentare + prezentare |
| **Săptămâna 14** | **PREZENTARE FINALĂ**  |

---

##  ROLURI ECHIPĂ

| Persoană | Algoritm | Rol secundar |
|----------|----------|--------------|
| **Irina ** | - | Coordonare + Environment + Experiments |
| **Ingrid ^^** | Q-Learning | Env research + Docs teoretice |
| **Matei** | DQN | Neural networks + Setup tehnic |
| **Iustin** | PPO | Experiments + Grafice + Analiză |

---

##  CE TREBUIE SĂ LIVRĂM

 **1 Environment** (Gymnasium, modificat cu reward shaping)  
 **3 Algoritmi RL** (Q-Learning, DQN, PPO)  
 **Experimente** (default vs modified reward, tuning)  
 **Rezultate** (grafice, tabele, metrici)  
 **Documentație** (PPT/PDF, **fără LLM pentru text!**)  
 **Prezentare** (săptămâna 14, în echipă)

---

##  TIPS RAPIDE

### Pentru citire:
- **Nu citiți tot dintr-o dată!** Start cu 05_TEAM_SUMMARY.md
- Marcați cu  ce ați citit
- Notați întrebările (ask la meeting)

### Pentru implementare:
- Start simplu → crește complexitatea
- Seed-uri fixe pentru reproducibilitate
- Git commits regulate
- Test frecvent (nu așteptați până la final)

### Pentru documentație:
- Scrieți în română simplă (**fără LLM!**)
- Grafice mari și clare
- Focus pe insights, nu doar formule
- E OK să includeți "ce nu a mers"

### Pentru prezentare:
- Fiecare prezintă partea la care a lucrat (3-5 min)
- Trebuie să puteți explica codul
- Rehearsal obligatoriu (minim 2x)
- Demo vizual = gold (video/gif)

---

##  RESURSE EXTERNE

### Documentație:
- **Gymnasium:** https://gymnasium.farama.org/
- **Stable Baselines3:** https://stable-baselines3.readthedocs.io/
- **PyTorch:** https://pytorch.org/tutorials/

### Cărți (PDF free):
- **Sutton & Barto** - "Reinforcement Learning: An Introduction"
  - Cap 6 (TD Learning) → Q-Learning
  - Cap 9 (Function Approximation) → DQN
  - Cap 13 (Policy Gradient) → PPO

### Cursuri video (opțional):
- **David Silver RL Course** (YouTube)
- **Stanford CS234** (Emma Brunskill)
- **Berkeley CS285** (Sergey Levine)

### Code examples:
- **CleanRL:** https://github.com/vwxyzjn/cleanrl
- **Spinning Up:** https://spinningup.openai.com/

---

##  ÎNTREBĂRI FRECVENTE

**Q: Ce document citesc primul?**  
A: `05_TEAM_SUMMARY.md` → apoi `06_VISUAL_OVERVIEW.md`

**Q: Când e Meeting #1?**  
A: Până pe 5 decembrie (Irina organizează)

**Q: Ce environment alegem?**  
A: Decidem la Meeting #1 (după research)

**Q: Cât timp ia proiectul?**  
A: ~40-50 ore/persoană pe 7-8 săptămâni

**Q: Putem folosi ChatGPT/Copilot?**  
A: DA pentru cod (dar trebuie să înțelegeți!), NU pentru documentație text

**Q: Ce facem dacă blocăm?**  
A: Ask în grup echipa → Irina → Prof. NU așteptați ultima clipă!

**Q: Trebuie să înțelegem tot codul?**  
A: DA! Proful poate întreba la prezentare.

---

##  CHECKLIST PROGRES

### Etapa 1: Research (29 nov - 5 dec)
- [ ] Citit toate documentele (start cu 05 și 06)
- [ ] Citit cursuri 1-3 RL
- [ ] Research 1-2 environments
- [ ] Completat secțiunea în `02_ENVIRONMENT_RESEARCH.md`
- [ ] Setup Python environment
- [ ] Meeting #1 done
- [ ] Irina scris profului

### Etapa 2: Setup (5 dec - 12 dec)
- [ ] Repo GitHub creat
- [ ] Structură proiect
- [ ] Wrapper reward implementat
- [ ] Testing environment

### Etapa 3-4: Implementare (12 dec - 2 ian)
- [ ] Q-Learning funcțional (Ingrid)
- [ ] DQN funcțional (Matei)
- [ ] PPO funcțional (Iustin)
- [ ] Logging + rezultate

### Etapa 5: Experimente (2 ian - 8 ian)
- [ ] Experimente default vs modified
- [ ] Multiple seeds
- [ ] Tuning hiperparametri
- [ ] Grafice + tabele

### Etapa 6: Documentare (8 ian - 15 ian)
- [ ] PowerPoint finalizat (**fără LLM!**)
- [ ] README.md complet
- [ ] Cod comentat
- [ ] Rehearsal (2x)

### Etapa 7: PREZENTARE! 
- [ ] Săptămâna 14
- [ ] În echipă
- [ ] Q&A cu proful

---

##  SUPORT

**În echipă:**
- Grup WhatsApp/Telegram
- Meeting-uri săptămânale
- Code review mutual

**Irina (coordonare):**
- Organizare + comunicare prof
- Clarificări cerințe
- Review final

**Prof:**
- stefan.iordache10@s.unibuc.ro
- Întrebări teoretice
- Clarificări algoritmi

**Online:**
- Stack Overflow
- Reddit r/reinforcementlearning
- GitHub issues

---

##  OBIECTIV FINAL

```

                                                           
  BUILD 3 RL AGENTS THAT LEARN!                           
                                                           
  • Q-Learning (Ingrid)                                   
  • DQN (Matei)                                           
  • PPO (Iustin)                                          
                                                           
  Compare them on the same environment with               
  different reward shaping strategies.                    
                                                           
  Show the world (and the professor) what you learned!   
                                                           

```

---

##  CONTACT

**Echipă:** Irina , Ingrid ^^, Matei, Iustin  
**Profesor:** stefan.iordache10@s.unibuc.ro  
**Curs:** Introducere în Reinforcement Learning 2025-2026  

---

##  NEXT ACTIONS (IMEDIAT!)

### ASTĂZI (29 noiembrie):
1.  Citit acest README
2.  Citit `05_TEAM_SUMMARY.md`
3.  Citit `06_VISUAL_OVERVIEW.md`
4.  Setup Python environment (început)

### MÂINE (30 noiembrie):
1. Citit curs 1 + 2 RL
2. Research environment (CartPole docs)
3. Test run random agent

### WEEKEND (30 nov - 1 dec):
1. Citit curs 3 RL
2. Research aprofundat 2 environments
3. Completat `02_ENVIRONMENT_RESEARCH.md`

### PÂNĂ PE 5 DECEMBRIE:
1. Toate taskurile research done
2. **MEETING #1** - alegem environment!

---

** REMEMBER:**
- Learning > Perfection
- Teamwork > Individual genius
- Progress > Procrastination
- Asking questions > Staying confused

---

** LET'S BUILD SOME INTELLIGENT AGENTS! **

---

_Last updated: 29 noiembrie 2025_  
_Version: 1.0_  
_Status: Ready to start! _

---

##  LICENSE

Documentație creată de și pentru echipa de proiect.  
Feel free to adapt and improve! 

**Credits:**
- Ingrid ^^ - Research & Documentation Lead
- Irina  - Project Coordination
- Matei - Deep RL Implementation
- Iustin - Experiments & Visualization

**Acknowledgments:**
- Prof. Ștefan Iordache & Prof. Ciprian Păduraru
- Sutton & Barto (pentru biblia RL)
- OpenAI Gymnasium team
- Stable Baselines3 contributors

---

**END OF README**

**Start with: `05_TEAM_SUMMARY.md`** 
