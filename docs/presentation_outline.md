# RL Project Presentation Outline

## Slide 1 – Title & team
- Project title, names, course, date.

## Slide 2 – Problem & environment
- Goal: learn to drive ego car safely on multi-lane road with traffic.
- State/action/reward definition (short bullets).
- Insert diagram: RL loop (from `docs/diagrams.tex`, `\RLLoopDiagram`).

## Slide 3 – Environment design
- Custom road layout (merging lane, curve, exit).
- Ego vs. traffic vehicles (yellow vs. blue IDM cars).
- Reward shaping: stay on road, centered, aligned; penalties for crash/off-road.

## Slide 4 – Agents implemented
- Tabular Q-Learning (discrete 6-dim state).
- DQN (25-dim state, MLP 256-128, replay, target net, epsilon-greedy).
- REINFORCE + value baseline (actor-critic flavour).
- Insert diagram: agent comparison (`\AgentComparisonDiagram`).

## Slide 5 – Training curves
- Show `results/training_comparison.png`.
- Comment briefly on convergence speed and stability for each agent.

## Slide 6 – Behaviour visualization
- Show `results/actions_dqn.png`, `actions_reinforce.png`, `actions_tabular.png`.
- Discuss qualitative behaviour: lane-keeping, smoothness, failures.

## Slide 7 – Evaluation & comparison
- Show `results/comparison/eval_barplot.png`.
- Mention average episodic return and variance per agent.
- One bullet on why DQN > REINFORCE > tabular (in your runs).

## Slide 8 – Hyperparameters & stability
- List key hyperparameters tried (lr, epsilon decay, network size).
- Mention what helped/hurt (e.g., reward shaping, termination conditions).

## Slide 9 – Lessons learned
- What worked well (DQN + kinematics state, custom env).
- What was hard (policy gradient stability, reward design).
- Possible future work (PPO, continuous actions, better critic).

## Slide 10 – Demo / Q&A
- Short demo video or live run (if allowed).
- Invite questions.
