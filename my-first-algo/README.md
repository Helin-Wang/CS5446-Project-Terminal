# My-First-Algo

## 🧩 Overview

This project implements a **simple reinforcement learning (RL) bot** designed to interact with a starter game environment.  
It focuses on lightweight experimentation — **no deep learning or environment simulation**, just local play, logging, and policy updates.

---

## 🚀 Stage 0 — Minimal Viable Bot

### 🎯 Goal

Develop a bot that can:

1. **Extract Observations**  
   - Capture a compact **observation vector** each turn from the game state.

2. **Select Actions**  
   - Choose **one macro-action** from a predefined set.  
   - Apply **feasibility masking** to filter out invalid options.

3. **Execute Macros**  
   - Use the **starter GameState API** to perform high-level actions  
     (e.g., building, spawning, etc.).

4. **Log Experiences**  
   - Record **(state, action, reward)** transitions to a `.jsonl` file for later training.

5. **Train Offline**  
   - Use a simple **REINFORCE** algorithm to train a lightweight policy:  
     - Model = softmax over macro-actions  
     - No neural networks — just basic parameter updates  
   - After training, **reload the updated weights** into the bot.

---

## 🧠 Key Principles

- **Simplicity First:** Avoid deep learning; start with interpretable, small-scale logic.  
- **Offline Learning:** Gather data from games played against the starter bot, then train offline.  
- **Reproducibility:** All logs and weights can be easily reloaded for iteration.  

---

## 📁 Outputs

- `trajectories.jsonl` — recorded (s, a, r) transitions  
- `policy_weights.json` — trained macro-policy parameters  

---

## 🕹️ Next Steps

- Improve observation encoding  
- Expand macro-action set  
- Add baselines or entropy regularization to stabilize REINFORCE  
