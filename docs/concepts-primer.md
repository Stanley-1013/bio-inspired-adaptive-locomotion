# SATA 相關概念教學 / Concepts Primer

這份文件是給 0 基礎讀者的中英雙語教學。預設你有基本工程學科能力：看得懂向量、函數、微分或最佳化的直覺，但不需要先學過強化學習、機器人控制或 Isaac Gym。

This bilingual primer assumes basic engineering literacy but no prior reinforcement learning, robot control, or Isaac Gym background.

## 這個專案在做什麼 / What This Project Is Doing

SATA 的任務可以用一句話概括：讓四足機器人在模擬器裡學會走路，並讓它的控制方式帶有仿生限制，例如肌肉活化延遲、力-速度關係與疲勞。

In one sentence, SATA trains a quadruped robot to walk in simulation while adding bio-inspired constraints such as activation delay, force-velocity limits, and fatigue.

更工程化地說：

In engineering terms:

- 機器人 / Robot: Unitree Go2，12 個可控制關節，稱為 12 DOF
- 環境 / Environment: Isaac Gym 中的崎嶇地形
- 控制目標 / Goal: 跟隨給定速度 `(v_x, v_y, ω_yaw)`
- 控制輸出 / Action: 每個關節的 torque command
- 學習方法 / Learning method: reinforcement learning，具體演算法是 PPO

## 強化學習 / Reinforcement Learning

強化學習（reinforcement learning, RL）研究的是：一個 agent 如何透過與 environment 互動，學到能最大化長期 reward 的行為。

Reinforcement learning studies how an agent learns behavior by interacting with an environment to maximize long-term reward.

在這個專案中：

In this project:

| RL 名詞 / RL Term | 在 SATA 中的意思 / Meaning in SATA |
|---|---|
| Agent | 學走路的 policy，也就是控制器 |
| Environment | Isaac Gym 裡的一個 Go2 機器人與地形 |
| Observation | 機器人當下可看到的狀態，例如速度、關節角、疲勞 |
| Action | policy 輸出的 12 維關節力矩命令 |
| Reward | 告訴 policy 這一步好不好的分數 |
| Episode | 一段模擬軌跡，通常到跌倒、超時或 reset 結束 |
| Policy | 從 observation 映射到 action 的神經網路 |

RL 的核心不是「直接告訴機器人每一步怎麼走」，而是設計 reward 和環境，讓 policy 透過 trial and error 自己找到可行策略。

The core idea is not to directly label the correct movement at every step. Instead, we design rewards and environments so the policy discovers useful behavior through trial and error.

## Observation、Action、Reward

### Observation

Observation 是 policy 的輸入。它不是完整真實世界，而是控制器在每個 timestep 能使用的資訊。

Observation is the input to the policy. It is not the entire world state, only the information available to the controller at each timestep.

SATA 的 Go2 task 使用 60 維 observation，包含：

SATA's Go2 task uses a 60-dimensional observation, including:

- 機身線速度與角速度 / base linear and angular velocity
- 重力方向投影 / projected gravity direction
- 速度命令 / commanded velocity
- 關節角與關節速度 / joint positions and velocities
- 施加的關節力矩 (τ) / applied joint torques (τ)
- 每個關節的 fatigue state / per-joint fatigue state

### Action

Action 是 policy 的輸出。在 position control 中，action 可能代表目標關節角；但這個 task 採用 torque control，因此 action 最終會變成每個關節的力矩。

Action is the policy output. In position control, actions may represent target joint angles. In this task, torque control is used, so actions eventually become joint torques.

注意：policy 的 action 不是馬上施加的 torque。它會先經過縮放、activation process、Hill model 和 fatigue update。

Important: the policy action is not directly applied as torque. It passes through scaling, activation processing, the Hill model, and fatigue update.

### Reward

Reward 是每一步的分數。設計 reward 就像定義「什麼是好的走路」。

Reward is the score at each step. Designing reward is like defining what good walking means.

SATA 的 reward 同時鼓勵：

SATA's reward encourages:

- 跟上命令速度 / tracking the commanded velocity
- 身體保持穩定與直立 / keeping the body stable and upright
- 不要側翻或上下亂震 / avoiding tipping and vertical oscillation
- 動作平滑 / smooth motion
- 降低不必要的疲勞 / reducing unnecessary fatigue

Reward 權重很重要。如果疲勞懲罰太大，機器人可能不敢用力；太小則可能完全不在意疲勞。SATA 把 fatigue penalty 設小，讓它只在「任務表現差不多」時影響策略選擇。

Reward weights matter. If fatigue penalty is too large, the robot may avoid pushing hard. If too small, it may ignore fatigue. SATA keeps the fatigue penalty small so it mainly affects choices when task performance is otherwise similar.

## Policy 與神經網路 / Policy and Neural Networks

Policy 是一個函數：

The policy is a function:

```text
action = policy(observation)
```

在 SATA 中，policy 是多層感知器（MLP）。它讀入 60 維 observation，輸出 12 維 action distribution。這裡使用 Gaussian distribution，代表 policy 不只輸出一個動作，也保留探索用的隨機性。

In SATA, the policy is a multilayer perceptron. It reads a 60-dimensional observation and outputs a 12-dimensional action distribution. A Gaussian distribution is used, so the policy keeps stochasticity for exploration rather than always outputting a single deterministic action.

訓練時會抽樣 action；測試或部署時通常使用平均值或較穩定的 action。

During training, actions are sampled. During evaluation or deployment, the mean or a more stable action is often used.

## Value Function 與 Critic

PPO 通常同時訓練 actor 和 critic。

PPO usually trains both an actor and a critic.

- Actor: policy，決定要做什麼 action
- Critic: value function，估計目前狀態未來大概能拿到多少 reward

Actor answers "what should I do?" The critic answers "how good is this state likely to be?"

Value function 可以寫成：

The value function can be written as:

```text
V(observation) = expected future return
```

Critic 不直接控制機器人，但它幫助訓練更穩定，因為 actor 可以知道某個 action 比預期更好還是更差。

The critic does not directly control the robot, but it stabilizes training by helping the actor estimate whether an action was better or worse than expected.

## PPO 是什麼 / What PPO Is

PPO 是 Proximal Policy Optimization。它是一種 policy gradient 演算法，常用於機器人 locomotion，因為實作相對穩定、能吃大量平行模擬資料。

PPO stands for Proximal Policy Optimization. It is a policy-gradient algorithm widely used for robot locomotion because it is relatively stable and works well with large batches from parallel simulation.

PPO 的直覺：

Intuition:

1. 先用目前 policy 跑一批 rollout，收集 observation、action、reward。
2. 估計哪些 action 比預期好，哪些比預期差。
3. 更新 policy，增加好 action 的機率、降低壞 action 的機率。
4. 用 clipping 限制 policy 一次不要改太多。

PPO clipping 的重點是「保守更新」。如果 policy 一次改太大，機器人控制很容易崩掉；clipping 讓每次更新比較像小步調整。

The key idea of PPO clipping is conservative updating. If the policy changes too much at once, robot control can collapse. Clipping keeps updates closer to small adjustments.

## Rollout、Transition、Iteration

一個 transition 通常包含：

A transition usually contains:

```text
observation_t, action_t, reward_t, observation_{t+1}, done_t
```

Rollout 是 policy 在環境中連續跑一段時間得到的一批 transitions。SATA 每個 PPO iteration 會收集：

A rollout is a batch of transitions collected by running the policy in the environment. In SATA, each PPO iteration collects:

```text
24 steps * 4096 envs ≈ 98,000 transitions
```

Iteration 則是「收資料 + 用這批資料更新 policy」的一個完整循環。

An iteration is one full cycle of collecting data and updating the policy with that data.

## 平行環境 / Parallel Environments

Isaac Gym 的強項是可以在 GPU 上同時模擬很多個環境。4096 envs 的意思是，同一張 GPU 上同時有 4096 隻 Go2 在不同地形或不同初始條件下練習。

Isaac Gym can simulate many environments in parallel on the GPU. 4096 envs means 4096 Go2 robots train at the same time under different terrains or initial conditions.

這能大幅增加資料量，也讓 policy 不只記住單一場景，而是學到比較泛化的走路策略。

This greatly increases data throughput and helps the policy learn behavior that generalizes beyond one fixed scene.

## Torque Control 與 Position Control

Position control 是告訴馬達「我要到這個角度」。底層控制器會自行產生力矩把關節拉過去。

Position control tells the motor "move to this target angle." A lower-level controller generates torque to reach that angle.

Torque control 是直接告訴馬達「我要施加這個力矩」。它更接近底層物理，也更難學，因為 policy 必須自己處理穩定性與動態。

Torque control directly commands "apply this torque." It is closer to the underlying physics and harder to learn because the policy must handle stability and dynamics itself.

SATA 選 torque control，是因為它想研究仿生力矩限制、肌肉式延遲與疲勞如何影響 locomotion。

SATA uses torque control because it studies how bio-inspired torque limits, muscle-like delays, and fatigue affect locomotion.

## DOF、Joint、Base

DOF 是 degree of freedom，自由度。Go2 有 4 條腿，每條腿 3 個可控關節，因此共有 12 DOF。

DOF means degree of freedom. Go2 has 4 legs with 3 controllable joints each, so it has 12 DOF.

Joint 指關節，例如髖關節或膝關節。Base 指機器人的機身主體，常用來描述機身速度、姿態、角速度。

Joint refers to a robot joint, such as hip or knee. Base refers to the robot body, often used for body velocity, orientation, and angular velocity.

## Hill Model 的直覺 / Intuition Behind the Hill Model

生物肌肉有一個重要現象：肌肉縮短得越快，能產生的力量越小。這稱為 force-velocity relation。

A key property of biological muscle is that the faster it shortens, the less force it can produce. This is the force-velocity relation.

SATA 用 Hill model 的簡化形式把這件事放進機器人控制：

SATA uses a simplified Hill model to add this effect to robot control:

```text
torque = a * tau_max * (1 - sign(a) * joint_velocity / omega_max)
```

`a` 是連續 activation（±1），`sign(a)` 只取它的方向（±1）。如果 action 想讓關節往某方向出力，而關節已經往同方向快速轉動，最大可用 torque 會下降。這讓控制器更像受肌肉限制的生物系統。

If the action requests torque in a direction where the joint is already moving quickly, available torque decreases. This makes the controller more like a biological system with muscle limitations.

## Activation Process 的直覺 / Intuition Behind Activation

真實肌肉不是接到命令後立刻產生最大力量，而是需要一點時間建立活化程度。SATA 用一階低通濾波模擬這種延遲。

Real muscle does not produce maximum force instantly after receiving a command. SATA models this delay with a first-order low-pass filter.

```text
new_activation = 0.6 * requested_activation + 0.4 * old_activation
```

這代表新的 activation 有 60% 來自現在的命令，40% 來自上一刻（程式碼 `(curr - prev) * 0.6 + prev`）。結果是力矩變化更平滑，也更接近生物肌肉。

This means the new activation is 60% from the current command and 40% from the previous activation (the code computes `(curr - prev) * 0.6 + prev`). The result is smoother torque changes and more muscle-like behavior.

## Fatigue 的直覺 / Intuition Behind Fatigue

Fatigue state 可以理解成「最近這個關節用了多少力」。它不是永久累積，而是會漏掉的累積器（leaky accumulator）。

The fatigue state can be understood as "how much this joint has recently worked." It is not accumulated forever; it is a leaky accumulator.

```text
fatigue = (fatigue + abs(torque) * dt) * 0.9
```

用力會增加 fatigue，乘上 `0.9` 則讓舊 fatigue 逐漸衰退。因為 fatigue 被放進 observation，policy 可以學到：某些腿或關節累了，就暫時把負載分給其他部位。

Using torque increases fatigue, while multiplying by `0.9` makes old fatigue decay. Because fatigue is included in the observation, the policy can learn to shift load away from tired joints.

## Curriculum Learning 與 Growth

Curriculum learning 是「先給簡單任務，再逐漸增加難度」。人類教學也常這樣做：先走平地，再走斜坡；先慢速，再快速。

Curriculum learning means starting with easier tasks and gradually increasing difficulty. Human teaching often works this way: flat ground before slopes, slow speed before fast speed.

SATA 的 growth curriculum 不只改任務難度，也模擬發育：

SATA's growth curriculum changes both task difficulty and developmental capacity:

- 前腿 torque ceiling 從低到高 / front-leg torque ceiling increases
- 控制頻率從 100 Hz 到 200 Hz / control frequency increases
- 速度命令範圍變大 / command range expands
- domain randomization 變強 / domain randomization becomes stronger

這能讓 policy 在早期先學穩定基礎，再面對完整能力與完整難度。

This lets the policy first learn basic stability, then face full capacity and full difficulty.

## Domain Randomization

Domain randomization 是在訓練時故意改變環境參數，例如摩擦係數、外力推動、地形形狀。目的不是讓訓練變難而已，而是避免 policy 只適應單一精準模擬設定。

Domain randomization intentionally varies environment parameters during training, such as friction, external pushes, and terrain. The goal is not merely to make training harder, but to prevent the policy from overfitting to one exact simulation setting.

對機器人來說，這很重要。真實世界的地面、摩擦、馬達響應永遠不會和模擬器完全一樣。

This matters for robotics because real ground, friction, and motor response never exactly match the simulator.

## Ablation Study

Ablation study 是研究「拿掉某個元件後會怎樣」。例如：

An ablation study asks what happens if one component is removed. Examples:

- 關掉 fatigue / disable fatigue
- 關掉 Hill model / disable the Hill model
- 關掉 activation process / disable activation processing
- 關掉 growth curriculum / disable the growth curriculum

如果完整 SATA 表現好，但拿掉 fatigue 後 load-shedding 行為消失，就能支持 fatigue 機制確實有貢獻。

If full SATA works well but load-shedding disappears after removing fatigue, that supports the claim that fatigue contributes to the behavior.

## 如何讀訓練結果 / How to Read Training Results

常見指標包括：

Common metrics include:

- Mean reward: 平均 reward，越高通常越好，但要搭配影片或細項 reward 解讀
- Episode length: episode 持續多久，太短可能代表常跌倒
- Tracking error: 命令速度與實際速度的差
- Smoothness: action 或關節加速度是否劇烈
- Fatigue-related metrics: 是否過度使用特定關節，是否能分散負載

只看 final reward 不一定夠。兩個 policy reward 接近，但一個可能動作平滑、疲勞分散，另一個可能靠激烈動作硬撐。因此 Phase 2 通常會搭配 ablation、影片、細項指標與統計檢定。

Final reward alone is not always enough. Two policies may have similar reward, but one may be smooth and fatigue-aware while the other relies on aggressive motion. Phase 2 should therefore combine ablations, videos, detailed metrics, and statistical checks.

## 最小閱讀路線 / Suggested Reading Path

如果你是第一次讀這個專案，建議順序如下：

If this is your first pass through the project, use this order:

1. 先讀本文件，建立 RL 與 locomotion 的共同語言。
2. 讀 [`training-internals.md`](./training-internals.md)，理解 SATA 訓練流程。
3. 讀 [`setup-sata.md`](./setup-sata.md)，了解如何實際跑訓練與播放結果。
4. 再看程式碼中的 config、reward、observation 和 torque pipeline。

The goal is not to memorize every formula. The useful mental model is: observation tells the policy what is happening, action tells the robot what to do, reward defines what good behavior means, and PPO repeatedly improves the policy using massive parallel simulation.
