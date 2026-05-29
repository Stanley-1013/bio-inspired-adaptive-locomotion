# SATA 訓練內部機制與 Phase 1 復現

本文件說明執行 `python scripts/train.py --task=go2_torque` 時，程式實際做了什麼，以及在本伺服器上復現 Phase 1 的結果。

This document explains what `python scripts/train.py --task=go2_torque` actually does in code, and records the Phase 1 reproduction results on this server.

相關文件 / Companion docs:
- [`setup-sata.md`](./setup-sata.md): 安裝與環境設定 / installation and setup
- [`concepts-primer.md`](./concepts-primer.md): 相關概念教學 / primer for RL, torque control, PPO, Hill model, and related terms

## 訓練目標 / What Gets Trained

一個 12 自由度（12-DOF）的 **Unitree Go2** 四足機器人會在 Isaac Gym 的崎嶇地形中學習追蹤速度命令 `(v_x, v_y, ω_yaw)`。策略網路（policy）輸出的是原始 **關節力矩**（joint torque）命令，而不是關節位置；接著一層仿生機制（bio-inspired layer）會在每個物理步（physics step）把原始力矩轉換成實際施加到馬達上的力矩。

A 12-DOF **Unitree Go2** quadruped learns to follow `(v_x, v_y, ω_yaw)` velocity commands on rough terrain inside Isaac Gym. The policy outputs raw **joint torque** commands rather than joint positions, and a bio-inspired layer translates those raw torques into actuated torques at every physics step.

- 機器人 / Robot: Unitree Go2，URDF 位於 `SATA/legged_gym/legged_gym/resources/robots/go2/urdf/go2_torque.urdf`
- 模擬器 / Simulator: Isaac Gym Preview 4，使用 `mesh_type='trimesh'` 的崎嶇地形
- 演算法 / Algorithm: SATA 客製化 `rsl_rl` 中的 PPO
- 每次訓練的平行環境數 / Parallel envs per run: 單張 A6000 上 4096 個環境
- 訓練迭代數 / Iterations: 3000 次；單 GPU 約 75 分鐘，與另外兩個 sibling runs 共用機器時約 65 分鐘

## 動作到關節力矩流程 / Action to Joint Torque Pipeline

策略輸出的 12 維動作 `a` 並不直接等於關節力矩。每個 step 會在 [go2_torque.py:221-248](/home/han/workspace/SATA/legged_gym/legged_gym/envs/go2/go2_torque/go2_torque.py#L221) 中經過四個階段。

The policy's 12-dimensional action `a` is not directly the joint torque. Four processing stages are applied each step in [go2_torque.py:221-248](/home/han/workspace/SATA/legged_gym/legged_gym/envs/go2/go2_torque/go2_torque.py#L221).

1. **縮放 / Scale**

   ```text
   a_scaled = a * action_scale    # action_scale = 5
   ```

   先把 policy 輸出的無單位動作放大到力矩控制比較合理的尺度。

   The dimensionless policy action is scaled to a torque-like range.

2. **肌肉活化過程 / Activation Process** (`activation_process=True`)

   ```text
   sign_curr = tanh(a_scaled / tau_max)
   sign      = 0.6 * sign_prev + 0.4 * sign_curr
   ```

   這是一階低通濾波（first-order low-pass）。直覺上，它模擬肌肉招募或馬達出力建立的延遲，避免力矩在正負最大值之間瞬間跳變。

   This first-order low-pass filter models muscle recruitment delay and prevents bang-bang torque switches.

3. **Hill 模型 / Hill Model** (`hill_model=True`)

   ```text
   tau = sign * tau_max * (1 - sign * omega / omega_max)
   ```

   Hill model 表達「同方向速度越快，能產生的最大力越小」的力-速度關係（force-velocity inverse relation）。如果關節正在同方向快速運動，可用力矩會下降；減速方向則不受這個限制。

   The Hill model captures an inverse force-velocity relation: maximum producible torque drops when the joint moves fast in the same direction, while deceleration is unaffected.

4. **疲勞更新 / Fatigue Update** (`motor_fatigue=True`)

   ```text
   fatigue = (fatigue + abs(tau) * dt) * 0.9
   ```

   每個自由度都有一個 leaky accumulator。有效時間常數約 10 個 control steps，在 200 Hz 下約 50 ms。疲勞向量會放進 observation，因此 policy 可以「看見」自己的疲勞狀態。

   Each DOF has a leaky accumulator. The effective time constant is about 10 control steps, or roughly 50 ms at 200 Hz. The fatigue vector is included in the observation, so the policy observes its own fatigue state.

以上三個仿生階段可在 [go2_torque_config.py:102-108](/home/han/workspace/SATA/legged_gym/legged_gym/envs/go2/go2_torque/go2_torque_config.py#L102) 中各自關閉，用於 ablation study。

Each biomechanical stage can be ablated independently by flipping its boolean in [go2_torque_config.py:102-108](/home/han/workspace/SATA/legged_gym/legged_gym/envs/go2/go2_torque/go2_torque_config.py#L102).

## 成長課程 / Growth Curriculum

`general_scale ∈ [0, 1]` 是控制「發育程度」的單一 scalar，會沿著 Gompertz curve 在 [go2_torque.py:183](/home/han/workspace/SATA/legged_gym/legged_gym/envs/go2/go2_torque/go2_torque.py#L183) 中逐漸上升。

`general_scale ∈ [0, 1]` is a single developmental-capacity scalar. It ramps along a Gompertz curve in [go2_torque.py:183](/home/han/workspace/SATA/legged_gym/legged_gym/envs/go2/go2_torque/go2_torque.py#L183).

```text
general_scale = exp(-exp(-k * (step - x0)))    # k=3e-5, x0=24,000
```

這個 step 指的是累積 environment step count，不是 PPO iteration count。它大約在 12k 到 60k steps 之間從接近 0 上升到接近 1。

The curve is driven by cumulative environment steps rather than PPO iterations. It rises from near 0 to near 1 over roughly 12k to 60k steps.

同一個 scalar 同時控制多個訓練條件：

The same scalar drives several training conditions:

- 前腳力矩上限 / Front-leg torque ceiling: `0.3 * tau_max` → `1.0 * tau_max`，後腳維持 `1.0 * tau_max`
- 控制頻率 / Control frequency: 100 Hz → 200 Hz
- 命令範圍 / Command range: 初期較小的 `(v_x, v_y, ω_yaw)`，後期完整範圍
- Domain randomization: 推力大小與摩擦係數擾動逐步增加

若要停用整個 curriculum，把 `control_type='TG'` 改成 `control_type='T'`。

To disable the entire curriculum, set `control_type='T'` instead of `control_type='TG'`.

## 獎勵函數 / Reward Function

獎勵設定來自 [go2_torque_config.py:129-139](/home/han/workspace/SATA/legged_gym/legged_gym/envs/go2/go2_torque/go2_torque_config.py#L129)。

The reward scales come from [go2_torque_config.py:129-139](/home/han/workspace/SATA/legged_gym/legged_gym/envs/go2/go2_torque/go2_torque_config.py#L129).

| 項目 / Term | Scale | 鼓勵或懲罰內容 / Incentive |
|---|---:|---|
| `forward` | +10 | 追蹤命令的 `v_x`，使用誤差的 exponential / track commanded `v_x` |
| `head_height` | +5 | 頭部保持高於地形且身體直立，隨成長縮放 / keep head above terrain and upright |
| `moving_y` | +5 | 追蹤命令的 `v_y`，隨成長縮放 / track commanded `v_y` |
| `moving_yaw` | +5 | 追蹤命令的 `ω_yaw` / track commanded yaw rate |
| `soft_dof_pos_limits` | -5 | 不要讓關節長時間接近機械極限 / avoid parking joints near mechanical limits |
| `roll` | -5 | 不要側翻 / avoid tipping sideways |
| `lin_vel_z` | -5 | 不要上下震盪 / avoid vertical oscillation |
| `dof_acc` | -1e-6 | 動作平滑 / smooth motion |
| `motor_fatigue` | **-0.05** | 對累積疲勞給小懲罰 / small penalty for cumulative fatigue |

`motor_fatigue` 的 scale 刻意設得很小，是 `-0.05` 而不是 `-5`。它會在沒有追蹤代價時把 policy 推向比較不疲勞的解，但不會禁止必要時用力推蹬。SATA 的主張是：這個小獎勵，加上 observation 中包含 fatigue，足以引導出 adaptive load-shedding behavior。

The `motor_fatigue` scale is intentionally small: `-0.05`, not `-5`. It nudges the policy toward non-fatiguing solutions when there is no tracking cost, but does not forbid hard pushes when needed. SATA's claim is that this small reward, combined with fatigue in the observation, is enough to elicit adaptive load-shedding behavior.

## 一次 PPO 迭代 / One PPO Iteration

核心流程在 [on_policy_runner.py:127-154](/home/han/workspace/SATA/rsl_rl/rsl_rl/runners/on_policy_runner.py#L127)。

The core loop is in [on_policy_runner.py:127-154](/home/han/workspace/SATA/rsl_rl/rsl_rl/runners/on_policy_runner.py#L127).

- Rollout: `24 steps * 4096 parallel envs = ~98k transitions / iteration`
- 每次 iteration 的更新量 / Updates per iteration: `5 epochs * 4 mini-batches = 20 gradient steps`
- Loss，見 [ppo.py:171](/home/han/workspace/SATA/rsl_rl/rsl_rl/algorithms/ppo.py#L171):

  ```text
  L = clipped_surrogate(epsilon=0.2)
      + value_loss_coef * clipped_value_loss
      - entropy_coef * entropy
  ```

- Adaptive KL: 目標 KL 為 `0.01`；每次 iteration 依據 KL 將 learning rate 加倍或減半，使更新幅度維持在合理範圍

到 3000 iterations 時，policy 約看過 `295M` 筆 environment transitions。

At 3000 iterations, the policy has seen about `295M` environment transitions.

## 網路形狀 / Network Shape

Actor 與 critic 都定義在 [actor_critic.py:38-80](/home/han/workspace/SATA/rsl_rl/rsl_rl/modules/actor_critic.py#L38)。

Both actor and critic are defined in [actor_critic.py:38-80](/home/han/workspace/SATA/rsl_rl/rsl_rl/modules/actor_critic.py#L38).

- 輸入 / Input: 60 維 observation；actor 與 critic 使用相同 observation，沒有 privileged information
- 隱藏層 / Hidden layers: `[512, 256, 128]`，activation 為 ELU
- 輸出 / Output: actor 輸出 12 維 Gaussian，並學習 diagonal std；critic 輸出 scalar value

總參數量約 400k。訓練時觀察到的 5.8 GB VRAM 主要花在 rollout buffer，而不是神經網路本身。Rollout buffer 需要保存 `24 * 4096 * ~60` 等多種 slot 的浮點資料。

Total parameters are about 400k. The observed 5.8 GB VRAM usage is mostly the rollout buffer, not the network itself.

### Observation 組成 / Observation Composition

Observation 來源見 [go2_torque.py:285-295](/home/han/workspace/SATA/legged_gym/legged_gym/envs/go2/go2_torque/go2_torque.py#L285)。

The observation is assembled in [go2_torque.py:285-295](/home/han/workspace/SATA/legged_gym/legged_gym/envs/go2/go2_torque/go2_torque.py#L285).

| Indices | 內容 / Content |
|---|---|
| 0:3 | Base linear velocity，body frame |
| 3:6 | Base angular velocity |
| 6:9 | Gravity vector projected to body frame |
| 9:12 | Velocity command `(v_x, v_y, ω_yaw)` |
| 12:24 | Joint positions - default angles，12 DOF |
| 24:36 | Joint velocities，12 DOF |
| 36:48 | Previous action |
| **48:60** | **Per-DOF fatigue state，SATA-specific addition** |

---

## Phase 1 復現結果 / Phase 1 Reproduction Results

三個 random seeds 已於 2026-05-25/26 在本機平行訓練。每個 seed 使用 published reference config：`go2_torque`、4096 envs、3000 PPO iterations、headless。每個 seed 使用一張 A6000，共使用 4 張 GPU 中的 3 張；第 4 張當時有其他使用者 workload。

Three random seeds were trained in parallel on 2026-05-25/26 using the published reference config: `go2_torque`, 4096 envs, 3000 PPO iterations, headless. Each seed used one A6000; 3 of 4 GPUs were used because the fourth was occupied by another tenant's workload.

| seed | Final mean reward | Wall time | Log dir |
|---:|---:|---:|---|
| 1 | 122 | 63.6 min | `logs/SATA/May25_23-12-14_ref_s1` |
| 2 | 108 | 64.7 min | `logs/SATA/May25_23-12-17_ref_s2` |
| 3 | 112 | 64.0 min | `logs/SATA/May25_23-12-22_ref_s3` |

**Iter 3000 的 mean ± std reward: 114 ± 6.**

**Mean ± std reward at iter 3000: 114 ± 6.**

觀察 / Observations:

- 三個 seeds 都收斂到相近 reward，表示 SATA 在此環境下可穩定復現，不是 one-shot result。
- Wall time variance 約 1 分鐘，主要來自三個平行 runs 之間的 NFS 與 CPU contention；每個 run throughput 約 `1.7 s/iter`，早期單 seed sanity check 約 `1.5 s/iter`。
- 每張 GPU memory 使用量約 5.8 GB / 48 GB，Phase 2 開始後有足夠空間跑 4 個平行 ablations。
- Checkpoint 每 50 iterations 儲存一次；每個 seed 約 `60 * 4.7 MB = 280 MB`，儲存在 NFS。
- gymtorch C++ extension 第一次啟動時編譯一次，快取於 `~/.cache/torch_extensions/py38_cu117/`；後續啟動會立即載入。

### 重播與延伸 / Replaying and Extending

```bash
# activate the env: handles conda, LD_LIBRARY_PATH, and cd
source ~/workspace/bio-inspired-adaptive-locomotion/scripts/sata-env.sh

# visualize a trained policy: needs VNC or a display
python scripts/play.py --task=go2_torque --load_run=ref_s1 --checkpoint=3000

# training curves for all three seeds: port 6006
tensorboard --logdir ~/workspace/SATA/legged_gym/logs --port 6006 --bind_all
```

Phase 2 ablation 的基本做法：每個 ablation 建立一個新的 config subclass，覆寫 `GO2TorqueCfg` 中的一個欄位，例如 `motor_fatigue=False`；再在 `legged_gym/envs/__init__.py` 加上一行 registration。Launch recipe 與 Phase 1 對稱：每張 GPU 跑一個 ablation，平行訓練約 65 分鐘。

For Phase 2 ablation, each ablation should be a new config subclass overriding one field on `GO2TorqueCfg`, such as `motor_fatigue=False`, plus a one-line registration in `legged_gym/envs/__init__.py`. The launch recipe is symmetric to Phase 1: one ablation per GPU, about 65 minutes in parallel.
