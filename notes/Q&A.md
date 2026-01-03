## Question1
我想请你浏览一下rsl_rl所提供的算法框架，然后为我总结一下，该框架中实现的对称性模块的功能、运行逻辑以及使用方法

---

**功能**
- 在PPO训练中利用任务/机器人左右、前后等对称性，把一条轨迹映射成多条"等价轨迹"，从而提升数据效率并让学到的策略更对称（ppo.py(line 229)）。
- 可选增加"mirror loss"（或仅作为指标记录），用来约束策略满足对称变换下的等变性：$\pi(T(s)) \approx T(\pi(s))$。

**运行逻辑**
- Runner构建算法前把环境对象塞进symmetry_cfg["_env"]，供对称增强函数按环境的观测/动作定义做变换（symmetry.py(line 11)、on_policy_runner.py(line 254)）。
- PPO初始化时解析data_augmentation_func（支持字符串"module: func"），并禁止在recurrent policy上使用（ppo.py(line 76)）。
- PPO.update每个mini-batch：
    - 若use_data_augmentation=True：调用data_augmentation_func(obs, actions, env)，把batch阔成batch_size * num_aug，并把returns/advantages/old_log_prob等按同样倍数repeat，再用扩增后的样本计算PPO loss（ppo.py(line 230)）。
    - 计算symmetry loss：用扩增后的obs_batch做一次act_inference得到均值动作；再用同一个函数把"原始均值动作"做对称变换得到目标动作；两者做MSE。若use_mirror_loss=True则按mirror_loss_coeff加到总loss，否则detach只用于记录（ppo.py(line 315)）。

**使用方法**
- 在训练配置里给algorithm.symmetry_cfg提供这些字段：use_data_augmentation、use_mirror_loss、mirror_loss_coeff、data_augmentation_func（可为callable或"module:func"字符串；解析逻辑键string_to_callable：utils.py(line 315)）。
- 实现data_augmentation_func(env, obs: TensorDict | None, actions: Tensor | None)：必须同时支持"只传obs"或"只传actions"，并保证扩增后的第1段是原始样本、后续段顺序与动作变换一致（mirror loss依赖这个假设）。
- 参考RobotLab的现成用法：配置里直接传callable（rsl_rl_ppo_cfg.py(line 50)），具体对称变换实现见compute_symmetric_states()（anymal.py(line 25)），命令行示例见README（README.md(line 275)）。

## Question2
请你浏览一下rsl_rl的算法框架，然后为我总结一下，如果使用RND模块，那么该模块在整个训练过程中的执行流程是什么

---

RND在rsl_rl里的训练执行流程（按时间顺序）
- 构建/配置阶段：Runner发现启用了algorithm.rnd_cfg后，会把rnd_state加进需要的观测集合（on_policy_runner.py(line 194)），并在构建算法前调用_resolve_rnd_config()补齐num_states / obs_groups、以及把weight乘上env.step_dt（on_policy_runner.py(line 249)，rnd.py(line 184)）。随后PPO初始化时创建RandomNetworkDistillation和rnd_optimizer（只优化predictor）（ppo.py(line 64)）。
- Rollout采样阶段（每个env step）：PPO.process_env_step()里先更新policy的归一化器，也更新RND的state normalizer（如果开了）（ppo.py(line 139)），然后用当前step的观测计算intrinsic reward：取rnd_state → （可选）归一化 → target/predictor输出做距离 → （可选）reward normalization → 按schedule更新weight → 缩放，最后把intrinsic reward加到extrinsic reward上并写入storage（rnd.py(line 117)，ppo.py(line 152)）。intrinsic_rewards另外保留给日志（on_policy_runner.py(line 94)）。
- 计算returns/advantages：compute_returns()用storage里"已叠加了intrinsic的reward"来做GAE/returns（ppo.py(line 170)）。
- 更新阶段（每个mini-batch）：PPO.update()正常算PPO的loss；同时算RND loss：对batch的观测提取rnd_state，让predictor去拟合target（MSE），反传并用rnd_optimizer.step()更新predictor（target永远固定）（ppo.py(line 347)）。如果你同时开了symmetry的数据增强，这里RND只用原始batch（original_batch_size那部分），不会用增强出来的样本训练RND（同上位置可看到[:original_batch_size]的切片）。
- 日志/保存：Logger会把extrinsic/intrinsic分开统计并打印/写入（logger.py(line 70)），Runner也会记录当前rnd_weight（on_policy_runner.py(line 113)）；保存/加载checkpoint时会额外保存/恢复rnd_state_dict和rnd_optimizer_state_dict（on_policy_runner.py(line 134)）。

## Question3
那你能试着为我分析一下target网络在RND模块中的意义吗？

---

在RND里，target网络的核心意义是：提供一个固定不变、随机初始化的"老师/标签生成器"f(s)，让可训练的predictor g(s;θ)去拟合它，而拟合误差就被当作"新奇度"奖励。

具体作用可以理解为：
- 把"新奇度"变成一个监督学习问题：target产生确定的随机特征f(s)，predictor学$g(s) \approx f(s)$；状态越常见，越容易被学会，误差越小；新状态误差大 → intrinsic reward大。
- 保证intrinsic reward的语义稳定：因为target冻结，误差会随着访问次数下降；如果target也跟着训练/变化，奖励目标会漂移，误差不再对应"见没见过"，甚至可能出现一直很大/很乱的非平稳内在奖励。
- 提供一个"随机特征空间/随机哈希"：随机网络相当于把状态映射到一个高维随机嵌入空间，不依赖手工特征；不同状态通常会得到不同的"签名"，误差就能作为密度/访问频次的代理信号。

它和DQN里的"target network（慢更新）"不是一类东西：RND的target按设计就是冻结的，只在初始化/加载checkpoint时改变；训练过程中只更新predictor。

## Question4
请问类EmpiricalDiscountedVariationNormalization，从强化学习理论的层面看，有什么功能和意义吗？

---

从RL理论/优化角度看，EmpiricalDiscountedVariationNormalization的意义就是做一种"按回报尺度自适应的reward缩放"，用来稳定训练、降低尺度敏感性。

它在做的事可以写成：
- 维护一个按折扣累积的量（近似每个环境的"折扣回报"）
    $R_t = \gamma R_{t-1} + r_t$（代码里是self.disc_avg.update(rew)）
- 用$R_t$的历史统计量估计一个运行中的标准差$\sigma_R$（self.emp_norm.update(avg)）
- 输出归一化后的奖励：$\tilde r_t = r_t / \sigma_R$（代码里self.emp_norm.std）

为什么这在理论上有用：
- 在理想MDP里，把奖励整体乘一个整数不会改变最优策略，但在PPO这类"有限步优化 + 函数逼近"中，reward / return的尺度会直接影响：
    - value target的量级 → value loss / 梯度大小
    - advantage的量级 → policy gradient的有效步长（等价于在改"有效学习率"）
    - PPO clip/entropy/value系数之间的相对平衡
- 用 $\sigma_R$ 做缩放，相当于把"典型回报的尺度"拉到相对稳定的范围，使训练对reward的绝对大小、以及reward随训练变化（尤其是RND intrinsic reward的非平稳）更不敏感，从而更稳、更容易调参。

补充：它只除以标准差、不减均值，常见动机是保持reward的符号/稀疏结构（比如intrinsic reward 非负）不被平移破坏。

## Question5
你觉得类EmpiricalDiscountedVariationNormalization，能用于每次agent与env交互得到的reward（即相对于intrinsic reward的extrinsic reward）吗？

---

可以用——这个类本质上就是一个"按回报尺度做自适应缩放"的模块，输入每一步的rew（无论是extrinsic还是intrisic）都会输出 $rew / \sigma$ ，其中 $\sigma$是它用折扣累积量 $R_t = \gamma R_{t-1} + rew_t$的运行标准差估出来的；这样通常能让value学习和PPO更新对reward绝对尺度不那么敏感、训练更稳。

但有两个实际注意点：
- 它没用dones重置折扣累积：_DiscountedAverage会一直做avg = avg * gamma + rew，episode结束也不清零，所以严格来说不是经典实现里"按每个episode的discounted return"来估计尺度；更像是跨时间的指数滑动累积。要更"标准"的return-normalization，一般会在done时把avg[dones]=0.
- 和RND同用时会改相对权重：如果你只对extrinsic做这种缩放，而intrinsic不缩放（或反之），两者的相对大小会变，等价于改变探索奖励权重，可能需要重新调rnd.weight / 系数。

## Question6
如果类_DiscountedAverage只用于RND，或者更准确的说只用于intrinsic reward，那是不是不在episode终止的时候清零在理论上也说得过去？

---

理论上"说得过去"，但取决于你把这个量当成在估计什么。
- _DiscountedAverage.update()这种 $avg_t = \gamma avg_{t-1} + r_t$（normalization.py(line 115)）如果不在done清零，它更像是在一个"连续任务（continuing）"上做指数滤波的折扣累积（跨episode也连续）。
- 经典的"用折扣回报尺度做归一化"的推导通常对应episodic return，那在理论上更一致的做法是对每个env在done=True时把累计量重置（否则你估计的就不是"每个episode的折扣回报"的方差了）。
- 但如果它只用于intrinsic reward的尺度归一化（而intrinsic本身也更像跨时间的非平稳信号，episode往往只是人为截断或重置），不清零可以理解为：让尺度估计更平滑、避免每次reset都人为打断统计。
- 代价是：当done很频繁或reset后分布变化很大时，不清零会把不同episode的历史"串起来"，可能让尺度估计偏离你期望的episodic语义。

## Question7
那从理论上说，每个env done=True时，需要把self.emp_norm._std也清零、重置吗？

---

不需要、也不建议在每个done=True时把self.emp_norm._std清零。
理论上emp_norm估计的是"折扣回报（或其近似量）分布"的全局尺度，用来长期稳定reward的量级；episode结束只是换了一条轨迹，不代表应该丢掉历史统计量。通常需要在done=True时重置的是每个env的折扣累积量（_DiscountedAverage.avg），而不是全局的运行均值/方差（emp_norm）。

## RslRlPpoActorCriticRecurrentCfg
self.alg.storage
    observations: {name_of_observation_group: [num_steps_per_env, num_envs, data_dim_of_observation_group]}
    rewards: [num_steps_per_env, num_envs, 1]
    actions: [num_steps_per_env, num_envs, num_actions]
    dones: [num_steps_per_env, num_envs, 1]
    values: [num_steps_per_env, num_envs, 1]
    actions_log_prob: [num_steps_per_env, num_envs, 1]
    mu: [num_steps_per_env, num_envs, num_actions]
    sigma: [num_steps_per_env, num_envs, num_actions]
    returns: [num_steps_per_env, num_envs, 1]
    advantages: [num_steps_per_env, num_envs, 1]
    saved_hidden_state_a: 
        [[num_steps_per_env, num_layers, num_envs, hidden_dim]] (GRU)
        /
        [**[num_steps_per_env, num_layers, num_envs, hidden_dim]**, **[num_steps_per_env, num_layers, num_envs, hidden_dim]**] (LSTM)
    saved_hidden_state_c:
        [[num_steps_per_env, num_layers, num_envs, hidden_dim]] (GRU)
        /
        [**[num_steps_per_env, num_layers, num_envs, hidden_dim]**, **[num_steps_per_env, num_layers, num_envs, hidden_dim]**] (LSTM)

self.alg.transition:
    hidden_states: 
        (**[num_layers, num_envs, hidden_dim]**, **[num_layers, num_envs, hidden_dim]**) (GRU)
        / 
        (**([num_layers, num_envs, hidden_dim], [num_layers, num_envs, hidden_dim])**, **([num_layers, num_envs, hidden_dim], [num_layers, num_envs, hidden_dim])**) (LSTM)
    actions: [num_envs, num_actions]
    values: [num_envs, 1]
    actions_log_prob: [num_envs, ]
    action_mean: [num_envs, num_actions]
    action_sigma: [num_envs, num_actions]
    observations: {name_of_observation_group: [num_envs, data_dim_of_observation_group]}
    rewards: [num_envs, ]
    dones: [num_envs, ]


class ActorCriticRecurrent:
    def act():
        obs: [num_envs, total_data_dim]
        out_mem: [num_envs, hidden_dim]

class Memory:
    def forward():
    if batch_mode:
        out: [num_steps_per_env, num_trajectories_of_the_mini_batch, total_data_dim]
    else:
        out: [1, num_envs, hidden_dim]
        self.hidden_state: [num_layers, num_envs, hidden_dim]

def split_and_pad_trajectories():
    dones: [num_steps_per_env, num_envs, 1]
    flat_dones: [num_envs * num_steps_per_env, 1]
    done_indices: [num_dones + 1, ]
    trajectory_lengths: [num_dones, ]
    trajectory_lengths_list: length = num_dones (list)

    padded_trajectories:
        For the key 'k' in tensor, padded_trajectories[k]: [the length of the longgest trajectory(namely, num_steps_per_env), num_trajectories, data_dim_of_observation_group]
    v.transpose(1, 0).flatten(0, 1): [num_envs * num_steps_per_env, data_dim_of_observation_group]
    trajectories: It's a tuple.
        The length of the tuple is the number of trajectories.
        The shape of the element of the tuple is [the length of the specific trajectory, data_dim_of_observation_group]
    tensor.transpose(1, 0).flatten(0, 1): [num_envs * num_steps_per_env, data_dim_of_tensor]

    trajectory_masks: [num_steps_per_env, num_dones]

def recurrent_mini_batch_generator():
    padded_obs_trajectories:
        For the key 'k' in self.observations, padded_trajectories[k]: [num_steps_per_env, num_trajectories, data_dim_of_observation_group]
    trajectory_masks: [num_steps_per_env, num_trajectories]

    first_traj = 0

    dones: [num_steps_per_env, num_envs]
    last_was_done: [num_steps_per_env, num_envs]
    trajectories_batch_size: tensor(num_trajectories_of_the_mini_batch)

    masks_batch: [num_steps_per_env, num_trajectories_of_the_mini_batch]
    obs_batch:
        For the key 'k' in self.observations, obs_batch[k]: [num_steps_per_env, num_trajectories_of_the_mini_batch, data_dim_of_observation_group]
    -------
    actions_batch: [num_steps_per_env, num_envs_of_the_mini_batch, num_actions]
    old_mu_batch: [num_steps_per_env, num_envs_of_the_mini_batch, num_actions]
    old_sigma_batch: [num_steps_per_env, num_envs_of_the_mini_batch, num_actions]
    returns_batch: [num_steps_per_env, num_envs_of_the_mini_batch, 1]
    advantages_batch: [num_steps_per_env, num_envs_of_the_mini_batch, 1]
    values_batch: [num_steps_per_env, num_envs_of_the_mini_batch, 1]
    old_actions_log_prob_batch: [num_steps_per_env, num_envs_of_the_mini_batch, 1]

    last_was_done.permute(1, 0): [num_envs, num_steps_per_env]
    hidden_state_a_batch:
        [[num_layers, num_trajectories_of_the_mini_batch, hidden_dim]]（GRU）
        /
        [**[num_layers, num_trajectories_of_the_mini_batch, hidden_dim]**, **[num_layers, num_trajectories_of_the_mini_batch, hidden_dim]**]（LSTM）
    hidden_state_c_batch:
        [[num_layers, num_trajectories_of_the_mini_batch, hidden_dim]]（GRU）
        /
        [**[num_layers, num_trajectories_of_the_mini_batch, hidden_dim]**, **[num_layers, num_trajectories_of_the_mini_batch, hidden_dim]**]（LSTM）
    
    summary: yield(
        **obs_batch**:
        For the key 'k' in self.observations, obs_batch[k]: [num_steps_per_env, num_trajectories_of_the_mini_batch, data_dim_of_observation_group]
        -------
        **actions_batch**: [num_steps_per_env, num_envs_of_the_mini_batch, num_actions]
        **values_batch**: [num_steps_per_env, num_envs_of_the_mini_batch, 1]
        **advantages_batch**: [num_steps_per_env, num_envs_of_the_mini_batch, 1]
        **returns_batch**: [num_steps_per_env, num_envs_of_the_mini_batch, 1]
        **old_actions_log_prob_batch**: [num_steps_per_env, num_envs_of_the_mini_batch, 1]
        **old_mu_batch**: [num_steps_per_env, num_envs_of_the_mini_batch, num_actions]
        **old_sigma_batch**: [num_steps_per_env, num_envs_of_the_mini_batch, num_actions]
        **hidden_state_a_batch**:
            [[num_layers, num_trajectories_of_the_mini_batch, hidden_dim]]（GRU）
            /
            [**[num_layers, num_trajectories_of_the_mini_batch, hidden_dim]**, **[num_layers, num_trajectories_of_the_mini_batch, hidden_dim]**]（LSTM）
        **hidden_state_c_batch**:
            [[num_layers, num_trajectories_of_the_mini_batch, hidden_dim]]（GRU）
            /
            [**[num_layers, num_trajectories_of_the_mini_batch, hidden_dim]**, **[num_layers, num_trajectories_of_the_mini_batch, hidden_dim]**]（LSTM）
        **maskes_batch**:[num_steps_per_env, num_trajectories_of_the_mini_batch]
    )

def unpad_trajectories():
    trajectories.transpose(1, 0): [num_trajectories_of_the_mini_batch, num_steps_per_env, hidden_dim]
    mask.transpose(1, 0): [num_trajectories_of_the_mini_batch, num_steps_per_env]
    trajectories.transpose(1, 0)[mask.transpose(1, 0)]: [valid_timestamps_of_the_mini_batch, hidden_dim]

    trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]): [num_envs_of_the_mini_batch, num_steps_per_env, hidden_dim]

    trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0): [num_steps_per_env, num_envs_of_the_mini_batch, hidden_dim]
