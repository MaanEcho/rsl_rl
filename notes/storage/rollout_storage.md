## Question
我不理解第92行代码保存observations的运行逻辑，相关的第57到61行代码，给我感觉self.observations是个Dict而不是个Tensor。请你为我讲解一下这其中的逻辑

---

这里self.observations不是Python的dict，而是tensordict.TensorDict（一个"字典形"的张量容器）。

- 在rollout_storage.py（line 57），它用{key: tensor}这种写法初始化，但外层包了一层TensorDict(...)，并且指定了batch_size=[T, N]（T=num_transitions_per_env, N=num_envs）。所以每个value的实际形状是[T, N, ...]。
- TensorDict的[]有两种常见用法：
    - td["key"]：按key取某个张量；
    - td[t]：按batch维切片（像张量一样切第一维）。
        所以self.observations[self.step]不是取某个key，而是取"第step个时间步"的子TensorDict，其batch_size变成[N]，里面每个value形状是[N, ...]。
- rollout_storage.py(line 92)的self.observations[self.step].copy_(transition.observations)就是在把transition.observations（一个[N, ...]的TensorDict，包含所有观测key）整体拷贝到buffer的第step帧里。

等价写法大概是（手动按key存）：
``` python
for k in self.observations.keys():
    self.observations[k][self.step].copy_(transition.observations[k])
```
这里只是用了TensorDict.copy_()一次性对所有key做了拷贝。

## 183-255行
函数recurrent_mini_batch_generator()的功能

---

recurrent_mini_batch_generator()（rollout_storage.py(line 184)）是给RNN/循环策略的PPO更新用的mini-batch生成器：它会把rollout按done切成若干条变长轨迹、padding成等长序列，并同时提供mask和每条轨迹的初始hidden state，让RNN在batch更新时能正确处理episode断点。

核心运行逻辑（设T=num_steps_per_env，N=num_envs，B=N/num_mini_batches）：
- 先把self.observations[T, N, ...]按self.dones切开并pad：得到
    - padded_obs_trajectories形状约为[T, M, ...]（M=本次rollout中所有env被切出的轨迹段总数）
    - trajectory_masks形状[T, M]（True=有效，False=padding）
- 按env维把数据分成num_mini_batches份（每份B个env）。对第i份env（start:stop）：
    - 构造last_was_done[T, N]：t=0或"上一时刻done"为True，用来标记"每段轨迹的起点"
    - trajectories_batch_size = sum(last_was_done[:, start:stop])：这B个env在T步里一共切出了多少条轨迹段（记作M_i）
    - 用指针first_traj:last_traj（在轨迹维上是连续的）切出本mini-batch对应的padded轨迹：
        - obs_batch = padded_obs_trajectories[:, first_traj:last_traj] → [T, M_i, ...]
        - masks_batch = trajectory_masks[:, first_traj:last_traj] → [T, M_i]
    - 其余非RNN的缓存（actions/returns/advantages/old_mu/old_sigma/log_prob等）直接按env切：形状仍是[T, B, ...]
    - 关键：从self.saved_hidden_state_a/c（rollout时每步保存的hidden state，形状约[T, num_layers, N, hidden_dim]）里，取出last_was_done==True那些"轨迹起点"的hidden state，再切片到[num_layers, M_i, hidden_dim]（LSTM是(h, c)两份；GRU退化成单个张量）
- yield出advantages,...,(hidden_state_a_batch, hidden_state_c_batch, masks_batch)

配合Memory.forward()的batch_mode：RNN先用[T, M_i, ...] + 初始hidden state跑序列，再用masks_batch把输出unpad回[T, B, ...]，从而能和actions_batch/returns_batch这些[T, B, ..]的数据一一对齐来算PPO loss。

## 197-199行
请问last_was_done这个变量的含义是什么，第198、199两行代码的功能又是什么

---

last_was_done（在rollout_storage.py(line 197)）是一个bool标记张量，形状与done系统（通常[T, N]）。它的含义是：
- last_was_done[t, e] == True表示在时间步t这一刻，这个env的"上一时刻"是episode结束（done），因此t是一段新轨迹（新episode段）的起点；同时t=0也被视为起点。
第198、199行做的就是把它构造出来：
- last_was_done[1:] = dones[:-1]：把dones向后平移1步（上一时刻的done变成这一时刻的标记）。所以如果dones[t-1, e]=1，那么last_was_done[t, e]=True。
- last_was_done[0]=True：强制把第一步都标成起点（因为没有t=-1，并且rollout的开头也应当作为一段序列的开始）。

这个标记用于后面统计/索引每个env在[start:stop]这组env里一共切出了多少条轨迹段，以及从保存的hidden state里取出每条轨迹段对应的"初始hidden state"。

## 200行
请问这句代码的功能是什么，为什么要对last_was_done求和，而且还是对[:, start:stop]这部分求和？

---

这句trajectories_batch_size = torch.sum(last_was_done[:, start:stop])的作用是：计算"当前这个mini-batch（选中的那一组env）里一共有多少条轨迹（trajectory）"。
- dones = self.dones.squeeze(-1)形状通常是[T, N]（T=num_steps_per_env, N=num_envs）。
- last_was_done也是[T, N]的bool mask，并且通过
    - last_was_done[1:] = dones[:-1]
    - last_was_done[0] = True
        把它变成"每条轨迹的起点标记"：对某个env来说，last_was_done[t, env]=True表示第t步是一个新episode/新轨迹片段的开始（t=0也算开始）。
- 对last_was_done求和就是在数True的个数：一个True对应一条轨迹的开始，所以总和就是轨迹条数（一个env在rollout里如果中途done多次，就会贡献多条轨迹）。
- 只对[:, start:stop]求和，是因为这个recurrent_mini_batch_generator()的一个mini-batch是按env分组的：start:stop这段env才是当前mini-batch；只统计它们各自包含的轨迹数量，用来算last_traj = first_traj + trajectories_batch_size，从而正确切出padded_obs_trajectories[:, first_traj:last_traj] / trajectory_masks[:, first_traj:last_traj]（这些张量的第2维是"轨迹编号"，不是env编号）

## 213-236行
请你为我讲解一下这部分代码的功能和运行逻辑

---

这段rollout_storage.py（line 213）到rollout_storagepy（line 236）的核心目的：给当前mini-batch里的每一条"轨迹段"（trajectory）准备对应的RNN初始hidden state（actor和critic各一份），这样在update时把padded_obs_trajectories喂给RNN，就能从正确的记忆状态开始还原rollout时的状态演化。

运行逻辑（带形状直觉）：
- rollout时保存的hidden state：self.saved_hidden_state_a[i] / self.saved_hidden_state_c[i]典型形状是[T, L, N, H]（时间步T，层数L，环境数N，hidden维H；LSTM会有i=0/1两份分别对应h和c，GRU只有一份）。
- last_was_done = last_was_done.permute(1, 0)：把last_was_done从[T, N]变成[N, T]，以便和后面permute后的hidden state前两维（env、time）对齐。
- 对每个saved_hidden_state做：
    - saved_hidden_state.permute(2, 0, 1, 3)：把[T, L, N, H]变成[N, T, L, H]（把env维提到最前面）。
    - [last_was_done]：用布尔mask选出所有"轨迹起点"（t=0或者上一时刻done）对应的hidden state，得到形状大致[num_traj_total, L, H]（把env+time展平成"轨迹条数"。
    - [first_traj:last_traj]：再切出当前env范围start:stop对应的那一段轨迹（因为轨迹在扁平化顺序里按env分块排列，所以可以用累计的first_traj/last_traj取连续片段）。
    - .transpose(1, 0)：把[batch_traj, L, H]变成[L, batch_traj, H]，这就是nn.LSTM期望的初始hidden state形状。
- \# Remove the tuple for GRU：如果只有一份hidden state（GRU），就把list解包成单个tensor；如果是LSTM（两份h/c），就保留两份一起传下去。
