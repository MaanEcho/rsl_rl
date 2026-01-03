## 88-142行
函数split_and_pad_trajectories()的功能

---

split_and_pad_trajectories()（utils.py(line 88)）的用途是：把rollout中按(time, num_envs, ...)存储的数据，依据dones把每个env的数据切成多段"episode"轨迹，然后把这些不同长度的轨迹拼成一个batch并用0 padding到同一长度，同时返回一个mask标记哪些位置是有效数据、哪些是padding，方便给RNN做batch训练。

- 输入
    - tensor: torch.Tensor 或 TensorDict，形状假设是[T, N, ...]（T=time, N=num_envs）
    - dones：形状一般是[T, N, 1]（或[T, N]），表示每个时间步每个env是否终止

- 输出
    - padded_trajectories：形状[T, M, ...](M=所有env切分后得到的轨迹段总数)，短轨迹后面补0
    - trajectory_masks：形状[T, M]的bool，True=真实数据，False=padding

- 运行逻辑
    1. dones = dones.clone; dones[-1] = 1：把最后一个时间步强制设为done，保证每个env的最后一段也能被"闭合"切出来。
    2. flat_dones = dones.transpose(1, 0).reshape(-1, 1)：把[T, N, ...]变成按env-major展开的[N*T, 1]，方便在一维上找done位置。
    3. 用flat_dones.nonzero()找到所有done的索引，再用相邻done索引之差得到每段轨迹长度trajectory_lengths。
    4. 对tensor（或TensorDict的每个key）做同样的transpose+flatten，然后按trajectory_lengths用torch.split切成若干段变长轨迹。
    5. 额外塞入一条长度为T的全0轨迹，调用pad_sequence统一pad到长度T，再把这条"强制对齐长度"的假轨迹删掉。
    6. 用trajectory_lengths生成trajectory_masks：时间索引 < length 的位置为True，其余为False。

### 112-115行
这部分代码的运行逻辑我没有理解，请你再为我详细解释一下

---

这几行是在把flat_dones里所有done==1的位置找出来，然后用"相邻done的间隔"算出每一段轨迹（episode片段）的长度（给后面的torch.split(..., trajectory_lengths_list)用）。

先明确flat_dones是什么：在前面代码里做了
- dones形状通常是[T, N, 1]（时间步T，并行环境数N）
- flat_dones = done.transpose(1, 0).reshape(-1, 1)之后变成[N*T, 1]，展开顺序是先env0的T个时间步，再env1的T个时间步...
    - 展开后第k个元素对应(env == k // T, t = k % T)
然后看你引用的三行（utils.py(line 112)）：
1. flat_dones.nonzero()[:, 0]
- flat_dones是(N*T, 1)，nonzero()会返回所有非零元素的坐标，形状大概是(#done, 2)，每行是[index_in_dim0, index_in_dim1]。
- 因为第二维永远是0（只有一列），所以取[:, 0]就得到所有done的"一维索引"：
    - 例如done出现在k=3, 5, 7, 11，那这里就是[3, 5, 7, 11]
2. done_indices = torch.cat((..., flat_dones.nonzero()[:, 0]))
- 这里在done索引前面人为加了一个-1：
    - flat_dones.new_tensor([-1], dtype=torch.int64)：在同设备上创建一个[-1]的int64张量
- 目的：让"第一段轨迹"的长度也能用同样的差分公式算出来。
    - 如果第一段轨迹从index 0开始，到第一个done出现在i0结束，那么长度应该是i0 - (-1) = i0 + 1，刚好用-1做哨兵就能得到。
3. trajectory_lengths = done_indices[1:] - done_indices[:-1]
- 这就是在算相邻done的间隔。
- 假设done_indices = [-1, i0, i1, i2, ...]，那么
    - 第一段长度：i0 - (-1)（对应索引0...i0）
    - 第二段长度：i1 - (i0)（对应索引i0+1...i1）
    - 第三段长度：i2 - (i1)（对应索引i1+1...i2）
- 注意：每段都包含done那个时间步（结束点是done的位置）。
4. trajectory_lengths_list = trajectory_lengths.tolist()
- torch.split(..., lengths)需要Python list/int序列，所以转成list。

一个具体例子（T=6，N=2）：
- env0的done（按时间）是[0, 0, 0, 1, 0, 1]
- env1的done是[0, 1, 0, 0, 0, 1]
- 展平后flat_dones（长度12）里done的索引是[3, 5, 7, 11]
- done_indices = [-1, 3, 5, 7, 11]
- trajectory_lengths = [4, 2, 2, 4]
    - 表示切出4段轨迹：env0长4、env0长2、env1长2、env1长4
这就是这段代码的"找done → 算每段长度"的完整逻辑。

## 119-130行
这部分代码的运行逻辑我没有理解，请你再为我详细解释一下

---

这段是在处理tensor为TensorDict的情况：对TensorDict里的每个字段（每个key对应一个张量）都按同一套dones切成若干条变长轨迹，然后padding成等长的[T, 轨迹条数, ...]。

假设输入tensor的batch维是[T, N]（T=time， N=num_envs），则对每个k, v in tensor.items()：
1. v.transpose(1, 0).flatten(0, 1)
- v原形状是[T, N, ...]
- transpose(1, 0)变成[N, T, ...]（先env后time）
- flatten(0, 1)变成[N*T, ...]（把每个env的时间序列按顺序串起来）
- 这么做的原因：前面trajectory_lengths_list是用同样的"env-major展平的"flat_dones算出来的，所以切分边界能对齐。
2. trajectories = torch.split(..., trajectory_lengths_list)
- 把[N*T, ...]按trajectory_lengths_list=[L1, L2, ...]切成很多段：
    - 第i段形状是[Li, ...]
- 每一段就对应一条"episode片段轨迹"（不同env、不同时刻结束都会产生不同长度的段）。
3. trajectories = (*trajectories, torch.zeros(v.shape[0], *v.shape[2:], ...))
- 这里额外加了一条全0的"假轨迹"，长度是v.shape[0]=T，形状[T, ...]
- 目的：保证后面的pad_sequence一定会pad到长度T（否则如果所有真实轨迹都比T短，pad_sequence会只pad到最长真实轨迹的长度，而不是固定T）。
4. padded_trajectories[k] = pad_sequence(trajectories)再[:, :-1]
- pad_sequence会把这些变长序列pad成同长，输出形状是[max_len, num_seq, ...]
- 因为我们加了那条长度为T的假轨迹，所以max_len=T，输出是[T, (真实轨迹条数+1), ...]
- [:, :-1]把最后那条假轨迹删掉，得到[T, 真实轨迹条数, ...]
5. 最后把padded_trajectories（一个普通dict：key → 张量）重新包成TensorDict
- batchsize=[tensor.batch_size[0], len(trajectory_lengths_list)]对应[T, 真实轨迹条数]
- 这样后续RNN训练就能把第二维当作"轨迹batch"，第一维当作"时间"。

## 185-287行
函数resolve_obs_groups()的功能和运行逻辑

---

resolve_obs_groups()（utils.py(line 185)）的作用是：校验obs_groups配置是否与环境返回的观测obs匹配，并把算法"必须存在"的observation set（default set）自动补齐，保证后续Actor/Critic/RND等模块取obs_groups[...]时不会缺key或引用不存在的group。

运行逻辑：
- 确保有"policy" set：若obs_groups里没有"policy"，但obs里存在同名group（obs["policy"]），则自动设为obs_groups["policy"]=["policy"]，并warnings.warn；否则直接ValueError。
- 校验每个set：obs_groups[set_name]不能为空列表；列表里每个group必须在obs的keys里存在，否则ValueError（并在"空列表+属于default_sets"时给出更有针对性的提示）。
- 补齐缺失的default_sets：对每个default_set_name，如果obs_groups没配置它：
    - 若obs里有同名group（obs[default_set_name]），默认用它：obs_groups[default_set_name]=[default_set_name]并warn；
    - 否则回退为使用和"policy"相同的group列表：obs_groups[default_set_name]=obs_groups["policy"].copy()并warn。
- 打印最终解析结果（print("Resolved observation sets:")）并返回（注意：会原地修改传入的obs_groups字典）。
