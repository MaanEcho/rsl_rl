## 194-203行
函数_get_default_obs_sets()的功能

---

_get_default_obs_sets()的核心作用是：告诉rsl_rl"除了policy之外，这个runner/算法还必须具备哪些observation set"，以便在启动时把obs_groups配置补全到算法可用的形式。
- 在OnPolicyRunner.__init__里，它被传给resolve_obs_groups(...)（on_policy_runner.py(line 43)），作为default_sets。
- 这个函数返回的默认set：永远包含"critic"；如果启用了RND(alg_cfg["rnd_cfg"] != None)则再加"rnd_state"（on_policy_runner.py(line 194)）。
- resolve_obs_groups会确保这些key一定存在：若环境观测里有同名group（比如obs["critic"]），就默认用它；否则就把"policy"的那些group复制给它（utils.py(line 185)）。
- 这样后续模块就不会因为缺key崩掉：ActorCritic需要obs_groups["critic"]来构建/评估value(actor_critic.py(line 48))，RND需要obs_groups["rnd_state"]来拼输入（rnd.py(line 155)）。

## 205-247行
函数_configure_multi_gpu()的功能

---

_configure_multi_gpu()的作用是：根据启动时的分布式环境变量，判断是否启用多GPU（分布式）训练，并在需要时完成rank/设备校验、初始化torch.distributed通信组、设置当前进程使用的GPU。

运行逻辑按顺序是：
- 读取WORLD_SIZE（默认是"1"），得到self.gpu_world_size；若WORLD_SIZE <= 1，则认为不是分布式：把self.gpu_local_rank = 0、self.gpu_global_rank = 0、self.multi_gpu_cfg = None，直接返回。
- 若WORLD_SIZE > 1，读取LOCAL_RANK（本机第几张卡）和RANK（全局第几个进程），分别写入self.gpu_local_rank / self.gpu_global_rank，并生成self.multi_gpu_cfg = {"global_rank", "local_rank", "world_size"}供后续模块使用。
- 做一致性/合法性检查：
    - 要求self.device必须等于f"cuda:{LOCAL_RANK}"，否则直接报错（防止"进程rank和实际用的GPU不一致"）。
    - 要求LOCAL_RANK < WORLD_SIZE 且 RANK < WORLD_SIZE，否则报错。
- 真正启用分布式：
    - 调用torch.distributed.init_process_group(backend="nccl", rank=RANK, world_size=WORLD_SIZE)建立NCCL通信组（通常配合torchrun自动提供的环境变量完成初始化）。
    - 调用torch.cuda.set_device(LOCAL_RANK)，把当前进程的默认CUDA设备固定到对应的本地GPU上。

## 208行
### Question
请问这句代码中的"WORLD_SIZE"是环境变量吗，如果是，通常它是如何设置的，典型代码指令是什么？

### Answer
是的，"WORLD_SIZE"是环境变量（environment variable）。在PyTorch分布式训练里它表示"总进程数"（通常 = num_nodes * num_proc_per_node），常和RANK、LOCAL_RANK、MASTER_ADDR、MASTER_PORT一起由启动器自动注入。

典型设置方式：
- 推荐：用torchrun自动设置（你一般不需要手动export）
    - 单机多卡：torchrun --standalone --nproc_per_node=4 robot_lab/scripts/reinforcement_learning/rsl_rl/my_training.py ...
    - 多机多卡（举例2台机器，每台8进程）：
        - 节点0：torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr=<IP> --master_port=29500 my_train.py ...
        - 节点1：torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 --master_addr=<IP> --master_port=29500 my_train.py ...
- 不推荐但可行：手动设置环境变量（每个进程一份）
    - export MASTER_ADDR=127.0.0.1; export MASTER_PORT=29500;
    - 其他进程把RANK/LOCAL_RANK改成对应值分别启动即可。

## 249-286行
函数_construct_algorithm()的功能

---

_construct_algorithm()（on_policy_runner.py(line 249)）的功能是：基于环境返回的obs（TensorDict）和训练配置train_cfg，把"策略网络+rollout缓存+算法本体（默认PPO）"组装成一个可训练的self.alg，并把可选模块（RND、对称性、多卡）对应的配置也整理好。

运行逻辑按顺序是：
- 处理RND配置：resolve_rnd_config(...)会在启用RND时根据obs+self.cfg["obs_group"]计算num_states、写回obs_groups，并按env.unwrapped.step_dt缩放RND reward权重；没启用则强制设为None（rnd.py(line 184)）。
- 处理对称性配置：resolve_symmetry_config(...)在启用symmetry时把_env=env塞进symmetry_cfg供数据增强/镜像函数使用；没启用则设为None（symmetry.py(line 10)）。
- 兼容旧配置：如果train_cfg里用了废弃的empirical_normalization，这里会发出弃用警告，并把它迁移到policy_cfg["actor_obs_normalization"] / ["critic_obs_normalization"]（仅当你没显式写这两个参数时）。
- 构建策略网络：从policy_cfg["class_name"]（字符串）用eval(...);pop("class_name")得到类并实例化（ActorCritic/...），传入参数是obs+obs_groups+num_actions+其他policy超参，然后.to(self.device)。
- 构建rollout缓存：创建RolloutStorage("rl", num_envs, num_steps_per_env, obs, [num_actions], device)用来存采样数据并给PPO update用（rollout_storage.py(line 33)）。
- 构建算法实例：同样用eval(self.alg_cfg.pop("class_name"))得到算法类（通常是PPO），然后alg_class(actor_critic, storage, device=..., **alg_cfg, multi_gpu_cfg=...)返回（ppo.py(line 24)）
补充：因为这里对self.policy_cfg / self.alg_cfg用了pop("class_name")，所以它们会被"就地修改"；通常没问题（只构建一次），但如果你想复用配置字典多次构建，需要提前copy()。
