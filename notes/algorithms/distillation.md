## 102-147行
### Question1
请问函数update()的当前实现，是会累计梯度之后再进行一次反向传播吗？

### Answer
严格说不是"先累计梯度再反向传播"（梯度只有在backward()时才会产生）。这份实现是：
- 先把连续gradient_length个batch的behavior_loss累加成一个总的loss（distillation.py(line 119)）。
- 每当cnt % gradient_length == 0时，才zero_grad()然后对这个"累加后的总loss"做一次loss.backward()（distillation.py(line 124)）
- 因为 $\nabla (\sum L_i) = \sum \nabla L_i$，所以效果等价于把这gradient_length个batch的梯度贡献加起来后更新一次参数（但这里是"累积loss/计算图 → 一次backward"，不是"多次backward累积.grad"）
- 注意它没有除以gradient_length，所以梯度规模会随gradient_length变大；另外会保留这段长度的计算图，显存开销更大；若最后剩余batch数不足gradient_length，那一段loss不会被反传更新。

### Question2
那我能否这样理解，当前函数update()的实现，还是相当于每个mini-batch进行一次反向传播更新？

### Answer
不完全等价。
- 这段update()不是"每个mini-batch都backward()+step()一次"；它是把连续gradient_length个mini-batch的bahavior_loss累加到loss里，只有当cnt%gradient_length == 0时才执行一次loss.backward()和optimizer.step()（distillation.py(lines 123-133)）。
- 因此更新频率是每gradient_length个mini-batch更新一次；更像"梯度累积/等效更大batch"，而不是标准的per-mini-batch SGD（后者会在每个mini-batch更新参数，导致后续梯度是在新参数上算的）。
- 对recurrent的StudentTeacherRecurrent来说，它还相当于做TBPTT：隐藏状态只在做完这次更新后才整体detach()（distillation.py(line 132)），所以梯度会跨这段长度回传。
- 额外注意：如果最后剩下不足gradient_length个mini-batch，当前实现不会再补一次backward()，那一段loss会被丢掉。

### Question3
那这种反向传播更新的方式，与per-mini-batch SGD相比，是否存在优势？

### Answer
有优势，但取决于你在蒸馏里用的是**非循环（student_teacher.py）还是循环（student_teacher_recurrent.py）**模型。
- 对循环/带memory的student（更关键）：现在这种做法相当于做TBPTT（截断反传），gradient_length就是反传穿过hidden state的时间长度。这样梯度能跨多个连续step传回去；如果改成per-mini-batch每个batch就是backward() + step()，通常会更"短视"，很难学到需要记忆的行为。
- 对非循环student：它更像"把gradient_length个mini-batch合成一个更大的batch再更新一次"，优点是梯度方差更小、更新更稳定，并且在多GPU下all_reduce次数更少（通信开销更低）；缺点是参数更新更少、可能收敛慢一些，而且当前实现是sum而不是mean(没除以gradient_length)，等效步长会变大，通常需要相应调小学习率/或改成除以gradient_length。
额外提醒：如果最后剩下不足gradient_length的那一段，当前实现不会补一次反传更新，会丢掉这部分数据。

### Question4
你的讲解中两次提到了TBPTT，请问什么是TBPTT，请你为我讲解一下

### Answer
TBPTT是Truncated Backpropagation Through Time（截断的时间反向传播），是训练RNN/LSTM/GRU这类"有隐藏状态"的模型时最常用的技巧。

背景：BPTT（完整时间反传）
- RNN在时间上会把每一步的hidden state传到下一步，因此第t步的输出依赖很多过去的输入。
- 如果你对一整段很长的序列做训练，理论上可以把这整段都展开，然后把loss的梯度从末尾一路反传回开头（BPTT）。
- 但序列很长时会非常耗显存/算力，而且梯度也容易不稳定（爆照/消失）。

TBPTT的核心做法
- 不对"整条长序列"做一次完整反传，而是把序列切成长度为K的小段（窗口）。
- 每处理K步（或K个mini-batch）就做一次反向传播和参数更新；
- 同时把hidden state在段与段之间detach（切断计算图），这样梯度不会跨越段边界继续往更早的时间传。
因此TBPTT的效果是：
- 梯度只在最近K步范围内传播（能学到有限长度的时序依赖）
- 显存更可控、训练更快、更稳定

和你看到的distillation代码的对应关系
- gradient_length就是TBPTT的"截断长度K"（每累计这么多步/批次才backward（）一次）。
- self.policy.detach_hidden_states()就是"把hidden state从计算图里摘掉"，防止梯度跨越更早的历史继续反传。

一句话：TBPTT="RNN训练时，按固定长度分段反传，段与段之间断开梯度"。

### Question5（108-145行）
请你为我讲解一下，当policy为StudentTeacherRecurrent时，这部分代码的功能和运行逻辑

### Answer
这段distillation.py（line 108）到distillation.py（line 145）在policy=StudentTeacherRecurrent时，本质是在用按时间步顺序回放rollout来训练"学生RNN+MLP"去拟合"老师动作"（行为克隆），并用TBPTT（截断反传）控制反传长度。
- for epoch ...：对同一段buffer做多轮优化（多次扫描同一段序列数据）。
- self.policy.reset(hidden_state=self.last_hidden_states)：把Memory的self.hidden_state（学生的memory_s，如果teacher_recurrent=True还有老师的memory_t）设为"这段rollout开始时的hidden state"，保证每个epoch都从同一初始记忆开始回放。
- for obs, _, privileged_actions, dones in self.storage.generator()（见rollout_storage.py(line 118)）：这里不是随机mini-batch，而是按t=0, ..., T-1逐步yield：
    - obs：当前时间步所有env的观测（典型batch是[num_envs, ...]的TensorDict）
    - privileged_actions：rollout时老师给出的动作（已detach，不参与梯度）
    - dones：当前时间步哪些env结束（[num_envs, 1]）
- actions = self.policy.act_inference(obs)：学生前向用于训练（不采样噪声）。对StudentTeacherRecurrent来说，会走memory_s(obs) -> student(MLP)并推进memory_s.hidden_state（把序列信息记进RNN）。
- loss = loss + behavior_loss + if cnt % gradient_length == 0: loss.backward()：把连续gradient_length个时间步的损失累加后一次反传，相当于对RNN做"长度为gradient_length的TBPTT"；self.policy.detach_hidden_states()用来切断计算图，避免反传跨越更长历史。
- slef.policy.reset(dones.view(-1))：对done=True的env，把对应hidden state清零（调用的是Memory.reset()的"按env重置"分支）。
- self.policy.detach_hidden_states(dones.view(-1))：把这些done env的hidden state在图上也切断，避免梯度跨episode泄露。
- 末尾self.last_hidden_states = self.policy.get_hidden_states()：保存"回放结束后的hidden state"，供下一轮/下一次迭代作为初始hidden state；self.storage.clear()清空buffer。

### Question6（112-113行）
请问这里为什么用的是self.policy.act_inference(obs)而不是self.policy.act(obs)？

### Answer
因为在distillation.py（line 112）这里需要用可反传梯度的"学生输出"来做行为克隆回归。
- StudentTeacherRecurrent.act()（student_teacher_recurrent.py(line 148)）会构造Normal(mean, std)然后用self.distribution.sample()采样动作；sample()默认是不保留梯度路径的（想要可导采样应使用rsample()），而且还会引入随机噪声。
- StudentTeacherRecurrent.act_inference()（student_teacher_recurrent.py(line 155)）直接返回self.student(out_mem)（动作均值/确定性输出），可导，适合拿来和privileged_actions做mse/huber并backward()更新学生网络。

### Question7（142行）
请问这里为什么要再次调用self.policy.detach_hidden_states()，为了保险吗？

### Answer
不是单纯"为了保险"，而是给update()的末尾再加一道"时间截断边界"。在StudentTeacherRecurrent里，act_inference()会走Memory.forward(..., masks=None)，每一步都会把memory_s.hidden_state更新成带计算图的张量（用来做TBPTT）。
虽然代码里已经在：
- epoch开始时detach一次，
- 每次optimizer.step()后detach一次，
- 每步对done的env只detach一次，
但当循环结束时仍可能存在这种情况：有些env没done、且最后一段没有触发cnt % gradient_length == 0的更新点，它们对应的hidden state还连着最近一段计算图。末尾这次self.policy.detach_hidden_states()就是把所有env的hidden state统一从图上断开，避免把autograd历史带出update()（否则可能造成额外显存占用，甚至下次训练时出现"跨段反传/二次反传"的问题）。
