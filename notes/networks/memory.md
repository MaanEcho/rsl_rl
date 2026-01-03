## 50-66行
函数reset()的功能

---

memory.py（line 50）这个reset()的作用是"管理/重置RNN的隐藏状态（hidden state）"，避免episode结束后把上一条轨迹的记忆带到下一条轨迹里。

它的运行逻辑分两类：
- dones is None：重置整个hidden state
    - hidden_state is None：self.hidden_state = None（彻底清空，让下一次RNN从零状态开始）
    - hidden_state is not None：self.hidden_state = hidden_state（用外部传入的初始状态覆盖，比如在更新阶段用buffer里保存的初始hidden state）
- dones is not None且self.hidden_state is not None：只重置"done的那些环境"的hidden state
    - hidden_state is None：
        - 若是LSTM（self.hidden_state是(h, c)的tuple）：对h和c都执行[..., dones==1, :] = 0
        - 若是GRU/RNN（单个张量）：直接self.hidden_state[..., dones==1, :] = 0
    - hidden_state is not None：代码"想"报错（NotImplemented），但这里少写了raise，所以目前实际上不会真正抛异常。

直观上，若GRU的self.hidden_state形状是[num_layers, num_envs, hidden_dim]， 这段代码做的就是对所有done的env：self.hidden_state[:, env_id, :] = 0（LSTM则同时把h、c对应位置清零）

## 68-80行
函数detach_hidden_state()的功能

---

memory.py（line 68）的detach_hidden_state()作用是：把RNN的hidden state从当前计算图中"摘出来"(detach)，从而控制反向传播的时间跨度，避免梯度一致穿过很长的历史（也能避免显存不断累积）。在多env并行时，还支持"只对done的env做detach"。

运行逻辑分两种：
- dones is None：对整个hidden state做detach
    - 如果是LSTM（self.hidden_state是(h, c)）：把h、c都替换为各自的detach()版本
        self.hidden_state = (h.detach(), c.detach())
    - 如果是GRU/RNN（单个张量）：self.hidden_state = self.hidden_state.detach()
- dones is not None：只对done的那些环境对应的hidden state做detach
    - 若是LSTM：对h和c分别做索引赋值
        hidden_state[..., dones==1, :] = hidden_state[..., dones==1, :].detach()
    - 若是GRU：同理对单个hidden state张量做这句

直观理解：
- reset(dones)是把done env的hidden置零（"清记忆"）；
- detach_hidden_state(dones)是把done env的hidden在计算图上断开（"不让梯度跨episode传播"）；
- detach_hidden_state()（无dones）常用于TBPTT：每隔固定步数把整段历史截断一次。
