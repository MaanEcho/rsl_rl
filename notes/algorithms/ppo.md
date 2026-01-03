## 431-459行
函数reduce_parameters()的功能

---

reduce_parameters()（ppo.py(line 431)）的作用是：在多GPU/多进程训练里，把每个rank用各自本地mini-batch算出来的梯度做同步，得到"所有GPU的平均梯度"，从而保证后面的optimizer.step()在每张卡上更新一致。

运行逻辑大概是三步：
- 收集本地梯度：把self.policy（以及启用RND时self.rnd）里所有param.grad is not None的梯度展平成一维，并torch.cat成一个大向量all_grads。
- 跨GPU求和再取平均：torch.distributed.all_reduce(all_grads, SUM)把所有rank的all_grads相加并把结果写回到每个rank；再all_grads /= world_size得到平均梯度。
- 把平均梯度写回各参数：按相同的参数遍历顺序，把all_grads逐段切片、reshape成每个参数梯度的形状，然后copy_回param.grad。

注意：这种"拼成一个大向量再all_reduce"的做法依赖各rank参数顺序一致、以及哪些参数有/没有梯度也要一致，否则切片会对不上。它相当于手动实现了DDP的梯度同步步骤。
