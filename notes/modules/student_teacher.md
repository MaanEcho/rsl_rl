## 167-204行
函数load_state_dict()的功能

---

这个load_state_dict()（student_teacher.py(line 167)）是一个"为蒸馏训练定制"的加载函数：它能识别两种不同来源的checkpoint，并把参数加载到StudentTeacher里的对应子网络里，同时用返回值告诉Runner这是"续训"还是"只是加载teacher作为初始化"。

运行逻辑分三种情况：
- 来自RL训练的checkpoint（ActorCritic）：如果state_dict里出现了"actor"相关key（student_teacher.py(line 180)），就认为这是PPO等RL训练保存的模型
    - 只提取actor.*权重，去掉前缀actor.后加载到self.teacher（student_teacher.py(line 185)）。
    - 只提取actor_obs_normalizer.*，去掉前缀后加载到self.teacher_obs_normalizer（student_teacher.py(line 187)）。
    - 不会加载critic，也不会加载student；然后把teacher置为eval()，并return False表示"不是续训"（student_teacher.py(line 195)）。
- 来自蒸馏训练的checkpoint（StudentTeacher自己保存的）：如果key里出现"student"（student_teacher.py(line 196)），就直接super.load_state_dict(...)把student/teacher/normalizer全部恢复，并return True表示"续训"（student_teacher.py(line 202)）。
- 两者都不是：直接报错（student_teacher.py(line 203)）。

strict参数会原样传给内部的load_state_dict，决定key是否必须严格匹配。
