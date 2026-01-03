## 259-290行
函数_store_code_state()的功能

---

_store_code_state()（logger.py(line 259)）的作用是：为了实验可复现性，把参与本次实验的代码仓库当前的git status和git diff存到日志目录里（并可选上传到wandb/neptune）。

运行逻辑：
- 只有在self.log_dir存在且当前进程允许写日志（not self.disable_logs）时才执行。
- 在log_dir下创建子目录git/（os.makedirs(..., exist_ok=True)）。
- 遍历self.git_status_repos里的每个路径（通常包含rsl_rl.__file__，也可由runner额外添加）：
    - 用git.Repo(path, search_parent_directories=True)向上查找并定位该路径所在的git仓库；找不到就打印提示并跳过。
    - 取仓库名repo_name=Path(repo.working_dir).name，目标文件为{repo_name}.diff；若文件已存在则跳过（避免重复写）。
    - 创建并写入diff文件（用"x"独占创建）：内容包含repo.git.status()和相对当前HEAD提交树的repo.git.diff(t)（其中t=repo.head.commit.tree）。
    - 把生成的diff文件路径加入file_paths。
- 如果使用的是wandb/neptune（self.writer存在且logger_type in ["wandb", "neptune"]）并且确实生成了文件，则逐个self.writer.save_file(path)上传。
