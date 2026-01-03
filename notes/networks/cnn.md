## 163-170行
函数_compute_padding()的功能

---

_compute_padding()（cnn.py(line 163)）的作用是：给特定输入特征图尺寸(H, w)、卷积的kernel/stride/dilation，计算一个对称的(pad_h, pad_w)，用于nn.Conv2d(padding=...)，让卷积后的输出空间尺寸尽量"只按stride缩小"，而不是再额外因为kernel变小。

它的推导来自Pytorch的输出尺寸公式（二维同理）：
out = floor((in + 2p - dilation*(k-1) - 1)/stride + 1)
这段实现选定的目标输出是：
out_target = floor(in / stride)
然后反解需要的总padding：
total_pad = stride * out_target - in - stride + dilation * (k-1) + 1
因为Conv2d的padding=p表示两侧各补p，所以每侧padding约为total_pad/2，用ceil()把它变成整数：
p = ceil(total_pad / 2)
代码里分别对H和W做同样计算，返回(pad_h, pad_w)。

一个直观例子：H=64，k=3，s=2，d=1
out_target=floor(64/2)=32，算出来pad_h=1，代回输出公式得到输出高度正好是32。

补充：当stride=1且有效kernel（d*(k-1) + 1）为偶数时，对称padding不可能做到"输出尺寸完全等于输入"，ceil()会倾向于多pad一点（输出可能比输入大1）；常见配置用奇数kernel时不会遇到这个问题。

## 173-192行
函数_compute_output_dim()的功能

---

_compute_output_dim()（cnn.py(line 173)）的功能是：根据一层卷积的超参数，推算该层输出特征图的空间尺寸（H_out, W_out）；如果该层后面还接了MaxPool2d(stride=2)，则再把尺寸按池化后的结果更新一次。

运行逻辑分两步：
1. 先算Conv2d的输出尺寸（分别对高和宽做同样计算）
    用的就是PyTorch文档里的公式：
    - h = floor((H + 2 * pad_h - dilation * (kernel - 1) - 1) / stride + 1)
    - w = floor((W + 2 * pad_w - dilation * (kernel - 1) - 1) / stride + 1)
    其中 dilation * (kernel - 1) + 1可以理解为"有效kernel大小"，padding是左右/上下对称各补pad_h/pad_w。
2. 如果is_max_pool = True，再算MaxPool2d之后的尺寸
    这里的CNN构建代码用的是固定的MaxPool2d(kernel_size=3, stride=2, padding=1)，它的输出尺寸等价于ceil(h/2)、ceil(w/2)，所以代码直接写：
    - h = ceil(h / 2)
    - w = ceil(w / 2)

最终返回(h, w)。
