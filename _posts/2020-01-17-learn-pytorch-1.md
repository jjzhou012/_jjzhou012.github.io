



### 1. Pytorch学习（1）: 自动微分

> 参考：
>
> - [http://pytorch123.com/SecondSection/autograd_automatic_differentiation/](http://pytorch123.com/SecondSection/autograd_automatic_differentiation/)
> - [https://zhuanlan.zhihu.com/p/67184419](https://zhuanlan.zhihu.com/p/67184419)

#### 1.1 requires_grad

当我们创建一个张量 (tensor) 的时候，如果没有特殊指定的话，那么这个张量默认是**不需要**求导的。**我们可以通过 `tensor.requires_grad` 来检查一个张量是否需要求导。**

**在张量间的计算过程中，如果在所有输入中，有一个输入需要求导，那么输出一定会需要求导；相反，只有当所有输入都不需要求导的时候，输出才会不需要**。

举一个比较简单的例子，比如我们在训练一个网络的时候，我们从 `DataLoader` 中读取出来的一个 mini-batch 的数据，这些输入默认是不需要求导的，其次，网络的输出没有特意指明需要求导，Ground Truth(数据标签) 也没有特意设置需要求导。那么 loss 如何自动求导呢？其实原因就是上边那条规则，虽然输入的训练数据是默认不求导的，但是，我们的 model 中的所有参数，它默认是求导的，这么一来，其中只要有一个需要求导，那么输出的网络结果必定也会需要求的。

相反，我们试试把网络参数的 `requires_grad` 设置为 False 会怎么样？我们可以通过这种方法，**在训练的过程中冻结部分网络**，**让这些层的参数不再更新**，这在迁移学习中很有用处。

qwerttyqwertyuiop[12345678990-asgdasdfghzxcvbnm,.lkjhgfdsazxcstdatdayzhoudjiajunsdfywfyuqwheawd123455678wSSSHKZKZZZZXCBSGDGUAIEQOIEOI/‘’‘[]09876123456ASDFGHZSDGFHJZXVWQFI’]






$$
\mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)
$$

$$
\mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right)
$$

$$
\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}
$$

$$
\mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n
$$

$$
\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N
$$

