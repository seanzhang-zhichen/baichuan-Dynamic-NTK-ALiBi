# 前言
**NTK-ALiBi**原理：[NTK-ALiBi：通过插值实现大模型ALiBi位置编码的长文本外推](https://zhuanlan.zhihu.com/p/647628295)


# 代码实现

打开百川模型文件夹中的`modeling_baichuan.py`

1、增加`build_dynamically_alibi_tensor`函数：

```
def build_dynamically_alibi_tensor(num_heads, max_pos) -> torch.Tensor:
    """Psuedo code for Dynamic NTK-ALiBi."""

    # dynamic ntk factor according to actual sequence length
    a0 = 1.0
    train_seq_len = 4096
    a = a0 * torch.tensor(max_pos) / train_seq_len  # [batch, 1]
    a = a.masked_fill(a < 1.0, 1.0)  # dynamic step 1: dynamic ntk scaling factor

    scale = a ** (1.0 / (num_heads-1))  # dynamic step 2: coefficient b, for computation convenience
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=torch.float32
    )
    base = base / scale  # dynamic step 3: divide b to alibi base
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)
    slopes = slopes * scale  # dynamic step 4: fix alibi bias m_h by multiplying b

    if closest_power_of_2 != num_heads:  # todo: fix ntk when num_heads is not power of 2
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes
```

2、修改`_gen_alibi_mask`函数内容

该函数只会在推理时调用

```
def _gen_alibi_mask(n_head, max_pos):
    """used in inference only"""
    slopes = torch.Tensor(build_dynamically_alibi_tensor(n_head, max_pos))
    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_pos).unsqueeze(0).unsqueeze(0).expand(
        n_head, -1, -1)
    alibi = alibi.view(n_head, 1, max_pos).to(torch.float16)
    alibi_mask = torch.triu(
        _fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1
    )
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask
```

麻烦各位点个`start`



参考：

1、[NTK-ALiBi：通过插值实现大模型ALiBi位置编码的长文本外推](https://zhuanlan.zhihu.com/p/647628295)

2、[https://github.com/keezen/ntk_alibi](https://github.com/keezen/ntk_alibi)