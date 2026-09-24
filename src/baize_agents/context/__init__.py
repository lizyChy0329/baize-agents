"""上下文治理：计量、截断、压缩。

长对话 agent 的硬门槛：上下文窗口是有限的，
不加治理，聊到几十轮必然撞上 context_length_exceeded。
"""
