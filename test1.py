s = "以下是#我的项目经历\n## 基于强化学习的模型工具调用能力微调\n• 项目描述:用 GRPO 算法在 Verl-Tool 框架下使用强化学习微调 Qwen-2.5-Math-1.5B 模型，大幅 提高基座模型调用 Python 代码解释器工具辅助数学推理的能力。• 训练:训练中采用 ToRL 式 prompt 风格，结合异步推理"
for c in s:
    print(c)
# task title
# type 

d = {"type": "","content":""}

i = 0
while i < len(s):
    c = s[i] 
    j = i
    if (i == 0 and c == "#") or (s[i - 1] == "\n" and c == "#") :
        while s[i] == "#":
            i += 1
        if s[i] == " ":
            i += 1
            c = s[i]
            d["type"] = "title"
            d["content"] = c
            print(d)
            while i < len(s) and s[i] != "\n" :
                i += 1
                c = s[i]
                if c != "\n":
                    d["type"] = "title"
                    d["content"] = c
                    print(d)
        else:
            for e in s[j:i + 1]:
                d["type"] = "text"
                d["content"] = c
    d["type"] = "text"
    d["cont
    ent"] = c
    print(d)
    i += 1

