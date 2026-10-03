import torch, common as C
agent = C.build_agent()
lm = agent.backbone.model.language_model
layer = lm.model.layers[0]
torch.manual_seed(0)
x = torch.randn(1, 3000, 1536, device="cuda", dtype=torch.bfloat16)
ns = (1, 2, 16, 291, 1024, 2536, 2787, 2800, 2827, 2843, 3000)
with torch.no_grad():
    for name, fn in (("gate_proj", layer.mlp.gate_proj), ("mlp", layer.mlp), ("q_proj", layer.self_attn.q_proj),
                     ("rmsnorm", layer.input_layernorm)):
        ref = fn(x[:, :2800])[0, :1]
        print(f"{name:10s}", " ".join(f"{n}:{'=' if torch.equal(fn(x[:, :n])[0, :1], ref) else 'x'}" for n in ns))
    e = torch.randn(1, 3000, 1536, device="cuda", dtype=torch.bfloat16)
    ref = lm.model(inputs_embeds=e[:, :2800], return_dict=True).last_hidden_state[0, :291]
    for n in (291, 2536, 2787, 2827, 2843):
        o = lm.model(inputs_embeds=e[:, :n], return_dict=True).last_hidden_state[0, :291]
        print(f"stock model, {n} rows, first 291 rows vs 2800-row pass:", "identical" if torch.equal(o, ref) else f"rel-L2 {((o.float()-ref.float()).norm()/ref.float().norm()).item():.4f}")
    o = lm.model(inputs_embeds=e[:, :2800], use_cache=True, return_dict=True).last_hidden_state[0, :291]
    print("stock model, 2800 rows, use_cache=True:", "identical" if torch.equal(o, ref) else f"rel-L2 {((o.float()-ref.float()).norm()/ref.float().norm()).item():.4f}")
    m = torch.ones(1, 2816, dtype=torch.long, device="cuda"); m[:, :16] = 0
    pos = m.cumsum(-1) - 1; pos.masked_fill_(m == 0, 1)
    ee = torch.cat((e[:, 2900:2916], e[:, :2800]), dim=1)
    o = lm.model(inputs_embeds=ee, attention_mask=m, position_ids=pos, return_dict=True).last_hidden_state[0, 16:16 + 291]
    print("stock model, 16 pads + 2800 rows:", "identical" if torch.equal(o, ref) else f"rel-L2 {((o.float()-ref.float()).norm()/ref.float().norm()).item():.4f}")
