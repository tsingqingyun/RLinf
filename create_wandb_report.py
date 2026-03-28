#!/usr/bin/env python3
import wandb
import wandb.apis.reports as wr

ENTITY_NAME = "niejc27"
PROJECT_NAME = "rlinf"

api = wandb.Api()

report = wr.Report(
    project=PROJECT_NAME,
    entity=ENTITY_NAME,
    title="RL 训练监控看板",
    description="包含结果、网络健康度与策略稳定性的监控。"
)

runset = wr.Runset(ENTITY_NAME, PROJECT_NAME)

report.blocks = [
    wr.H1("行1：结果 (Results)"),
    wr.PanelGrid(
        runsets=[runset],
        panels=[
            wr.LinePlot(x="Step", y=["success_once"],    title="Success Once"),
            wr.LinePlot(x="Step", y=["returns_mean"],    title="Returns Mean"),
        ]
    ),

    wr.H1("行2：健康 (Health)"),
    wr.PanelGrid(
        runsets=[runset],
        panels=[
            wr.LinePlot(x="Step", y=["actor/grad_norm"],    title="Actor Grad Norm"),
            wr.LinePlot(x="Step", y=["critic/value_loss"],  title="Critic Value Loss"),
        ]
    ),

    wr.H1("行3：稳定 (Stability)"),
    wr.PanelGrid(
        runsets=[runset],
        panels=[
            wr.LinePlot(x="Step", y=["actor/ratio_abs"],        title="Actor Ratio Abs"),
            wr.LinePlot(x="Step", y=["actor/policy_loss_abs"],  title="Actor Policy Loss Abs"),
        ]
    ),
]

report.save()
print(f"看板已创建！链接: {report.url}")
