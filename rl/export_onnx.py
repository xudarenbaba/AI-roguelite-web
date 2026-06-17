"""
export_onnx.py — 将训练好的 SB3 PPO 模型导出为 ONNX。

用法：
    # 导出最优模型（推荐）
    python -m rl.export_onnx

    # 导出指定检查点
    python -m rl.export_onnx --model rl/checkpoints/assault_v1_final.zip

    # 导出并验证（对比 SB3 推理和 ONNX 推理结果）
    python -m rl.export_onnx --verify

输出：
    rl/assault_policy.onnx        — 供浏览器 onnxruntime-web 使用的模型
    rl/assault_policy_meta.json   — obs 维度、动作数、版本信息（前端读取用）

注意：
    导出的是 actor（策略网络）部分，不包含 value head，最小化模型体积。
    输入：float32[1, OBS_DIM]
    输出：float32[1, 9]（9个动作的 logit，取 argmax 得到动作）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from rl.env import OBS_DIM, ACTION_VECTORS


# ── Actor 包装：只保留策略网络，输出 logit ─────────────────────────────────

class ActorOnly(nn.Module):
    """
    包装 SB3 ActorCriticPolicy 的 actor 部分。

    前向传播：obs → logit（9维）
    调用方在 JS 端 argmax 得到动作索引。
    """

    def __init__(self, sb3_policy: "ActorCriticPolicy") -> None:  # type: ignore[name-defined]
        super().__init__()
        # SB3 MlpPolicy 结构：mlp_extractor（共享层）+ action_net（actor head）
        self.mlp_extractor = sb3_policy.mlp_extractor
        self.action_net    = sb3_policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent_pi, _ = self.mlp_extractor(obs)
        return self.action_net(latent_pi)  # logit，shape [batch, 9]


# ── 主函数 ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export PPO policy to ONNX")
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="SB3 模型路径（.zip）。默认自动查找 rl/checkpoints/assault_best/best_model.zip",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(_ROOT / "rl" / "assault_policy.onnx"),
        help="输出 ONNX 路径",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="导出后验证 ONNX 与 SB3 推理结果的一致性",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset 版本（onnxruntime-web 推荐 17）",
    )
    return p.parse_args()


def find_model(model_arg: str | None) -> Path:
    if model_arg:
        p = Path(model_arg)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        return p
    # 自动查找优先级：best > final
    candidates = [
        _ROOT / "rl" / "checkpoints" / "assault_best" / "best_model.zip",
        *sorted((_ROOT / "rl" / "checkpoints").glob("assault_v*_final.zip")),
        *sorted((_ROOT / "rl" / "checkpoints").glob("assault_*_steps.zip")),
    ]
    for c in candidates:
        if c.exists():
            print(f"Auto-selected model: {c}")
            return c
    raise FileNotFoundError(
        "No model found. Run `python -m rl.train` first, "
        "or specify --model path/to/model.zip"
    )


def main() -> None:
    args = parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("ERROR: stable-baselines3 not installed. Run: pip install stable-baselines3")
        sys.exit(1)

    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnx/onnxruntime not installed. Run: pip install onnx onnxruntime")
        sys.exit(1)

    model_path = find_model(args.model)
    output_path = Path(args.output)
    meta_path = output_path.with_suffix(".json").with_name(
        output_path.stem + "_meta.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {model_path}")
    model = PPO.load(str(model_path), device="cpu")
    policy = model.policy.to("cpu")
    policy.eval()

    actor = ActorOnly(policy)
    actor.eval()

    # 构造 dummy 输入
    dummy_obs = torch.zeros(1, OBS_DIM, dtype=torch.float32)

    print(f"Exporting to ONNX (opset {args.opset})...")
    torch.onnx.export(
        actor,
        dummy_obs,
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["logits"],
        dynamic_axes={
            "obs":    {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    # 验证 ONNX 结构
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model check passed.")

    # 模型大小
    size_kb = output_path.stat().st_size / 1024
    print(f"Model size: {size_kb:.1f} KB")

    # ── 可选：推理结果对比验证 ────────────────────────────────────────────────
    if args.verify:
        print("\nVerifying ONNX vs SB3 inference...")
        ort_session = ort.InferenceSession(str(output_path))

        rng = np.random.default_rng(42)
        errors = []
        for _ in range(100):
            obs_np = rng.uniform(-1, 1, (1, OBS_DIM)).astype(np.float32)
            obs_t  = torch.from_numpy(obs_np)

            with torch.no_grad():
                sb3_logits = actor(obs_t).numpy()

            ort_logits = ort_session.run(
                ["logits"], {"obs": obs_np}
            )[0]

            err = np.abs(sb3_logits - ort_logits).max()
            errors.append(err)

        max_err = max(errors)
        mean_err = np.mean(errors)
        print(f"Max abs error: {max_err:.2e}")
        print(f"Mean abs error: {mean_err:.2e}")
        if max_err < 1e-4:
            print("Verification PASSED (error < 1e-4)")
        else:
            print(f"WARNING: error {max_err:.2e} is higher than expected.")

    # ── 写元数据 JSON ──────────────────────────────────────────────────────────
    meta = {
        "obs_dim": OBS_DIM,
        "n_actions": 9,
        "action_vectors": ACTION_VECTORS,
        "model_file": output_path.name,
        "opset": args.opset,
        "description": (
            "assault stance RL policy. "
            "Input: float32[1, obs_dim]. "
            "Output: float32[1, 9] logits. "
            "Take argmax to get action index."
        ),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nMeta written to: {meta_path}")
    print(f"ONNX model written to: {output_path}")
    print("\nNext step: copy assault_policy.onnx to game/ and run the game.")


if __name__ == "__main__":
    main()
