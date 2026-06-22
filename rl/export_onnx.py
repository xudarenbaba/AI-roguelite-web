"""
export_onnx.py — 将训练好的 RecurrentPPO (LSTM) 模型导出为 ONNX。

用法：
    python -m rl.export_onnx --verify
    python -m rl.export_onnx --model rl/checkpoints/assault_lstm_v1_final.zip --verify

输出：
    rl/assault_policy.onnx        — 供浏览器 onnxruntime-web 使用
    rl/assault_policy_meta.json   — obs 维度、动作数、LSTM 维度等元信息

LSTM 模型的 ONNX 接口（关键，浏览器需逐帧管理 hidden state）：
    输入：
        obs   : float32[1, OBS_DIM]
        h_in  : float32[1, 1, LSTM_HIDDEN]   上一帧 LSTM hidden state
        c_in  : float32[1, 1, LSTM_HIDDEN]   上一帧 LSTM cell state
    输出：
        logits: float32[1, 9]                动作 logits（argmax 取动作）
        h_out : float32[1, 1, LSTM_HIDDEN]   本帧 hidden state（下一帧作为 h_in）
        c_out : float32[1, 1, LSTM_HIDDEN]   本帧 cell state（下一帧作为 c_in）

    浏览器首帧用全零 hidden state；episode 重置（切换姿态/重开）时也重置为零。
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

from rl.env import OBS_DIM, ACTION_VECTORS, N_ACTIONS


# ── LSTM Actor 包装：obs + (h,c) → logits + (h',c') ─────────────────────────

class LstmActorOnly(nn.Module):
    """
    包装 RecurrentPPO 的 actor 推理路径（含 LSTM），用于 ONNX 导出。

    数据流（与 sb3-contrib RecurrentActorCriticPolicy 对齐）：
        obs[1,D] → lstm_actor → latent[1,H] → policy_net → action_net → logits[1,9]

    LSTM 的 hidden state 作为显式输入输出，使浏览器可以逐帧传递。
    """

    def __init__(self, sb3_policy) -> None:  # noqa: ANN001
        super().__init__()
        self.lstm          = sb3_policy.lstm_actor      # nn.LSTM(D, H)
        self.policy_net    = sb3_policy.mlp_extractor.policy_net
        self.action_net    = sb3_policy.action_net
        self.hidden_size   = self.lstm.hidden_size
        self.num_layers    = self.lstm.num_layers

    def forward(
        self,
        obs: torch.Tensor,    # [1, D]
        h_in: torch.Tensor,   # [num_layers, 1, H]
        c_in: torch.Tensor,   # [num_layers, 1, H]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # LSTM 期望输入 [seq_len=1, batch=1, D]
        lstm_in = obs.unsqueeze(0)                      # [1, 1, D]
        lstm_out, (h_out, c_out) = self.lstm(lstm_in, (h_in, c_in))
        latent = lstm_out.squeeze(0)                    # [1, H]
        latent_pi = self.policy_net(latent)             # [1, 256]
        logits = self.action_net(latent_pi)            # [1, 9]
        return logits, h_out, c_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export RecurrentPPO (LSTM) policy to ONNX")
    p.add_argument("--model", type=str, default=None,
                   help="模型路径（.zip）。默认自动查找 best/final 检查点")
    p.add_argument("--output", type=str,
                   default=str(_ROOT / "rl" / "assault_policy.onnx"))
    p.add_argument("--verify", action="store_true",
                   help="导出后验证 ONNX 与 PyTorch 推理一致性")
    p.add_argument("--opset", type=int, default=12,
                   help="ONNX opset（onnxruntime-web wasm 推荐 12）")
    return p.parse_args()


def find_model(model_arg: str | None) -> Path:
    if model_arg:
        p = Path(model_arg)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        return p
    candidates = [
        _ROOT / "rl" / "checkpoints" / "assault_best" / "best_model.zip",
        *sorted((_ROOT / "rl" / "checkpoints").glob("assault_lstm*_final.zip")),
        *sorted((_ROOT / "rl" / "checkpoints").glob("assault*_final.zip")),
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
        from sb3_contrib import RecurrentPPO
    except ImportError:
        print("ERROR: sb3-contrib not installed. Run: pip install sb3-contrib")
        sys.exit(1)

    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnx/onnxruntime not installed. Run: pip install onnx onnxruntime")
        sys.exit(1)

    model_path = find_model(args.model)
    output_path = Path(args.output)
    meta_path = output_path.with_name(output_path.stem + "_meta.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {model_path}")
    model = RecurrentPPO.load(str(model_path), device="cpu")
    policy = model.policy.to("cpu")
    policy.eval()

    actor = LstmActorOnly(policy)
    actor.eval()

    H = actor.hidden_size
    L = actor.num_layers
    print(f"LSTM: hidden_size={H}, num_layers={L}")

    # dummy 输入
    dummy_obs = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    dummy_h   = torch.zeros(L, 1, H, dtype=torch.float32)
    dummy_c   = torch.zeros(L, 1, H, dtype=torch.float32)

    print(f"Exporting to ONNX (opset {args.opset})...")
    # dynamo=False 走 TorchScript 路径以支持 opset 12（onnxruntime-web wasm 兼容性最好）
    with torch.no_grad():
        torch.onnx.export(
            actor,
            (dummy_obs, dummy_h, dummy_c),
            str(output_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["obs", "h_in", "c_in"],
            output_names=["logits", "h_out", "c_out"],
            dynamo=False,
        )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed.")
    print(f"Model size: {output_path.stat().st_size / 1024:.1f} KB")

    # ── 验证：PyTorch vs ONNX，且模拟多帧 hidden state 递推 ──────────────────
    if args.verify:
        print("\nVerifying ONNX vs PyTorch (10-step rollout with hidden state)...")
        sess = ort.InferenceSession(str(output_path))
        rng = np.random.default_rng(0)

        h_t = np.zeros((L, 1, H), dtype=np.float32)
        c_t = np.zeros((L, 1, H), dtype=np.float32)
        h_p = torch.zeros(L, 1, H)
        c_p = torch.zeros(L, 1, H)

        max_err = 0.0
        for _ in range(10):
            obs_np = rng.uniform(-1, 1, (1, OBS_DIM)).astype(np.float32)

            # ONNX 推理
            ort_logits, h_t, c_t = sess.run(
                ["logits", "h_out", "c_out"],
                {"obs": obs_np, "h_in": h_t, "c_in": c_t},
            )
            # PyTorch 推理
            with torch.no_grad():
                pt_logits, h_p, c_p = actor(torch.from_numpy(obs_np), h_p, c_p)

            err = np.abs(pt_logits.numpy() - ort_logits).max()
            max_err = max(max_err, err)

        print(f"Max abs error over 10-step rollout: {max_err:.2e}")
        if max_err < 1e-4:
            print("Verification PASSED (error < 1e-4)")
        else:
            print(f"WARNING: error {max_err:.2e} higher than expected.")

    # ── 元数据 ────────────────────────────────────────────────────────────────
    meta = {
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "action_vectors": ACTION_VECTORS,
        "lstm_hidden": H,
        "lstm_layers": L,
        "model_file": output_path.name,
        "opset": args.opset,
        "recurrent": True,
        "description": (
            "assault stance RecurrentPPO (LSTM) policy. "
            "Inputs: obs[1,obs_dim], h_in[lstm_layers,1,lstm_hidden], c_in[...]. "
            "Outputs: logits[1,n_actions], h_out, c_out. "
            "Browser must feed previous frame's h_out/c_out as next h_in/c_in; "
            "reset to zeros when episode/stance restarts."
        ),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nMeta written to: {meta_path}")
    print(f"ONNX model written to: {output_path}")
    print("\nNext step: cp rl/assault_policy.onnx game/ && reload the game.")


if __name__ == "__main__":
    main()
