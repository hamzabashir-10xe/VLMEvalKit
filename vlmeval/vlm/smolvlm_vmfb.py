"""
SmolVLM-500M VMFB wrapper for VLMEvalKit.
Reuses IreePagedLLM and prepare_smolvlm_inputs_numpy from e2e_runner.py.
Drop-in replacement for the HuggingFace SmolVLM-500M baseline.

Usage:
    python run.py --data MMStar --model SmolVLM-500M-VMFB
"""

import json
import sys
import warnings
import tempfile
import os
from pathlib import Path

import numpy as np
from PIL import Image

# ── Suppress numpy 2.0 copy keyword DeprecationWarning ───────────────────────
warnings.filterwarnings(
    "ignore",
    message=".*copy keyword.*",
    category=DeprecationWarning,
)

# ── Paths — adjust if your layout differs ────────────────────────────────────
SMOLVLM_DEMO_DIR  = Path("/home/lpt-10xe/Downloads/smolVLM-demo")
SMOLVLM_DIR       = SMOLVLM_DEMO_DIR / "SmolVLM"

VMFB_PATH         = SMOLVLM_DIR / "smolvlm-full-500m-working-matmul-fp16-3.vmfb"
WEIGHT_PATHS      = [
    SMOLVLM_DIR / "SmolVLM-500M-Instruct-f16.gguf",
    SMOLVLM_DIR / "mmproj-SmolVLM-500M-Instruct-f16.gguf",
]
TOKENIZER_DIR     = SMOLVLM_DIR / "tokenizer"
VLM_CONFIG_PATH   = SMOLVLM_DIR / "vlm_config.json"

RUNTIME_DEVICE    = "local-task"
RUNTIME_THREADS   = 8
RUNTIME_STACK_SIZE = 131072
# MAX_NEW_TOKENS    = 32          # keep short for benchmark speed
MAX_NEW_TOKENS    = 5          # for MMStar, which is very short-answer. Adjust as needed for other benchmarks.

# ── Import e2e_runner from its location ───────────────────────────────────────
sys.path.insert(0, str(SMOLVLM_DEMO_DIR))
from e2e_runner import (
    IreePagedLLM,
    prepare_smolvlm_inputs_numpy,
    resolve_smolvlm_runtime_config,
    _load_json_if_exists,
)

import iree.runtime as ireert
from transformers import AutoTokenizer
from vlmeval.vlm.base import BaseModel


def _load_runtime_config():
    """Read vlm_config.json and preprocessor_config.json — same logic as e2e_runner.main()."""
    cfg = {}
    if VLM_CONFIG_PATH.exists():
        with open(VLM_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    paged            = cfg.get("paged_kv_cache", {})
    kv_cache_dim     = int(paged.get("paged_kv_block_size_elements_per_device", [655360])[0])
    block_size       = int(paged.get("block_seq_stride", 32))
    max_blocks       = int(paged.get("device_block_count", 512))
    has_prefill_pos  = bool(cfg.get("has_prefill_position", False))
    image_seq_len    = int(cfg.get("image_seq_len", 64))
    final_image_seq_len = int(cfg.get("final_image_seq_len", image_seq_len))

    processor_cfg_path = TOKENIZER_DIR / "processor_config.json"
    if processor_cfg_path.exists():
        try:
            with open(processor_cfg_path, "r", encoding="utf-8") as f:
                pcfg = json.load(f)
            image_seq_len = int(pcfg.get("image_seq_len", image_seq_len))
        except Exception:
            pass

    return kv_cache_dim, block_size, max_blocks, image_seq_len, has_prefill_pos, final_image_seq_len


class SmolVLMVMFB(BaseModel):
    """
    VLMEvalKit model wrapper for IREE-compiled SmolVLM-500M VMFB.
    Produces identical benchmark results format to the HF SmolVLM-500M baseline.
    """

    INSTALL_REQ = False
    INTERLEAVE  = False

    def __init__(self):
        super().__init__()

        # ── Set IREE runtime flags (threads / stack) ──────────────────────
        ireert.flags.parse_flags(
            f"--task_topology_group_count={RUNTIME_THREADS}",
            f"--task_worker_stack_size={RUNTIME_STACK_SIZE}",
        )

        # ── Load runtime config ───────────────────────────────────────────
        (
            self.kv_cache_dim,
            self.block_size,
            self.max_blocks,
            self.image_seq_len,
            self.has_prefill_position,
            self.final_image_seq_len,
        ) = _load_runtime_config()

        # ── Load tokenizer ────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(TOKENIZER_DIR),
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,
        )

        # ── Load preprocessor config ──────────────────────────────────────
        self.preprocessor_cfg = _load_json_if_exists(
            TOKENIZER_DIR / "preprocessor_config.json",
            {
                "do_convert_rgb": True,
                "do_image_splitting": False,
                "do_normalize": True,
                "do_pad": True,
                "do_rescale": True,
                "do_resize": True,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std":  [0.5, 0.5, 0.5],
                "max_image_size": {"longest_edge": 512},
                "resample": 1,
                "rescale_factor": 1.0 / 255.0,
                "size": {"longest_edge": 2048},
            },
        )
        self.preprocessor_cfg["do_image_splitting"] = False
        self.preprocessor_cfg["max_image_size"] = {"longest_edge": 256} 

        # ── Load VMFB + weights via IreePagedLLM ─────────────────────────
        self.model = IreePagedLLM(
            vmfb_path=str(VMFB_PATH),
            weight_files=[str(p) for p in WEIGHT_PATHS],
            device=RUNTIME_DEVICE,
            kv_cache_dim=self.kv_cache_dim,
            block_size=self.block_size,
            max_blocks=self.max_blocks,
            model_type="smolvlm",
            has_prefill_position=self.has_prefill_position,
        )

        print("[SmolVLMVMFB] Model loaded and ready.", file=sys.stderr)

    # ── VLMEvalKit interface ──────────────────────────────────────────────────

    PUREMCQ_DATASETS = {
        "MMStar", "SEEDBench_IMG", "AI2D_TEST", "ScienceQA_VAL", "ScienceQA_TEST",
    }

    def generate_inner(self, message, dataset=None):
        """
        Called by VLMEvalKit for every benchmark sample.
        message format: list of dicts with 'type' and 'value' keys.
        """
        if dataset in self.PUREMCQ_DATASETS:
            prompt, image_path = self._parse_message_puremcq(message)
        else:
            prompt, image_path = self._parse_message(message)

        if image_path is None:
            # Text-only fallback (shouldn't happen in MMStar but handle gracefully)
            return "A"

        return self._run_inference(prompt, image_path)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _parse_message(self, message):
        """Extract text prompt and image path from VLMEvalKit message."""
        image_path = None
        texts      = []

        for item in message:
            if item["type"] == "image":
                image_path = item["value"]   # VLMEvalKit gives a file path
            elif item["type"] == "text":
                texts.append(item["value"])

        prompt = " ".join(texts).strip()
        return prompt, image_path

    def _parse_message_puremcq(self, message):
        """MCQ text formatting matching HF SmolVLM's build_prompt_puremcq.
        Chat template and <image> insertion are handled by e2e_runner's _render_vlm_chat_prompt.
        """
        replace_mapping = {
            "\nOptions:": "\nChoices:",
            "Please select the correct answer from the options above.": "Answer with the letter.",
        }

        image_path = None
        texts = []

        for item in message:
            if item["type"] == "image":
                image_path = item["value"]
            elif item["type"] == "text":
                text = item["value"].strip()
                for k, v in replace_mapping.items():
                    text = text.replace(k, v)
                texts.append(text)

        prompt = " ".join(texts).strip()
        return prompt, image_path

    def _run_inference(self, prompt: str, image_path: str) -> str:
        """
        Full prefill → greedy decode loop using IreePagedLLM.
        Mirrors the per-image loop in e2e_runner.main().
        """
        # Reset KV cache for each sample
        self.model.reset_cache()

        # ── Preprocess ────────────────────────────────────────────────────
        vlm_inputs = prepare_smolvlm_inputs_numpy(
            tokenizer=self.tokenizer,
            tokenizer_dir=str(TOKENIZER_DIR),
            image_path=image_path,
            text=prompt,
            block_size=self.block_size,
            image_seq_len=self.image_seq_len,
            final_image_seq_len=self.final_image_seq_len,
            preprocessor_cfg=self.preprocessor_cfg,
        )

        input_ids    = vlm_inputs["prompt_ids_raw"][0].tolist()

        # ── Prefill ───────────────────────────────────────────────────────
        next_token = self.model.prefill_vlm(vlm_inputs)
        generated  = [int(next_token)]
        input_ids.append(int(next_token))

        # ── Decode loop ───────────────────────────────────────────────────
        eos_id = self.tokenizer.eos_token_id

        for _ in range(MAX_NEW_TOKENS - 1):
            if eos_id is not None and next_token == eos_id:
                break
            start_pos  = len(input_ids) - 1
            next_token = self.model.decode(next_token, start_pos)
            input_ids.append(int(next_token))
            generated.append(int(next_token))

        # ── Decode tokens → text ──────────────────────────────────────────
        return self.tokenizer.decode(generated, skip_special_tokens=True)