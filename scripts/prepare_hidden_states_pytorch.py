"""
This script generates hidden states using PyTorch/transformers directly,
without depending on SpecForge. It uses Qwen3VLForConditionalGeneration
with a wrapper to extract hidden states.

Compatible with prepare_hidden_states.py arguments (except sglang-specific args).

Usage:
torchrun --nproc_per_node=8 \
    scripts/prepare_hidden_states_pytorch.py \
    --target-model-path Qwen/Qwen3-VL-7B \
    --enable-aux-hidden-states \
    --data-path ./cache/dataset/data.jsonl \
    --output-path ./cache/hidden_states \
    --chat-template qwen3-vl \
    --max-length 2048 \
    --batch-size 1 \
    --is-vlm

For pre-formatted data (with chat template already applied), add --is-preformatted:
torchrun --nproc_per_node=8 \
    scripts/prepare_hidden_states_pytorch.py \
    --target-model-path Qwen/Qwen3-VL-7B \
    --enable-aux-hidden-states \
    --data-path ./cache/dataset/preformatted_data.jsonl \
    --output-path ./cache/hidden_states \
    --chat-template qwen3-vl \
    --is-preformatted \
    --max-length 2048
"""

import argparse
import gc
import hashlib
import json
import os
import datetime
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    Qwen3VLForConditionalGeneration,
)

try:
    from qwen_vl_utils import process_vision_info

    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    process_vision_info = None


@dataclass
class DataPoint:
    """Data point structure for saving hidden states."""

    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    hidden_state: torch.Tensor
    aux_hidden_state: Optional[torch.Tensor] = None


# Chat template configurations (matching SpecForge templates)
# NOTE: end_of_turn includes trailing newline to match actual decoded text
CHAT_TEMPLATES = {
    "qwen3-vl": {
        "system_prompt": "You are a helpful assistant.",
        "end_of_turn": "<|im_end|>\n",
        "user_header": "<|im_start|>user\n",
        "assistant_header": "<|im_start|>assistant\n",
    },
    "qwen2-vl": {
        "system_prompt": "You are a helpful assistant.",
        "end_of_turn": "<|im_end|>\n",
        "user_header": "<|im_start|>user\n",
        "assistant_header": "<|im_start|>assistant\n",
    },
    "qwen": {
        "system_prompt": "You are a helpful assistant.",
        "end_of_turn": "<|im_end|>\n",
        "user_header": "<|im_start|>user\n",
        "assistant_header": "<|im_start|>assistant\n",
    },
    "llama3": {
        "system_prompt": "",
        "end_of_turn": "<|eot_id|>",
        "user_header": "<|start_header_id|>user<|end_header_id|>\n\n",
        "assistant_header": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
}


class HiddenStatesModelWrapper:
    """
    A wrapper that extracts hidden states from Qwen3VL transformer models.
    Uses hooks to capture hidden states from specified layers.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        aux_hidden_states_layers: Optional[List[int]] = None,
    ):
        self.model = model
        self.aux_hidden_states_layers = aux_hidden_states_layers or []
        self._hidden_states_cache: Dict[int, torch.Tensor] = {}
        self._hooks = []
        self._setup_hooks()

    def _setup_hooks(self):
        """Set up forward hooks to capture hidden states from specified layers."""
        if not self.aux_hidden_states_layers:
            return

        layers = self._get_transformer_layers()
        if layers is None:
            print("Warning: Could not find transformer layers for hook registration.")
            return

        print(f"Setting up hooks for layers {self.aux_hidden_states_layers} (total layers: {len(layers)})")

        for layer_idx in self.aux_hidden_states_layers:
            if 0 <= layer_idx < len(layers):
                hook = layers[layer_idx].register_forward_hook(
                    self._create_hook(layer_idx)
                )
                self._hooks.append(hook)
                print(f"  Registered hook for layer {layer_idx}")
            else:
                print(f"  Warning: Layer index {layer_idx} out of range [0, {len(layers)})")

    def _get_transformer_layers(self):
        """Get the transformer layers from the Qwen3VL model."""
        model = self.model

        # For Qwen3VL specifically: model.model.layers
        # Structure: Qwen3VLForConditionalGeneration -> model (Qwen3VLModel) -> layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
            if hasattr(layers, "__len__") and len(layers) > 0:
                return layers

        # Alternative: For VLM models with language_model attribute
        if hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                layers = lm.model.layers
                if hasattr(layers, "__len__") and len(layers) > 0:
                    return layers
            # Try direct layers access
            if hasattr(lm, "layers"):
                layers = lm.layers
                if hasattr(layers, "__len__") and len(layers) > 0:
                    return layers

        # Fallback: try common layer attribute names
        layer_attrs = ["layers", "h", "block", "blocks", "decoder", "encoder"]

        for attr in layer_attrs:
            if hasattr(model, attr):
                layers = getattr(model, attr)
                if hasattr(layers, "__len__") and len(layers) > 0:
                    return layers
            if hasattr(model, "model") and hasattr(model.model, attr):
                layers = getattr(model.model, attr)
                if hasattr(layers, "__len__") and len(layers) > 0:
                    return layers

        return None

    def _get_final_norm(self):
        """Get the final layer normalization from the model.

        For Qwen3VL, the structure is:
        model.model.language_model.norm (Qwen3VLTextRMSNorm)
        """
        model = self.model

        # Qwen3VL specific path
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            lm = model.model.language_model
            if hasattr(lm, "norm"):
                return lm.norm

        # Alternative paths for other model architectures
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm

        if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            return model.transformer.ln_f

        if hasattr(model, "model") and hasattr(model.model, "final_layer_norm"):
            return model.model.final_layer_norm

        return None

    def _create_hook(self, layer_idx: int):
        """Create a forward hook for a specific layer."""

        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output
            self._hidden_states_cache[layer_idx] = hidden_state.detach()

        return hook

    def set_aux_hidden_states_layers(self, layers: Optional[List[int]]):
        """Update the auxiliary hidden states layers dynamically."""
        # Remove existing hooks
        self.remove_hooks()
        # Set new layers
        self.aux_hidden_states_layers = layers or []
        # Re-setup hooks
        self._setup_hooks()

    def clear_cache(self):
        self._hidden_states_cache.clear()

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @torch.no_grad()
    def extend(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        return_last_hidden_states: bool = True,
        return_logits: bool = False,
        **kwargs,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
    ]:
        """
        Forward pass that returns hidden states in a format compatible with the original script.

        Returns:
            Tuple of (logits, None, aux_hidden_states_list, last_hidden_states_list)
            where each list contains one tensor per sample in the batch.
        """
        self.clear_cache()

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
        }

        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            model_inputs["image_grid_thw"] = image_grid_thw

        outputs = self.model(**model_inputs, **kwargs)

        # Extract last hidden states and apply final layer norm
        # The outputs.hidden_states[-1] is the raw output before final norm
        # We need to apply the norm to match SGLang's output
        last_hidden_states = None
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden_states = outputs.hidden_states[-1]
            # Apply final layer norm (for Qwen3VL: model.model.language_model.norm)
            final_norm = self._get_final_norm()
            if final_norm is not None:
                last_hidden_states = final_norm(last_hidden_states)
        else:
            print("Warning: Could not extract hidden states from model outputs.")

        # Extract auxiliary hidden states from hooks
        aux_hidden_states = None
        if self.aux_hidden_states_layers and self._hidden_states_cache:
            aux_list = []
            for layer_idx in sorted(self.aux_hidden_states_layers):
                if layer_idx in self._hidden_states_cache:
                    aux_list.append(self._hidden_states_cache[layer_idx])

            if aux_list:
                # Concatenate along hidden dim to match SGLang format: [batch, seq, num_layers * hidden]
                aux_hidden_states = torch.cat(aux_list, dim=-1)

        self.clear_cache()

        # Convert to list format (one tensor per sample) to match original interface
        batch_size = input_ids.size(0)
        aux_hidden_states_list = []
        last_hidden_states_list = []

        for i in range(batch_size):
            if aux_hidden_states is not None:
                aux_hidden_states_list.append(aux_hidden_states[i])
            else:
                aux_hidden_states_list.append(None)

            if last_hidden_states is not None:
                last_hidden_states_list.append(last_hidden_states[i])
            else:
                last_hidden_states_list.append(None)

        logits = outputs.logits if return_logits and hasattr(outputs, "logits") else None

        return logits, None, aux_hidden_states_list, last_hidden_states_list

    def __del__(self):
        self.remove_hooks()


class VLMDataset(Dataset):
    """Dataset for VLM models with image processing using qwen_vl_utils."""

    def __init__(
        self,
        data_path: str,
        processor,
        chat_template: str = "qwen3-vl",
        max_length: int = 2048,
        num_samples: Optional[int] = None,
        is_preformatted: bool = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ):
        if not HAS_QWEN_VL_UTILS:
            raise ImportError(
                "qwen_vl_utils is required for VLM preprocessing. "
                "Install it with: pip install qwen-vl-utils"
            )

        self.processor = processor
        self.max_length = max_length
        self.is_preformatted = is_preformatted
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Get chat template config
        if chat_template not in CHAT_TEMPLATES:
            print(f"Warning: Unknown chat template '{chat_template}', using qwen3-vl")
            chat_template = "qwen3-vl"
        self.template = CHAT_TEMPLATES[chat_template]

        # Load data
        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
                if num_samples and len(self.data) >= num_samples:
                    break

    def __len__(self):
        return len(self.data)

    def _apply_loss_mask_from_chat_template(
        self,
        text: str,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Apply loss mask to identify assistant response spans."""
        loss_mask = torch.zeros(len(offsets), dtype=torch.long)

        user_sep = f"{self.template['end_of_turn']}{self.template['user_header']}"
        assistant_sep = (
            f"{self.template['end_of_turn']}{self.template['assistant_header']}"
        )

        assistant_pattern = (
            re.escape(assistant_sep) + r"(.*?)(?=" + re.escape(user_sep) + "|$)"
        )

        for match in re.finditer(assistant_pattern, text, re.DOTALL):
            start_char = match.start(1)
            end_char = match.end(1)

            for idx, (token_start, token_end) in enumerate(offsets):
                if token_end <= start_char:
                    continue
                if token_start > end_char:
                    continue
                loss_mask[idx] = 1

        return loss_mask

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.data[idx]

        # Handle preformatted data
        if self.is_preformatted:
            text = item.get("text", item.get("content", ""))
            image_path = item.get("image", "")

            # For preformatted VLM data, we still need to process images
            if image_path:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path,
                            },
                            {"type": "text", "text": text},
                        ],
                    }
                ]
                image_inputs, video_inputs = process_vision_info(messages)

                encoding = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                )
            else:
                encoding = self.processor(
                    text=[text],
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                )
                image_inputs = None

            input_ids = encoding.input_ids[0]
            attention_mask = encoding.attention_mask[0]

            # For preformatted data, use simple loss mask (all 1s except first token)
            loss_mask = torch.ones(len(input_ids), dtype=torch.long)
            loss_mask[0] = 0

            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
            }

            if hasattr(encoding, "pixel_values") and encoding.pixel_values is not None:
                result["pixel_values"] = encoding.pixel_values
            if (
                hasattr(encoding, "image_grid_thw")
                and encoding.image_grid_thw is not None
            ):
                result["image_grid_thw"] = encoding.image_grid_thw[0]

            return result

        # Non-preformatted data handling
        image_path = item.get("image", "")
        source = item.get("conversations", item.get("messages", []))

        messages = [{"role": "system", "content": self.template["system_prompt"]}]

        if source and source[0]["role"] != "user":
            source = source[1:]

        for j, sentence in enumerate(source):
            role = sentence["role"]
            if role == "user":
                # Add image to ALL user messages to match SpecForge behavior
                messages.append(
                    {
                        "role": role,
                        "content": [
                            {
                                "type": "image",
                                "image": image_path,
                            },
                            {"type": "text", "text": sentence["content"]},
                        ],
                    }
                )
            else:
                messages.append({"role": role, "content": sentence["content"]})

        conversation = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        encoding = self.processor(
            text=[conversation],
            images=image_inputs,
            videos=video_inputs,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        input_ids = encoding.input_ids[0]
        attention_mask = encoding.attention_mask[0]
        offsets = encoding.offset_mapping[0]
        pixel_values = encoding.pixel_values
        image_grid_thw = encoding.image_grid_thw[0]

        decoded_conversation = self.processor.tokenizer.decode(
            input_ids, skip_special_tokens=False
        )

        loss_mask = self._apply_loss_mask_from_chat_template(
            decoded_conversation, offsets
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


class TextDataset(Dataset):
    """Dataset for text-only models."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        chat_template: str = "llama3",
        max_length: int = 2048,
        num_samples: Optional[int] = None,
        is_preformatted: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_preformatted = is_preformatted

        if chat_template not in CHAT_TEMPLATES:
            print(f"Warning: Unknown chat template '{chat_template}', using llama3")
            chat_template = "llama3"
        self.template = CHAT_TEMPLATES[chat_template]

        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
                if num_samples and len(self.data) >= num_samples:
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.data[idx]

        if self.is_preformatted:
            text = item.get("text", item.get("content", ""))
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            loss_mask = [0] + [1] * (len(input_ids) - 1)
        else:
            messages = item.get("conversations", item.get("messages", []))
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            loss_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
        }


def collate_fn_text(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for text-only batching."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    attention_mask = []
    loss_mask = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        input_ids.append(
            torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        loss_mask.append(
            torch.cat([item["loss_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "loss_mask": torch.stack(loss_mask),
    }


def collate_fn_vlm(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for VLM batching (batch_size=1 only)."""
    assert len(batch) == 1, "VLM collate only supports batch_size=1"

    item = batch[0]
    result = {
        "input_ids": item["input_ids"].unsqueeze(0),
        "attention_mask": item["attention_mask"].unsqueeze(0),
        "loss_mask": item["loss_mask"].unsqueeze(0),
    }

    if "pixel_values" in item:
        result["pixel_values"] = item["pixel_values"]
    if "image_grid_thw" in item:
        result["image_grid_thw"] = item["image_grid_thw"].unsqueeze(0)

    return result


class HiddenStatesGenerator:
    """
    Generator for creating and saving hidden states based on the target model.
    This is compatible with the original prepare_hidden_states.py generator.
    """

    def __init__(
        self,
        target_model: HiddenStatesModelWrapper,
        enable_aux_hidden_states: bool = True,
        num_io_threads: int = 4,
        io_queue_size: int = 50,
        file_group_size: int = 2000,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = target_model
        self.enable_aux_hidden_states = enable_aux_hidden_states
        self.num_io_threads = num_io_threads
        self.io_queue_size = io_queue_size
        self.file_group_size = file_group_size
        self.rank = rank
        self.world_size = world_size
        self.show_progress = rank == 0

        self.io_executor = None
        self.pending_futures = []

    def __enter__(self):
        """Initializes resources when entering a 'with' block."""
        self.io_executor = ThreadPoolExecutor(max_workers=self.num_io_threads)
        self.pending_futures = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleans up resources when exiting a 'with' block."""
        if self.io_executor is not None:
            if self.show_progress:
                print("\nWaiting for all async I/O operations to complete...")
            self._wait_all_saves()
            self.io_executor.shutdown(wait=True)
            self.io_executor = None

        if self.world_size > 1:
            dist.barrier()

    def _save_tensor_sync(self, data_point: DataPoint, output_file: str) -> None:
        """Save a data point to a file synchronously."""
        if data_point.hidden_state is not None and torch.any(
            torch.isnan(data_point.hidden_state)
        ):
            print(
                f"Warning: NaN found in hidden_state for {output_file}. Skipping save."
            )
            return

        if data_point.aux_hidden_state is not None and torch.any(
            torch.isnan(data_point.aux_hidden_state)
        ):
            print(
                f"Warning: NaN found in aux_hidden_state for {output_file}. Skipping save."
            )
            return

        torch.save(asdict(data_point), output_file)

    def _save_tensor_async(self, data_point: DataPoint, output_file: str) -> None:
        """Submit a job to save the data point asynchronously."""
        if len(self.pending_futures) >= self.io_queue_size:
            self.pending_futures = [f for f in self.pending_futures if not f.done()]
            if len(self.pending_futures) >= self.io_queue_size:
                self.pending_futures.pop(0).result()

        future = self.io_executor.submit(
            self._save_tensor_sync, data_point, output_file
        )
        self.pending_futures.append(future)

    def _wait_all_saves(self):
        """Ensure that all submitted jobs are completed."""
        if self.pending_futures:
            for future in tqdm(
                self.pending_futures,
                desc="Finalizing Writes",
                disable=not self.show_progress,
            ):
                future.result()
            self.pending_futures.clear()

    def _prepare_output_dirs(
        self, output_path: str, start_idx: int, total_samples: int
    ) -> None:
        """Prepare output directories organized into groups of files."""
        if total_samples == 0:
            return

        start_group = (start_idx // self.file_group_size) * self.file_group_size
        end_sample_idx = start_idx + total_samples - 1
        end_group = (end_sample_idx // self.file_group_size) * self.file_group_size

        for group_start_idx in range(start_group, end_group + 1, self.file_group_size):
            grouped_subdir = (
                f"rows_{group_start_idx}-{group_start_idx + self.file_group_size}"
            )
            output_dir = os.path.join(output_path, grouped_subdir)
            os.makedirs(output_dir, exist_ok=True)

    def _check_existing_files_batch(
        self, output_path: str, global_indices: List[int]
    ) -> List[bool]:
        """Check if the files for the given global indices exist."""

        def check_single_file(idx):
            return os.path.exists(self._get_file_path(output_path, idx))

        with ThreadPoolExecutor(max_workers=self.num_io_threads) as executor:
            exists = list(executor.map(check_single_file, global_indices))
        return exists

    def _get_file_path(self, output_path: str, idx: int) -> str:
        """Get the standard file path for the data point with the given index."""
        group_idx = (idx // self.file_group_size) * self.file_group_size
        grouped_subdir = f"rows_{group_idx}-{group_idx + self.file_group_size}"
        return os.path.join(output_path, grouped_subdir, f"data_{idx}.ckpt")

    @torch.no_grad()
    def generate(
        self,
        data_loader: DataLoader,
        output_path: str,
        start_idx: int = 0,
        samples_per_dp: int = 0,
        is_vlm: bool = False,
    ):
        """
        Generate hidden states for all samples.
        Prioritizes minimal CPU RAM usage by processing samples one-by-one.
        """
        self._prepare_output_dirs(output_path, start_idx, samples_per_dp)

        global_idx = start_idx
        total_skipped, total_processed = 0, 0

        progress_bar = tqdm(
            data_loader,
            disable=not self.show_progress,
            desc="Generating Hidden States",
            position=self.rank,
            leave=True,
        )

        for batch_idx, batch in enumerate(progress_bar):
            batch_size = batch["input_ids"].size(0)
            current_batch_indices = list(range(global_idx, global_idx + batch_size))

            # Check which files already exist
            exists_list = self._check_existing_files_batch(
                output_path, current_batch_indices
            )

            valid_indices_in_batch = [
                i for i, exists in enumerate(exists_list) if not exists
            ]
            sample_global_indices = [
                current_batch_indices[i] for i in valid_indices_in_batch
            ]
            num_valid = len(valid_indices_in_batch)
            total_skipped += batch_size - num_valid

            global_idx += batch_size

            if num_valid == 0:
                if self.show_progress:
                    progress_bar.set_postfix(
                        {
                            "processed": total_processed,
                            "skipped": total_skipped,
                            "pending_io": len(self.pending_futures),
                        }
                    )
                continue

            # Filter batch before moving to GPU to save memory
            if is_vlm:
                # For VLM, we don't filter because batch_size=1
                filtered_batch = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "loss_mask": batch["loss_mask"],
                }
                input_ids_gpu = batch["input_ids"].cuda(non_blocking=True)
                attention_mask_gpu = batch["attention_mask"].cuda(non_blocking=True)

                model_kwargs = {
                    "input_ids": input_ids_gpu,
                    "attention_mask": attention_mask_gpu,
                    "return_last_hidden_states": True,
                    "return_logits": False,
                }

                if "pixel_values" in batch:
                    model_kwargs["pixel_values"] = batch["pixel_values"].cuda(
                        non_blocking=True
                    )
                if "image_grid_thw" in batch:
                    model_kwargs["image_grid_thw"] = batch["image_grid_thw"].cuda(
                        non_blocking=True
                    )
            else:
                filtered_batch = {
                    "input_ids": batch["input_ids"][valid_indices_in_batch],
                    "attention_mask": batch["attention_mask"][valid_indices_in_batch],
                    "loss_mask": batch["loss_mask"][valid_indices_in_batch],
                }

                model_kwargs = {
                    "input_ids": filtered_batch["input_ids"].cuda(non_blocking=True),
                    "attention_mask": filtered_batch["attention_mask"].cuda(
                        non_blocking=True
                    ),
                    "return_last_hidden_states": True,
                    "return_logits": False,
                }

            del batch 
            assert 'pixel_values' in model_kwargs, "pixel_values is not in model_kwargs"
            assert 'image_grid_thw' in model_kwargs, "image_grid_thw is not in model_kwargs"
            _, _, aux_hidden_states_list, last_hidden_states_list = self.model.extend(
                **model_kwargs
            )

            # Process samples one at a time to minimize CPU RAM footprint
            for i, (
                current_global_idx,
                aux_hidden_states,
                last_hidden_states,
            ) in enumerate(
                zip(
                    sample_global_indices,
                    aux_hidden_states_list,
                    last_hidden_states_list,
                )
            ):
                # Transfer only the required slice for one sample to CPU
                aux_hs_cpu = (
                    aux_hidden_states.cpu().clone().unsqueeze(0)
                    if aux_hidden_states is not None
                    else None
                )
                last_hs_cpu = (
                    last_hidden_states.cpu().clone().unsqueeze(0)
                    if last_hidden_states is not None
                    else None
                )

                data_point = DataPoint(
                    input_ids=filtered_batch["input_ids"][i].clone(),
                    loss_mask=filtered_batch["loss_mask"][i].clone(),
                    hidden_state=last_hs_cpu,
                    aux_hidden_state=aux_hs_cpu if self.enable_aux_hidden_states else None,
                )

                output_file = self._get_file_path(output_path, current_global_idx)
                self._save_tensor_async(data_point, output_file)

                del last_hs_cpu, aux_hs_cpu

            total_processed += len(sample_global_indices)

            # Clean up GPU and CPU batch data
            del aux_hidden_states_list, last_hidden_states_list, filtered_batch
            del model_kwargs

            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            if self.show_progress:
                progress_bar.set_postfix(
                    {
                        "processed": total_processed,
                        "skipped": total_skipped,
                        "pending_io": len(self.pending_futures),
                    }
                )

        if self.show_progress:
            print(
                f"\nGeneration loop finished. Processed: {total_processed}, Skipped: {total_skipped}"
            )

        if self.world_size > 1:
            dist.barrier()


def parse_args():
    """Parse arguments - compatible with prepare_hidden_states.py (except sglang args)."""
    parser = argparse.ArgumentParser(
        description="Generate hidden states using PyTorch/transformers (Qwen3VL)"
    )

    # model-related arguments
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--is-vlm", action="store_true", help="Whether the target model is a VLM"
    )
    model_group.add_argument("--enable-aux-hidden-states", action="store_true")
    model_group.add_argument("--aux-hidden-states-layers", type=str, default=None)

    # data-related arguments
    data_group = parser.add_argument_group("data")
    data_group.add_argument("--data-path", type=str, required=True)
    data_group.add_argument("--max-length", type=int, default=2048)
    data_group.add_argument("--chat-template", type=str, default="qwen3-vl")
    data_group.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Whether the input data is preformatted text with the chat template already applied.",
    )
    data_group.add_argument("--num-samples", type=int, default=None)
    data_group.add_argument("--build-dataset-num-proc", type=int, default=8)

    # inference-related arguments
    inference_group = parser.add_argument_group("inference")
    inference_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism size (not used in PyTorch version, kept for compatibility)",
    )
    inference_group.add_argument("--batch-size", type=int, default=1)

    # others arguments
    others_group = parser.add_argument_group("others")
    others_group.add_argument("--cache-dir", type=str, default="./cache")
    others_group.add_argument("--output-path", type=str, default=None)
    others_group.add_argument(
        "--model-download-dir",
        type=str,
        default=None,
        help="The directory to download the target model to",
    )
    others_group.add_argument(
        "--dist-timeout",
        type=int,
        default=2000,
        help="Timeout for collective communication in minutes",
    )
    others_group.add_argument(
        "--num-io-threads",
        type=int,
        default=4,
        help="Number of threads for async I/O operations",
    )
    others_group.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for DataLoader"
    )
    others_group.add_argument(
        "--io-queue-size",
        type=int,
        default=50,
        help="Max number of pending I/O futures.",
    )
    others_group.add_argument(
        "--file-group-size",
        type=int,
        default=2000,
        help="Number of files per subdirectory.",
    )

    # VLM-specific arguments (for image processing)
    vlm_group = parser.add_argument_group("vlm")
    vlm_group.add_argument(
        "--min-pixels",
        type=int,
        default=256 * 28 * 28,
        help="Minimum pixels for image processing",
    )
    vlm_group.add_argument(
        "--max-pixels",
        type=int,
        default=1280 * 28 * 28,
        help="Maximum pixels for image processing",
    )

    return parser.parse_args()


def print_with_rank(message: str, rank: int = None):
    """Print message with rank prefix."""
    if rank is None:
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
    print(f"[Rank {rank}] {message}")


def main():
    args = parse_args()

    # Parse aux hidden states layers
    if args.aux_hidden_states_layers is not None:
        args.aux_hidden_states_layers = [
            int(x) for x in args.aux_hidden_states_layers.split(",")
        ]

    # Initialize distributed environment
    if "RANK" in os.environ:
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=1000),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    print_with_rank(f"DP Rank {rank}, World Size {world_size}", rank)

    # Set default output path
    if args.output_path is None:
        args.output_path = os.path.join(
            Path(__file__).parent.parent, "cache", "hidden_states"
        )

    # Check data path exists
    assert os.path.exists(
        args.data_path
    ), f"Dataset path {args.data_path} does not exist"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path,
        trust_remote_code=True,
        cache_dir=args.model_download_dir,
    )

    # Load processor for VLM
    processor = None
    if args.is_vlm:
        processor = AutoProcessor.from_pretrained(
            args.target_model_path,
            trust_remote_code=True,
            cache_dir=args.model_download_dir,
        )

    # Load model config to get dtype
    model_config = AutoConfig.from_pretrained(
        args.target_model_path,
        trust_remote_code=True,
        cache_dir=args.model_download_dir,
    )

    # Get dtype from config
    torch_dtype = None
    if hasattr(model_config, "dtype") and model_config.dtype is not None:
        torch_dtype = model_config.dtype
    elif hasattr(model_config, "torch_dtype") and model_config.torch_dtype is not None:
        torch_dtype = model_config.torch_dtype
    elif hasattr(model_config, "text_config"):
        text_config = model_config.text_config
        if hasattr(text_config, "dtype") and text_config.dtype is not None:
            torch_dtype = text_config.dtype
        elif (
            hasattr(text_config, "torch_dtype") and text_config.torch_dtype is not None
        ):
            torch_dtype = text_config.torch_dtype

    # Default to bfloat16 if no dtype found
    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    elif isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype, torch.bfloat16)

    print_with_rank(f"Loading model from {args.target_model_path}...", rank)

    # Load Qwen3VL model (hardcoded as per requirements)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.target_model_path,
        dtype=torch_dtype,
        device_map=f"cuda:{local_rank}",
        trust_remote_code=True,
        cache_dir=args.model_download_dir,
    )
    model.eval()

    print_with_rank(f"Model loaded. Type: {type(model).__name__}", rank)

    # Compute default aux_hidden_states_layers if enabled but not specified
    aux_layers = None
    if args.enable_aux_hidden_states:
        if args.aux_hidden_states_layers is not None:
            aux_layers = args.aux_hidden_states_layers
        else:
            # Default layers for EAGLE3: [1, num_layers//2-1, num_layers-4]
            if hasattr(model_config, "text_config"):
                num_layers = model_config.text_config.num_hidden_layers
            elif hasattr(model_config, "num_hidden_layers"):
                num_layers = model_config.num_hidden_layers
            else:
                raise ValueError(
                    f"Cannot determine num_hidden_layers from config: {model_config}"
                )
            aux_layers = [1, num_layers // 2 - 1, num_layers - 4]
            print_with_rank(
                f"Using default aux_hidden_states_layers: {aux_layers} "
                f"(num_layers={num_layers})",
                rank,
            )

    # Create model wrapper for hidden states extraction
    target_model = HiddenStatesModelWrapper(
        model=model,
        aux_hidden_states_layers=aux_layers,
    )

    # Build cache key for dataset
    cache_params_string = f"{args.data_path}-{args.max_length}-{args.chat_template}-{args.target_model_path}-{args.num_samples}-{args.is_preformatted}"
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    print_with_rank(f"Dataset cache key: {cache_key}", rank)

    # Create dataset
    if args.is_vlm:
        dataset = VLMDataset(
            data_path=args.data_path,
            processor=processor,
            chat_template=args.chat_template,
            max_length=args.max_length,
            num_samples=args.num_samples,
            is_preformatted=args.is_preformatted,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        collate_fn = collate_fn_vlm
    else:
        dataset = TextDataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            num_samples=args.num_samples,
            is_preformatted=args.is_preformatted,
        )
        collate_fn = collate_fn_text

    print_with_rank(f"Dataset prepared with {len(dataset)} samples.", rank)

    # Create dataloader with distributed sampler
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    else:
        sampler = None

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print_with_rank(
        f"DataLoader created for DP Rank {rank}. Number of batches: {len(data_loader)}",
        rank,
    )

    # Calculate starting index and sample count for current rank
    total = len(dataset)
    dp_rank = rank
    dp_size = world_size

    # Calculate samples per DP rank (handle non-divisible case)
    samples_per_dp = total // dp_size
    remainder = total % dp_size

    # Earlier ranks handle one extra sample if there's a remainder
    if dp_rank < remainder:
        samples_per_dp += 1
        start_idx = dp_rank * samples_per_dp
    else:
        start_idx = dp_rank * samples_per_dp + remainder

    print_with_rank(
        f"DP Rank {dp_rank} will process {samples_per_dp} samples, "
        f"starting from index {start_idx}",
        rank,
    )

    # Generate hidden states
    try:
        with HiddenStatesGenerator(
            target_model,
            args.enable_aux_hidden_states,
            num_io_threads=args.num_io_threads,
            io_queue_size=args.io_queue_size,
            file_group_size=args.file_group_size,
            rank=rank,
            world_size=world_size,
        ) as hidden_states_generator:
            hidden_states_generator.generate(
                data_loader,
                output_path=args.output_path,
                start_idx=start_idx,
                samples_per_dp=samples_per_dp,
                is_vlm=args.is_vlm,
            )

    finally:
        print_with_rank("All hidden states generated or job finished.", rank)
        target_model.remove_hooks()

        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
