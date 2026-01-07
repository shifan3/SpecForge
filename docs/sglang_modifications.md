# SGLang Package Modifications for Qwen3-VL Eagle3 Support

This document describes the modifications made to the SGLang package to support Eagle3 training with Qwen3-VL models.

## Modified Files

### 1. `/usr/local/lib/python3.11/dist-packages/sglang/srt/models/qwen3_vl.py`

**Change**: Added `set_eagle3_layers_to_capture` method to `Qwen3VLForConditionalGeneration` class.

**Location**: After the `forward` method, before the `load_weights` method (around line 768).

**Code Added**:
```python
    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        self.capture_aux_hidden_states = True
        self.model.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]
```

**Purpose**: This method enables Eagle3 auxiliary hidden state capture for Qwen3-VL models. Eagle3 requires capturing hidden states from specific intermediate layers (low, mid, high) during forward passes to train the draft model.

**Reference**: The implementation mirrors the existing `set_eagle3_layers_to_capture` method in `qwen2_5_vl.py` (line 722-733).

## Why These Changes Were Needed

The SGLang package has Eagle3 support for several models including:
- `qwen2.py`
- `qwen2_5_vl.py`
- `qwen3.py`
- `llama.py`
- etc.

However, `qwen3_vl.py` was missing this method, causing the error:
```
AttributeError: 'Qwen3VLForConditionalGeneration' object has no attribute 'set_eagle3_layers_to_capture'
```

## How to Apply This Patch

If you need to reapply this modification after upgrading SGLang:

```bash
# Find the file
SGLANG_QWEN3_VL=$(python3 -c "import sglang; print(sglang.__path__[0])")/srt/models/qwen3_vl.py

# Add the method before the load_weights method
# Insert after line containing "return self.pooler(hidden_states, forward_batch)"
# and before line containing "def load_weights"
```

Or use this patch command:
```bash
cat << 'EOF' | patch -p0
--- /usr/local/lib/python3.11/dist-packages/sglang/srt/models/qwen3_vl.py.orig
+++ /usr/local/lib/python3.11/dist-packages/sglang/srt/models/qwen3_vl.py
@@ -765,6 +765,18 @@
         else:
             return self.pooler(hidden_states, forward_batch)

+    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
+        self.capture_aux_hidden_states = True
+        self.model.capture_aux_hidden_states = True
+        if layer_ids is None:
+            num_layers = self.config.num_hidden_layers
+            self.model.layers_to_capture = [
+                2,
+                num_layers // 2,
+                num_layers - 3,
+            ]  # Specific layers for EAGLE3 support
+        else:
+            self.model.layers_to_capture = [val + 1 for val in layer_ids]
+
     def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
EOF
```
