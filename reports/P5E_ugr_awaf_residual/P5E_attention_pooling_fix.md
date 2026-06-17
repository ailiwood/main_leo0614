# P5E Attention Pooling Fix

## Bug

Line 75 was: `return pooled, attn if return_weights else pooled`

Python parses this as: `return pooled, (attn if return_weights else pooled)` — which ALWAYS returns a tuple.

When `return_weights=False`, it returned `(pooled, pooled)` instead of just `pooled`.

## Fix

Changed to:
```python
if return_weights:
    return pooled, attn
return pooled
```

## Impact

This bug affected all callers that used `return_weights=False`. The V2 model doesn't use that code path (it always uses `return_weights=True`), but the V1/P5D model did in some places. Fixed for safety.
