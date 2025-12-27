# Chapter 5: Signals and Wait Operations

This chapter covers signal-based synchronization, which enables fine-grained
producer-consumer patterns without global barriers.

## Learning Objectives

1. Understand signal-based synchronization
2. Learn to use put_signal for efficient pipelining
3. Implement producer-consumer patterns

## Key Concepts

### Signals

Signals are special variables used for point-to-point synchronization:

```cpp
// Allocate a signal (must be 64-bit aligned)
uint64_t *signal = (uint64_t *)nvshmem_calloc(1, sizeof(uint64_t));

// Wait until signal meets condition
nvshmem_signal_wait_until(signal, NVSHMEM_CMP_EQ, expected_value);

// Set a remote signal
nvshmemx_signal_op(signal, value, NVSHMEM_SIGNAL_SET, peer);
```

### Put with Signal

Combine data transfer with notification:
```cpp
// Put data and set signal atomically
nvshmem_float_put_signal(dest, src, count, signal, value,
                          NVSHMEM_SIGNAL_SET, peer);
```

This guarantees the signal is visible AFTER the data is delivered.

### Comparison Operators

| Operator | Meaning |
|----------|---------|
| `NVSHMEM_CMP_EQ` | Equal |
| `NVSHMEM_CMP_NE` | Not equal |
| `NVSHMEM_CMP_GT` | Greater than |
| `NVSHMEM_CMP_GE` | Greater or equal |
| `NVSHMEM_CMP_LT` | Less than |
| `NVSHMEM_CMP_LE` | Less or equal |

### Signal Operations

| Operation | Effect |
|-----------|--------|
| `NVSHMEM_SIGNAL_SET` | signal = value |
| `NVSHMEM_SIGNAL_ADD` | signal += value |

## Examples

### 1. signal_wait.cu - Basic Signal Wait
### 2. put_signal.cu - Put with Signal

## Use Cases

- Pipeline stages in multi-GPU workflows
- Ring communication without global barriers
- Producer-consumer patterns

## Next Chapter

In [Chapter 6](../06_atomics/), we'll learn about atomic operations.
