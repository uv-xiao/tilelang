# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Token system for tracking asynchronous distributed operations.

Tokens are handles that represent pending asynchronous operations.
They enable fine-grained synchronization and computation-communication overlap.

Key concepts:
- Token: Handle returned by async operations (put_async, get_async, put_signal)
- consume_token: Ensures async operation completed before using result
- wait_token: Explicitly waits for token completion

The token system allows overlapping computation with communication:

    token = T.put_async(src, dst, peer_pe)  # Start async transfer
    # ... do computation while transfer in flight ...
    T.consume_token(dst, token)  # Ensure transfer completed before using dst
"""

from __future__ import annotations

from typing import List, Any
import tilelang.language as T
from tvm.tir import PrimExpr


class Token:
    """
    Represents a handle to an asynchronous distributed operation.

    Tokens are returned by async primitives (put_async, get_async, put_signal)
    and must be consumed or waited on to ensure completion.

    Attributes:
        handle: The underlying TVM expression handle
        op_type: Type of operation ("put", "get", "put_signal", etc.)
        dst_pe: Target PE for the operation (if applicable)
        scope: Communication scope used
    """

    def __init__(
        self,
        handle: PrimExpr,
        op_type: str = "unknown",
        dst_pe: PrimExpr | None = None,
        scope: str = "global",
    ):
        """
        Create a new Token.

        Args:
            handle: The underlying expression handle from the async call
            op_type: Type of async operation
            dst_pe: Target PE (for debugging/optimization)
            scope: Communication scope
        """
        self.handle = handle
        self.op_type = op_type
        self.dst_pe = dst_pe
        self.scope = scope

    def __repr__(self) -> str:
        return f"Token({self.op_type}, scope={self.scope})"


def consume_token(value: Any, token: Token) -> Any:
    """
    Consume a token, ensuring the associated async operation has completed.

    This creates a data dependency that forces the compiler to wait for
    the async operation before using the value. The returned value is
    identical to the input but with the dependency encoded.

    Args:
        value: The value/buffer that depends on the async operation
        token: Token from the async operation

    Returns:
        The value with the dependency on the token

    Example:
        >>> token = T.get_async(remote_buf, local_buf, src_pe)
        >>> # ... other computation ...
        >>> local_buf_ready = T.consume_token(local_buf, token)
        >>> T.gemm(local_buf_ready, B, C)  # Uses data from async get
    """
    # The consume creates a dependency but doesn't actually modify the value
    # This is lowered to appropriate sync by the compiler
    if isinstance(token, Token):
        handle = token.handle
    else:
        handle = token

    # Call the consume intrinsic
    # The compiler will ensure the async op completes before using value
    return T.call_extern(
        "handle",
        "tl_dist_consume_token",
        value if isinstance(value, PrimExpr) else T.address_of(value),
        handle
    )


def wait_token(token: Token) -> None:
    """
    Explicitly wait for a token to complete.

    This blocks until the async operation associated with the token
    has completed. Use this when you need to ensure completion
    without immediately using the result.

    Args:
        token: Token from an async operation

    Example:
        >>> token = T.put_async(local_buf, remote_buf, dst_pe)
        >>> T.wait_token(token)  # Ensure put completed
        >>> # Now safe to modify local_buf
    """
    if isinstance(token, Token):
        handle = token.handle
    else:
        handle = token

    T.call_extern("handle", "tl_dist_wait_token", handle)


def wait_tokens(tokens: List[Token]) -> None:
    """
    Wait for multiple tokens to complete.

    This is more efficient than calling wait_token multiple times
    when waiting for several async operations.

    Args:
        tokens: List of tokens from async operations

    Example:
        >>> tokens = []
        >>> for peer in peers:
        >>>     tokens.append(T.put_async(local_buf, T.remote(buf, peer), peer))
        >>> T.wait_tokens(tokens)  # Wait for all puts to complete
    """
    for token in tokens:
        wait_token(token)


def create_token(handle: PrimExpr, op_type: str = "unknown", **kwargs) -> Token:
    """
    Create a Token from an expression handle.

    This is an internal helper used by async primitives.

    Args:
        handle: The expression handle from the async call
        op_type: Type of operation
        **kwargs: Additional token metadata

    Returns:
        A new Token instance
    """
    return Token(handle, op_type=op_type, **kwargs)
