"""Capability checks shared by manifold-learning algorithms."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class LearningCapabilityError(ValueError):
    """Raised when an algorithm needs unavailable exact geometry operations."""


@runtime_checkable
class EquivariantEmbeddingProtocol(Protocol):
    """Geometry protocol for algorithms that require Euclidean embeddings."""

    def embed(self, x: Any) -> Any:
        """Map a represented point to equivariant Euclidean coordinates."""
        ...


def operation_kind(manifold: Any, operation: str) -> str:
    """Return the certified status of one geometry operation."""
    if hasattr(manifold, "operation_kind"):
        try:
            return str(manifold.operation_kind(operation))
        except ValueError:
            return "unknown"
    return "exact" if bool(getattr(manifold, f"{operation}_is_exact", False)) else "unknown"


def require_exact_operations(manifold: Any, method: str, *operations: str) -> None:
    """Require exact named operations and report every failed capability."""
    unavailable = [
        f"{name}={operation_kind(manifold, name)}"
        for name in operations
        if operation_kind(manifold, name) != "exact"
    ]
    if unavailable:
        geometry = type(manifold).__name__
        details = ", ".join(unavailable)
        raise LearningCapabilityError(
            f"{method} requires exact manifold operations on {geometry}; received {details}."
        )


__all__ = [
    "EquivariantEmbeddingProtocol",
    "LearningCapabilityError",
    "operation_kind",
    "require_exact_operations",
]
