"""Resource conflict model for fine-grained action parallelism.

Implements a multi-granularity resource tree where every action maps to a
set of (path, mode) pairs.  Two actions conflict iff any pair of their
WRITE paths share a common prefix (write-write conflict).  READ locks
never conflict with anything.

Resource hierarchy (see README for full tree)::

    vm
    ├── display:<N>
    │   ├── clipboard
    │   └── chrome:<profile>
    ├── fs
    │   └── <path nodes nest naturally>
    └── cloud
        └── gdrive
            ├── doc:<id> / chars[start:end]
            ├── sheet:<id> / sheet:<tab> / cells[<range>]
            └── slides:<id> / slide:<slide_id> / element:<elem_id>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ------------------------------------------------------------------
# Access modes
# ------------------------------------------------------------------

class AccessMode(Enum):
    READ = "READ"
    WRITE = "WRITE"


# ------------------------------------------------------------------
# Resource paths
# ------------------------------------------------------------------

class ResourcePath:
    """A hierarchical resource path like ``vm/display:2/clipboard``.

    Paths are tuples of string segments.  Two paths overlap when one is
    a prefix of the other (i.e. they share the same subtree).

    For range-typed leaves (cells, chars) the last segment encodes the
    range and overlap is checked via range intersection.
    """

    __slots__ = ("_segments",)

    def __init__(self, *segments: str) -> None:
        if len(segments) == 1 and "/" in segments[0]:
            self._segments: Tuple[str, ...] = tuple(segments[0].split("/"))
        else:
            self._segments = tuple(segments)

    @property
    def segments(self) -> Tuple[str, ...]:
        return self._segments

    def __repr__(self) -> str:
        return f"ResourcePath({'/'.join(self._segments)})"

    def __str__(self) -> str:
        return "/".join(self._segments)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResourcePath):
            return NotImplemented
        return self._segments == other._segments

    def __hash__(self) -> int:
        return hash(self._segments)

    def is_prefix_of(self, other: ResourcePath) -> bool:
        """True if *self* is a prefix of (or equal to) *other*."""
        if len(self._segments) > len(other._segments):
            return False
        return other._segments[: len(self._segments)] == self._segments

    def overlaps(self, other: ResourcePath) -> bool:
        """True if the two paths share a subtree (prefix relationship).

        Special handling for range-typed leaves: if the paths match up to
        the range segment but the ranges themselves don't intersect, they
        do NOT overlap.
        """
        shorter, longer = (self, other) if len(self._segments) <= len(other._segments) else (other, self)

        for i, seg in enumerate(shorter._segments):
            if i >= len(longer._segments):
                break

            if seg == longer._segments[i]:
                continue

            # Check range overlap (cells[A1:B2] vs cells[C3:D4])
            r1 = _parse_range_segment(seg)
            r2 = _parse_range_segment(longer._segments[i])
            if r1 is not None and r2 is not None and r1[0] == r2[0]:
                return _ranges_overlap(r1[1], r2[1])

            return False

        return True


# ------------------------------------------------------------------
# Range parsing and overlap
# ------------------------------------------------------------------

_RANGE_RE = re.compile(
    r"^(?P<kind>cells|chars)\[(?P<range>[^\]]+)\]$"
)

_CELL_RE = re.compile(
    r"^(?P<col1>[A-Z]+)(?P<row1>\d+):(?P<col2>[A-Z]+)(?P<row2>\d+)$"
)

_CHAR_RE = re.compile(
    r"^(?P<start>\d+):(?P<end>\d+)$"
)


def _parse_range_segment(seg: str) -> Optional[Tuple[str, str]]:
    """Parse ``cells[A1:B2]`` into ``("cells", "A1:B2")`` or None."""
    m = _RANGE_RE.match(seg)
    if m:
        return (m.group("kind"), m.group("range"))
    return None


def _col_to_num(col: str) -> int:
    """Convert Excel-style column letter to number: A=0, B=1, ..., Z=25, AA=26."""
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def _ranges_overlap(r1: str, r2: str) -> bool:
    """Check whether two range strings overlap.

    Supports:
    - Cell ranges: ``A1:B2`` vs ``C3:D4``
    - Character ranges: ``0:100`` vs ``50:150``
    - Column-only ranges: ``A:C`` vs ``B:D``
    """
    # Cell ranges (e.g., A1:B2)
    m1 = _CELL_RE.match(r1)
    m2 = _CELL_RE.match(r2)
    if m1 and m2:
        c1_start = _col_to_num(m1.group("col1"))
        c1_end = _col_to_num(m1.group("col2"))
        r1_start = int(m1.group("row1"))
        r1_end = int(m1.group("row2"))

        c2_start = _col_to_num(m2.group("col1"))
        c2_end = _col_to_num(m2.group("col2"))
        r2_start = int(m2.group("row1"))
        r2_end = int(m2.group("row2"))

        cols_overlap = c1_start <= c2_end and c2_start <= c1_end
        rows_overlap = r1_start <= r2_end and r2_start <= r1_end
        return cols_overlap and rows_overlap

    # Character/numeric ranges (e.g., 0:100) — exclusive end: [start, end)
    cm1 = _CHAR_RE.match(r1)
    cm2 = _CHAR_RE.match(r2)
    if cm1 and cm2:
        s1, e1 = int(cm1.group("start")), int(cm1.group("end"))
        s2, e2 = int(cm2.group("start")), int(cm2.group("end"))
        return s1 < e2 and s2 < e1

    # Fallback: if we can't parse, assume overlap (conservative)
    return True


# ------------------------------------------------------------------
# Resource footprint
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ResourceLock:
    """A single lock: a path + access mode."""
    path: ResourcePath
    mode: AccessMode


class ResourceFootprint:
    """The set of resource locks an action holds.

    Two footprints conflict if any pair of locks are both WRITE and
    their paths overlap.
    """

    __slots__ = ("_locks",)

    def __init__(self, locks: Optional[Set[ResourceLock]] = None) -> None:
        self._locks: FrozenSet[ResourceLock] = frozenset(locks or set())

    @property
    def locks(self) -> FrozenSet[ResourceLock]:
        return self._locks

    def reads(self) -> FrozenSet[ResourceLock]:
        return frozenset(lk for lk in self._locks if lk.mode == AccessMode.READ)

    def writes(self) -> FrozenSet[ResourceLock]:
        return frozenset(lk for lk in self._locks if lk.mode == AccessMode.WRITE)

    def conflicts_with(self, other: ResourceFootprint) -> bool:
        """True if any WRITE lock in *self* overlaps with any WRITE lock
        in *other*."""
        my_writes = self.writes()
        their_writes = other.writes()

        for w1 in my_writes:
            for w2 in their_writes:
                if w1.path.overlaps(w2.path):
                    return True
        return False

    def merge(self, other: ResourceFootprint) -> ResourceFootprint:
        """Return a new footprint combining both sets of locks."""
        return ResourceFootprint(set(self._locks) | set(other._locks))

    def __repr__(self) -> str:
        parts = []
        for lk in sorted(self._locks, key=lambda l: (str(l.path), l.mode.value)):
            parts.append(f"{lk.mode.value}({lk.path})")
        return f"ResourceFootprint({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResourceFootprint):
            return NotImplemented
        return self._locks == other._locks

    def __hash__(self) -> int:
        return hash(self._locks)


# ------------------------------------------------------------------
# Convenience builders
# ------------------------------------------------------------------

def read(path: str) -> ResourceLock:
    """Shorthand for a READ lock on a path string."""
    return ResourceLock(path=ResourcePath(path), mode=AccessMode.READ)


def write(path: str) -> ResourceLock:
    """Shorthand for a WRITE lock on a path string."""
    return ResourceLock(path=ResourcePath(path), mode=AccessMode.WRITE)


def footprint(*locks: ResourceLock) -> ResourceFootprint:
    """Build a ResourceFootprint from individual locks."""
    return ResourceFootprint(set(locks))


# Common resource path builders

def display_path(display_num: int) -> str:
    return f"vm/display:{display_num}"


def clipboard_path(display_num: int) -> str:
    return f"vm/display:{display_num}/clipboard"


def chrome_path(display_num: int) -> str:
    return f"vm/display:{display_num}/chrome"


def fs_path(filepath: str) -> str:
    clean = filepath.rstrip("/")
    return f"vm/fs{clean}"


def sheet_path(doc_id: str, tab: int = 0, cell_range: Optional[str] = None) -> str:
    base = f"vm/cloud/gdrive/sheet:{doc_id}/sheet:{tab}"
    if cell_range:
        return f"{base}/cells[{cell_range}]"
    return base


def doc_path(doc_id: str, char_range: Optional[str] = None, section: Optional[str] = None) -> str:
    base = f"vm/cloud/gdrive/doc:{doc_id}"
    if section:
        return f"{base}/section:{section}"
    if char_range:
        return f"{base}/chars[{char_range}]"
    return base


def slide_path(
    doc_id: str,
    slide_id: Optional[str] = None,
    element_id: Optional[str] = None,
) -> str:
    base = f"vm/cloud/gdrive/slides:{doc_id}"
    if slide_id:
        base = f"{base}/slide:{slide_id}"
        if element_id:
            base = f"{base}/element:{element_id}"
    return base


# ------------------------------------------------------------------
# Resource table (tracks all active locks)
# ------------------------------------------------------------------

class ResourceTable:
    """Tracks all active resource locks across running actions.

    Used by the scheduler to check whether a proposed action can be
    dispatched without conflicting with currently-running actions.
    """

    def __init__(self) -> None:
        self._active: Dict[str, ResourceFootprint] = {}  # node_id -> footprint

    def acquire(self, node_id: str, fp: ResourceFootprint) -> bool:
        """Try to acquire locks for *node_id*.

        Returns True if no write-write conflicts exist with currently
        active locks.  On success, the footprint is recorded.
        On failure, nothing changes.
        """
        for other_id, other_fp in self._active.items():
            if fp.conflicts_with(other_fp):
                return False
        self._active[node_id] = fp
        return True

    def release(self, node_id: str) -> None:
        """Release all locks held by *node_id*."""
        self._active.pop(node_id, None)

    def can_acquire(self, fp: ResourceFootprint) -> bool:
        """Check if *fp* could be acquired without actually acquiring."""
        for other_fp in self._active.values():
            if fp.conflicts_with(other_fp):
                return False
        return True

    def conflicts_with_holder(self, fp: ResourceFootprint) -> Optional[str]:
        """Return the node_id of the first conflicting holder, or None."""
        for other_id, other_fp in self._active.items():
            if fp.conflicts_with(other_fp):
                return other_id
        return None

    def active_count(self) -> int:
        return len(self._active)

    def active_nodes(self) -> List[str]:
        return list(self._active.keys())

    def get_footprint(self, node_id: str) -> Optional[ResourceFootprint]:
        return self._active.get(node_id)

    def clear(self) -> None:
        self._active.clear()
