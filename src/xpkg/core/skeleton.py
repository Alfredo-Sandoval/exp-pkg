"""
SIESTA skeleton loader/normalizer (schema v1.1.0)

 - Link field is `links` (pairs of KEYPOINT NAMES; undirected).
 - Accept aliases on load: {edges, segments, bones, skeleton}; normalize to `links`.
 - Accept keypoint container aliases:
   {keypoints, bodyparts, markers, nodes, landmarks}; normalize to `keypoints`.
 - Do NOT store symmetry pairs; derive L<->R from per-keypoint `side`
   + *_left/_right naming.
 - Optionally normalize names to snake_case and expand L/R tokens.
 - Export ALWAYS with `links` and `keypoints` only (drop alias containers).
 - Provide validation, adjacency/incidence helpers,
   and simple analytics resolution (pairs/triples).
"""

from __future__ import annotations

import hashlib
import re
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xpkg.config.definitions import ROLE_ENUM, SIDE_TOKENS, get_skeleton_def
from xpkg.core.json_utils import parse_json_dict
from xpkg.core.logging_utils import get_logger

logger = get_logger(__name__)

SNAKE_RE1 = re.compile(r"([a-z0-9])([A-Z])")
NON_ALNUM_RE = re.compile(r"[^a-z0-9_]+")

SCHEMA_VERSION = "1.1.0"

LINK_FIELDS = ("links",)
KP_FIELDS = ("keypoints",)


def to_snake(name: str) -> str:
    """Convert to snake_case with minimal heuristics.

    Args:
        name: The string to convert.

    Returns:
        The snake_case version of the string.
    """
    s = SNAKE_RE1.sub(r"\1_\2", name)
    s = s.replace("-", "_").replace(" ", "_")
    s = s.lower()
    s = NON_ALNUM_RE.sub("_", s)
    s = re.sub(r"__+", "_", s).strip("_")
    if s.endswith("_l"):
        s = s[:-2] + "_left"
    elif s.endswith("_r"):
        s = s[:-2] + "_right"
    return s


def infer_side(name: str, default: str = "unknown") -> str:
    """Infer the side (left/right/midline) from a keypoint name.

    Args:
        name: The keypoint name to analyze.
        default: The default side if none is inferred.

    Returns:
        The inferred side string.
    """
    toks = name.split("_")
    for t in reversed(toks):
        if t in SIDE_TOKENS:
            return SIDE_TOKENS[t]
    return default


def base_lr_name(name: str) -> tuple[str, str | None]:
    """Split a normalized name into base portion and explicit side suffix.

    Args:
        name: The normalized keypoint name.

    Returns:
        A tuple of (base_name, side_suffix).
    """
    if name.endswith("_left"):
        return name[:-5], "left"
    if name.endswith("_right"):
        return name[:-6], "right"
    if name.startswith("left_"):
        return name[5:], "left"
    if name.startswith("right_"):
        return name[6:], "right"
    if name.startswith("lf_"):
        return "f_" + name[3:], "left"
    if name.startswith("rf_"):
        return "f_" + name[3:], "right"
    if name.startswith("lh_"):
        return "h_" + name[3:], "left"
    if name.startswith("rh_"):
        return "h_" + name[3:], "right"
    if name.startswith("fl_"):
        return "f_" + name[3:], "left"
    if name.startswith("fr_"):
        return "f_" + name[3:], "right"
    if name.startswith("hl_"):
        return "h_" + name[3:], "left"
    if name.startswith("hr_"):
        return "h_" + name[3:], "right"
    if name.startswith("l_"):
        return name[2:], "left"
    if name.startswith("r_"):
        return name[2:], "right"
    return name, None


def uniq(seq: Iterable[Any]) -> list[Any]:
    """Return the unique elements of `seq`, preserving input order.

    Args:
        seq: The input sequence.

    Returns:
        A list of unique elements.
    """
    seen: set[Any] = set()
    out: list[Any] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


@dataclass
class Keypoint:
    """Normalized descriptor for a skeleton keypoint."""

    id: int
    name: str
    side: str = "unknown"
    role: str | None = None
    entity: str | None = None
    mirror_partner: str | None = None

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class Skeleton:
    """Schema-aware skeleton model with normalization/validation helpers."""

    name: str
    keypoints: list[Keypoint]
    links_ids: list[tuple[int, int]]
    schema_version: str = SCHEMA_VERSION
    description: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    analytics: dict[str, Any] = field(default_factory=dict)
    units: dict[str, Any] | None = None
    coordinate_system: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    license: str | None = None
    validation: dict[str, Any] | None = None
    constraints: dict[str, Any] | None = None
    notes: str | None = None
    preview_image: str | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        normalize_names: bool = True,
    ) -> Skeleton:
        """Create a normalized Skeleton from parsed YAML/JSON.

        Args:
            data: The dictionary containing skeleton data.
            normalize_names: Whether to normalize keypoint names to snake_case.

        Returns:
            A normalized Skeleton instance.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        d = dict(data)

        sv = str(d.get("schema_version", SCHEMA_VERSION))

        if "keypoints" not in d:
            raise ValueError("Missing keypoints.")
        kps_raw = d["keypoints"]

        kps_norm: list[Keypoint]
        if not kps_raw:
            kps_norm = []
        elif isinstance(kps_raw[0], dict):
            kps_norm = []
            has_id_all = all("id" in kp for kp in kps_raw)
            if has_id_all:
                kps_raw = sorted(kps_raw, key=lambda x: int(x["id"]))
            for i, kp in enumerate(kps_raw):
                kid = int(kp.get("id", i))
                nm = str(kp["name"])
                nm_canon = to_snake(nm) if normalize_names else nm
                side = str(kp.get("side", infer_side(nm_canon))).lower()
                role = kp.get("role")
                if role and role not in ROLE_ENUM and not str(role).startswith("custom:"):
                    warnings.warn(
                        f"Unknown role '{role}' for '{nm_canon}'. Consider custom:role.",
                        stacklevel=2,
                    )
                entity = kp.get("entity")
                mirror_partner = kp.get("mirror_partner")
                kps_norm.append(Keypoint(kid, nm_canon, side, role, entity, mirror_partner))
        else:
            kps_norm = []
            for i, nm in enumerate(map(str, kps_raw)):
                nm_canon = to_snake(nm) if normalize_names else nm
                kps_norm.append(Keypoint(i, nm_canon, infer_side(nm_canon)))

        kps_norm.sort(key=lambda x: x.id)
        for i, kp in enumerate(kps_norm):
            kp.id = i
        names = [kp.name for kp in kps_norm]
        if len(names) != len(set(names)):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate keypoint names after normalization: {sorted(set(dupes))}")

        name2id = {kp.name: kp.id for kp in kps_norm}

        links_raw = d.get("links") or []
        links_name_pairs: list[tuple[str, str]] = []
        for pair in links_raw:
            if not isinstance(pair, list | tuple) or len(pair) != 2:
                raise ValueError(f"Invalid link pair: {pair!r}")
            a, b = pair[0], pair[1]
            if isinstance(a, int) and isinstance(b, int):
                a_name = kps_norm[a].name
                b_name = kps_norm[b].name
            else:
                a_name = to_snake(str(a)) if normalize_names else str(a)
                b_name = to_snake(str(b)) if normalize_names else str(b)
            if a_name == b_name:
                raise ValueError(f"Self-link not allowed: {a_name}")
            links_name_pairs.append((a_name, b_name))

        seen_undirected: set[frozenset[int]] = set()
        links_ids: list[tuple[int, int]] = []
        for a_name, b_name in links_name_pairs:
            if a_name not in name2id or b_name not in name2id:
                raise ValueError(f"Unknown keypoint in link: [{a_name}, {b_name}]")
            a_id, b_id = name2id[a_name], name2id[b_name]
            if a_id == b_id:
                raise ValueError(f"Self-link not allowed: {a_name}")
            und = frozenset({a_id, b_id})
            if und in seen_undirected:
                continue
            seen_undirected.add(und)
            links_ids.append((a_id, b_id))

        d.pop("symmetry_pairs", None)
        d.pop("symmetries", None)

        extras: dict[str, Any] = {}
        for key in (
            "extras",
            "groups",
            "colors",
            "render",
            "link_overrides",
            "link_sets",
        ):
            if key in d:
                extras[key] = d[key]
        analytics = d.get("analytics") or {}
        units = d.get("units")
        coordinate_system = d.get("coordinate_system")
        calibration = d.get("calibration")
        metadata = d.get("metadata")
        license_ = d.get("license")
        validation = d.get("validation")
        constraints = d.get("constraints")
        notes = d.get("notes")
        preview_image = d.get("preview_image")

        skel = cls(
            name=d.get("name", "unnamed"),
            keypoints=kps_norm,
            links_ids=links_ids,
            schema_version=sv,
            description=d.get("description"),
            extras=extras,
            analytics=analytics,
            units=units,
            coordinate_system=coordinate_system,
            calibration=calibration,
            metadata=metadata,
            license=license_,
            validation=validation,
            constraints=constraints if isinstance(constraints, dict) else None,
            notes=str(notes) if isinstance(notes, str) else None,
            preview_image=(str(preview_image) if isinstance(preview_image, str) else None),
        )
        skel.validate(strict=True)
        return skel

    @property
    def n_keypoints(self) -> int:
        """Return the total number of keypoints for this skeleton.

        Returns:
            The number of keypoints.
        """
        return len(self.keypoints)

    @property
    def keypoint_names(self) -> list[str]:
        """Return the list of normalized keypoint names in order.

        Returns:
            A list of keypoint names.
        """
        return [kp.name for kp in self.keypoints]

    def __len__(self) -> int:
        return len(self.keypoints)

    def name_to_id(self) -> dict[str, int]:
        """Map keypoint names to their integer IDs.

        Returns:
            A dictionary mapping names to IDs.
        """
        return {kp.name: kp.id for kp in self.keypoints}

    def id_to_name(self) -> dict[int, str]:
        """Map keypoint IDs back to their normalized names.

        Returns:
            A dictionary mapping IDs to names.
        """
        return {kp.id: kp.name for kp in self.keypoints}

    def matches(self, other: Skeleton) -> bool:
        """Lightweight structural equivalence check.

        Two skeletons match if they have the same name, the same ordered
        keypoint names, and the same set of undirected links.

        Args:
            other: The other skeleton to compare against.

        Returns:
            True if the skeletons match, False otherwise.
        """
        if str(self.name) != str(other.name):
            return False
        if self.keypoint_names != other.keypoint_names:
            return False
        return sorted(self.links_ids) == sorted(other.links_ids)

    def content_hash(self) -> str:
        """Return a stable content-addressable hash of the skeleton structure.

        The hash covers:
        - Skeleton name
        - Ordered keypoint names
        - Sorted undirected links (as id pairs)

        Use this to detect skeleton schema mismatches between training and inference.
        Two skeletons with the same content_hash are structurally equivalent.

        Returns:
            A 16-character hex string hash.
        """
        h = hashlib.sha256()
        h.update(self.name.encode("utf-8"))
        for kp in self.keypoints:
            h.update(kp.name.encode("utf-8"))
        for a, b in sorted(self.links_ids):
            h.update(f"{a},{b}".encode())
        return h.hexdigest()[:16]

    def __contains__(self, item: object) -> bool:
        """Membership check for keypoints by name, index, or Keypoint.

        Supports patterns used by higher-level code like:
            if "nose" in skeleton
            if kp in skeleton
            if 3 in skeleton
        """
        if isinstance(item, str):
            return any(kp.name == item for kp in self.keypoints)
        if isinstance(item, Keypoint):
            return any(
                (item is kp) or (kp.id == item.id) or (kp.name == item.name)
                for kp in self.keypoints
            )
        if isinstance(item, int):
            return 0 <= int(item) < len(self.keypoints)
        return False

    def compute_hash(self) -> str:
        """Return a stable SHA256 hash of keypoint names and links.

        This hash is used in model metadata to guard against skeleton mismatch,
        so keep it stable across releases.

        Returns:
            A SHA256 hex string hash.
        """

        h = hashlib.sha256()

        for kp in self.keypoints:
            h.update(kp.name.encode("utf-8"))

        for a, b in sorted(self.links_ids):
            h.update(f"{a}-{b}".encode())
        return h.hexdigest()

    def get_keypoint_by_name(self, name: str) -> Keypoint | None:
        """Return keypoint metadata by its normalized name.

        Args:
            name: The name of the keypoint to retrieve.

        Returns:
            The Keypoint instance if found, else None.
        """
        for kp in self.keypoints:
            if kp.name == name:
                return kp
        alt = to_snake(str(name))
        for kp in self.keypoints:
            if kp.name == alt:
                return kp
        return None

    def get_keypoint_by_id(self, kp_id: int) -> Keypoint | None:
        """Return keypoint metadata for the given integer ID.

        Args:
            kp_id: The integer ID of the keypoint.

        Returns:
            The Keypoint instance if found, else None.
        """
        if 0 <= kp_id < len(self.keypoints):
            return self.keypoints[kp_id]
        return None

    def keypoint_to_index(self, kp: Keypoint | str | int) -> int:
        """Normalize a keypoint reference (object, name, or index) into its index.

        Args:
            kp: A Keypoint object, name string, or integer index.

        Returns:
            The integer index of the keypoint.

        Raises:
            KeyError: If the keypoint reference is unknown.
        """
        if isinstance(kp, int):
            if 0 <= kp < len(self.keypoints):
                return kp
            raise KeyError(f"Unknown keypoint index: {kp}")
        if isinstance(kp, Keypoint):
            return kp.id
        n2i = self.name_to_id()
        name = str(kp)
        if name in n2i:
            return n2i[name]
        alt = to_snake(name)
        if alt in n2i:
            return n2i[alt]
        raise KeyError(f"Unknown keypoint: {kp}")

    def has_keypoint(self, name: str) -> bool:
        """Return True if the given keypoint name exists in the skeleton.

        Args:
            name: The name of the keypoint to check.

        Returns:
            True if the keypoint exists, False otherwise.
        """
        n2i = self.name_to_id()
        n = str(name)
        return n in n2i or to_snake(n) in n2i

    def lr_partner_map(self) -> dict[int, int | None]:
        """Derived left-right partner map by id using side + *_left/_right naming.

        Robust to side label aliases (e.g., 'l'/'r').

        Returns:
            A dictionary mapping keypoint IDs to their symmetric partner IDs.
        """
        buckets: dict[tuple[str | None, str], dict[str, int]] = {}
        for kp in self.keypoints:
            base, side = base_lr_name(kp.name)
            side_norm = SIDE_TOKENS.get((kp.side or "").lower(), None)
            if side_norm not in ("left", "right"):
                side_norm = SIDE_TOKENS.get((side or "").lower(), None)
            if side_norm not in ("left", "right"):
                side_norm = "unknown"
            ent = kp.entity
            buckets.setdefault((ent, base), {})
            buckets[(ent, base)][side_norm] = kp.id

        lr: dict[int, int | None] = {}
        for kp in self.keypoints:
            if kp.mirror_partner:
                lr[kp.id] = self.name_to_id().get(kp.mirror_partner)
                continue
            sid = SIDE_TOKENS.get((kp.side or "").lower(), kp.side)
            if sid not in ("left", "right"):
                _, inferred = base_lr_name(kp.name)
                sid = SIDE_TOKENS.get((inferred or "").lower(), inferred or sid)
            if sid not in ("left", "right"):
                lr[kp.id] = None
                continue
            base, _ = base_lr_name(kp.name)
            partner = buckets.get((kp.entity, base), {}).get("right" if sid == "left" else "left")
            lr[kp.id] = partner
        return lr

    def links_by_names(self) -> list[tuple[str, str]]:
        """Return the list of links expressed as (name, name) pairs.

        Returns:
            A list of tuples containing keypoint name pairs.
        """
        id2n = self.id_to_name()
        return [(id2n[a], id2n[b]) for (a, b) in self.links_ids]

    def add_link_by_name(self, src: str, dst: str) -> tuple[int, int]:
        """Add a link between the two named keypoints (if not present).

        Args:
            src: The name of the source keypoint.
            dst: The name of the destination keypoint.

        Returns:
            A tuple of the integer IDs of the linked keypoints.

        Raises:
            KeyError: If either keypoint name is unknown.
            ValueError: If a self-link is attempted.
        """
        n2i = self.name_to_id()
        if src not in n2i or dst not in n2i:
            raise KeyError(f"Unknown keypoints: '{src}', '{dst}'")
        a, b = n2i[src], n2i[dst]
        if a == b:
            raise ValueError("Self-link not allowed")
        lo, hi = (a, b) if a < b else (b, a)
        if (lo, hi) not in self.links_ids:
            self.links_ids.append((lo, hi))
            self.links_ids.sort()
        return (lo, hi)

    def remove_link_by_name(self, src: str, dst: str) -> None:
        """Remove a link between `src` and `dst`, if it exists.

        Args:
            src: The name of the source keypoint.
            dst: The name of the destination keypoint.
        """
        n2i = self.name_to_id()
        if src not in n2i or dst not in n2i:
            return
        a, b = n2i[src], n2i[dst]
        lo, hi = (a, b) if a < b else (b, a)
        self.links_ids = [(x, y) for (x, y) in self.links_ids if not (x == lo and y == hi)]

    def get_symmetry_name(self, keypoint_name: str) -> str:
        """Return symmetric partner name using overrides or derived mapping; empty if none.

        Args:
            keypoint_name: The name of the keypoint.

        Returns:
            The name of the symmetric partner keypoint, or an empty string if none.
        """
        kp = self.get_keypoint_by_name(keypoint_name)
        if kp is None:
            return ""
        if kp.mirror_partner and self.get_keypoint_by_name(kp.mirror_partner):
            return kp.mirror_partner
        lr = self.lr_partner_map()
        partner_id = lr.get(kp.id)
        if partner_id is None:
            return ""
        return self.id_to_name().get(partner_id, "")

    def add_symmetry_by_name(self, left: str, right: str) -> None:
        """Set explicit mirror_partner overrides for a pair of names (both directions).

        Args:
            left: The name of the left keypoint.
            right: The name of the right keypoint.

        Raises:
            KeyError: If either keypoint name is unknown.
        """
        a = self.get_keypoint_by_name(left)
        b = self.get_keypoint_by_name(right)
        if a is None or b is None:
            raise KeyError(f"Unknown keypoints: '{left}', '{right}'")
        a.mirror_partner = b.name
        b.mirror_partner = a.name

    def clear_symmetry_by_name(self, name: str, partner: str) -> None:
        """Clear explicit overrides for a given symmetric pair if set.

        Args:
            name: The name of the first keypoint.
            partner: The name of the second keypoint.
        """
        a = self.get_keypoint_by_name(name)
        b = self.get_keypoint_by_name(partner)
        if a and a.mirror_partner == partner:
            a.mirror_partner = None
        if b and b.mirror_partner == name:
            b.mirror_partner = None

    def validate(self, *, strict: bool = True) -> None:
        """Validate skeleton consistency; strict mode enforces contiguous IDs.

        Args:
            strict: Whether to enforce strict validation rules.

        Raises:
            ValueError: If the skeleton is inconsistent.
        """
        ids = [kp.id for kp in self.keypoints]
        if ids != list(range(len(self.keypoints))):
            raise ValueError("Keypoint ids must be contiguous 0..K-1.")
        names = [kp.name for kp in self.keypoints]
        if len(names) != len(set(names)):
            raise ValueError("Keypoint names must be unique.")
        keypoint_count = len(self.keypoints)
        for a, b in self.links_ids:
            if not (0 <= a < keypoint_count and 0 <= b < keypoint_count):
                raise ValueError(f"Link id out of range: {(a, b)}")
            if a == b:
                raise ValueError(f"Self-link not allowed: {(a, b)}")
        for kp in self.keypoints:
            s = kp.side
            if s and s not in SIDE_TOKENS.values():
                if strict:
                    raise ValueError(f"Invalid side '{s}' for {kp.name}")
                else:
                    warnings.warn(f"Invalid side '{s}' for {kp.name}", stacklevel=2)

    def degree(self, kp: str | int | Keypoint) -> int:
        """Return the degree (number of connected links) of a keypoint.

        Args:
            kp: A Keypoint object, name string, or integer index.

        Returns:
            The number of links connected to the keypoint.
        """
        idx = self.keypoint_to_index(kp)
        count = 0
        for a, b in self.links_ids:
            if a == idx or b == idx:
                count += 1
        return count

    def is_connected(self) -> bool:
        """Return True if the skeleton graph is connected.

        Returns:
            True if the graph is connected, False otherwise.
        """
        if not self.keypoints:
            return True
        if not self.links_ids and len(self.keypoints) > 1:
            return False

        adj: dict[int, list[int]] = {i: [] for i in range(len(self.keypoints))}
        for a, b in self.links_ids:
            adj[a].append(b)
            adj[b].append(a)

        visited = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(adj[node])

        return len(visited) == len(self.keypoints)

    def to_dict(self, *, keep_names: bool = True) -> dict[str, Any]:
        """Serialize to dict; writes 'links' by names (or ids when keep_names=False).

        Args:
            keep_names: Whether to use keypoint names for links instead of IDs.

        Returns:
            A dictionary representation of the skeleton.
        """
        out: dict[str, Any] = {}
        out["schema_version"] = self.schema_version
        out["name"] = self.name
        if self.description:
            out["description"] = self.description
        if self.license:
            out["license"] = self.license
        if self.metadata:
            out["metadata"] = self.metadata
        if self.units:
            out["units"] = self.units
        if self.coordinate_system:
            out["coordinate_system"] = self.coordinate_system
        if self.calibration:
            out["calibration"] = self.calibration
        if self.preview_image:
            out["preview_image"] = self.preview_image
        out["keypoints"] = [
            {
                "id": kp.id,
                "name": kp.name,
                **({"side": kp.side} if kp.side else {}),
                **({"role": kp.role} if kp.role else {}),
                **({"entity": kp.entity} if kp.entity else {}),
                **({"mirror_partner": kp.mirror_partner} if kp.mirror_partner else {}),
            }
            for kp in self.keypoints
        ]
        if keep_names:
            out["links"] = [
                [self.keypoints[a].name, self.keypoints[b].name] for (a, b) in self.links_ids
            ]
        else:
            out["links"] = [[a, b] for (a, b) in self.links_ids]
        out.update(self.extras or {})
        if self.analytics:
            out["analytics"] = self.analytics
        if self.validation:
            out["validation"] = self.validation
        if self.constraints:
            out["constraints"] = self.constraints
        if self.notes:
            out["notes"] = self.notes
        out.pop("symmetry_pairs", None)
        out.pop("symmetries", None)
        return out

    def to_keypoint_config(self) -> dict[str, Any]:
        """Build keypoint config dict for training/inference.

        Returns essential skeleton information in a format suitable for
        the Siesta training stack and other frameworks.

        Returns:
            A dictionary containing keypoint names, count, links, and skeleton name.
        """
        return {
            "keypoint_names": self.keypoint_names,
            "num_keypoints": self.n_keypoints,
            "links": list(self.links_by_names()),
            "skeleton_name": self.name,
        }

    def to_minimal_dict(self) -> dict[str, Any]:
        """Return compact, human-diffable dict. Links by names.

        Returns:
            A minimal dictionary representation of the skeleton.
        """
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "keypoints": [kp.name for kp in self.keypoints],
            "links": [
                [self.keypoints[a].name, self.keypoints[b].name] for (a, b) in self.links_ids
            ],
            "description": self.description,
            "extras": self.extras or {},
        }

    def dump(self, path: Path, *, fmt: str | None = None, keep_names: bool = True) -> None:
        """Save to JSON.

        Args:
            path: The file path to save to.
            fmt: The format to use (currently only 'json' is supported).
            keep_names: Whether to use keypoint names for links.

        Raises:
            ValueError: If an unknown format is specified.
        """
        from xpkg.io.skeleton_io import dump_skeleton

        dump_skeleton(self, path, fmt=fmt, keep_names=keep_names)

    @classmethod
    def load(cls, src: Path, **kwargs) -> Skeleton:
        """Load a skeleton from JSON file.

        Args:
            src: The path to the JSON file.
            **kwargs: Additional arguments passed to `from_dict`.

        Returns:
            A Skeleton instance.

        Raises:
            ValueError: If the file is not a JSON file.
        """
        from xpkg.io.skeleton_io import load_skeleton

        return load_skeleton(src, **kwargs)

    @classmethod
    def load_builtin(cls, name: str, **kwargs) -> Skeleton:
        """Load a built-in skeleton by name.

        Args:
            name: The name of the built-in skeleton.
            **kwargs: Additional arguments passed to `from_dict`.

        Returns:
            A Skeleton instance.
        """
        from xpkg.io.skeleton_io import load_builtin_skeleton

        return load_builtin_skeleton(name, **kwargs)

    @classmethod
    def load_any(
        cls,
        path: str | Path,
        *,
        format: str | None = None,
        **kwargs,
    ) -> Skeleton:
        """Load skeleton from any supported external format.

        Auto-detects format by file extension. Supports:
        - .json: Siesta JSON format
        - .pkg.slp: SLEAP package files
        - .yaml/.yml: DLC, SLEAP, or Ultralytics format (auto-detected)

        Args:
            path: Path to skeleton file.
            format: Optional format override ('dlc', 'sleap', 'ultralytics').
            **kwargs: Additional arguments (reserved for future use).

        Returns:
            Skeleton: Loaded and normalized skeleton.

        Raises:
            ValueError: If file format is unsupported or parsing fails.
            ImportError: If optional dependency is required but missing.
        """
        from xpkg.io.skeleton_io import load_any_skeleton

        return load_any_skeleton(path, format=format, **kwargs)

    @classmethod
    def from_json(cls, text: str, **kwargs) -> Skeleton:
        """Load a skeleton from JSON text.

        Args:
            text: The JSON string.
            **kwargs: Additional arguments passed to `from_dict`.

        Returns:
            A Skeleton instance.
        """
        return cls.from_dict(parse_json_dict(text), **kwargs)

    def resolve_pairs(self, pairs: Sequence[Sequence[Any]]) -> list[tuple[int, int]]:
        """Resolve keypoint names/ids into integer index pairs.

        Args:
            pairs: A sequence of pairs (names or IDs).

        Returns:
            A list of integer ID pairs.
        """
        n2i = self.name_to_id()
        out: list[tuple[int, int]] = []
        for a, b in pairs:
            ai = int(a) if isinstance(a, int) else n2i[str(a)]
            bi = int(b) if isinstance(b, int) else n2i[str(b)]
            out.append((ai, bi))
        return out

    def resolve_triples(self, triples: Sequence[Sequence[Any]]) -> list[tuple[int, int, int]]:
        """Resolve keypoint triples into integer combinations for analytics.

        Args:
            triples: A sequence of triples (names or IDs).

        Returns:
            A list of integer ID triples.
        """
        n2i = self.name_to_id()
        out: list[tuple[int, int, int]] = []
        for a, b, c in triples:
            ai = int(a) if isinstance(a, int) else n2i[str(a)]
            bi = int(b) if isinstance(b, int) else n2i[str(b)]
            ci = int(c) if isinstance(c, int) else n2i[str(c)]
            out.append((ai, bi, ci))
        return out

    def rename_keypoint(self, old_name: str, new_name: str) -> None:
        """Rename a keypoint; preserves order/ids.

        Also enforces uniqueness of names. Links are index-based and do not need updates.

        Args:
            old_name: The current name of the keypoint.
            new_name: The new name for the keypoint.

        Raises:
            ValueError: If the new name is invalid or already exists.
            KeyError: If the old name is not found.
        """
        if not new_name or not isinstance(new_name, str):
            raise ValueError("new_name must be a non-empty string")
        new_name_snake = to_snake(new_name)
        if any(kp.name == new_name_snake for kp in self.keypoints):
            if old_name != new_name_snake:
                raise ValueError(f"Keypoint with name '{new_name_snake}' already exists")
        kp = self.get_keypoint_by_name(old_name)
        if kp is None:
            raise KeyError(f"Keypoint '{old_name}' not found")
        kp.name = new_name_snake

    def add_keypoint(
        self,
        name: str,
        *,
        side: str = "unknown",
        role: str | None = None,
        position: int | None = None,
    ) -> Keypoint:
        """Add a new keypoint to the skeleton.

        If ``position`` is provided, insert at that index and renumber existing ids;
        otherwise append at the end. Updates internal link id pairs to remain valid.

        Args:
            name: The name of the new keypoint.
            side: The side of the keypoint.
            role: The role of the keypoint.
            position: The optional index to insert at.

        Returns:
            The newly created Keypoint instance.

        Raises:
            ValueError: If a keypoint with the same name already exists.
        """
        nm = to_snake(str(name))
        if any(kp.name == nm for kp in self.keypoints):
            raise ValueError(f"Keypoint with name '{nm}' already exists")

        if position is None:
            kid = len(self.keypoints)
            kp = Keypoint(
                id=kid,
                name=nm,
                side=str(side or "unknown"),
                role=role or None,
            )
            self.keypoints.append(kp)
            return kp

        pos = max(0, min(int(position), len(self.keypoints)))
        for k in self.keypoints:
            if k.id >= pos:
                k.id += 1
        new_links: list[tuple[int, int]] = []
        for a, b in self.links_ids:
            na = a + 1 if a >= pos else a
            nb = b + 1 if b >= pos else b
            lo, hi = (na, nb) if na < nb else (nb, na)
            if lo != hi and (lo, hi) not in new_links:
                new_links.append((lo, hi))
        self.links_ids = sorted(new_links)
        kp = Keypoint(
            id=pos,
            name=nm,
            side=str(side or "unknown"),
            role=role or None,
        )
        self.keypoints.insert(pos, kp)
        for i, k in enumerate(self.keypoints):
            k.id = i
        return kp

    def remove_keypoint(self, kp_or_name: Keypoint | str) -> None:
        """Remove a keypoint by object or name; updates ids and links.

        Args:
            kp_or_name: The Keypoint instance or name string to remove.

        Raises:
            KeyError: If the keypoint is not found.
        """
        if isinstance(kp_or_name, Keypoint):
            target_name = kp_or_name.name
        else:
            target_name = to_snake(str(kp_or_name))
        idx = next((i for i, k in enumerate(self.keypoints) if k.name == target_name), None)
        if idx is None:
            raise KeyError(f"Keypoint '{target_name}' not found")
        del self.keypoints[idx]
        for i, k in enumerate(self.keypoints):
            k.id = i
        new_links: list[tuple[int, int]] = []
        for a, b in self.links_ids:
            if a == idx or b == idx:
                continue
            na = a - 1 if a > idx else a
            nb = b - 1 if b > idx else b
            lo, hi = (na, nb) if na < nb else (nb, na)
            if lo != hi and (lo, hi) not in new_links:
                new_links.append((lo, hi))
        self.links_ids = sorted(new_links)


def build_keypoint_skeleton(keypoint_names: list[str], *, name: str = "imported") -> Skeleton:
    """Build a skeleton from keypoint names with no links."""
    keypoints = [Keypoint(id=i, name=kp) for i, kp in enumerate(keypoint_names)]
    return Skeleton(name=name, keypoints=keypoints, links_ids=[])


SIESTA_SKELETON_NAME = "mouse_bottom_up_v2"
_SIESTA_SKELETON = Skeleton.from_dict(get_skeleton_def(SIESTA_SKELETON_NAME))

KEYPOINT_NAMES: list[str] = list(_SIESTA_SKELETON.keypoint_names)
SKELETON_CONNECTIONS: list[tuple[int, int]] = list(_SIESTA_SKELETON.links_ids)
