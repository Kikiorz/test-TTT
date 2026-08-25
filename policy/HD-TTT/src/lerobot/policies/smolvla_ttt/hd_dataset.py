"""Loading and attaching offline HD-TTT supervision.

Hindsight supervision is generated offline because the causal teacher and the
counterfactual histories are not available in deployment.  This module keeps
that artifact separate from the LeRobot dataset itself and exposes a small
``Dataset`` wrapper which merges the labels into each sample under ``hd_*``
keys.

The canonical on-disk representation is a columnar mapping saved as ``.pt``,
``.npz`` or ``.json``.  Every label column has one row per dataset frame (the
first dimension), for example::

    {
        "hd_teacher_velocity": Tensor[N, chunk, action_dim],
        "hd_teacher_true_velocity": Tensor[N, chunk, action_dim],
        "hd_teacher_wrong_velocity": Tensor[N, chunk, action_dim],
        "hd_rho": Tensor[N],
        "hd_write_gate": Tensor[N],
        "hd_counterfactual_write_gate": Tensor[N],
    }

For convenience, records with ``episode_index``/``frame_index`` (or a global
``index``) are accepted as well.  A top-level episode mapping is also
supported, which is useful when labels are generated one episode at a time.
Labels are deliberately kept out of ``LeRobotDataset.meta``: they are training
annotations, not physical observations or action features.

For bounded recurrent training, a payload may instead contain a ``windows``
list.  Each record stores the complete context for one target window, keyed by
its source ``target_global_index``.  ``TailPreservingSequenceDataset`` calls
``get_window_labels`` to select that context; this preserves distinct labels
for overlapping windows and is required for counterfactual gate replay.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


# Raw names emitted by small standalone hindsight-generation scripts.  The
# canonical names are what the SmolVLA-TTT model consumes.
_LABEL_ALIASES = {
    "teacher_velocity": "hd_teacher_velocity",
    "teacher_true_velocity": "hd_teacher_true_velocity",
    "teacher_wrong_velocity": "hd_teacher_wrong_velocity",
    "attribution": "hd_attribution",
    "hd_C": "hd_attribution",
    "rho": "hd_rho",
    "u": "hd_write_gate",
    "write_gate": "hd_write_gate",
    "hd_u": "hd_write_gate",
    "hd_counterfactual_gate": "hd_counterfactual_write_gate",
    "counterfactual_write_gate": "hd_counterfactual_write_gate",
    "local_key": "hd_local_key",
    "local_value": "hd_local_value",
    "local_prediction": "hd_local_prediction",
    "local_query": "hd_local_query",
    # The full pair matrix is normally reduced to a future-column score before
    # training.  Keeping this alias makes a columnar ``C`` artifact usable when
    # it already contains one score per frame.
    "C": "hd_attribution",
}

_INDEX_KEYS = ("dataset_index", "global_index", "index")
_EPISODE_KEYS = ("episode_index", "episode")
_FRAME_KEYS = ("frame_index", "frame")
_CONTAINER_KEYS = ("labels", "records", "samples", "data")


def _canonical_label_key(key: Any) -> str | None:
    """Return the canonical ``hd_*`` name, or ``None`` for metadata."""

    if not isinstance(key, str):
        return None
    if key in _LABEL_ALIASES:
        return _LABEL_ALIASES[key]
    if key.startswith("hd_"):
        return key
    return _LABEL_ALIASES.get(key)


def _as_cpu_tensor(value: Any) -> Tensor:
    """Convert a numeric label value to a detached CPU tensor."""

    if isinstance(value, Tensor):
        return value.detach().cpu()
    try:
        return torch.as_tensor(value)
    except Exception as exc:  # pragma: no cover - helpful error path
        raise TypeError(f"HD labels must be numeric/tensor values, got {type(value).__name__}") from exc


def _scalar_int(value: Any, *, name: str) -> int:
    tensor = _as_cpu_tensor(value)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be scalar, got shape {tuple(tensor.shape)}")
    return int(tensor.reshape(()).item())


def _mapping_value(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _read_label_file(path: str | Path) -> Any:
    """Read a supported label artifact without touching the source dataset."""

    path = Path(path).expanduser()
    if path.is_dir():
        candidates = [path / name for name in ("labels.pt", "labels.pth", "labels.npz", "labels.json")]
        existing = [candidate for candidate in candidates if candidate.is_file()]
        if len(existing) != 1:
            raise FileNotFoundError(
                f"HD label directory {path} must contain exactly one of labels.pt/.pth/.npz/.json; "
                f"found {[candidate.name for candidate in existing]}"
            )
        path = existing[0]
    if not path.is_file():
        raise FileNotFoundError(f"HD label file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        # HD artifacts are expected to be tensor-only dictionaries, so the
        # safe weights-only loader is sufficient and avoids executing pickle
        # payloads supplied accidentally from an unrelated checkpoint.
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # torch < 2.0 has no weights_only argument
            return torch.load(path, map_location="cpu")
    if suffix == ".npz":
        import numpy as np

        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    raise ValueError(f"Unsupported HD label format {suffix!r}; use .pt, .pth, .npz, or .json")


class HindsightLabelDataset(Dataset):
    """Merge an offline HD-TTT label artifact into a frame dataset.

    Args:
        dataset: A map-style LeRobot dataset.  Its episode metadata is used to
            resolve records carrying ``episode_index`` and ``frame_index``.
        label_path: Path to a ``.pt``, ``.npz`` or ``.json`` artifact, or a
            directory containing one of ``labels.*``.
        strict: Require every frame to have every label column.  ``True`` is
            the safe training default; ``False`` is useful for inspecting a
            sparse artifact and fills missing rows with zeros.

    The wrapper forwards ``meta``, ``episodes`` and other attributes to the
    underlying dataset, so it can be passed to the existing LeRobot training
    and sequence samplers transparently.
    """

    def __init__(self, dataset: Dataset, label_path: str | Path, *, strict: bool = True) -> None:
        self.dataset = dataset
        self.label_path = Path(label_path).expanduser()
        self.strict = bool(strict)
        self.label_metadata: dict[str, Any] = {}
        self.hd_window_local = False
        self.hd_window_keyed = False
        self._episode_locations = self._build_episode_locations(dataset)
        self._absolute_frame_indices = self._build_absolute_frame_indices(dataset)
        self._source_to_local = (
            {source: local for local, source in enumerate(self._absolute_frame_indices)}
            if self._absolute_frame_indices is not None
            else {}
        )
        self._records: dict[int, dict[str, Tensor]] = {}
        # Window-local artifacts keep one complete replay context per target
        # window.  A frame can belong to several overlapping windows, so a
        # single frame record is not expressive enough for counterfactual
        # gates/noise/state labels.
        self._window_records: dict[int, dict[str, Any]] = {}
        self._label_keys: tuple[str, ...] = ()
        self._templates: dict[str, Tensor] = {}

        payload = _read_label_file(self.label_path)
        if isinstance(payload, Mapping) and isinstance(payload.get("metadata"), Mapping):
            self.label_metadata = dict(payload["metadata"])
            self.hd_window_local = bool(self.label_metadata.get("window_local", False))
        if isinstance(payload, Mapping) and "windows" in payload:
            self._ingest_window_records(payload["windows"])
            # A window artifact may also carry ordinary frame columns for
            # inspection/backward compatibility.  Do not send the structural
            # ``windows`` list through the columnar parser.
            ordinary_payload = {key: value for key, value in payload.items() if key != "windows"}
            if any(_canonical_label_key(key) is not None for key in ordinary_payload):
                self._ingest_payload(ordinary_payload)
        else:
            self._ingest_payload(payload)
        if not self._label_keys:
            raise ValueError(f"HD label artifact {self.label_path} contains no hd_* label columns")
        self.hd_window_keyed = bool(self._window_records)

        missing = sorted(set(range(len(dataset))) - set(self._records))
        if missing and self.strict and not self._window_records:
            preview = missing[:8]
            raise ValueError(
                f"HD labels at {self.label_path} do not cover {len(missing)} of {len(dataset)} frames; "
                f"first missing dataset indices: {preview}. Use aligned labels or strict=False."
            )
        if self.strict and not self._window_records:
            incomplete = [
                index
                for index, record in self._records.items()
                if any(key not in record for key in self._label_keys)
            ]
            if incomplete:
                preview = incomplete[:8]
                raise ValueError(
                    f"HD labels at {self.label_path} have missing columns on {len(incomplete)} frames; "
                    f"first incomplete dataset indices: {preview}. Use aligned labels or strict=False."
                )

    # ------------------------------------------------------------------
    # Dataset delegation and public metadata
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        record = self._records.get(int(index))
        if record is None:
            if self.strict and not self._window_records:  # checked eagerly, retained for defensive use
                raise IndexError(f"No HD labels for dataset index {index}")
            record = {}
            if not self._window_records:
                record = {key: torch.zeros_like(template) for key, template in self._templates.items()}
        # Window-keyed artifacts are attached by TailPreservingSequenceDataset
        # after it knows the target/history context.  A direct frame lookup has
        # no unambiguous record, so return the physical sample unchanged here.
        if self._window_records and not record:
            return sample
        for key in self._label_keys:
            value = record.get(key)
            if value is None:
                if self.strict:
                    raise IndexError(f"HD label column {key!r} missing at dataset index {index}")
                value = torch.zeros_like(self._templates[key])
            sample[key] = value
        return sample

    def __getattr__(self, name: str) -> Any:
        # ``Dataset`` has no useful __getattr__; this forwards LeRobot fields
        # such as meta, episodes, num_frames and repo_id without copying them.
        # Use ``object.__getattribute__`` here rather than ``self.dataset`` so
        # that partially-constructed instances still raise a normal
        # ``AttributeError`` instead of recursively calling ``__getattr__``.
        if name in {
            "dataset",
            "_records",
            "_window_records",
            "_label_keys",
            "_templates",
            "_source_to_local",
            "_absolute_frame_indices",
        }:
            raise AttributeError(name)
        try:
            dataset = object.__getattribute__(self, "dataset")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(dataset, name)

    @property
    def label_keys(self) -> tuple[str, ...]:
        return self._label_keys

    def get_window_labels(
        self,
        target_start: int,
        history_start: int,
        window_length: int,
    ) -> dict[int, dict[str, Tensor]] | None:
        """Return labels for the exact sequence window requested by the sampler.

        ``target_start``/``history_start`` are indices in the selected (and
        possibly episode-subset) dataset view.  Artifacts are keyed by source
        ``global_index`` so they remain valid when LeRobot reindexes a subset.
        ``None`` means this is an ordinary frame-level artifact or the window
        was not generated by the collector.
        """

        if not self._window_records:
            return None
        if target_start < 0 or target_start >= len(self.dataset):
            raise IndexError(f"target_start {target_start} is outside the dataset")
        if self._absolute_frame_indices is None:
            source_target = int(target_start)
        else:
            source_target = int(self._absolute_frame_indices[target_start])
        window = self._window_records.get(source_target)
        if window is None:
            if self.strict:
                raise KeyError(
                    "No window-local HD labels for target source frame "
                    f"{source_target}; regenerate labels with the same sampler settings"
                )
            return None
        expected_length = int(window["length"])
        if expected_length != int(window_length):
            raise ValueError(
                "Window-local HD label length mismatch: artifact="
                f"{expected_length}, sampler={window_length}"
            )
        expected_history = int(window["history_start_source"])
        actual_history = (
            int(self._absolute_frame_indices[history_start])
            if self._absolute_frame_indices is not None
            else int(history_start)
        )
        if expected_history != actual_history:
            raise ValueError(
                "Window-local HD history mismatch: artifact="
                f"{expected_history}, sampler={actual_history}"
            )
        source_indices = window["source_indices"]
        labels = window["labels"]
        result: dict[int, dict[str, Tensor]] = {}
        for row, source_index_value in enumerate(source_indices):
            source_index = _scalar_int(source_index_value, name="window global_index")
            local_index = self._source_to_local.get(source_index, source_index)
            if not 0 <= local_index < len(self.dataset):
                raise ValueError(
                    f"Window-local label source frame {source_index} is not in the selected dataset"
                )
            result[local_index] = {
                key: _as_cpu_tensor(value)[row]
                for key, value in labels.items()
            }
        return result

    # ------------------------------------------------------------------
    # Episode/index resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _episode_table(dataset: Dataset) -> tuple[list[int], list[int], list[int]] | None:
        meta = getattr(dataset, "meta", None)
        episodes = getattr(meta, "episodes", None)
        if episodes is None:
            return None
        try:
            if isinstance(episodes, Mapping):
                starts = episodes["dataset_from_index"]
                ends = episodes["dataset_to_index"]
            else:
                # A few lightweight test/third-party datasets expose episode
                # metadata as a list of dictionaries rather than a HF Dataset.
                starts = [row["dataset_from_index"] for row in episodes]
                ends = [row["dataset_to_index"] for row in episodes]

            selected = getattr(dataset, "episodes", None)
            if selected is None:
                episode_ids = list(range(len(starts)))
            else:
                # Avoid ``selected or ...``: numpy arrays and torch tensors do
                # not have a scalar truth value.
                if isinstance(selected, Tensor):
                    selected = selected.detach().cpu().tolist()
                elif hasattr(selected, "tolist") and not isinstance(selected, (list, tuple, range)):
                    selected = selected.tolist()
                episode_ids = [int(ep) for ep in selected]
            starts = [int(_as_cpu_tensor(starts[int(ep)]).item()) for ep in episode_ids]
            ends = [int(_as_cpu_tensor(ends[int(ep)]).item()) for ep in episode_ids]
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        # A LeRobot dataset with an episode subset is re-indexed contiguously.
        lengths = [end - start for start, end in zip(starts, ends, strict=True)]
        return [int(ep) for ep in episode_ids], starts, lengths

    @classmethod
    def _build_episode_locations(cls, dataset: Dataset) -> dict[int, tuple[int, int]]:
        table = cls._episode_table(dataset)
        if table is None:
            return {}
        episode_ids, _absolute_starts, lengths = table
        locations: dict[int, tuple[int, int]] = {}
        local_start = 0
        for episode_id, length in zip(episode_ids, lengths, strict=True):
            if length <= 0:
                raise ValueError(f"Episode {episode_id} has non-positive length {length}")
            locations[episode_id] = (local_start, length)
            local_start += length
        if local_start != len(dataset):
            raise ValueError(
                f"Dataset episode metadata covers {local_start} frames, but dataset length is {len(dataset)}"
            )
        return locations

    @classmethod
    def _build_absolute_frame_indices(cls, dataset: Dataset) -> list[int] | None:
        """Return source-dataset indices for selected episode frames.

        ``LeRobotDataset(episodes=[...])`` reindexes its reader contiguously,
        while the metadata boundaries retain absolute frame offsets.  Keeping
        this map lets a full-dataset label artifact be reused for an episode
        subset without requiring a second copy of the labels.
        """

        table = cls._episode_table(dataset)
        if table is None:
            return None
        _episode_ids, starts, lengths = table
        return [
            start + offset
            for start, length in zip(starts, lengths, strict=True)
            for offset in range(length)
        ]

    def _resolve_index(self, mapping: Mapping[str, Any], fallback: int | None = None) -> int:
        raw_global_index = mapping.get("global_index")
        if raw_global_index is not None:
            source_index = _scalar_int(raw_global_index, name="global_index")
            if self._source_to_local and source_index not in self._source_to_local:
                raise ValueError(f"HD label global index {source_index} is not in the selected dataset episodes")
            index = self._source_to_local.get(source_index, source_index)
        else:
            raw_index = _mapping_value(mapping, ("dataset_index", "index"))
            if raw_index is not None:
                index = _scalar_int(raw_index, name="dataset_index")
                # For a selected-episode dataset, an out-of-range index can
                # still be a valid absolute source index from the artifact.
                if not 0 <= index < len(self.dataset) and index in self._source_to_local:
                    index = self._source_to_local[index]
            else:
                raw_episode = _mapping_value(mapping, _EPISODE_KEYS)
                raw_frame = _mapping_value(mapping, _FRAME_KEYS)
                if raw_episode is None or raw_frame is None:
                    if fallback is None:
                        raise ValueError("Each HD label record needs index or episode_index+frame_index")
                    index = int(fallback)
                else:
                    episode = _scalar_int(raw_episode, name="episode_index")
                    frame = _scalar_int(raw_frame, name="frame_index")
                    if episode not in self._episode_locations:
                        raise ValueError(f"HD label references unknown episode {episode}")
                    start, length = self._episode_locations[episode]
                    if frame < 0 or frame >= length:
                        raise ValueError(f"HD label frame {frame} is outside episode {episode} length {length}")
                    index = start + frame
        if index < 0 or index >= len(self.dataset):
            raise ValueError(f"HD label dataset index {index} is outside [0, {len(self.dataset)})")
        return index

    # ------------------------------------------------------------------
    # Payload parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_container(payload: Any) -> Any:
        if not isinstance(payload, Mapping):
            return payload
        # Prefer an explicit labels/records container.  Metadata next to it is
        # merged into the nested mapping/list before parsing; otherwise a
        # common ``{"labels": {...}, "episode_index": ...}`` artifact would
        # silently lose its index columns.
        for key in _CONTAINER_KEYS:
            if key in payload and isinstance(payload[key], (Mapping, list, tuple)):
                nested = payload[key]
                metadata: dict[Any, Any] = {}
                for raw_key, value in payload.items():
                    if raw_key == key:
                        continue
                    if raw_key == "metadata" and isinstance(value, Mapping):
                        metadata.update(value)
                    elif (
                        _canonical_label_key(raw_key) is None
                        and isinstance(raw_key, str)
                        and raw_key in _INDEX_KEYS + _EPISODE_KEYS + _FRAME_KEYS
                    ):
                        metadata[raw_key] = value
                if isinstance(nested, Mapping):
                    merged = dict(nested)
                    for raw_key, value in metadata.items():
                        merged.setdefault(raw_key, value)
                    return merged
                if metadata:
                    # Convert a record list plus columnar metadata into a
                    # regular list of records.  Scalar metadata is broadcast;
                    # vector metadata is indexed by record position.
                    records = list(nested)
                    merged_records: list[Any] = []
                    for row, record in enumerate(records):
                        if not isinstance(record, Mapping):
                            merged_records.append(record)
                            continue
                        merged_record = dict(record)
                        for raw_key, value in metadata.items():
                            merged_record.setdefault(raw_key, value)
                            if isinstance(value, (list, tuple)) and len(value) == len(records):
                                merged_record[raw_key] = value[row]
                            else:
                                try:
                                    value_tensor = _as_cpu_tensor(value)
                                except TypeError:
                                    value_tensor = None
                                if (
                                    value_tensor is not None
                                    and value_tensor.ndim > 0
                                    and value_tensor.shape[0] == len(records)
                                ):
                                    merged_record[raw_key] = value_tensor[row]
                        merged_records.append(merged_record)
                    return merged_records
                return nested
        return payload

    @staticmethod
    def _is_episode_mapping(mapping: Mapping[Any, Any]) -> bool:
        if not mapping:
            return False
        for key, value in mapping.items():
            try:
                int(key)
            except (TypeError, ValueError):
                return False
            if not isinstance(value, Mapping):
                return False
        return True

    @staticmethod
    def _label_columns(mapping: Mapping[Any, Any]) -> dict[str, Any]:
        columns: dict[str, Any] = {}
        for raw_key, value in mapping.items():
            key = _canonical_label_key(raw_key)
            if key is not None:
                # ``C[event, future]`` is often serialized directly from the
                # hindsight attribution object.  The online loss consumes one
                # future-time weight per frame, so collapse the event axis at
                # load time to obtain ``rho[future]``.  Keep already-reduced
                # vectors/matrices unchanged.
                if raw_key in {"C", "hd_C"}:
                    tensor = _as_cpu_tensor(value)
                    if tensor.ndim >= 2 and tensor.shape[-2] == tensor.shape[-1]:
                        value = tensor.sum(dim=-2)
                columns[key] = value
        return columns

    @staticmethod
    def _metadata_columns(mapping: Mapping[Any, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for raw_key, value in mapping.items():
            if isinstance(raw_key, str) and (
                raw_key in _INDEX_KEYS + _EPISODE_KEYS + _FRAME_KEYS
            ):
                metadata[raw_key] = value
        return metadata

    def _register(self, index: int, values: Mapping[str, Any]) -> None:
        record = self._records.setdefault(index, {})
        for key, value in values.items():
            tensor = _as_cpu_tensor(value)
            if key in record and tuple(record[key].shape) != tuple(tensor.shape):
                raise ValueError(
                    f"Conflicting HD label shapes for {key!r} at index {index}: "
                    f"{tuple(record[key].shape)} vs {tuple(tensor.shape)}"
                )
            record[key] = tensor
            if key not in self._templates:
                self._templates[key] = tensor
        self._label_keys = tuple(sorted(set(self._label_keys).union(values)))

    def _ingest_window_records(self, windows: Any) -> None:
        """Load a window-keyed artifact without collapsing it to frame rows."""

        if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes, bytearray)):
            raise ValueError("Window-local HD artifact 'windows' must be a record sequence")
        for position, raw_window in enumerate(windows):
            if not isinstance(raw_window, Mapping):
                raise ValueError(f"Window-local record {position} must be a mapping")
            raw_target = _mapping_value(
                raw_window,
                ("target_global_index", "target_index", "global_index"),
            )
            if raw_target is None:
                raise ValueError(
                    f"Window-local record {position} needs target_global_index"
                )
            target_source = _scalar_int(raw_target, name="target_global_index")
            if self._source_to_local and target_source not in self._source_to_local:
                # The artifact may intentionally contain a superset of the
                # selected episodes.  Ignore windows outside this view rather
                # than making a subset training run impossible.
                continue
            source_values = raw_window.get("source_indices", raw_window.get("global_indices"))
            if source_values is None:
                raise ValueError(f"Window-local record {position} needs source_indices")
            source_tensor = _as_cpu_tensor(source_values).reshape(-1)
            row_count = int(source_tensor.numel())
            if row_count <= 0:
                raise ValueError(f"Window-local record {position} has no context rows")
            raw_labels = raw_window.get("labels", raw_window)
            if not isinstance(raw_labels, Mapping):
                raise ValueError(f"Window-local record {position} labels must be a mapping")
            columns = self._label_columns(raw_labels)
            if not columns:
                raise ValueError(f"Window-local record {position} has no hd_* labels")
            labels: dict[str, Tensor] = {}
            for key, value in columns.items():
                tensor = _as_cpu_tensor(value)
                if tensor.ndim == 0:
                    tensor = tensor.expand(row_count)
                elif int(tensor.shape[0]) != row_count:
                    raise ValueError(
                        f"Window-local label {key!r} in record {position} has first dimension "
                        f"{tensor.shape[0]}, expected {row_count}"
                    )
                labels[key] = tensor
                if key not in self._templates:
                    self._templates[key] = tensor[0]
            if target_source not in set(int(value) for value in source_tensor.tolist()):
                raise ValueError(
                    f"Window-local target {target_source} is not present in its source_indices"
                )
            history_source = raw_window.get("history_start_source", source_tensor[0])
            history_source = _scalar_int(history_source, name="history_start_source")
            length = _scalar_int(raw_window.get("length", row_count), name="window length")
            if length != row_count:
                raise ValueError(
                    f"Window-local record {position} length={length} but has {row_count} rows"
                )
            if target_source in self._window_records:
                raise ValueError(
                    f"Duplicate window-local target source frame {target_source}"
                )
            self._window_records[target_source] = {
                "length": length,
                "history_start_source": history_source,
                "source_indices": source_tensor,
                "labels": labels,
            }
            self._label_keys = tuple(sorted(set(self._label_keys).union(labels)))

    @staticmethod
    def _row_value(value: Any, row: int, row_count: int) -> Any:
        tensor = _as_cpu_tensor(value)
        if tensor.ndim > 0 and tensor.shape[0] == row_count:
            return tensor[row]
        if tensor.ndim == 0 or row_count == 1:
            return tensor
        raise ValueError(
            f"HD label column has first dimension {tensor.shape[0] if tensor.ndim else 0}, "
            f"expected {row_count}"
        )

    def _ingest_records(self, records: Sequence[Any]) -> None:
        for position, raw_record in enumerate(records):
            if not isinstance(raw_record, Mapping):
                raise ValueError(f"HD label record {position} must be a mapping")
            labels = self._label_columns(raw_record)
            if not labels:
                continue
            index = self._resolve_index(raw_record, fallback=position)
            self._register(index, labels)

    def _ingest_episode_mapping(self, mapping: Mapping[Any, Any]) -> None:
        for raw_episode, raw_payload in mapping.items():
            episode = int(raw_episode)
            if episode not in self._episode_locations:
                raise ValueError(f"HD label references unknown episode {episode}")
            start, length = self._episode_locations[episode]
            if isinstance(raw_payload, Sequence) and not isinstance(raw_payload, (str, bytes, bytearray)):
                self._ingest_records(
                    [
                        {**record, "dataset_index": start + offset}
                        for offset, record in enumerate(raw_payload)
                        if isinstance(record, Mapping)
                    ]
                )
                continue
            if not isinstance(raw_payload, Mapping):
                raise ValueError(f"HD episode {episode} payload must be a mapping or record list")
            columns = self._label_columns(raw_payload)
            if not columns:
                continue
            for key, value in columns.items():
                tensor = _as_cpu_tensor(value)
                if tensor.ndim > 0 and tensor.shape[0] == length:
                    for frame in range(length):
                        self._register(start + frame, {key: tensor[frame]})
                else:
                    for frame in range(length):
                        self._register(start + frame, {key: tensor})

    def _ingest_columnar(self, mapping: Mapping[Any, Any]) -> None:
        columns = self._label_columns(mapping)
        if not columns:
            raise ValueError("No recognized hd_* columns in label mapping")
        metadata = self._metadata_columns(mapping)
        # Prefer the dataset/episode cardinality when it is unambiguous.  A
        # one-frame dataset can legitimately store one teacher vector with
        # shape ``[chunk, action_dim]``; treating ``chunk`` as the row count
        # would otherwise reject that artifact.
        if len(self.dataset) == 1:
            row_count = 1
        else:
            row_count = None
            expected_counts = {len(self.dataset), len(self._episode_locations)}
            for value in columns.values():
                tensor = _as_cpu_tensor(value)
                if tensor.ndim > 0 and int(tensor.shape[0]) in expected_counts:
                    row_count = int(tensor.shape[0])
                    break
            if row_count is None:
                for value in columns.values():
                    tensor = _as_cpu_tensor(value)
                    if tensor.ndim > 0:
                        row_count = int(tensor.shape[0])
                        break
            if row_count is None:
                row_count = 1

        # Explicit metadata is the most robust representation and supports
        # sparse labels or datasets with a selected episode subset.
        if metadata:
            for row in range(row_count):
                row_metadata = {
                    key: self._row_value(value, row, row_count) for key, value in metadata.items()
                }
                labels = {
                    key: self._row_value(value, row, row_count) for key, value in columns.items()
                }
                self._register(self._resolve_index(row_metadata, fallback=row), labels)
            return

        if row_count == len(self.dataset):
            for row in range(row_count):
                self._register(
                    row,
                    {key: self._row_value(value, row, row_count) for key, value in columns.items()},
                )
            return

        # A plain columnar artifact may describe all source frames while the
        # current training config selects only a subset of episodes.  Use the
        # absolute metadata offsets in that case.
        if self._absolute_frame_indices and row_count > len(self.dataset):
            max_source_index = max(self._absolute_frame_indices)
            if max_source_index < row_count:
                for local_index, source_index in enumerate(self._absolute_frame_indices):
                    self._register(
                        local_index,
                        {
                            key: _as_cpu_tensor(value)[source_index]
                            for key, value in columns.items()
                        },
                    )
                return

        # Episode-packed columnar files use shape [num_episodes, T, ...].
        if self._episode_locations and row_count == len(self._episode_locations):
            for episode_position, (episode, (start, length)) in enumerate(self._episode_locations.items()):
                del episode
                for frame in range(length):
                    frame_values: dict[str, Tensor] = {}
                    for key, value in columns.items():
                        tensor = _as_cpu_tensor(value)
                        if tensor.ndim > 1 and tensor.shape[1] >= length:
                            frame_values[key] = tensor[episode_position, frame]
                        else:
                            frame_values[key] = tensor[episode_position]
                    self._register(start + frame, frame_values)
            return

        raise ValueError(
            f"HD labels do not cover the dataset: columns have {row_count} rows but dataset has {len(self.dataset)} frames; "
            "include index/episode_index+frame_index metadata or save one row per frame"
        )

    def _ingest_payload(self, payload: Any) -> None:
        payload = self._unwrap_container(payload)
        if isinstance(payload, Mapping) and self._is_episode_mapping(payload):
            self._ingest_episode_mapping(payload)
        elif isinstance(payload, Mapping):
            self._ingest_columnar(payload)
        elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            self._ingest_records(payload)
        else:
            raise ValueError(
                f"HD label payload must be a mapping or record sequence, got {type(payload).__name__}"
            )


__all__ = ["HindsightLabelDataset"]
