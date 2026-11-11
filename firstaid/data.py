from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import openpyxl

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WORKBOOK = PROJECT_DIR / "scenarios_redcross1.xlsx"

# Higher number = more urgent. Used to spot "under-triage" (predicting something less urgent than it is).
URGENCY_RANK = {"self-care": 0, "see a doctor": 1, "emergency": 2}


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    text: str
    label: str
    split: str


@dataclass(frozen=True)
class KitItem:
    item_id: int
    name: str


@dataclass(frozen=True)
class Guidance:
    label: str
    kit_items: tuple[KitItem, ...]
    steps: str
    urgency: str
    source: str

    @property
    def urgency_rank(self) -> int:
        return URGENCY_RANK[self.urgency]

    @property
    def is_emergency(self) -> bool:
        return self.urgency == "emergency"


@dataclass(frozen=True)
class Dataset:
    scenarios: list[Scenario]
    guidance: dict[str, Guidance]  # keyed by label, in Sheet2 order

    @property
    def labels(self) -> list[str]:
        return list(self.guidance)

    def split(self, name: str) -> list[Scenario]:
        """The scenarios whose split column is `name` ('train' or 'test')."""
        chosen = []
        for scenario in self.scenarios:
            if scenario.split == name:
                chosen.append(scenario)
        return chosen


def clean_text(text: str) -> str:
    """Normalise curly apostrophes to ' and collapse whitespace, so typed and pasted text match."""
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    return " ".join(text.split())


def load_dataset(path: Path | str = DEFAULT_WORKBOOK) -> Dataset:
    """Read the three sheets, check that they match up, and return them as one Dataset."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    kit_items_by_id = _load_kit_items(workbook["Sheet3"])
    guidance = _load_guidance(workbook["Sheet2"], kit_items_by_id)
    scenarios = _load_scenarios(workbook["Sheet1"], guidance)
    workbook.close()
    return Dataset(scenarios=scenarios, guidance=guidance)


# ---- Sheet3: kit items ----

def _load_kit_items(sheet) -> dict[int, KitItem]:
    kit_items_by_id = {}
    for row in _get_rows(sheet):
        item_id = int(row["medic_kit_item_id"])
        kit_items_by_id[item_id] = KitItem(item_id, str(row["medik_kit_item_name"]).strip())
    return kit_items_by_id


# ---- Sheet2: guidance per label ----

def _load_guidance(sheet, kit_items_by_id: dict[int, KitItem]) -> dict[str, Guidance]:
    guidance = {}
    for row in _get_rows(sheet):
        entry = _guidance_from_row(row, kit_items_by_id)
        guidance[entry.label] = entry
    return guidance


def _guidance_from_row(row: dict, kit_items_by_id: dict[int, KitItem]) -> Guidance:
    label = str(row["label"]).strip()
    item_ids = _parse_ids(row["kit_item_ids"])
    _check_kit_ids(label, item_ids, kit_items_by_id)
    urgency = str(row["urgency"]).strip()
    _check_urgency(label, urgency)
    return Guidance(
        label=label,
        kit_items=_kit_items_with_ids(item_ids, kit_items_by_id),
        steps=str(row["steps"]).strip(),
        urgency=urgency,
        source=_cell_text(row["red_cross_source"]),
    )


def _parse_ids(value) -> list[int]:
    if value is None:
        return []
    item_ids = []
    for part in str(value).replace(";", ",").split(","):
        if part.strip():
            item_ids.append(int(float(part)))
    return item_ids


def _check_kit_ids(label: str, item_ids: list[int], kit_items_by_id: dict[int, KitItem]) -> None:
    """Stop with a clear message if Sheet2 uses kit item ids that Sheet3 does not have."""
    unknown = []
    for item_id in item_ids:
        if item_id not in kit_items_by_id:
            unknown.append(item_id)
    if unknown:
        raise ValueError(f"Sheet2 '{label}' uses kit item id(s) {unknown} that are not in Sheet3")


def _check_urgency(label: str, urgency: str) -> None:
    if urgency not in URGENCY_RANK:
        raise ValueError(f"Sheet2 '{label}' has unknown urgency '{urgency}'")


def _kit_items_with_ids(item_ids: list[int], kit_items_by_id: dict[int, KitItem]) -> tuple[KitItem, ...]:
    items = []
    for item_id in item_ids:
        items.append(kit_items_by_id[item_id])
    return tuple(items)


# ---- Sheet1: training and test sentences ----

def _load_scenarios(sheet, guidance: dict[str, Guidance]) -> list[Scenario]:
    scenarios = []
    for row in _get_rows(sheet):
        scenarios.append(_scenario_from_row(row, guidance))
    _check_unique_ids(scenarios)
    return scenarios


def _check_unique_ids(scenarios: list[Scenario]) -> None:
    """Stop with a clear message if two Sheet1 rows share a scenario_id."""
    seen = set()
    duplicates = set()
    for scenario in scenarios:
        if scenario.scenario_id in seen:
            duplicates.add(scenario.scenario_id)
        seen.add(scenario.scenario_id)
    if duplicates:
        raise ValueError(f"Sheet1 has duplicate scenario_id(s) {sorted(duplicates)}: give every row its own number")


def _scenario_from_row(row: dict, guidance: dict[str, Guidance]) -> Scenario:
    label = str(row["label"]).strip()
    if label not in guidance:
        raise ValueError(f"Sheet1 scenario {row['scenario_id']} has label '{label}' that is not in Sheet2")
    return Scenario(
        scenario_id=int(row["scenario_id"]),
        text=clean_text(str(row["use_case_description"])),
        label=label,
        split=str(row["split"]).strip(),
    )


# ---- reading any sheet ----

def _get_rows(sheet) -> list[dict]:
    """Return each non-empty row as {header: value}."""
    rows = sheet.iter_rows(values_only=True)
    headers = []
    for header in next(rows):
        headers.append(_cell_text(header))

    result = []
    for row in rows:
        if not _is_empty(row):
            result.append(dict(zip(headers, row)))
    return result


def _is_empty(row: tuple) -> bool:
    for value in row:
        if value is not None:
            return False
    return True


def _cell_text(value) -> str:
    """Makes a cell a string, and empty cell becone '' """
    if value is None:
        return ""
    return str(value).strip()
