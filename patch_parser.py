"""
Patch note parser for Valorant balance updates

usage: (start venv and install all dependencies via requirements.txt)
python3 patch_parser.py

Goals:
- Parse scraped HTML patch notes (from patch_scraper) and extract structured balance changes
- Focus on Agent balance updates; skip irrelevant sections (general updates, bug fixes, competitive, esports, gameplay systems, console-only)
- Capture both structured values (old/new/unit) and qualitative direction/magnitude using regex + keyword heuristics
- Output a DataFrame that can be used for accuracy/precision/recall/F1 evaluation and for visualizing trends across patch versions
"""

import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pandas as pd
from bs4 import BeautifulSoup, NavigableString

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# ========================== [Enums / Data Model] ==========================


class ChangeDirection(Enum):
    BUFF = "buff"
    NERF = "nerf"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class Magnitude(Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    UNKNOWN = "unknown"


@dataclass
class BalanceChange:
    """Represents a single balance change extracted from patch notes."""

    patch_version: str
    agent: str
    ability: Optional[str]
    description: str
    direction: ChangeDirection
    magnitude: Magnitude
    old_value: Optional[float] = None
    new_value: Optional[float] = None
    unit: Optional[str] = None
    line_num: Optional[int] = None


# ========================== [Parser] ==========================


class PatchNoteParser:
    """Parser for extracting structured information from Valorant patch notes."""

    # Common Valorant agents
    AGENTS = [
        "Raze",
        "Jett",
        "Phoenix",
        "Sage",
        "Sova",
        "Brimstone",
        "Viper",
        "Cypher",
        "Reyna",
        "Killjoy",
        "Breach",
        "Omen",
        "Skye",
        "Yoru",
        "Astra",
        "KAY/O",
        "Chamber",
        "Neon",
        "Fade",
        "Harbor",
        "Gekko",
        "Deadlock",
        "Iso",
        "Clove",
        "Waylay",
        "Vyse",
        "Veto",
    ]

    ABILITY_KEYWORDS: Dict[str, str] = {
        # Raze
        "boom bot": "Boom Bot",
        "blast pack": "Blast Pack",
        "paint shells": "Paint Shells",
        "showstopper": "Showstopper",
        # Jett
        "cloudburst": "Cloudburst",
        "updraft": "Updraft",
        "tailwind": "Tailwind",
        "blade storm": "Blade Storm",
        # Phoenix
        "curveball": "Curveball",
        "blaze": "Blaze",
        "hot hands": "Hot Hands",
        "run it back": "Run It Back",
        # Sage
        "barrier orb": "Barrier Orb",
        "slow orb": "Slow Orb",
        "healing orb": "Healing Orb",
        "resurrection": "Resurrection",
        # Sova
        "shock bolt": "Shock Bolt",
        "owl drone": "Owl Drone",
        "recon bolt": "Recon Bolt",
        "hunter's fury": "Hunter's Fury",
        # Brimstone
        "incendiary": "Incendiary",
        "sky smoke": "Sky Smoke",
        "stim beacon": "Stim Beacon",
        "orbital strike": "Orbital Strike",
        # Viper
        "snake bite": "Snake Bite",
        "poison cloud": "Poison Cloud",
        "toxic screen": "Toxic Screen",
        "viper's pit": "Viper's Pit",
        # Cypher
        "trapwire": "Trapwire",
        "cyber cage": "Cyber Cage",
        "spycam": "Spycam",
        "neural theft": "Neural Theft",
        # Reyna
        "leer": "Leer",
        "dismiss": "Dismiss",
        "devour": "Devour",
        "empress": "Empress",
        # Killjoy
        "alarmbot": "Alarmbot",
        "nanoswarm": "Nanoswarm",
        "turret": "Turret",
        "lockdown": "Lockdown",
        # Breach
        "aftershock": "Aftershock",
        "flashpoint": "Flashpoint",
        "fault line": "Fault Line",
        "rolling thunder": "Rolling Thunder",
        # Omen
        "paranoia": "Paranoia",
        "dark cover": "Dark Cover",
        "shrouded step": "Shrouded Step",
        "from the shadows": "From the Shadows",
        # Skye
        "guiding light": "Guiding Light",
        "trailblazer": "Trailblazer",
        "regrowth": "Regrowth",
        "seekers": "Seekers",
        # Yoru
        "fakeout": "Fakeout",
        "blindside": "Blindside",
        "gatecrash": "Gatecrash",
        "dimensional drift": "Dimensional Drift",
        # Astra
        "gravity well": "Gravity Well",
        "nova pulse": "Nova Pulse",
        "nebula": "Nebula",
        "astral form": "Astral Form",
        "cosmic divide": "Cosmic Divide",
        # KAY/O
        "frag/ment": "FRAG/ment",
        "fragment": "FRAG/ment",
        "zero/point": "ZERO/point",
        "zero point": "ZERO/point",
        "flash/drive": "FLASH/drive",
        "flash drive": "FLASH/drive",
        "null/cmd": "NULL/cmd",
        "null cmd": "NULL/cmd",
        # Chamber
        "rendezvous": "Rendezvous",
        "headhunter": "Headhunter",
        "trademark": "Trademark",
        "tour de force": "Tour De Force",
        # Neon
        "fast lane": "Fast Lane",
        "relay bolt": "Relay Bolt",
        "high gear": "High Gear",
        "overdrive": "Overdrive",
        # Fade
        "prowler": "Prowler",
        "seize": "Seize",
        "haunt": "Haunt",
        "nightfall": "Nightfall",
        # Harbor
        "cascade": "Cascade",
        "cove": "Cove",
        "high tide": "High Tide",
        "reckoning": "Reckoning",
        # Gekko
        "dizzy": "Dizzy",
        "wingman": "Wingman",
        "mosh pit": "Mosh Pit",
        "thrash": "Thrash",
        # Deadlock
        "gravnet": "GravNet",
        "sonic sensor": "Sonic Sensor",
        "barrier mesh": "Barrier Mesh",
        "annihilation": "Annihilation",
        # Iso
        "double tap": "Double Tap",
        "undercut": "Undercut",
        "contingency": "Contingency",
        "kill contract": "Kill Contract",
        # Clove
        "pick-me-up": "Pick-Me-Up",
        "pick me up": "Pick-Me-Up",
        "meddle": "Meddle",
        "ruse": "Ruse",
        "not dead yet": "Not Dead Yet",
    }

    # Sections we should ignore entirely
    SKIP_SECTIONS = [
        "general updates",
        "competitive updates",
        "esports updates",
        "gameplay systems",
        "bug fixes",
        "console only",
        "pc only",
    ]

    # Direction keywords
    BUFF_KEYWORDS = [
        "increased",
        "increases",
        "increase",
        "improved",
        "improves",
        "improve",
        "enhanced",
        "enhances",
        "enhance",
        "boosted",
        "boosts",
        "boost",
        "faster",
        "more",
        "higher",
        "longer",
        "greater",
        "added",
        "adds",
        "extended",
        "reduce cost",
        "cheaper",
    ]

    NERF_KEYWORDS = [
        "decreased",
        "decreases",
        "decrease",
        "reduced",
        "reduces",
        "reduce",
        "lowered",
        "lowers",
        "lower",
        "weakened",
        "weakens",
        "weaken",
        "slower",
        "less",
        "shorter",
        "removed",
        "removes",
        "removal",
        "increased cost",
        "more expensive",
        "longer cooldown",
    ]

    NEUTRAL_KEYWORDS = [
        "adjusted",
        "adjusts",
        "adjust",
        "changed",
        "changes",
        "change",
        "updated",
        "updates",
        "update",
        "fixed",
        "fixes",
        "fix",
        "standardized",
        "standardizes",
        "standardize",
        "normalized",
        "tuned",
        "cleaned up",
    ]

    def __init__(self):
        # numerical changes: "X >>> Y", "X -> Y", "X to Y", etc
        # NOTE FOR units or symbols between the number and arrow (e.g., "40s → 60s", "15% → 10%").
        self.number_pattern = re.compile(
            r"(\d+\.?\d*)\s*[a-z%]*\s*(?:>>>|->|→| to | from )\s*(\d+\.?\d*)",
            re.IGNORECASE,
        )

        # modifiers (seconds, damage, etc.)
        self.unit_pattern = re.compile(
            r"\b(seconds?|sec|damage|hp|health|armor|cost|credits?|cooldown|duration|range|radius|ms|m|percent|point|points?)\b",
            re.IGNORECASE,
        )

        # VALORANT Patch Notes 11.09 -> 11.09
        self.version_pattern = re.compile(
            r"patch\s+notes\s+(\d+\.\d+)", re.IGNORECASE
        )

        # Regex to find section labels in headings
        self.section_normalizer = re.compile(r"\s+")

    # -------------------------- [Extractors] --------------------------

    def extract_patch_version(self, text: str, fallback: str = "unknown") -> str:
        match = self.version_pattern.search(text)
        if match:
            return match.group(1)
        return fallback

    def normalize_text(self, text: str) -> str:
        return self.section_normalizer.sub(" ", text).strip()

    def extract_agent(self, text: str) -> Optional[str]:
        """Extract agent name from text."""
        text_upper = text.upper()
        for agent in self.AGENTS:
            if agent.upper() in text_upper:
                return agent
        return None

    def extract_ability(self, text: str, agent: Optional[str]) -> Optional[str]:
        """Extract ability name from text using bold markers or title casing."""
        text_lower = text.lower()
        for key in sorted(self.ABILITY_KEYWORDS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(key)}\b", text_lower):
                return self.ABILITY_KEYWORDS[key]

        ability_patterns = [
            r"\*\*([^*]+)\*\*",  # Markdown bold, no longer in use but kept
            r"<strong>([^<]+)</strong>",  # HTML bold
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
        ]

        for pattern in ability_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                cleaned = match.strip()
                if not cleaned or cleaned == cleaned.upper():
                    # skip all-caps headings likely to be agents/sections
                    continue
                if agent and cleaned.lower() == agent.lower():
                    continue
                if cleaned.lower() in ["agent", "ability", "update", "weapon", "patch"]:
                    continue
                return cleaned
        return None

    def extract_values(self, text: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """Extract old and new values from text."""
        old_val, new_val, unit = None, None, None

        num_match = self.number_pattern.search(text)
        if num_match:
            try:
                old_val = float(num_match.group(1))
                new_val = float(num_match.group(2))
            except ValueError:
                pass
        else:
            #increases by percentages
            by_pattern = re.search(r"by\s+(\d+\.?\d*)%", text, re.IGNORECASE)
            if by_pattern:
                try:
                    new_val = float(by_pattern.group(1))
                except ValueError:
                    pass
            else:
                #Single number case
                single_match = re.search(r"(\d+\.?\d*)", text)
                if single_match:
                    try:
                        new_val = float(single_match.group(1))
                    except ValueError:
                        pass

        unit_match = self.unit_pattern.search(text)
        if unit_match:
            unit = unit_match.group(1).lower()

        return old_val, new_val, unit

    # -------------------------- [Direction/Magnitude] --------------------------

    def detect_direction(
        self, text: str, old_val: Optional[float] = None, new_val: Optional[float] = None, unit: Optional[str] = None
    ) -> ChangeDirection:
        """Detect if change is a buff, nerf, or neutral using keywords and numerical context."""
        text_lower = text.lower()

        #Special case of decay consistency is always a buff 
        if "decay" in text_lower and "consistent" in text_lower and "instead of" in text_lower:
            return ChangeDirection.BUFF
        
        #Special case of weapon buffs with percentage context
        if ("weapon draw speed" in text_lower or "weapon recovery speed" in text_lower) and "increase" in text_lower and ("percent" in text_lower or "%" in text_lower):
            return ChangeDirection.BUFF
        
        #Special case of Trapwire activation
        if "trapwire" in text_lower and ("pc:" in text_lower or "console:" in text_lower):
            # Extract values to check direction
            if old_val is None or new_val is None:
                old_val, new_val, unit = self.extract_values(text)
            if old_val is not None and new_val is not None and new_val < old_val:
                return ChangeDirection.BUFF
        
        #Special case of projectile speed increases
        if "projectile speed" in text_lower and "increased" in text_lower:
            return ChangeDirection.BUFF
        
        #Special case of activation time decreases
        if ("activation" in text_lower or "activation time" in text_lower) and "decreased" in text_lower:
            return ChangeDirection.BUFF

        if old_val is None or new_val is None:
            old_val, new_val, unit = self.extract_values(text)
        if old_val is None or new_val is None:
            return ChangeDirection.UNKNOWN

        buff_count = sum(1 for keyword in self.BUFF_KEYWORDS if keyword in text_lower)
        nerf_count = sum(1 for keyword in self.NERF_KEYWORDS if keyword in text_lower)
        neutral_count = sum(1 for keyword in self.NEUTRAL_KEYWORDS if keyword in text_lower)

        # Guard against obvious sign flips when no numeric parse is found.
        if "cooldown" in text_lower and "increase" in text_lower and "decrease" not in text_lower:
            nerf_count += 2
        if "cost" in text_lower and "increase" in text_lower and "decrease" not in text_lower:
            nerf_count += 2

        # Attributes where an increase is usually negative
        negative_attributes = [
            "cooldown",
            "cost",
            "equip time",
            "activation delay",
            "activation time",
            "delay",
            "recharge time",
            "battery recharge",
            "time to recharge",
            "reload time",
            "reactivation time",
            "cast time",
            "ultimate points",
            "ultimate point",
            "ult points",
            "price",
            "nearsight",
            "decay",
            "windup",
            "initial windup",
        ]
        # Attributes where an increase is usually positive
        positive_attributes = [
            "damage",
            "duration",
            "range",
            "health",
            "hp",
            "armor",
            "speed",
            "movement speed",
            "sprint time",
            "radius",
            "width",
            "size",
            "amount",
            "count",
            "shield",
            "window",
            "m",
            "flash",
            "concuss",
            "hindered",
            "marked",
            "deafen",
            "projectile speed",
            "weapon draw speed",
            "weapon recovery speed",
            "fire rate",
            "firing rate",
        ]

        if old_val is not None and new_val is not None and old_val != new_val:
            is_increase = new_val > old_val
            is_decrease = new_val < old_val

            has_negative_attr = any(word in text_lower for word in negative_attributes)
            has_positive_attr = any(word in text_lower for word in positive_attributes)
            
            # Special case: contextless time values (just "PC:" or "Console:") - decrease = buff
            is_contextless_time = ("pc:" in text_lower or "console:" in text_lower) and not has_negative_attr and not has_positive_attr

            if has_negative_attr:
                if is_increase:
                    nerf_count += 5
                if is_decrease:
                    buff_count += 5
            elif has_positive_attr:
                if is_increase:
                    buff_count += 5
                if is_decrease:
                    nerf_count += 5
            elif is_contextless_time:
                # Contextless times: decrease = buff (activation becomes faster)
                if is_decrease:
                    buff_count += 5
                if is_increase:
                    nerf_count += 5
            else:
                if is_increase:
                    buff_count += 1
                if is_decrease:
                    nerf_count += 1

        if buff_count > nerf_count and buff_count > 0:
            return ChangeDirection.BUFF
        if nerf_count > buff_count and nerf_count > 0:
            return ChangeDirection.NERF
        if neutral_count > 0:
            return ChangeDirection.NEUTRAL
        return ChangeDirection.NEUTRAL

    def estimate_magnitude(
        self, text: str, old_val: Optional[float] = None, new_val: Optional[float] = None
    ) -> Magnitude:
        """Estimate the magnitude of change using percent deltas and special case handling."""
        text_lower = text.lower()
        
        #Special case of decay consistency is always unknown
        if "decay" in text_lower and "consistent" in text_lower:
            return Magnitude.UNKNOWN
        
        #Special case for ult/ultimate points where single point is minor and more is moderate
        if ("ult" in text_lower and "points" in text_lower) or ("ultimate" in text_lower and "points" in text_lower):
            if old_val is not None and new_val is not None and old_val != new_val:
                point_diff = abs(old_val - new_val)
                if point_diff == 1:
                    return Magnitude.MINOR
                return Magnitude.MODERATE
        
        #Sspecial case for cost changes - 100 credit is moderate, 50 credit is minor, 150+ is significant
        if "cost" in text_lower and old_val is not None and new_val is not None:
            cost_diff = abs(old_val - new_val)
            if cost_diff >= 150:
                return Magnitude.SIGNIFICANT
            if cost_diff >= 100:
                return Magnitude.MODERATE
            if cost_diff >= 50:
                return Magnitude.MINOR
        
        #Special case of trapwire activation time with PC/Console format
        if ("trapwire" in text_lower or "activation" in text_lower) and ("pc:" in text_lower or "console:" in text_lower) and old_val is not None and new_val is not None:
            time_diff = abs(old_val - new_val)
            if time_diff >= 0.6:  #0.6s or more
                return Magnitude.SIGNIFICANT
            if time_diff >= 0.3:  #0.3s or more
                return Magnitude.MODERATE
            return Magnitude.MINOR
        
        #Special case of weapon buffs with percentage context
        if ("weapon draw speed" in text_lower or "weapon recovery speed" in text_lower) and "by" in text_lower and ("%" in text_lower or "percent" in text_lower):
            if new_val is not None or old_val is not None:
                val = new_val if new_val is not None else old_val
                if val <= 10:
                    return Magnitude.MINOR
                if val <= 30:
                    return Magnitude.MODERATE
                return Magnitude.SIGNIFICANT
            text_val, _, _ = self.extract_values(text)
            if text_val is not None:
                if text_val <= 10:
                    return Magnitude.MINOR
                if text_val <= 30:
                    return Magnitude.MODERATE
                return Magnitude.SIGNIFICANT
        
        if old_val is None or new_val is None or old_val == 0:
            return Magnitude.UNKNOWN

        percent_change = abs((new_val - old_val) / old_val) * 100

        if percent_change <= 10:
            return Magnitude.MINOR
        if percent_change <= 30:
            return Magnitude.MODERATE
        return Magnitude.SIGNIFICANT

    # -------------------------- [Parsing Helpers] --------------------------

    def _section_is_skipped(self, section_name: Optional[str]) -> bool:
        if not section_name:
            return False
        section_lower = section_name.lower()
        return any(skip in section_lower for skip in self.SKIP_SECTIONS)

    def _is_relevant_line(self, text: str) -> bool:
        """Heuristic: keep lines that contain numbers or direction keywords."""
        has_number = bool(re.search(r"\d", text))
        return has_number

    def _heading_agent(self, heading_text: str) -> Optional[str]:
        """Try to derive agent from heading text."""
        return self.extract_agent(heading_text)

    # -------------------------- [Main parsing] --------------------------

    def parse_patch_html(self, patch_html: str, fallback_version: str = "unknown") -> List[BalanceChange]:
        """Parse a single HTML patch note and return balance changes."""
        if BeautifulSoup is None:
            raise ImportError("beautifulsoup4 is required to parse HTML. Please install bs4.")
        soup = BeautifulSoup(patch_html, "html.parser")
        body = soup.body or soup

        full_text = body.get_text(" ", strip=True)
        patch_version = self.extract_patch_version(full_text, fallback=fallback_version)

        changes: List[BalanceChange] = []
        seen_entries = set()

        lines_cache = patch_html.splitlines()

        def find_line_for_text(marker: str) -> Optional[int]:
            for idx, line in enumerate(lines_cache, start=1):
                if marker and marker in line:
                    return idx
            return None

        # Narrow to AGENT UPDATES section.
        agent_h3 = soup.find(lambda tag: tag.name in {"h2", "h3"} and "agent updates" in tag.get_text(" ", strip=True).lower())
        if not agent_h3:
            return changes
        nodes_between = []
        node = agent_h3.next_sibling
        while node:
            if getattr(node, "name", None) in {"h2", "h3"} and "agent updates" not in node.get_text(" ", strip=True).lower():
                break
            nodes_between.append(node)
            node = node.next_sibling
        fragment_html = "".join(str(n) for n in nodes_between)
        fragment = BeautifulSoup(fragment_html, "html.parser")

        def direct_text(li_tag):
            parts = []
            for child in li_tag.contents:
                if isinstance(child, NavigableString):
                    txt = str(child).strip()
                    if txt:
                        parts.append(txt)
            return " ".join(parts).strip()

        def process_change_li(li_tag, agent_name: str, ability_name: str):
            text = self.normalize_text(direct_text(li_tag))
            if text and self._is_relevant_line(text):
                old_val, new_val, unit = self.extract_values(text)
                if new_val is not None:
                    direction = self.detect_direction(text, old_val=old_val, new_val=new_val, unit=unit)
                    magnitude = self.estimate_magnitude(text, old_val, new_val)
                    html_line_num = find_line_for_text(text)
                    key = (agent_name, ability_name, text)
                    if key not in seen_entries:
                        seen_entries.add(key)
                        changes.append(
                            BalanceChange(
                                patch_version=patch_version,
                                agent=agent_name,
                                ability=ability_name,
                                description=text,
                                direction=direction,
                                magnitude=magnitude,
                                old_value=old_val,
                                new_value=new_val,
                                unit=unit,
                                line_num=html_line_num,
                            )
                        )

            child_ul = li_tag.find("ul", recursive=False)
            if child_ul:
                for child_li in child_ul.find_all("li", recursive=False):
                    process_change_li(child_li, agent_name, ability_name)

        for agent_li in fragment.find_all("li"):
            agent_name = direct_text(agent_li)
            if agent_name not in self.AGENTS:
                continue

            ability_ul = agent_li.find("ul", recursive=False)
            if not ability_ul:
                continue

            for ability_li in ability_ul.find_all("li", recursive=False):
                strong = ability_li.find("strong")
                ability_name = (
                    self.normalize_text(strong.get_text(" ", strip=True)) if strong else direct_text(ability_li) or "General"
                )
                change_ul = ability_li.find("ul", recursive=False)
                if not change_ul:
                    continue
                for change_li in change_ul.find_all("li", recursive=False):
                    process_change_li(change_li, agent_name, ability_name)

        return changes

    # -------------------------- [DataFrame helpers] --------------------------

    def to_dataframe(self, changes: List[BalanceChange]) -> "pd.DataFrame":
        """Converts a list of BalanceChange objects into a Pandas DataFrame."""
        if pd is None:
            raise ImportError("pandas is required to convert changes into a DataFrame. Please install pandas.")
        data_dicts = [asdict(change) for change in changes]
        df = pd.DataFrame(data_dicts)

        if df.empty:
            return df

        # Drop rows that have no numeric information at all.
        df = df[~(df["old_value"].isna() & df["new_value"].isna())]

        df["direction"] = df["direction"].apply(lambda x: x.value if hasattr(x, "value") else x)
        df["magnitude"] = df["magnitude"].apply(lambda x: x.value if hasattr(x, "value") else x)
        if "line_num" in df.columns:
            df = df.rename(columns={"line_num": "lineNum"})

        return df

    # -------------------------- [Batch helpers] --------------------------

    def parse_directory(self, directory_path: str) -> "pd.DataFrame":
        """Parse all HTML files in a directory and return a combined DataFrame."""
        from pathlib import Path

        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        all_changes: List[BalanceChange] = []
        for path in directory.glob("*.html"):
            html_text = path.read_text(encoding="utf-8")
            version_from_filename = path.stem.split("-")[-1]  # e.g., valorant-patch-notes-11-09 -> 11-09
            version_from_filename = version_from_filename.replace("-", ".")
            changes = self.parse_patch_html(html_text, fallback_version=version_from_filename)
            all_changes.extend(changes)
            print(f'[PARSER] parsed path {path}.')

        return self.to_dataframe(all_changes)


def main():
    """
    Example run: parse all scraped HTML under public/patch-notes-html/ and print a quick summary.
    """
    parser = PatchNoteParser()
    try:
        df = parser.parse_directory("public/patch-notes-html/")
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return

    if df.empty:
        print("[INFO] No balance changes found.")
        return

    print("\n=== Entire Dataframe ===")
    print(df)

    print("\n=== Parsed Balance Changes ===")
    print(df[["lineNum", "patch_version", "agent", "ability", "direction", "magnitude", "old_value", "new_value", "unit"]])

    print("\n=== Direction counts per agent ===")
    print(df.groupby(["agent", "direction"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
