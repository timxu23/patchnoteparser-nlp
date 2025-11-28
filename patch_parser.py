"""
Prototype: Summarization of Valorant Patch Notes

This version integrates Pandas DataFrames for structured historical tracking and summary.

Improvements:
- Added 'patch_version' to BalanceChange dataclass.
- Added method to convert changes into a Pandas DataFrame.
- Updated main() to use DataFrame for categorization and summary statistics.
"""

import re
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import sys

                    # ========== [Heuristics] ===========

class ChangeDirection(Enum):
    BUFF = "buff"
    NERF = "nerf"
    NEUTRAL = "neutral"


class Magnitude(Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"


@dataclass
class BalanceChange:
    """Represents a single balance change extracted from patch notes."""
    # added patch_version for historical tracking
    patch_version: str 
    agent: str
    ability: Optional[str]
    description: str
    direction: ChangeDirection
    magnitude: Magnitude
    old_value: Optional[float] = None
    new_value: Optional[float] = None
    unit: Optional[str] = None


class PatchNoteParser:
    """Parser for extracting structured information from Valorant patch notes."""
    
    # Common Valorant agents
    AGENTS = [
        "Raze", "Jett", "Phoenix", "Sage", "Sova", "Brimstone", "Viper",
        "Cypher", "Reyna", "Killjoy", "Breach", "Omen", "Skye", "Yoru",
        "Astra", "KAY/O", "Chamber", "Neon", "Fade", "Harbor", "Gekko",
        "Deadlock", "Iso", "Clove"
    ]
    
    # Keywords that indicate direction
    BUFF_KEYWORDS = [
        "increased", "increases", "increase", "improved", "improves", "improve",
        "enhanced", "enhances", "enhance", "boosted", "boosts", "boost",
        "faster", "more", "higher", "longer", "greater", "added", "adds"
    ]
    
    NERF_KEYWORDS = [
        "decreased", "decreases", "decrease", "reduced", "reduces", "reduce",
        "lowered", "lowers", "lower", "weakened", "weakens", "weaken",
        "slower", "less", "shorter", "removed", "removes", "removal"
    ]
    
    NEUTRAL_KEYWORDS = [
        "adjusted", "adjusts", "adjust", "changed", "changes", "change",
        "updated", "updates", "update", "fixed", "fixes", "fix",
        "standardized", "standardizes", "standardize", "normalized"
    ]
    
    def __init__(self):
        # Pattern to match numerical changes: "X >>> Y", "X -> Y", "X to Y", etc.
        self.number_pattern = re.compile(
            r'(\d+\.?\d*)\s*(?:>>>|->|→|to|from\s+\d+\.?\d*\s+to)\s*(\d+\.?\d*)',
            re.IGNORECASE
        )
        
        # Pattern to match units (seconds, damage, etc.)
        self.unit_pattern = re.compile(r'\b(seconds?|sec|damage|hp|health|armor|cost|cooldown|duration|range|radius|ms|m)\b', re.IGNORECASE)
    
    def extract_agent(self, text: str) -> Optional[str]:
        """Extract agent name from text."""
        text_upper = text.upper()
        for agent in self.AGENTS:
            if agent.upper() in text_upper:
                return agent
        return None
    
    def extract_ability(self, text: str, agent: str) -> Optional[str]:
        """Extract ability name from text (usually in bold or after agent name)."""
        ability_patterns = [
            r'\*\*([^*]+)\*\*',  # Bold text
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Title case words
        ]
        
        for pattern in ability_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match and match != agent and len(match) > 2:
                    if match.lower() not in ['agent', 'ability', 'weapon', 'update', 'patch']:
                        return match.strip()
        return None
    
    def detect_direction(self, text: str) -> ChangeDirection:
        """Detect if change is a buff, nerf, or neutral using keywords and numerical context."""
        text_lower = text.lower()
        
        buff_count = sum(1 for keyword in self.BUFF_KEYWORDS if keyword in text_lower)
        nerf_count = sum(1 for keyword in self.NERF_KEYWORDS if keyword in text_lower)
        neutral_count = sum(1 for keyword in self.NEUTRAL_KEYWORDS if keyword in text_lower)
        
        # Things where higher = worse (nerf if increased): costs, delays, cooldowns
        negative_attributes = ['cooldown', 'cost', 'equip time', 'activation delay', 
                             'delay', 'recharge time', 'reload time', 'cast time', 'ms']
        # Things where higher = better (buff if increased): damage, duration, range, health
        positive_attributes = ['damage', 'duration', 'range', 'health', 'hp', 'armor', 'speed', 
                             'radius', 'size', 'amount', 'count', 'shield', 'window', 'm']

        # Prioritize numerical change + context over keywords
        old_val, new_val, _ = self.extract_values(text)
        
        if old_val is not None and new_val is not None and old_val != new_val:
            is_increase = new_val > old_val
            is_decrease = new_val < old_val
            
            has_negative_attr = any(word in text_lower for word in negative_attributes)
            has_positive_attr = any(word in text_lower for word in positive_attributes)

            # Weight numerical changes heavily based on context
            if has_negative_attr:
                if is_increase: nerf_count += 5  # Cost increase = Nerf
                if is_decrease: buff_count += 5  # Cooldown decrease = Buff
            elif has_positive_attr:
                if is_increase: buff_count += 5  # Damage increase = Buff
                if is_decrease: nerf_count += 5  # Health decrease = Nerf
            else:
                # Fallback to general numerical direction if context is unknown
                if is_increase: buff_count += 1
                if is_decrease: nerf_count += 1

        # Determine final direction
        if buff_count > nerf_count and buff_count > 0:
            return ChangeDirection.BUFF
        elif nerf_count > buff_count and nerf_count > 0:
            return ChangeDirection.NERF
        else:
            return ChangeDirection.NEUTRAL
    
    def estimate_magnitude(self, text: str, old_val: Optional[float] = None, 
                          new_val: Optional[float] = None) -> Magnitude:
        """Estimate the magnitude of change."""
        text_lower = text.lower()
        
        # Check for magnitude keywords
        if any(word in text_lower for word in ['slightly', 'minor', 'small', 'marginal']):
            return Magnitude.MINOR
        if any(word in text_lower for word in ['significantly', 'major', 'large', 'dramatically', 'huge']):
            return Magnitude.SIGNIFICANT
        
        # Calculate percentage change if we have values
        if old_val is not None and new_val is not None and old_val != 0:
            percent_change = abs((new_val - old_val) / old_val) * 100
            
            if percent_change < 10:
                return Magnitude.MINOR
            elif percent_change < 30:
                return Magnitude.MODERATE
            else:
                return Magnitude.SIGNIFICANT
        
        # Default to moderate if we can't determine
        return Magnitude.MODERATE
    
    def extract_values(self, text: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """Extract old and new values from text."""
        old_val, new_val, unit = None, None, None
        
        # Try to match numerical pattern
        num_match = self.number_pattern.search(text)
        if num_match:
            try:
                old_val = float(num_match.group(1))
                new_val = float(num_match.group(2))
            except ValueError:
                pass # Catch cases where conversion to float fails
        
        # Extract unit
        unit_match = self.unit_pattern.search(text)
        if unit_match:
            unit = unit_match.group(1).lower()
        
        return old_val, new_val, unit
    
    def parse_patch_note(self, patch_version: str, patch_text: str) -> List[BalanceChange]:
        """Parse patch notes and extract all balance changes."""
        changes = []
        
        current_agent = None
        current_ability = None
        
        lines = patch_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # --- Context Tracking (Agent/Ability) ---
            
            # Check for agent name (bold or standalone)
            agent_match = re.search(r'\*\*([A-Z][A-Z\s]+)\*\*', line)
            if agent_match:
                potential_agent = agent_match.group(1).strip()
                for agent in self.AGENTS:
                    if agent.upper() == potential_agent.upper():
                        current_agent = agent
                        current_ability = None
                        continue

            # Check for agent name without bold
            agent = self.extract_agent(line)
            if agent and len(line) < 20:
                current_agent = agent
                current_ability = None
                continue
            
            if not current_agent:
                continue
            
            # Check for ability name
            ability_match = re.search(r'\*\*([^*]+)\*\*', line)
            if ability_match:
                potential_ability = ability_match.group(1).strip()
                if potential_ability.upper() != current_agent.upper():
                    current_ability = potential_ability
                    continue
            
            # --- Change Extraction ---

            # Clean bullet points/formatting
            if line.startswith('-') or line.startswith('•'):
                change_text = line[1:].strip()
            elif re.match(r'^[-•]\s+', line):
                change_text = re.sub(r'^[-•]\s+', '', line).strip()
            else:
                change_text = line
            
            if len(change_text) < 5 or change_text.upper() in [a.upper() for a in self.AGENTS]:
                continue
            
            # Extract and classify
            old_val, new_val, unit = self.extract_values(change_text)
            direction = self.detect_direction(change_text)
            magnitude = self.estimate_magnitude(change_text, old_val, new_val)
            
            # Create change object
            change = BalanceChange(
                patch_version=patch_version,
                agent=current_agent,
                ability=current_ability,
                description=change_text,
                direction=direction,
                magnitude=magnitude,
                old_value=old_val,
                new_value=new_val,
                unit=unit
            )
            
            changes.append(change)
        
        return changes
    
    def to_dataframe(self, changes: List[BalanceChange]) -> pd.DataFrame:
        """Converts a list of BalanceChange objects into a Pandas DataFrame."""
        # Convert list of dataclass objects to list of dictionaries
        data_dicts = [asdict(change) for change in changes]
        
        df = pd.DataFrame(data_dicts)
        
        # Convert enum types to simple strings for easier plotting/export
        if not df.empty:
            df['direction'] = df['direction'].apply(lambda x: x.value)
            df['magnitude'] = df['magnitude'].apply(lambda x: x.value)
            
        return df

    def build_history_dataframe(self, patch_notes: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Parse multiple patch notes and return a consolidated historical DataFrame.

        Args:
            patch_notes: List of (patch_version, patch_text) tuples.
        """
        all_changes: List[BalanceChange] = []
        for patch_version, patch_text in patch_notes:
            all_changes.extend(self.parse_patch_note(patch_version, patch_text))

        return self.to_dataframe(all_changes)

def main():
    """Example usage with sample Valorant patch notes or scraped directory."""

    cli = argparse.ArgumentParser(description="Parse Valorant patch notes into a dataframe.")
    cli.add_argument(
        "--notes-dir",
        type=Path,
        default=Path("public/patch-notes-test"),
        help="Directory of scraped patch note .txt files to parse (default: public/patch-notes-test)",
    )
    cli.add_argument(
        "--use-sample",
        action="store_true",
        help="Run with built-in sample patch notes instead of reading from disk.",
    )
    args = cli.parse_args()
    
    parser = PatchNoteParser()

    # saved the old sample parsing as well here as a legacy
    if args.use_sample:
        PATCH_VERSION_1 = "v8.01"
        PATCH_VERSION_2 = "v8.02"
        
        patch_note_1 = """
        **RAZE**
        
        **Showstopper**
        - Equip Time increased 1.1 >>> 1.4 seconds
        - Quick Equip Time increased 0.5 >>> 0.7 seconds
        - VFX reduced when firing rocket (Neutral)
        
        **Blast Pack**
        - Damage decreased 75 >>> 50 damage
        - Damage to objects now consistently does 600 damage
        """
        
        patch_note_2 = """
        **JETT**
        
        **Tailwind (Dash)**
        - Dash window decreased 12m >>> 7.5m
        - Dash cooldown increase from 20 to 25 seconds
        - Activation delay increased 0.75 >>> 1.0 seconds
        
        **Cloudburst (Smoke)**
        - Duration increased 4.0s >>> 4.5s
        - Smoke now blocks vision more effectively (Buff)
        """

        patch_history_df = parser.build_history_dataframe([
            (PATCH_VERSION_1, patch_note_1),
            (PATCH_VERSION_2, patch_note_2),
        ])
    else:
        # In scraped mode, read every .txt file in the directory, using the filename stem as patch_version.
        if not args.notes_dir.exists() or not args.notes_dir.is_dir():
            print(f"[ERROR] notes directory not found: {args.notes_dir}")
            sys.exit(1)

        patch_inputs: List[Tuple[str, str]] = []
        txt_files = sorted(args.notes_dir.glob("*.txt"))

        if not txt_files:
            print(f"[ERROR] no .txt files found in {args.notes_dir}")
            sys.exit(1)

        for file_path in txt_files:
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                print(f"[WARN] could not read {file_path}: {exc}")
                continue

            patch_version = file_path.stem or "unknown_patch"
            patch_inputs.append((patch_version, text))

        if not patch_inputs:
            print(f"[ERROR] no readable patch notes in {args.notes_dir}")
            sys.exit(1)

        patch_history_df = parser.build_history_dataframe(patch_inputs)

    print("\n--- PATCH HISTORY DATA FRAME ---\n")
    print(patch_history_df[['patch_version', 'agent', 'ability', 'direction', 'magnitude', 'old_value', 'new_value', 'unit']])

    print("\n\nMASTER DATAFRAME SUMMARY (All Patches)")
    print("=" * 70)
    
    agent_summary = patch_history_df.groupby(['agent', 'direction']).size().unstack(fill_value=0)
    agent_summary['Total'] = agent_summary.sum(axis=1)
    
    print("\n--- Balance Summary by Agent ---")
    print(agent_summary.sort_values(by='Total', ascending=False))
    
    print("\n--- Jett's Historical Changes (Buffs vs. Nerfs) ---")
    jett_history = patch_history_df[patch_history_df['agent'] == 'Jett']
    history_pivot = jett_history.pivot_table(
        index='patch_version', 
        columns='direction', 
        aggfunc='size', 
        fill_value=0
    )
    print(history_pivot)


if __name__ == "__main__":
    main()
