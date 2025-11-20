"""
Prototype: Summarization of Valorant Patch Notes

This prototype extracts entities and classifies balance changes (buffs, nerfs, neutral)
from Valorant patch notes with approximate magnitude estimation.

Notes:
next want to also categorize total changes by agent 
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


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
        
        # Pattern to match percentage changes
        self.percent_pattern = re.compile(r'(\d+\.?\d*)%', re.IGNORECASE)
        
        # Pattern to match units (seconds, damage, etc.)
        self.unit_pattern = re.compile(r'\b(seconds?|sec|damage|hp|health|armor|cost|cooldown|duration|range|radius)\b', re.IGNORECASE)
    
    def extract_agent(self, text: str) -> Optional[str]:
        """Extract agent name from text."""
        text_upper = text.upper()
        for agent in self.AGENTS:
            if agent.upper() in text_upper:
                return agent
        return None
    
    def extract_ability(self, text: str, agent: str) -> Optional[str]:
        """Extract ability name from text (usually in bold or after agent name)."""
        # Look for common ability patterns
        ability_patterns = [
            r'\*\*([^*]+)\*\*',  # Bold text
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Title case words
        ]
        
        for pattern in ability_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match and match != agent and len(match) > 2:
                    # Filter out common non-ability words
                    if match.lower() not in ['agent', 'ability', 'weapon', 'update', 'patch']:
                        return match.strip()
        return None
    
    def detect_direction(self, text: str) -> ChangeDirection:
        """Detect if change is a buff, nerf, or neutral."""
        text_lower = text.lower()
        
        buff_count = sum(1 for keyword in self.BUFF_KEYWORDS if keyword in text_lower)
        nerf_count = sum(1 for keyword in self.NERF_KEYWORDS if keyword in text_lower)
        neutral_count = sum(1 for keyword in self.NEUTRAL_KEYWORDS if keyword in text_lower)
        
        # Check for numerical direction
        num_match = self.number_pattern.search(text)
        if num_match:
            old_val = float(num_match.group(1))
            new_val = float(num_match.group(2))
            
            # Determine direction from numerical change
            # Context matters: for cooldown/equip time (negative attributes), increase = nerf, decrease = buff
            # For damage/duration/range/health (positive attributes), increase = buff, decrease = nerf
            context = text_lower
            
            # Things where higher = worse (nerf if increased)
            # These are costs, delays, and cooldowns - things you want less of
            negative_attributes = ['cooldown', 'cost', 'equip time', 'activation delay', 
                                 'delay', 'recharge time', 'reload time', 'cast time']
            
            # Things where higher = better (buff if increased)
            # These are benefits - things you want more of
            positive_attributes = ['damage', 'duration', 'range', 'health', 'hp', 'armor', 'speed', 
                                 'radius', 'size', 'amount', 'count', 'shield', 'window']
            
            # Determine direction based on attribute type and change direction
            # For negative attributes (cooldown, cost, delays): increase = nerf, decrease = buff
            # For positive attributes (damage, duration, range, health): increase = buff, decrease = nerf
            
            has_negative_attr = any(word in context for word in negative_attributes)
            has_positive_attr = any(word in context for word in positive_attributes)
            
            if has_negative_attr:
                # For negative attributes: higher value = worse
                if new_val > old_val:
                    nerf_count += 3
                elif new_val < old_val:
                    buff_count += 3
            elif has_positive_attr:
                # For positive attributes: higher value = better
                if new_val > old_val:
                    buff_count += 3
                elif new_val < old_val:
                    nerf_count += 3
            else:
                # Unknown attribute: use keyword hints or default
                if new_val > old_val:
                    buff_count += 1
                elif new_val < old_val:
                    nerf_count += 1
        
        # Determine final direction
        if buff_count > nerf_count and buff_count > neutral_count:
            return ChangeDirection.BUFF
        elif nerf_count > buff_count and nerf_count > neutral_count:
            return ChangeDirection.NERF
        else:
            return ChangeDirection.NEUTRAL
    
    def estimate_magnitude(self, text: str, old_val: Optional[float] = None, 
                          new_val: Optional[float] = None) -> Magnitude:
        """Estimate the magnitude of change."""
        text_lower = text.lower()
        
        # Check for magnitude keywords
        if any(word in text_lower for word in ['slightly', 'minor', 'small', 'smaller']):
            return Magnitude.MINOR
        if any(word in text_lower for word in ['significantly', 'major', 'large', 'dramatically']):
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
            old_val = float(num_match.group(1))
            new_val = float(num_match.group(2))
        
        # Extract unit
        unit_match = self.unit_pattern.search(text)
        if unit_match:
            unit = unit_match.group(1).lower()
        
        return old_val, new_val, unit
    
    def parse_patch_note(self, patch_text: str) -> List[BalanceChange]:
        """Parse patch notes and extract all balance changes."""
        changes = []
        
        current_agent = None
        current_ability = None
        
        # Process line by line to track agent and ability context
        lines = patch_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is an agent name (bold or standalone)
            agent_match = re.search(r'\*\*([A-Z][A-Z\s]+)\*\*', line)
            if agent_match:
                potential_agent = agent_match.group(1).strip()
                # Find matching agent from list to preserve proper casing
                for agent in self.AGENTS:
                    if agent.upper() == potential_agent.upper():
                        current_agent = agent
                        current_ability = None
                        continue
            
            # Also check for agent name without bold
            agent = self.extract_agent(line)
            if agent and len(line) < 20:  # Likely just an agent header
                current_agent = agent  # extract_agent already returns proper case
                current_ability = None
                continue
            
            # Skip if no agent found yet
            if not current_agent:
                continue
            
            # Check if this line is an ability name (bold text that's not an agent)
            ability_match = re.search(r'\*\*([^*]+)\*\*', line)
            if ability_match:
                potential_ability = ability_match.group(1).strip()
                if potential_ability.upper() != current_agent.upper():
                    current_ability = potential_ability
                    continue
            
            # Check if this is a change line (starts with bullet point)
            if line.startswith('-') or line.startswith('•'):
                change_text = line[1:].strip()
            elif re.match(r'^[-•]\s+', line):
                change_text = re.sub(r'^[-•]\s+', '', line).strip()
            else:
                # Might still be a change line without bullet
                change_text = line
            
            # Skip if too short or looks like a header
            if len(change_text) < 5:
                continue
            
            # Skip if it's just an agent or ability name
            if change_text.upper() in [a.upper() for a in self.AGENTS]:
                continue
            
            # Extract values
            old_val, new_val, unit = self.extract_values(change_text)
            
            # Detect direction
            direction = self.detect_direction(change_text)
            
            # Estimate magnitude
            magnitude = self.estimate_magnitude(change_text, old_val, new_val)
            
            # Create change object
            change = BalanceChange(
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
    
    def format_output(self, changes: List[BalanceChange]) -> str:
        """Format changes as a structured table."""
        output = []
        output.append("=" * 100)
        output.append("STRUCTURED PATCH NOTE SUMMARY")
        output.append("=" * 100)
        output.append("")
        output.append(f"{'Agent':<15} {'Ability':<20} {'Direction':<10} {'Magnitude':<12} {'Change':<40}")
        output.append("-" * 100)
        
        for change in changes:
            change_desc = change.description[:38] + "..." if len(change.description) > 40 else change.description
            if change.old_value and change.new_value:
                change_desc = f"{change.old_value} → {change.new_value} {change.unit or ''}"
            
            output.append(
                f"{change.agent:<15} "
                f"{(change.ability or 'N/A'):<20} "
                f"{change.direction.value.upper():<10} "
                f"{change.magnitude.value.upper():<12} "
                f"{change_desc:<40}"
            )
        
        output.append("")
        output.append("=" * 100)
        
        return "\n".join(output)


def main():
    """Example usage with sample Valorant patch notes."""
    
    # Sample patch note 1: Raze changes
    patch_note_1 = """
    **RAZE**
    
    **Showstopper**
    - Equip Time increased 1.1 >>> 1.4
    - Quick Equip Time increased 0.5 >>> 0.7
    - VFX reduced when firing rocket
    - VFX on rocket's trail slightly reduced
    
    **Blast Pack**
    - Damage decreased 75 >>> 50
    - Damage to objects now consistently does 600
    """
    
    # Sample patch note 2: Jett changes
    patch_note_2 = """
    **JETT**
    
    **Tailwind (Dash)**
    - Dash window decreased 12 >>> 7.5 seconds
    - Dash cooldown increase from 20 to 25 seconds
    - Activation delay increased 0.75 >>> 1.0 seconds
    
    **Cloudburst (Smoke)**
    - Duration increased 4.0 >>> 4.5 seconds
    - Smoke now blocks vision more effectively
    """
    
    parser = PatchNoteParser()
    
    print("Processing Patch Note 1: Raze Changes")
    print()
    changes_1 = parser.parse_patch_note(patch_note_1)
    print(parser.format_output(changes_1))
    print()
    
    print("\nProcessing Patch Note 2: Jett Changes")
    print()
    changes_2 = parser.parse_patch_note(patch_note_2)
    print(parser.format_output(changes_2))
    print()
    
    # Summary statistics
    print("\nSUMMARY STATISTICS")
    print("=" * 50)
    all_changes = changes_1 + changes_2
    buffs = sum(1 for c in all_changes if c.direction == ChangeDirection.BUFF)
    nerfs = sum(1 for c in all_changes if c.direction == ChangeDirection.NERF)
    neutral = sum(1 for c in all_changes if c.direction == ChangeDirection.NEUTRAL)
    
    print(f"Total changes: {len(all_changes)}")
    print(f"Buffs: {buffs}")
    print(f"Nerfs: {nerfs}")
    print(f"Neutral: {neutral}")
    print()
    
    # Agents affected
    agents = set(c.agent for c in all_changes)
    print(f"Agents affected: {', '.join(sorted(agents))}")


if __name__ == "__main__":
    main()
