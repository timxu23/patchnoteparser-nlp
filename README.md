# Buffs, Nerfs, and Beyond: Structured Summarization of Game Patch Notes

## Project Overview

This project develops a system that automatically detects and classifies gameplay balance changes from official Valorant patch notes. The system extracts entities (agents, abilities, weapons) and classifies each change by direction (buff, nerf, neutral) and approximate magnitude.

## Prototype Description

This is a **simple prototype** that demonstrates the core functionality on 1-2 Valorant patch notes. The final system will be more sophisticated and handle multiple patches with improved accuracy.

### Features

- **Entity Extraction**: Identifies agents and abilities from patch notes
- **Direction Classification**: Classifies changes as buff, nerf, or neutral
- **Magnitude Estimation**: Estimates change magnitude (minor, moderate, significant)
- **Value Extraction**: Extracts numerical changes (e.g., "75 → 50")
- **Structured Output**: Produces tabular summaries of all changes

### How It Works

1. **Parsing**: Splits patch notes into sections by agent
2. **Entity Recognition**: Uses keyword matching to identify agents and abilities
3. **Change Detection**: Uses regex patterns to find numerical changes and keyword analysis for direction
4. **Classification**: Combines numerical analysis with keyword matching to determine buff/nerf/neutral
5. **Magnitude**: Estimates based on percentage change and descriptive keywords

## Usage

```bash
python patch_parser.py
```

The script includes two sample patch notes (Raze and Jett changes) and will output structured summaries.

## Example Output

```
====================================================================================================
STRUCTURED PATCH NOTE SUMMARY
====================================================================================================

Agent           Ability              Direction   Magnitude    Change                                    
----------------------------------------------------------------------------------------------------
Raze            Showstopper          NERF        MODERATE     1.1 → 1.4 sec
Raze            Showstopper          NERF        MINOR        0.5 → 0.7 sec
Raze            Showstopper          NEUTRAL     MINOR        VFX reduced when firing rocket
Raze            Showstopper          NEUTRAL     MINOR        VFX on rocket's trail slightly reduced
Raze            Blast Pack           NERF        SIGNIFICANT  75 → 50 damage
Raze            Blast Pack           NEUTRAL     MODERATE     Damage to objects now consistently does 600
```

## Limitations

- Basic regex patterns for numerical changes
- Heuristic-based direction detection
- Limited handling of edge cases and ambiguous phrasing

## Future Improvements

- Machine learning models for entity extraction (NER)
- Fine-tuned classification models for direction and magnitude
- Better handling of context-dependent changes
- Support for more games (League of Legends, Dota 2, Overwatch)
- Trend visualization over multiple patches
- Pull patch notes from official Valorant website to summarize past changes

## Requirements

- Python 3.7+
- No external dependencies (uses only standard library)

## Team Members

- Tim: Paper writing
- Joy: Programming/Software development
- Asaad: Linguistics and theoretical issues
- Mahabub: Evaluation

