from bs4 import BeautifulSoup, NavigableString
import pandas as pd
import re

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

AGENT_NAMES = {
    "Breach", "Fade", "Gekko", "KAY/O", "Sova",
    "Iso", "Neon", "Reyna", "Waylay", "Yoru", "Raze",
    "Astra", "Brimstone", "Omen", "Viper",
    "Vyse", "Cypher", "Killjoy", "Deadlock", "Sage",
}

BENEFICIAL_WHEN_HIGH = {
    "health", "hp",
    "range", "radius", "length", "width",
    "duration", "time", "sprint time",
    "speed", "movement speed",
    "fire rate", "firing rate",
    "spread recovery",
    "weapon draw speed", "weapon recovery speed",
    "flash duration", "concuss duration",
    "hindered duration", "deafen", "marked",
}

BENEFICIAL_WHEN_LOW = {
    "cooldown",
    "cost", "credits",
    "windup", "initial windup",
    "decay",
    "time to recharge",
    "fortification delay",
    "reload time",
}

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def get_direct_text(li):
    """Return the direct text of an <li>, excluding text from nested tags."""
    parts = []
    for child in li.contents:
        if isinstance(child, NavigableString):
            s = child.strip()
            if s:
                parts.append(s)
    return " ".join(parts).strip()


def parse_numeric_change(description):
    """
    Parse numeric changes from a description string.

    Returns:
        old_value (float or 'unknown'),
        new_value (float or 'unknown'),
        unit (str or 'unknown')
    """
    text = description.replace("\u2192", "→")
    text = text.replace("&gt;&gt;&gt;", ">>>")

    two_num_pattern = re.compile(
        r'(?P<old>\d+(\.\d+)?)\s*(?P<unit>[a-zA-Z%]+)?\s*'
        r'(?:→|to|>>>)+\s*'
        r'(?P<new>\d+(\.\d+)?)\s*(?P<unit2>[a-zA-Z%]+)?'
    )

    m = two_num_pattern.search(text)
    if m:
        old = float(m.group("old"))
        new = float(m.group("new"))
        unit = m.group("unit2") or m.group("unit") or "unknown"
        return old, new, unit

    one_num_pattern = re.compile(r'(?P<value>\d+(\.\d+)?)\s*(?P<unit>[a-zA-Z%]+)')
    m1 = one_num_pattern.search(text)
    if m1:
        new = float(m1.group("value"))
        unit = m1.group("unit") or "unknown"
        return "unknown", new, unit

    return "unknown", "unknown", "unknown"


def compute_magnitude(old, new):
    """
    Your rule:
      - <=10%  -> minor
      - <=30%  -> moderate
      - >30%   -> significant
    """
    if isinstance(old, str) or isinstance(new, str):
        return "unknown"
    if old == 0:
        return "unknown"

    pct_change = abs(new - old) / old * 100.0
    if pct_change <= 10:
        return "minor"
    if pct_change <= 30:
        return "moderate"
    return "significant"


def infer_direction(description, old, new):
    """
    Heuristic buff/nerf classifier based on:
      - numeric up/down
      - whether stat is better when high or low
    """
    desc = description.lower()

    if isinstance(old, str) or isinstance(new, str):
        return "unknown"

    if new > old:
        numeric_up = True
    elif new < old:
        numeric_up = False
    else:
        return "unknown"

    stat_type = None
    for kw in BENEFICIAL_WHEN_HIGH:
        if kw in desc:
            stat_type = ("high", kw)
            break
    if stat_type is None:
        for kw in BENEFICIAL_WHEN_LOW:
            if kw in desc:
                stat_type = ("low", kw)
                break

    if stat_type is None:
        return "unknown"

    better_when, _ = stat_type

    if numeric_up:
        return "buff" if better_when == "high" else "nerf"
    else:
        return "buff" if better_when == "low" else "nerf"


# -------------------------------------------------------------------
# Core parser with lineNum support
# -------------------------------------------------------------------

def extract_agent_updates_dataframe(html: str, patch_version: str | None = None) -> pd.DataFrame:
    html_raw = html
    soup = BeautifulSoup(html, "lxml")

    def find_agent_line_from_text(html_raw: str, agent: str) -> int | str:
        lines = html_raw.splitlines()
        for i, line in enumerate(lines, start=1):
            if agent not in line:
                continue

            stripped = line.strip()
            if stripped == agent:
                return i
            if stripped.startswith("<li") and agent in stripped:
                return i

        return "unknown"


    agent_h3 = soup.find("h3", string=lambda s: s and "AGENT UPDATES" in s)
    if not agent_h3:
        raise ValueError("Could not find 'AGENT UPDATES' header in HTML.")

    stop_h3 = soup.find("h3", string=lambda s: s and "MAP UPDATES" in s)

    nodes_between = []
    node = agent_h3.next_sibling
    while node and node is not stop_h3:
        nodes_between.append(node)
        node = node.next_sibling

    rows = []

    for li in soup.find_all("li"):
        agent_name = get_direct_text(li)
        if agent_name not in AGENT_NAMES:
            continue

        current_agent = agent_name
        agent_line = find_agent_line_from_text(html_raw, current_agent)

        inner_ul = li.find("ul", recursive=False)
        if not inner_ul:
            continue

        for abl_li in inner_ul.find_all("li", recursive=False):
            strong = abl_li.find("strong")
            if strong:
                ability_name = strong.get_text(strip=True)
            else:
                ability_name = get_direct_text(abl_li) or "General"

            details_ul = abl_li.find("ul")
            if not details_ul:
                continue

            for dli in details_ul.find_all("li"):
                description = dli.get_text(" ", strip=True)
                if not re.search(r"\d", description):
                    continue

                old_val, new_val, unit = parse_numeric_change(description)
                direction = infer_direction(description, old_val, new_val)
                magnitude = compute_magnitude(old_val, new_val)

                old_val_out = old_val if old_val != "unknown" else "unknown"
                new_val_out = new_val if new_val != "unknown" else "unknown"
                unit_out = unit if unit != "unknown" else "unknown"
                direction_out = direction or "unknown"
                magnitude_out = magnitude or "unknown"

                row = {
                    "lineNum": agent_line if agent_line is not None else "unknown",
                    "agent": current_agent,
                    "ability": ability_name,
                    "description": description,
                    "direction": direction_out,
                    "magnitude": magnitude_out,
                    "old_value": old_val_out,
                    "new_value": new_val_out,
                    "unit": unit_out,
                }
                if patch_version is not None:
                    row["patch_version"] = patch_version

                rows.append(row)

    cols = ["lineNum"]
    if rows and "patch_version" in rows[0]:
        cols.append("patch_version")
    cols += [
        "agent",
        "ability",
        "description",
        "direction",
        "magnitude",
        "old_value",
        "new_value",
        "unit",
    ]

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df[cols]
    return df


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    html_path = "public/patch-notes-html/valorant-patch-notes-11-08.html"
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    df = extract_agent_updates_dataframe(html, patch_version="11.08")
    print(df)
    # df.to_csv("agent_updates_11_08.csv", index=False)
