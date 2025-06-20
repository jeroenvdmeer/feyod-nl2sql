import re
import unicodedata
import logging
from typing import List
from common.database import get_distinct_player_names, get_distinct_club_names
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

FUZZY_THRESHOLD = 85

def normalize_name(name: str) -> str:
    """Lowercase and remove accents/diacritics from a name."""
    return unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8').lower().strip()

def extract_entity_candidates(user_query: str, all_names: set[str]) -> List[str]:
    """Return all substrings in the user query that could match a known name (case-insensitive, normalized)."""
    # Clean up input query - normalize whitespace and line endings
    user_query = ' '.join(user_query.split())
    norm_query = normalize_name(user_query)
    candidates = set()
    
    # Find all substrings in the user query that match normalized known names
    for name in all_names:
        norm_name = normalize_name(name)
        # Use word boundaries to ensure we match complete words/names
        pattern = r'\b' + re.escape(norm_name) + r'\b'
        for match in re.finditer(pattern, norm_query, flags=re.IGNORECASE):
            # Get the original text from the user query for this match
            start, end = match.span()
            # Adjust indices to find word boundaries in original query
            while start > 0 and user_query[start-1].isalnum():
                start -= 1
            while end < len(user_query) and user_query[end].isalnum():
                end += 1
            candidates.add(user_query[start:end].strip())
    
    return list(set([c for c in candidates if c.strip()]))

async def find_ambiguous_entities(user_query: str) -> List[str]:
    """
    Returns a list of ambiguous or unmatched player/club names in the user query.
    If a close match is found, logs and returns the canonical name.
    """
    player_names = await get_distinct_player_names()
    club_names = await get_distinct_club_names()
    all_names = set(player_names) | set(club_names)
    norm_name_map = {normalize_name(n): n for n in all_names}
    candidates = extract_entity_candidates(user_query, all_names)
    ambiguous = []
    for candidate in candidates:
        norm_candidate = normalize_name(candidate)
        # Step C: Try exact normalized match
        if norm_candidate in norm_name_map:
            logger.info(f"Matched candidate '{candidate}' to canonical '{norm_name_map[norm_candidate]}' (exact normalized match)")
            continue  # Not ambiguous
        # Step B: Fuzzy match on normalized names
        player_matches = process.extract(norm_candidate, [normalize_name(n) for n in player_names], scorer=fuzz.ratio, limit=3)
        club_matches = process.extract(norm_candidate, [normalize_name(n) for n in club_names], scorer=fuzz.ratio, limit=3)
        best_player = player_matches[0] if player_matches else (None, 0)
        best_club = club_matches[0] if club_matches else (None, 0)
        # Step D: If a close match, log canonical name
        if best_player[1] >= FUZZY_THRESHOLD:
            idx = [normalize_name(n) for n in player_names].index(best_player[0])
            canonical = player_names[idx]
            logger.info(f"Fuzzy matched candidate '{candidate}' to player '{canonical}' (score={best_player[1]})")
            continue
        if best_club[1] >= FUZZY_THRESHOLD:
            idx = [normalize_name(n) for n in club_names].index(best_club[0])
            canonical = club_names[idx]
            logger.info(f"Fuzzy matched candidate '{candidate}' to club '{canonical}' (score={best_club[1]})")
            continue
        # If no good match or multiple close matches, mark as ambiguous
        ambiguous.append(candidate)
        logger.warning(f"Could not confidently match candidate '{candidate}' to any known name.")
    return ambiguous

async def resolve_entities(user_query: str) -> dict[str, str]:
    """
    Returns a mapping of detected entity mentions in the user query to their canonical database names.
    """
    player_names = await get_distinct_player_names()
    club_names = await get_distinct_club_names()
    all_names = set(player_names) | set(club_names)
    norm_name_map = {normalize_name(n): n for n in all_names}
    candidates = extract_entity_candidates(user_query, all_names)
    resolved = {}
    for candidate in candidates:
        norm_candidate = normalize_name(candidate)
        if norm_candidate in norm_name_map:
            resolved[candidate] = norm_name_map[norm_candidate]
            continue
        player_matches = process.extract(norm_candidate, [normalize_name(n) for n in player_names], scorer=fuzz.ratio, limit=1)
        club_matches = process.extract(norm_candidate, [normalize_name(n) for n in club_names], scorer=fuzz.ratio, limit=1)
        best_player = player_matches[0] if player_matches else (None, 0)
        best_club = club_matches[0] if club_matches else (None, 0)
        if best_player[1] >= FUZZY_THRESHOLD:
            idx = [normalize_name(n) for n in player_names].index(best_player[0])
            resolved[candidate] = player_names[idx]
            continue
        if best_club[1] >= FUZZY_THRESHOLD:
            idx = [normalize_name(n) for n in club_names].index(best_club[0])
            resolved[candidate] = club_names[idx]
            continue
    return resolved