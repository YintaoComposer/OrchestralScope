from typing import Dict


def parse_palette(spec: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for token in spec.split(','):
        if not token:
            continue
        k, v = token.split(':')
        mapping[k.strip()] = v.strip()
    return mapping


