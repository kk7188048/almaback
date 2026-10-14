"""Repo map: tree-sitter symbol definitions/references -> file graph -> PageRank -> signatures within a token budget.
The approach is the one popularised by Aider's repo map; here the ranking is a plain power-iteration PageRank."""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

# our language name -> tree-sitter grammar name
GRAMMAR = {
    "python": "python", "javascript": "javascript", "typescript": "typescript", "tsx": "tsx", "java": "java",
    "go": "go", "rust": "rust", "ruby": "ruby", "php": "php", "csharp": "csharp", "c": "c", "cpp": "cpp",
}
DEF_TYPES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition", "generator_function_declaration"},
    "typescript": {"function_declaration", "class_declaration", "method_definition", "interface_declaration", "type_alias_declaration", "enum_declaration"},
    "tsx": {"function_declaration", "class_declaration", "method_definition", "interface_declaration", "type_alias_declaration", "enum_declaration"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration", "method_declaration", "constructor_declaration"},
    "go": {"function_declaration", "method_declaration", "type_spec"},
    "rust": {"function_item", "struct_item", "enum_item", "trait_item"},
    "ruby": {"method", "class", "module", "singleton_method"},
    "php": {"function_definition", "class_declaration", "method_declaration", "interface_declaration"},
    "csharp": {"class_declaration", "interface_declaration", "method_declaration", "struct_declaration", "enum_declaration"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
}
IDENT_TYPES = {"identifier", "property_identifier", "type_identifier", "field_identifier", "constant", "name"}
MAX_NODES = 250_000
_FALLBACK_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:def|class|function|func|fun|fn|interface|struct|enum|trait|module|object)\s+([A-Za-z_]\w*)", re.M)


@dataclass
class Definition:
    name: str
    line: int
    signature: str
    depth: int


@dataclass
class FileTags:
    defs: list[Definition] = field(default_factory=list)
    refs: Counter = field(default_factory=Counter)


@lru_cache(maxsize=None)
def _parser(grammar: str):
    try:
        from tree_sitter_language_pack import get_parser

        return get_parser(grammar)
    except Exception:
        return None


def _declarator_name(node):
    """C/C++: the function name is buried inside the `declarator` chain."""
    decl = node.child_by_field_name("declarator")
    while decl is not None:
        if decl.type in ("identifier", "field_identifier", "qualified_identifier", "type_identifier"):
            return decl
        nxt = decl.child_by_field_name("declarator")
        if nxt is None:
            for child in decl.children:
                if child.type in ("identifier", "field_identifier", "qualified_identifier"):
                    return child
            return None
        decl = nxt
    return None


def extract_tags(source: bytes, lang: str) -> FileTags | None:
    grammar = GRAMMAR.get(lang)
    parser = _parser(grammar) if grammar else None
    if parser is None:
        return _fallback_tags(source)
    try:
        tree = parser.parse(source)
    except Exception:
        return _fallback_tags(source)
    def_types = DEF_TYPES.get(lang, set())
    tags = FileTags()
    name_spans: set[tuple[int, int]] = set()
    stack = [(tree.root_node, 0)]
    visited = 0
    while stack:
        node, depth = stack.pop()
        visited += 1
        if visited > MAX_NODES:
            break
        child_depth = depth
        if node.type in def_types:
            name_node = node.child_by_field_name("name") or (_declarator_name(node) if lang in ("c", "cpp") else None)
            if name_node is not None:
                name = source[name_node.start_byte:name_node.end_byte].decode("utf-8", "replace")
                first_line = source[node.start_byte:node.end_byte].split(b"\n", 1)[0].decode("utf-8", "replace").strip()
                tags.defs.append(Definition(name, node.start_point[0] + 1, first_line[:160], depth))
                name_spans.add((name_node.start_byte, name_node.end_byte))
                child_depth = depth + 1
        elif node.type in IDENT_TYPES and (node.start_byte, node.end_byte) not in name_spans:
            tags.refs[source[node.start_byte:node.end_byte].decode("utf-8", "replace")] += 1
        stack.extend((c, child_depth) for c in reversed(node.children))
    tags.defs.sort(key=lambda d: d.line)
    return tags


def _fallback_tags(source: bytes) -> FileTags:
    text = source.decode("utf-8", "replace")
    tags = FileTags()
    for m in _FALLBACK_RE.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        tags.defs.append(Definition(m.group(1), line, m.group(0).strip()[:160], 0))
    for word in re.findall(r"[A-Za-z_]\w{2,}", text):
        tags.refs[word] += 1
    return tags


def pagerank(nodes: list[str], edges: dict[tuple[str, str], float], personalization: dict[str, float], damping: float = 0.85, iters: int = 40) -> dict[str, float]:
    if not nodes:
        return {}
    total_p = sum(personalization.get(n, 1.0) for n in nodes)
    base = {n: personalization.get(n, 1.0) / total_p for n in nodes}
    out_weight: dict[str, float] = defaultdict(float)
    for (src, _dst), w in edges.items():
        out_weight[src] += w
    rank = dict(base)
    for _ in range(iters):
        dangling = sum(rank[n] for n in nodes if out_weight[n] == 0)
        new = {n: (1 - damping) * base[n] + damping * dangling * base[n] for n in nodes}
        for (src, dst), w in edges.items():
            new[dst] += damping * rank[src] * w / out_weight[src]
        rank = new
    return rank


@dataclass
class RepoMap:
    text: str
    ranking: list[tuple[str, float]]
    parsed_files: int
    symbol_count: int


ENTRYPOINT_HINT = re.compile(r"(^|/)(main|app|index|server|manage|__main__|cli|wsgi|asgi)\.[a-z]+$|(^|/)readme", re.I)


def build_repo_map(files: list[tuple[str, str, bytes]], token_budget: int) -> RepoMap:
    """files: (path, lang, source bytes) for code files only."""
    tags_by_file: dict[str, FileTags] = {}
    for path, lang, data in files:
        tags = extract_tags(data, lang)
        if tags is not None and (tags.defs or tags.refs):
            tags_by_file[path] = tags

    definers: dict[str, set[str]] = defaultdict(set)
    for path, tags in tags_by_file.items():
        for d in tags.defs:
            definers[d.name].add(path)

    edges: dict[tuple[str, str], float] = defaultdict(float)
    for path, tags in tags_by_file.items():
        for name, count in tags.refs.items():
            targets = definers.get(name)
            if not targets or len(name) < 3:
                continue
            weight = min(count, 5) / len(targets)
            for target in targets:
                if target != path:
                    edges[(path, target)] += weight

    nodes = list(tags_by_file)
    personalization = {p: (10.0 if ENTRYPOINT_HINT.search(p) else 1.0) for p in nodes}
    rank = pagerank(nodes, edges, personalization)
    ranking = sorted(rank.items(), key=lambda kv: (-kv[1], kv[0]))

    char_budget = token_budget * 4
    lines: list[str] = []
    used = 0
    symbols = 0
    for path, _score in ranking:
        defs = tags_by_file[path].defs[:40]
        block = [f"{path}:"] + [f"  {'  ' * d.depth}{d.signature}" for d in defs]
        text = "\n".join(block)
        if used + len(text) + 1 > char_budget:
            break
        lines.append(text)
        used += len(text) + 1
        symbols += len(defs)
    return RepoMap("\n".join(lines), ranking, len(tags_by_file), symbols)
