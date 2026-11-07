"""
Seeded synthetic strings for tests, demos and benchmarks.

    random_text        uniform characters over an alphabet
    dna_sequence       A/C/G/T with a chosen GC content, optional
                       repeated segments (genomes are repetitive)
    english_like_text  words drawn from a Zipf distribution over an
                       English vocabulary, with punctuation and
                       sentence breaks — the statistics that matter for
                       string search (skewed letters, many short repeats)
    log_lines          timestamped log records (long shared prefixes)
    periodic_string    u^k + tail — worst cases for naive matching
    fibonacci_string   F_k = F_{k−1}F_{k−2}: maximally repetitive
    plant              insert patterns at random positions
    mutate             substitute a given number of characters
    sample_patterns    present and absent query patterns from a text
"""
import random
from typing import Dict, List, Optional, Sequence, Tuple

DNA = "ACGT"
LOWER = "abcdefghijklmnopqrstuvwxyz"

# ~240 common English words (function words first, so Zipf weights
# give them the highest frequencies, as in real text)
VOCABULARY = """the of and to a in is it you that he was for on are with as I his
they be at one have this from or had by hot word but what some we can out
other were all there when up use your how said an each she which do their
time if will way about many then them write would like so these her long
make thing see him two has look more day could go come did number sound no
most people my over know water than call first who may down side been now
find any new work part take get place made live where after back little only
round man year came show every good me give our under name very through just
form sentence great think say help low line differ turn cause much mean
before move right boy old too same tell does set three want air well also
play small end put home read hand port large spell add even land here must
big high such follow act why ask men change went light kind off need house
picture try us again animal point mother world near build self earth father
head stand own page should country found answer school grow study still learn
plant cover food sun four between state keep eye never last let thought city
tree cross farm hard start might story saw far sea draw left late run while
press close night real life few north open seem together next white children
begin got walk example ease paper group always music those both mark often
letter until mile river car feet care second book carry took science eat room
friend began idea fish mountain stop once base hear horse cut sure watch color
face wood main enough plain girl usual young ready above ever red list though
feel talk bird soon body dog family direct pose leave song measure door
product black short numeral class wind question happen complete ship area half
rock order fire south problem piece told knew pass since top whole king space
heard best hour better true during hundred five remember step early hold west
ground interest reach fast verb sing listen six table travel less morning ten
simple several vowel toward war lay against pattern slow center love person
money serve appear road map rain rule govern pull cold notice voice unit power
town fine certain fly fall lead cry dark machine note wait plan figure star box
noun field rest correct able pound done beauty drive stood contain front teach
week final gave green oh quick develop ocean warm free minute strong special
mind behind clear tail produce fact street inch multiply nothing course stay
wheel full force blue object decide surface deep moon island foot system busy
test record boat common gold possible plane stead dry wonder laugh thousand ago
ran check game shape equate miss brought heat snow tire bring yes distant fill
east paint language among""".split()


def _rng(seed) -> random.Random:
    return seed if isinstance(seed, random.Random) else random.Random(seed)


def random_text(n: int, alphabet: str = LOWER, seed=None) -> str:
    rng = _rng(seed)
    return "".join(rng.choices(alphabet, k=n))


def dna_sequence(n: int, gc_content: float = 0.5, repeat_fraction: float = 0.0,
                 repeat_length: Tuple[int, int] = (50, 500), seed=None) -> str:
    """
    Random genome of length n. A fraction `repeat_fraction` of it is
    made of copies of earlier segments (lengths in repeat_length),
    imitating transposons and duplications.
    """
    if not 0 <= gc_content <= 1:
        raise ValueError("gc_content must be in [0, 1]")
    rng = _rng(seed)
    at, gc = (1 - gc_content) / 2, gc_content / 2
    seq = rng.choices(DNA, weights=[at, gc, gc, at], k=n)
    copied = 0
    target = int(repeat_fraction * n)
    lo, hi = repeat_length
    while copied < target and n > 2 * hi:
        length = rng.randint(lo, hi)
        src = rng.randrange(0, n - length)
        dst = rng.randrange(0, n - length)
        seq[dst:dst + length] = seq[src:src + length]
        copied += length
    return "".join(seq)


def english_like_text(n: int, seed=None, vocabulary: Sequence[str] = VOCABULARY,
                      zipf_s: float = 1.07) -> str:
    """About n characters of Zipf-distributed words in sentences."""
    rng = _rng(seed)
    weights = [1 / (r + 1) ** zipf_s for r in range(len(vocabulary))]
    words = rng.choices(vocabulary, weights=weights, k=n // 4 + 16)
    out, size, sentence = [], 0, 0
    for w in words:
        if sentence == 0:
            w = w.capitalize()
        sentence += 1
        end = sentence >= rng.randint(6, 18)
        token = w + (". " if end else (", " if rng.random() < 0.06 else " "))
        if end:
            sentence = 0
        out.append(token)
        size += len(token)
        if size >= n:
            break
    return "".join(out)[:n]


def log_lines(n: int, seed=None) -> str:
    """About n characters of synthetic server log records."""
    rng = _rng(seed)
    levels = ["INFO", "INFO", "INFO", "DEBUG", "WARN", "ERROR"]
    paths = ["/api/users", "/api/orders", "/login", "/static/app.js",
             "/api/search", "/health"]
    out, size, t = [], 0, 1_700_000_000
    while size < n:
        t += rng.randint(0, 3)
        line = (f"2026-11-07T{(t // 3600) % 24:02d}:{(t // 60) % 60:02d}:"
                f"{t % 60:02d} {rng.choice(levels)} {rng.choice(paths)} "
                f"status={rng.choice([200, 200, 200, 201, 404, 500])} "
                f"ms={rng.randint(1, 900)}\n")
        out.append(line)
        size += len(line)
    return "".join(out)[:n]


def periodic_string(unit: str, repeats: int, tail: str = "") -> str:
    return unit * repeats + tail


def fibonacci_string(k: int, a: str = "a", b: str = "b") -> str:
    """F_1 = b, F_2 = a, F_k = F_{k−1} + F_{k−2}."""
    if k < 1:
        raise ValueError("k must be >= 1")
    prev, cur = b, a
    if k == 1:
        return b
    for _ in range(k - 2):
        prev, cur = cur, cur + prev
    return cur


def plant(text: str, patterns: Sequence[str], seed=None) -> Tuple[str, Dict[str, List[int]]]:
    """Overwrite random, non-overlapping positions of `text` with each
    pattern; returns the new text and where each was planted."""
    rng = _rng(seed)
    chars = list(text)
    taken: List[Tuple[int, int]] = []
    where: Dict[str, List[int]] = {p: [] for p in patterns}
    for p in patterns:
        for _ in range(1000):
            i = rng.randrange(0, len(chars) - len(p) + 1)
            if all(i + len(p) <= a or i >= b for a, b in taken):
                chars[i:i + len(p)] = p
                taken.append((i, i + len(p)))
                where[p].append(i)
                break
        else:
            raise ValueError("no room left to plant the patterns")
    return "".join(chars), where


def mutate(seq: str, mutations: int, alphabet: str = DNA, seed=None) -> Tuple[str, List[int]]:
    """Substitute `mutations` distinct positions with a different letter."""
    rng = _rng(seed)
    if mutations > len(seq):
        raise ValueError("more mutations than positions")
    positions = sorted(rng.sample(range(len(seq)), mutations))
    chars = list(seq)
    for i in positions:
        chars[i] = rng.choice([c for c in alphabet if c != chars[i]])
    return "".join(chars), positions


def sample_patterns(text: str, count: int, length: int, present_fraction: float = 0.5,
                    alphabet: Optional[str] = None, seed=None) -> List[str]:
    """Patterns cut from the text (present) mixed with random strings
    over the text's alphabet (usually absent)."""
    rng = _rng(seed)
    alphabet = alphabet or "".join(sorted(set(text)))
    out = []
    for k in range(count):
        if k < round(present_fraction * count) and len(text) >= length:
            i = rng.randrange(0, len(text) - length + 1)
            out.append(text[i:i + length])
        else:
            out.append("".join(rng.choices(alphabet, k=length)))
    rng.shuffle(out)
    return out
