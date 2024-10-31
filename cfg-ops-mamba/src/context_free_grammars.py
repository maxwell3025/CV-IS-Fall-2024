import random
import logging
import time
    
logger = logging.getLogger(__name__)
logger.disabled = True

def sentence_to_string(sentence: list["CFGSymbol"]):
    return "".join(str(symbol) for symbol in sentence)

class CFGSymbol:
    def __init__(self, terminal: bool, string_repr: str) -> None:
        self.rules: list[list[CFGSymbol]] = []
        self.derivation_count: list[int] = []
        self.alphabet: list[CFGSymbol] = []
        self.alphabet_nonterm: list[CFGSymbol] = []
        self.cache: dict[tuple["CFGSymbol", ...], list[int]] = {}
        self.terminal = terminal
        self.string_repr = string_repr
        self.possibly_empty = False
        self.enumeration: dict[CFGSymbol, int] = {}

    def __str__(self) -> str:
        return self.string_repr

    def __repr__(self) -> str:
        return self.string_repr

    def add_rule(self, *args: tuple["CFGSymbol", ...]):
        self.rules.append(list(args))

    def _find_neighbors(self):
        searched: set[CFGSymbol] = set()
        to_search: set[CFGSymbol] = set((self,))
        while len(to_search) != 0:
            current_value = to_search.pop()
            searched.add(current_value)
            yield current_value
            for rule in current_value.rules:
                for symbol in rule:
                    if symbol not in searched:
                        to_search.add(symbol)

    def _init_alphabet(self):
        self.alphabet = []
        self.alphabet_nonterm = []
        for current_value in self._find_neighbors():
            if current_value.terminal:
                self.alphabet.append(current_value)
            else:
                self.alphabet_nonterm.append(current_value)

    def _calculate_emptyness(self):
        for symbol in self._find_neighbors():
            for rule in symbol.rules:
                if len(rule) == 0:
                    symbol.possibly_empty = True
                else:
                    symbol.possibly_empty = False
        changing = True
        while changing:
            changing = False
            for symbol in self._find_neighbors():
                if symbol.possibly_empty:
                    continue
                for rule in symbol.rules:
                    if all([rule_symbol.possibly_empty for rule_symbol in rule]):
                        symbol.possibly_empty = True
                        changing = True

    def _init_cache(self, max_length: int):
        self.cache = {}
        for symbol in self._find_neighbors():
            for rule in symbol.rules:
                for slice_index in range(2, len(rule) + 1):
                    self.cache[tuple(rule[0:slice_index])] = [0] * (max_length + 1)
            if symbol.terminal:
                self.cache[(symbol,)] = [0] + [1] + [0] * (max_length - 1)
            else:
                self.cache[(symbol,)] = [0] * (max_length + 1)
            self.cache[()] = [1] + [0] * max_length
        changed = True
        while changed:
            changed = False
            for substring in self.cache:
                if len(substring) == 0:
                    continue
                if len(substring) == 1:
                    (item,) = substring
                    if item.terminal:
                        continue
                    for target_length in range(max_length + 1):
                        derivations = sum(self.cache[tuple(rule)][target_length] for rule in item.rules)
                        if derivations < self.cache[substring][target_length]:
                            logger.info("Warning: invariant violated")
                        if derivations > self.cache[substring][target_length]:
                            self.cache[substring][target_length] = derivations
                            changed = True
                            logger.info(f"There are at least {derivations} ways of deriving "
                                f"{''.join(str(symbol) for symbol in substring)} with"
                                f" {target_length} terminals")
                    continue
                for target_length in range(max_length + 1):
                    derivations = sum(
                        self.cache[substring[:-1]][prefix_len] *
                        self.cache[(substring[-1],)][target_length - prefix_len]
                        for prefix_len in range(target_length)
                    )
                    if derivations < self.cache[substring][target_length]:
                        logger.info("Warning: invariant violated")
                    if derivations > self.cache[substring][target_length]:
                        self.cache[substring][target_length] = derivations
                        changed = True
                        logger.info(f"There are at least {derivations} ways of deriving "
                            f"{''.join(str(symbol) for symbol in substring)} with"
                            f" {target_length} terminals")

    def _propagate_neighbors(self):
        for symbol in self._find_neighbors():
            symbol.alphabet = self.alphabet
            symbol.alphabet_nonterm = self.alphabet_nonterm
            symbol.cache = self.cache

    def _init_enumeration(self):
        for i, term in enumerate(self.alphabet):
            self.enumeration[term] = i
        
    def init(self, max_length: int):
        self._init_alphabet()
        self._calculate_emptyness()
        self._init_cache(max_length)
        self._propagate_neighbors()
        self._init_enumeration()

    def test(self, string: list["CFGSymbol"]):
        segment_multibundle: list[list[tuple[tuple["CFGSymbol", ...], int]]] = [[] for _ in range(len(string))]
        for symbol_index, symbol in enumerate(string):
            segment_multibundle[symbol_index].append(((symbol,), symbol_index + 1))
        change = True
        while change:
            change = False
            for segment_bundle_index, segment_bundle in enumerate(segment_multibundle):
                for segment in segment_bundle:
                    next_bunch: list[tuple[tuple["CFGSymbol", ...], int]] = None
                    if segment[1] < len(segment_multibundle):
                        next_bunch = segment_multibundle[segment[1]]
                    else:
                        next_bunch = []
                    for continuation_segment in next_bunch:
                        merged_subsentence = segment[0] + continuation_segment[0]
                        merged_segment = (merged_subsentence, continuation_segment[1])
                        if merged_subsentence in self.cache and merged_segment not in segment_bundle:
                            change = True
                            segment_bundle.append(merged_segment)
                            logger.info(f"{sentence_to_string(string[segment_bundle_index:merged_segment[1]])} matches {sentence_to_string(merged_subsentence)}")
                    for reduce_goal in self.alphabet_nonterm:
                        # logger.info(f"Trying to reduce {segment_bundle_index} -{sentence_to_string(segment[0])}-> {segment[1]} using {reduce_goal}")
                        for rule in reduce_goal.rules:
                            reduced_segment = ((reduce_goal,), segment[1])
                            if segment[0] == tuple(rule) and reduced_segment not in segment_bundle:
                                segment_bundle.append(reduced_segment)
                                logger.info(f"{sentence_to_string(string[segment_bundle_index:reduced_segment[1]])} reduces to {reduce_goal}")
                                change = True
        return ((self,), len(string)) in segment_multibundle[0]

    def sample_random(self, length: int):
        if self.terminal: return [self]
        if sum(self.cache[tuple(rule)][length] for rule in self.rules) == 0:
            return None
        logger.info("BEGIN")
        logger.info(f"Deriving {self}")
        # Pick a random production rule
        choice_index = random.randint(0, sum(self.cache[tuple(rule)][length] for rule in self.rules) - 1)
        chosen_rule: tuple[CFGSymbol] = None
        for candidate_rule in self.rules:
            candidate_rule = tuple(candidate_rule)
            choice_index -= self.cache[tuple(candidate_rule)][length]
            if choice_index < 0:
                chosen_rule = candidate_rule
                break
        logger.info(f"Chose rule {self} -> {''.join(str(symbol) for symbol in chosen_rule)}")
        # Start prefix sampling down
        prefix_len = length
        symbol_lengths: list[int] = []
        for prefix_size in range(len(chosen_rule) - 1, -1, -1):
            prefix = chosen_rule[0:prefix_size]
            suffix = (chosen_rule[prefix_size],)
            derivation_counts = [
                self.cache[prefix][split_index] *
                self.cache[suffix][prefix_len - split_index]
                for split_index in range(prefix_len + 1)
            ]
            choice_index = random.randint(0, sum(derivation_counts) - 1)
            chosen_split = None
            for candidate_split in range(prefix_len + 1):
                choice_index -= derivation_counts[candidate_split]
                if choice_index < 0:
                    chosen_split = candidate_split
                    break
            logger.info(f"{chosen_rule[prefix_size]} will have length {prefix_len - chosen_split}")
            symbol_lengths = [prefix_len - chosen_split] + symbol_lengths
            prefix_len = chosen_split
        # Recursively sample
        symbol_list = []
        for symbol_index in range(len(chosen_rule)):
            symbol_len = symbol_lengths[symbol_index]
            symbol_list = symbol_list + chosen_rule[symbol_index].sample_random(symbol_len)
        logger.info("END")
        return symbol_list

    def sample_random_inv(self, length: int):
        if sum(self.cache[tuple(rule)][length] for rule in self.rules) == len(self.alphabet)**length:
            return None
        while True:
            sample = [random.choice(self.alphabet) for _ in range(length)]
            if not self.test(sample): return sample
        
def get_json_cfg():
    value = CFGSymbol(False, "<value>")
    obj = CFGSymbol(False, "<Object>")
    obj_inner = CFGSymbol(False, "<entry list>")
    arr = CFGSymbol(False, "<Array>")
    arr_inner = CFGSymbol(False, "<inner list")
    num = CFGSymbol(True, "0")
    str = CFGSymbol(True, "\"s\"")
    id = CFGSymbol(True, "p")
    colon = CFGSymbol(True, ":")
    lcurly = CFGSymbol(True, "{")
    rcurly = CFGSymbol(True, "}")
    lsquare = CFGSymbol(True, "[")
    rsquare = CFGSymbol(True, "]")
    comma = CFGSymbol(True, ",")
    
    value.add_rule(obj)
    value.add_rule(arr)
    value.add_rule(str)
    value.add_rule(num)
    obj.add_rule(lcurly,obj_inner,rcurly)
    obj_inner.add_rule(obj_inner,id,colon,value,comma)
    obj_inner.add_rule()
    arr.add_rule(lsquare,obj_inner,rsquare)
    arr_inner.add_rule(arr_inner,value,comma)
    arr_inner.add_rule()

    obj.init(64)
    return obj

def get_list_cfg():
    L = CFGSymbol(False, "L")
    i = CFGSymbol(True, "i")
    L.add_rule()
    L.add_rule(L, i)
    L.init(64)
    return L

def get_arithmetic_expr():
    G = CFGSymbol(False, "G")
    E = CFGSymbol(False, "E")
    T = CFGSymbol(False, "T")
    F = CFGSymbol(False, "F")
    X = CFGSymbol(False, "X")
    plus = CFGSymbol(True, "+")
    times = CFGSymbol(True, "*")
    lparen = CFGSymbol(True, "(")
    rparen = CFGSymbol(True, ")")
    x = CFGSymbol(True, "x")
    prime = CFGSymbol(True, "'")
    
    G.rules.append([E])
    E.rules.append([T])
    E.rules.append([E,plus,T])
    T.rules.append([F])
    T.rules.append([T,times,F])
    F.rules.append([X])
    F.rules.append([lparen,E,rparen])
    X.rules.append([X, prime])
    X.rules.append([x])
    G.init(64)
    return G
    
def a_or_bb():
    G = CFGSymbol(False, "G")
    a = CFGSymbol(True, "a")
    b = CFGSymbol(True, "b")
    G.add_rule(G, a)
    G.add_rule(G, b, b)
    G.add_rule(a)
    G.add_rule(b, b)
    G.init(64)
    return G
    
def parity():
    Even = CFGSymbol(False, "Even")
    Odd = CFGSymbol(False, "Odd")
    a = CFGSymbol(True, "a")
    b = CFGSymbol(True, "b")
    Even.add_rule(Even, a)
    Even.add_rule(Odd, b)
    Even.add_rule(a)
    Odd.add_rule(Odd, a)
    Odd.add_rule(Even, b)
    Odd.add_rule(b)
    Even.init(64)
    return Even

language_list = dict(
    json=get_json_cfg(),
    list=get_json_cfg(),
    arithmetic=get_arithmetic_expr(),
    a_or_bb=a_or_bb(),
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET)
    G = parity()
    # for i in range(64):
    #     print(f"{i}: {G.cache[(G,)][i]}")
    for i in range(64):
        print(G.sample_random_inv(64))
    exit(0)
    # logger.disabled = False
    print("starting timer")
    perf_time = time.time()
    m = 640
    n = 256
    for i in range(m):
        sentence = G.sample_random(n)
        # print("".join(str(tok) for tok in sentence))
    perf_time = time.time() - perf_time
    print(perf_time / m / n * 1000)
    # print(counters)
