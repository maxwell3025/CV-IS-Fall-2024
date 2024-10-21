import random
import logging
    
logger = logging.getLogger(__name__)

def sentence_to_string(sentence: list["CFGSymbol"]):
    return "".join(str(symbol) for symbol in sentence)
class CFGSymbol:
    def __init__(self, terminal: bool, string_repr: str) -> None:
        self.rules: list[list[CFGSymbol]] = []
        self.derivation_count: list[int] = []
        self.alphabet: list[CFGSymbol] = []
        self.alphabet_nonterm: list[CFGSymbol] = []
        self.terminal = terminal
        self.string_repr = string_repr
        self.possibly_empty = False
        self.cache = None
    def __str__(self) -> str:
        return self.string_repr
    def __repr__(self) -> str:
        return self.string_repr
    def findNeighbors(self):
        self.alphabet = []
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
    def fill_alphabet(self):
        for current_value in self.findNeighbors():
            if current_value.terminal:
                self.alphabet.append(current_value)
            else:
                self.alphabet_nonterm.append(current_value)
    def _calculateEmptyness(self):
        for symbol in self.findNeighbors():
            for rule in symbol.rules:
                if len(rule) == 0:
                    symbol.possibly_empty = True
        changing = True
        while changing:
            changing = False
            for symbol in self.findNeighbors():
                for rule in symbol.rules:
                    if all([rule_symbol.possibly_empty for rule_symbol in rule]):
                        symbol.possibly_empty = True
                        changing = True
    def _fill_cache(self, cache: dict[tuple["CFGSymbol"], list[int]], max_length: int):
        for symbol in self.findNeighbors():
            for rule in symbol.rules:
                for slice_index in range(2, len(rule) + 1):
                    cache[tuple(rule[0:slice_index])] = [0] * (max_length + 1)
            if symbol.terminal:
                cache[(symbol,)] = [0] + [1] + [0] * (max_length - 1)
            else:
                cache[(symbol,)] = [0] * (max_length + 1)
            cache[()] = [1] + [0] * max_length
        changed = True
        while changed:
            changed = False
            for substring in cache:
                if len(substring) == 0:
                    continue
                if len(substring) == 1:
                    (item,) = substring
                    if item.terminal:
                        continue
                    for target_length in range(max_length + 1):
                        derivations = sum(cache[tuple(rule)][target_length] for rule in item.rules)
                        if derivations < cache[substring][target_length]:
                            logger.info("Warning: invariant violated")
                        if derivations > cache[substring][target_length]:
                            cache[substring][target_length] = derivations
                            changed = True
                            logger.info(f"There are at least {derivations} ways of deriving "
                                f"{''.join(str(symbol) for symbol in substring)} with"
                                f" {target_length} terminals")
                    continue
                for target_length in range(max_length + 1):
                    derivations = sum(
                        cache[substring[:-1]][prefix_len] *
                        cache[(substring[-1],)][target_length - prefix_len]
                        for prefix_len in range(target_length)
                    )
                    if derivations < cache[substring][target_length]:
                        logger.info("Warning: invariant violated")
                    if derivations > cache[substring][target_length]:
                        cache[substring][target_length] = derivations
                        changed = True
                        logger.info(f"There are at least {derivations} ways of deriving "
                            f"{''.join(str(symbol) for symbol in substring)} with"
                            f" {target_length} terminals")
    def sample_random(self, cache: dict[tuple["CFGSymbol", ...], list[int]], length: int):
        if self.terminal: return [self]
        logger.info("BEGIN")
        logger.info(f"Deriving {self}")
        # Pick a random production rule
        choice_index = random.randint(0, sum(cache[tuple(rule)][length] for rule in self.rules) - 1)
        chosen_rule: tuple[CFGSymbol] = None
        for candidate_rule in self.rules:
            candidate_rule = tuple(candidate_rule)
            choice_index -= cache[tuple(candidate_rule)][length]
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
                cache[prefix][split_index] *
                cache[suffix][prefix_len - split_index]
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
            symbol_list = symbol_list + chosen_rule[symbol_index].sample_random(cache, symbol_len)
        logger.info("END")
        return symbol_list
    def test(self, cache: dict[tuple["CFGSymbol", ...], list[int]], string: list["CFGSymbol"]):
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
                        if merged_subsentence in cache and merged_segment not in segment_bundle:
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
    def sample_inv_random(self, length: int):
        while True:
            sample = [random.choice(self.alphabet) for _ in range(length)]
            if self.test(cache, sample): return sample
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET)
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
    
    logger.disabled = True
    G.rules.append([E])
    E.rules.append([T])
    E.rules.append([E,plus,T])
    T.rules.append([F])
    T.rules.append([T,times,F])
    F.rules.append([X])
    F.rules.append([lparen,E,rparen])
    X.rules.append([X, prime])
    X.rules.append([x])
    cache = {}
    G._fill_cache(cache, 64)
    logger.disabled = False
    counters = {}
    G.fill_alphabet()
    print(G.test(cache, [
        x,
        plus,
        x
    ]))
    # for i in range(100):
    #     sentence = G.sample_random(cache, 5)
    #     sentence = tuple(sentence)
    #     if sentence not in counters:
    #         counters[sentence] = 0
    #     counters[sentence] += 1
    # print(counters)

# def populate_cache(sentence: tuple[CFGSymbol], cache: map[tuple[CFGSymbol], list[int]], length: int):
