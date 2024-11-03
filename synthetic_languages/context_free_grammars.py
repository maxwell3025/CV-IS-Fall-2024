import random
import logging
import time

from synthetic_languages.language_select_task import LanguageSelectTask
    
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
        
class CFGRecognizerTask(LanguageSelectTask):
    """Represents a language recognition task
    """
    def __init__(self, grammar: CFGSymbol, max_length=64):
        self.grammar = grammar
        self.grammar.init(max_length)

    def sample(self, length: int, type: int) -> list[int] | None:
        if(type == 0):
            symbol_list = self.grammar.sample_random(length)
            if(symbol_list == None): return None
            return [self.grammar.enumeration[symbol] for symbol in symbol_list]
        if(type == 1):
            symbol_list = self.grammar.sample_random_inv(length)
            if(symbol_list == None): return None
            return [self.grammar.enumeration[symbol] for symbol in symbol_list]
            
    def repr_sample(self, length: int, type: int) -> str | None:
        if(type == 0):
            symbol_list = self.grammar.sample_random(length)
            if(symbol_list == None): return None
            output_string = [repr(symbol) for symbol in symbol_list]
            output_string = "".join(output_string)
            return f"{self.grammar}: {output_string}"
        if(type == 1):
            symbol_list = self.grammar.sample_random_inv(length)
            if(symbol_list == None): return None
            output_string = [repr(symbol) for symbol in symbol_list]
            output_string = "".join(output_string)
            return f"Not {self.grammar}: {output_string}"

    def language_count(self) -> int:
        return 2

    def alphabet_size(self) -> int:
        return len(self.grammar.alphabet)

class CFGDiscriminationTask(LanguageSelectTask):
    """Represents a language discrimination task
    """
    def __init__(self, grammar_list: list[CFGSymbol], max_length=64):
        self.grammar_list = grammar_list
        union_enumeration = set()
        for grammar in self.grammar_list:
            if len(grammar.cache) == 0:
                grammar.init(max_length)
            union_enumeration.update(*grammar.alphabet)
        self.enumeration = {}
        for i, symbol in enumerate(union_enumeration):
            self.enumeration[symbol] = i

    def sample(self, length: int, type: int) -> list[int] | None:
            symbol_list = self.grammar_list[type].sample_random(length=length)
            if(symbol_list == None): return None
            return [self.grammar.enumeration[symbol] for symbol in symbol_list]
            
    def repr_sample(self, length: int, type: int) -> str | None:
            symbol_list = self.grammar_list[type].sample_random(length=length)
            if(symbol_list == None): return None
            output_string = [repr(symbol) for symbol in symbol_list]
            output_string = "".join(output_string)
            return f"{self.grammar}: {output_string}"

    def language_count(self) -> int:
        return len(self.grammar_list)

    def alphabet_size(self) -> int:
        return len(self.enumeration)

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
