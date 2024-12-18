import random


class DfaState:
    def __init__(self, alphabet: set[str], name="Unnamed State") -> None:
        self.alphabet = alphabet
        self.transitions = {}
        self.name = name
        self.suffixes = []
        self.suffixes_inv = []
        for character in alphabet:
            self.transitions[character] = self

def iterate(string: str, start: DfaState):
    current_state = start
    for i in range(len(string)):
        character = string[i]
        print(current_state.name, f"-{character}->", end=" ")
        current_state = current_state.transitions[character]
    print(current_state.name)
    return current_state

def precomputeSuffixCounts(machine: set[DfaState], accepting: set[DfaState], limit: int):
    # `state.suffixes[i] == j` means:
    # if you are in state `state`, there are `j` distinct suffixes of length `i`
    # that lead to an accepting state
    for state in machine:
        if state in accepting:
            state.suffixes = [1]
            state.suffixes_inv = [0]
        else:
            state.suffixes = [0]
            state.suffixes_inv = [1]
    for i in range(limit):
        # suffixes[i] is already calculated
        for state in machine:
            current_sum = 0
            current_sum_inv = 0
            for character in state.alphabet:
                current_sum += state.transitions[character].suffixes[i]
                current_sum_inv += state.transitions[character].suffixes_inv[i]
            state.suffixes.append(current_sum)
            state.suffixes_inv.append(current_sum_inv)

def sampleRandom(machine: set[DfaState], start: DfaState, length: int):
    current_string = ""
    current_state = start
    for i in range(length):
        value = random.randint(0, current_state.suffixes[length - i] - 1)
        random_character = None
        for character in current_state.alphabet:
            value -= current_state.transitions[character].suffixes[length - i - 1]
            if value < 0:
                random_character = character
                current_state = current_state.transitions[character]
                break
        current_string = current_string + random_character
    return current_string

def sampleRandomInv(machine: set[DfaState], start: DfaState, length: int):
    current_string = ""
    current_state = start
    for i in range(length):
        value = random.randint(0, current_state.suffixes_inv[length - i] - 1)
        random_character = None
        for character in current_state.alphabet:
            value -= current_state.transitions[character].suffixes_inv[length - i - 1]
            if value < 0:
                random_character = character
                current_state = current_state.transitions[character]
                break
        current_string = current_string + random_character
    return current_string

class RegularLanguage:
    def __init__(self, machine: set[DfaState], start: DfaState, enumeration: dict[str, int]) -> None:
        self.machine = machine
        self.start = start
        self.enumeration = enumeration
    def sampleRandom(self, length: int):
        sentence = sampleRandom(self.machine, self.start, length=length)
        return [self.enumeration[char] for char in list(sentence)]
    def sampleRandomInv(self, length: int):
        sentence = sampleRandom(self.machine, self.start, length=length)
        return [self.enumeration[char] for char in list(sentence)]

def get_example_1(limit: int):
    """(a|bb)+
    """
    alphabet = {"a", "b"}
    enumeration = {"a": 0, "b": 1}
    S = DfaState(alphabet, "none")
    F = DfaState(alphabet, "(a|bb)+")
    B = DfaState(alphabet, "(a|bb)*b")
    E = DfaState(alphabet, "fail")
    S.transitions["a"] = F
    S.transitions["b"] = B
    F.transitions["a"] = F
    F.transitions["b"] = B
    B.transitions["a"] = E
    B.transitions["b"] = F
    E.transitions["a"] = E
    E.transitions["b"] = E
    machine = {S, F, B, E}
    accepting = {F}
    precomputeSuffixCounts(machine, accepting, limit)
    return RegularLanguage(machine, S, enumeration)

def get_example_2(limit: int):
    """multiple of 7
    """
    alphabet = { str(i) for i in range(10) }
    enumeration = { str(i): i for i in range(10) }
    state_list = [DfaState(alphabet, str(i)) for i in range(7)]
    for state_index in range(7):
        state = state_list[state_index]
        for digit in range(10):
            state.transitions[str(digit)] = state_list[(state_index * 10 + digit) % 7]
    S_Null = DfaState(alphabet, "empty string")
    S_Null.transitions = state_list[0].transitions

    machine = {
        S_Null,
        *state_list,
    }
    accepting = {state_list[0]}
    precomputeSuffixCounts(machine, accepting, limit)
    return RegularLanguage(machine, S_Null, enumeration)

def get_example_3(limit: int):
    """multiple of 3
    """
    alphabet = { str(i) for i in range(10) }
    enumeration = { str(i): i for i in range(10) }
    state_list = [DfaState(alphabet, str(i)) for i in range(3)]
    for state_index in range(3):
        state = state_list[state_index]
        for digit in range(10):
            state.transitions[str(digit)] = state_list[(state_index * 10 + digit) % 3]
    S_Null = DfaState(alphabet, "empty string")
    S_Null.transitions = state_list[0].transitions

    machine = {
        S_Null,
        *state_list,
    }
    accepting = {state_list[0]}
    precomputeSuffixCounts(machine, accepting, limit)
    return RegularLanguage(machine, S_Null, enumeration)

def get_example_4(limit: int):
    """counting: mod 3
    """
    alphabet = { "a", "b" }
    enumeration = { "a": 0, "b": 1 }
    S_0 = DfaState(alphabet, "0")
    S_1 = DfaState(alphabet, "1")
    S_2 = DfaState(alphabet, "2")
    S_0.transitions["a"] = S_0
    S_1.transitions["a"] = S_1
    S_2.transitions["a"] = S_2
    S_0.transitions["b"] = S_1
    S_1.transitions["b"] = S_2
    S_2.transitions["b"] = S_0

    machine = {
        S_0,
        S_1,
        S_2,
    }
    accepting = {S_0}
    precomputeSuffixCounts(machine, accepting, limit)
    return RegularLanguage(machine, S_0, enumeration)

def get_example_5(limit: int):
    """counting: mod 3 with negatives
    """
    alphabet = { "a", "b", "c" }
    enumeration = { "a": 0, "b": 1, "c": 2 }
    S_0 = DfaState(alphabet, "0")
    S_1 = DfaState(alphabet, "1")
    S_2 = DfaState(alphabet, "2")
    S_0.transitions["a"] = S_0
    S_1.transitions["a"] = S_1
    S_2.transitions["a"] = S_2
    S_0.transitions["b"] = S_1
    S_1.transitions["b"] = S_2
    S_2.transitions["b"] = S_0
    S_0.transitions["c"] = S_2
    S_1.transitions["c"] = S_0
    S_2.transitions["c"] = S_1

    machine = {
        S_0,
        S_1,
        S_2,
    }
    accepting = {S_0}
    precomputeSuffixCounts(machine, accepting, limit)
    return RegularLanguage(machine, S_0, enumeration)

def get_example_6(limit: int):
    """counting: parity
    """
    alphabet = { "a", "b" }
    enumeration = { "a": 0, "b": 1 }
    S_0 = DfaState(alphabet, "0")
    S_1 = DfaState(alphabet, "1")
    S_0.transitions["a"] = S_0
    S_1.transitions["a"] = S_1
    S_0.transitions["b"] = S_1
    S_1.transitions["b"] = S_0

    machine = {
        S_0,
        S_1,
    }
    accepting = {S_0}
    precomputeSuffixCounts(machine, accepting, limit)
    return RegularLanguage(machine, S_0, enumeration)


if __name__ == "__main__":
    machine, start, enumeration = get_example_5(64)
    mysample = sampleRandom(machine, start, 10)
    print(mysample)
    iterate(mysample, start)

    alphabet = {"a", "b"}
    # a goes to A
    # b goes to B

    A = DfaState(alphabet, "A")
    B = DfaState(alphabet, "B")

    B.transitions["a"] = A
    A.transitions["b"] = B

    iterate("abababa", A)

    precomputeSuffixCounts({A, B}, {B}, 8)
    # print("A:", A.suffixes)
    # print("B:", B.suffixes)
    print(sampleRandom({A, B}, A, 8))
    print(sampleRandomInv({A, B}, A, 8))
