import random
from synthetic_languages.context_free_grammars import CFGSymbol, CFGDiscriminationTask, CFGRecognizerTask
from synthetic_languages.language_select_task import LanguageSelectTask
from synthetic_languages.regular_languages import DfaState, RegularRecognizerTask, precomputeSuffixCounts

def a_or_bb_plus(limit: int) -> LanguageSelectTask:
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
    return RegularRecognizerTask(machine, S, enumeration)

def decimal_multiple_7(limit: int) -> LanguageSelectTask:
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
    return RegularRecognizerTask(machine, S_Null, enumeration)

def decimal_multiple_3(limit: int) -> LanguageSelectTask:
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
    return RegularRecognizerTask(machine, S_Null, enumeration)

def count_mod_3(limit: int) -> LanguageSelectTask:
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
    return RegularRecognizerTask(machine, S_0, enumeration)

def sum_mod_3(limit: int) -> LanguageSelectTask:
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
    return RegularRecognizerTask(machine, S_0, enumeration)

def parity(limit: int) -> LanguageSelectTask:
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
    return RegularRecognizerTask(machine, S_0, enumeration)

def recognize_json(limit: int) -> LanguageSelectTask:
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

    return CFGRecognizerTask(obj, limit)

def get_list_cfg(limit: int) -> LanguageSelectTask:
    L = CFGSymbol(False, "L")
    i = CFGSymbol(True, "i")
    L.add_rule()
    L.add_rule(L, i)

    return CFGRecognizerTask(L, limit)

def get_arithmetic_expr(limit: int) -> LanguageSelectTask:
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

    return CFGRecognizerTask(G, limit)

def get_arithmetic_expr_all(limit: int) -> LanguageSelectTask:
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

    return CFGDiscriminationTask([G, T], limit)

def dyck_1(limit: int) -> LanguageSelectTask:
    G = CFGSymbol(False, "G")
    a = CFGSymbol(True, "(")
    b = CFGSymbol(True, ")")
    G.add_rule(a, G, b, G)
    G.add_rule(a, G, b)
    G.add_rule(a, b, G)
    G.add_rule(a, b)

    def is_dyck_1(string: list[CFGSymbol]):
        counter = 0
        for char in string:
            if char == a:
                counter += 1
            elif char == b:
                counter -= 1
            else:
                return False
            if counter < 0:
                return False
        return counter == 0
    G.provide_test(is_dyck_1)
    return CFGRecognizerTask(G, limit)
    
def dyck_1_modified(limit: int) -> LanguageSelectTask:
    G = CFGSymbol(False, "G")
    a = CFGSymbol(True, "(")
    b = CFGSymbol(True, ")")
    G.add_rule(a, G, b, G)
    G.add_rule(a, G, b)
    G.add_rule(a, b, G)
    G.add_rule(a, b)

    def sample_random_inv_modified(length: int):
        if length % 2 != 0 or length < 6:
            return CFGSymbol.sample_random_inv(G, length)
        split = random.randrange(length // 6, length // 3)
        return G.sample_random(split * 2) + [b, a] + G.sample_random(length - split * 2 - 2)
    G.sample_random_inv = sample_random_inv_modified

    def is_dyck_1(string: list[CFGSymbol]):
        counter = 0
        for char in string:
            if char == a:
                counter += 1
            elif char == b:
                counter -= 1
            else:
                return False
            if counter < 0:
                return False
        return counter == 0
    G.provide_test(is_dyck_1)
    return CFGRecognizerTask(G, limit)
    