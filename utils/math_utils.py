import asyncio
import re
from itertools import islice, zip_longest

from sympy.parsing.latex import parse_latex

from math_verify import parse, verify


def repeatness(s: str):
    # n-gram based repetition ratio across multiple n values
    n = len(s)
    if n <= 1:
        return 0

    min_n = 2
    max_n = min(5, n)

    total_ngrams = 0
    total_repeated = 0

    for gram_len in range(min_n, max_n + 1):
        window_count = n - gram_len + 1
        if window_count <= 0:
            continue

        freq = {}
        for i in range(window_count):
            g = s[i : i + gram_len]
            freq[g] = freq.get(g, 0) + 1

        # number of repeated n-grams counted as extra occurrences beyond the first
        repeated = sum(count - 1 for count in freq.values() if count > 1)

        total_ngrams += window_count
        total_repeated += repeated

    if total_ngrams == 0:
        return 0

    ratio = total_repeated / total_ngrams
    return ratio > 0.2


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]


REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.
    This code comes from https://arxiv.org/pdf/2206.14858.pdf, page18.
    """
    # final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def latex_eval(latex):
    sym = parse_latex(latex)
    val = sym.evalf()
    return sym, val


def _is_latex_equal(str1, str2):
    try:
        str1,str2 = extract_last_boxed(str1), extract_last_boxed(str2)
        # print(str1,str2)
        sym1, val1 = latex_eval(str1)
        sym2, val2 = latex_eval(str2)
        if sym1 == sym2 or val1 == val2:
            return True
        else:
            raise ValueError
    except Exception:  # noqa
        # Handle the case where str1 or str2 might be None
        if str1 is None or str2 is None:
            return False
        try:
            norm1, norm2 = normalize_final_answer(str1), normalize_final_answer(str2)
            sym1, val1 = latex_eval(norm1)
            sym2, val2 = latex_eval(norm2)
            if sym1 == sym2 or val1 == val2:
                return True
            else:
                return norm1 == norm2
        except Exception:  # noqa
            pass
    return False


async def is_latex_equal(str1, str2, executor, math_mode="legacy"):
    if math_mode == "legacy":
        if (len(str1) > 128 and repeatness(str1)) or (len(str2) > 128 and repeatness(str2)):
            return False

        try:
            loop = asyncio.get_event_loop()
            task = loop.run_in_executor(executor, _is_latex_equal, str1, str2)
            result = await asyncio.wait_for(task, timeout=1.0)
            return result
        except asyncio.exceptions.TimeoutError:
            return False
    elif math_mode == "math_verify":
        try:
            loop = asyncio.get_event_loop()
            task = loop.run_in_executor(executor, verify, parse(str1), parse(str2))
            result = await asyncio.wait_for(task, timeout=1.0)
            return result
        except asyncio.exceptions.TimeoutError:
            return False
    else:
        raise NotImplementedError(f"Math mode {math_mode} is not implemented")


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:  # noqa
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:  # noqa
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace(",", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False) -> bool:
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        try:
            return float(ss1) == (float(ss2))
        except Exception:  # noqa
            return ss1 == ss2
    except Exception:  # noqa
        return str1 == str2


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def get_answer_str(s: str) -> str:
    res = remove_boxed(last_boxed_only_string(s))
    if res is not None:
        return res
    return s


async def is_equal(str1, str2, executor, math_mode="legacy"):
    first_equal = is_equiv(str1, str2)
    if first_equal:
        return True
    return await is_latex_equal(str1, str2, executor, math_mode)


def solution2answer(solution: str, math_mode="eval_peeking") -> str:
    answer = solution
    if math_mode == "eval_peeking":
        answer = get_answer_str(solution)
    else:
        raise ValueError(f"Invalid math_mode: {math_mode}")
    return answer


def get_final_answer(output: str) -> str:
    output = output.replace("is:", "is").replace("answer:", "answer is").strip()
    if output.endswith("."):
        output = output[:-1]
    if ".$" in output:
        output = output.replace(".$", "$")
    pattern_list = [
        r"answer is (-?\d+\.?\d*)$",
        r"answer is (.+?)$",
    ]
    matches = []
    for pat in pattern_list:
        matches = re.findall(pat, output, re.S)
        if matches:
            return get_answer_str(matches[0])

    return get_answer_str(output)

def extract_last_boxed(s: str) -> str | None:
    # 避免正则表达式灾难性回溯，使用手动解析
    tag = r'boxed{'
    start = s.rfind(tag)
    if start == -1:
        return None

    i = start + len(tag)
    depth = 1
    buf = []
    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1
    return ''.join(buf) if depth == 0 else None


def equation(str1, str2, math_mode="math_verify"):
    if math_mode == "legacy":
        if (len(str1) > 128 and repeatness(str1)) or (len(str2) > 128 and repeatness(str2)):
            return False

        try:
            # 直接调用，不走 executor 或 asyncio
            result = _is_latex_equal(str1, str2)
            return result
        except Exception:
            # 原逻辑超时返回 False，现在任何异常也返回 False 保持行为一致
            return False

    elif math_mode == "math_verify":
        try:
            if "boxed" not in str2:
                str2 = "\\boxed{" + str2 + "}"
            # print(parse(str1),parse(str2))
            p1 = parse(str1)
            p2 = parse(str2)
            if p1 is None or p2 is None:
                return False
            try:
                result = verify(p1, p2, timeout_seconds=30)
            except Exception as e:
                print("error",e)
                return False
            result_add = _is_latex_equal(str1, str2)
            # print("get",result_add)
            return result|result_add
        except Exception as e:
            print(e)
            return False

    else:
        raise NotImplementedError(f"Math mode {math_mode} is not implemented")
