import ast
import math
import operator


# -------------------------------
# CONSTANTS
# -------------------------------

CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


# -------------------------------
# OPERATORS
# -------------------------------

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


# -------------------------------
# SCIENTIFIC FUNCTIONS
# Trigonometric functions use radians.
# Degree-specific versions are also provided.
# -------------------------------

FUNCTIONS = {
    # Trigonometry
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,

    # Degree-based trigonometry
    "sind": lambda x: math.sin(math.radians(x)),
    "cosd": lambda x: math.cos(math.radians(x)),
    "tand": lambda x: math.tan(math.radians(x)),

    # Hyperbolic
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "acosh": math.acosh,
    "atanh": math.atanh,

    # Logarithms and exponential
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "ln": math.log,
    "exp": math.exp,

    # Roots and powers
    "sqrt": math.sqrt,
    "cbrt": math.cbrt,
    "degrees": math.degrees,
    "radians": math.radians,

    # Other mathematical functions
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "comb": math.comb,
    "perm": math.perm,
    "round": round,
}


# -------------------------------
# SAFE EXPRESSION EVALUATOR
# -------------------------------

def _evaluate(node):

    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise ValueError("Only numeric values are allowed.")
        return node.value

    if isinstance(node, ast.Name):
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]

        raise ValueError(f"Unknown constant: {node.id}")

    if isinstance(node, ast.BinOp):

        operation = OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported operation.")

        left = _evaluate(node.left)
        right = _evaluate(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is too large.")

        result = operation(left, right)

        return result

    if isinstance(node, ast.UnaryOp):

        operation = OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported unary operation.")

        return operation(_evaluate(node.operand))

    if isinstance(node, ast.Call):

        if not isinstance(node.func, ast.Name):
            raise ValueError("Invalid function call.")

        function_name = node.func.id
        function = FUNCTIONS.get(function_name)

        if function is None:
            raise ValueError(f"Unsupported function: {function_name}")

        if node.keywords:
            raise ValueError("Keyword arguments are not supported.")

        arguments = [_evaluate(arg) for arg in node.args]

        return function(*arguments)

    raise ValueError("Invalid mathematical expression.")


# -------------------------------
# PUBLIC CALCULATOR FUNCTION
# -------------------------------

def calculate(expression: str):

    if not expression or not expression.strip():
        raise ValueError("Expression cannot be empty.")

    if len(expression) > 250:
        raise ValueError("Expression is too long.")

    try:
        tree = ast.parse(expression, mode="eval")

        if sum(1 for _ in ast.walk(tree)) > 100:
            raise ValueError("Expression is too complex.")

        result = _evaluate(tree.body)

        if isinstance(result, (int, float)):
            if not math.isfinite(result):
                raise ValueError("Result is not finite.")

            if isinstance(result, int) and result.bit_length() > 4096:
                raise ValueError("Result is too large.")

        return result

    except ZeroDivisionError:
        raise ValueError("Cannot divide by zero.")

    except (SyntaxError, TypeError, OverflowError) as error:
        raise ValueError("Invalid expression or mathematical domain.") from error

    except ValueError:
        raise

    except Exception as error:
        raise ValueError(
            f"Calculation failed: {type(error).__name__}"
        ) from error