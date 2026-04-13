"""
Safe math evaluation tool.

CONCEPT: LLMs are bad at arithmetic. Tools fix that.
We use a safe evaluator — never use eval() on user input directly.
"""

import ast
import operator
from langchain_core.tools import tool


# Map AST node types to safe operator functions
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(node) -> float:
    """
    Recursively evaluate AST nodes using only safe operators.

    This prevents code injection via the calculator.

    Args:
        node: AST node to evaluate

    Returns:
        float: Result of evaluation

    Raises:
        ValueError: If operation is not supported
    """
    if isinstance(node, ast.Expression):
        return safe_eval(node.body)

    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        return SAFE_OPERATORS[op_type](left, right)

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = safe_eval(node.operand)
        return SAFE_OPERATORS[op_type](operand)

    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression. Supports +, -, *, /, ** (power).
    Use this for any calculation, unit conversion, or numerical reasoning.
    Input must be a valid math expression like '(15 * 8) / 2' or '2 ** 10'.
    """
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')

        # Safely evaluate using only allowed operators
        result = safe_eval(tree)

        # Format the result nicely
        if result == int(result):
            return f"Result: {int(result)}"
        return f"Result: {result:.6f}".rstrip('0').rstrip('.')

    except SyntaxError:
        return f"Error: Invalid expression syntax. Use format like '(15 * 8) / 2'"

    except ValueError as e:
        return f"Error: {str(e)}"

    except ZeroDivisionError:
        return "Error: Division by zero"

    except Exception as e:
        return f"Error: {str(e)}"
