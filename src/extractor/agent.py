from __future__ import annotations

import ast
import operator
import re
from pathlib import Path
from typing import Any, Callable


class ToolAgent:
    """A small autonomous agent that can do local reasoning and file-based tool calls."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[[str], str]] = {
            "calculate": self._calculate,
            "list_files": self._list_files,
            "read_file": self._read_file,
            "summarize": self._summarize_text,
        }

    def run(self, problem: str, max_steps: int = 5) -> str:
        if not problem or not problem.strip():
            raise ValueError("The task description must not be empty.")

        steps = 0
        current = problem.strip()
        while steps < max_steps:
            steps += 1
            lowered = current.lower()
            if self._contains_math_expression(current) or any(
                keyword in lowered
                for keyword in ("add", "plus", "sum", "subtract", "minus", "multiply", "times", "divide", "divided")
            ):
                return self._calculate(current)
            for name, tool in self._tools.items():
                result = self._try_tool(name, current)
                if result is not None:
                    return result
            if current.lower().startswith("read ") or "read file" in current.lower():
                return self._read_file(current)
            return (
                "I can solve arithmetic tasks, inspect files in the workspace, and read local text files. "
                f"Current task: {problem}"
            )

        raise RuntimeError("The agent could not complete the requested task within the step budget.")

    def _try_tool(self, name: str, problem: str) -> str | None:
        lowered = problem.lower()
        if name == "calculate" and self._contains_math_expression(problem):
            return self._calculate(problem)
        if name == "list_files" and ("list files" in lowered or "show files" in lowered or "directory" in lowered):
            return self._list_files(problem)
        if name == "read_file" and ("read" in lowered or "open" in lowered):
            return self._read_file(problem)
        if name == "summarize" and ("summarize" in lowered or "summary" in lowered):
            return self._summarize_text(problem)
        return None

    @staticmethod
    def _contains_math_expression(text: str) -> bool:
        return bool(re.search(r"\d+\s*[+\-*/%]\s*\d+|\d+\s*\*\s*\d+|\d+\s*\+\s*\d+", text))

    @staticmethod
    def _evaluate_expression(expression: str) -> float:
        cleaned = expression
        for token in ("equals", "is", "result", "answer"):
            cleaned = cleaned.replace(token, "")
        cleaned = re.sub(r"[^0-9.\s+\-*/%()]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            raise ValueError("No numerical expression was found in the task.")
        node = ast.parse(cleaned, mode="eval")

        def _eval(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                ops: dict[type[ast.operator], Any] = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.Mod: operator.mod,
                    ast.Pow: operator.pow,
                }
                op_type = type(node.op)
                if op_type not in ops:
                    raise ValueError(f"Unsupported operator in expression: {ast.dump(node.op)}")
                return ops[op_type](left, right)
            if isinstance(node, ast.UnaryOp):
                value = _eval(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return +value
                if isinstance(node.op, ast.USub):
                    return -value
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Could not evaluate expression: {ast.dump(node)}")

        return _eval(node)

    def _calculate(self, problem: str) -> str:
        expressions = re.findall(r"-?\d+(?:\.\d+)?\s*[+\-*/%]\s*-?\d+(?:\.\d+)?", problem)
        if expressions:
            return f"Result: {self._evaluate_expression(expressions[0]):g}"

        lowered = problem.lower()
        chunks = re.finditer(r"-?\d+(?:\.\d+)?|add|plus|sum|subtract|minus|difference|multiply|times|product|divide|divided|then|by|and", lowered)
        result: float | None = None
        current_op = "add"

        for match in chunks:
            token = match.group(0)
            if re.fullmatch(r"-?\d+(?:\.\d+)?", token):
                value = float(token)
                if result is None:
                    result = value
                    continue
                if current_op == "add":
                    result += value
                elif current_op == "subtract":
                    result -= value
                elif current_op == "multiply":
                    result *= value
                elif current_op == "divide":
                    if value == 0:
                        raise ValueError("Division by zero is not allowed.")
                    result /= value
                current_op = "add"
                continue

            if token in {"add", "plus", "sum"}:
                current_op = "add"
            elif token in {"subtract", "minus", "difference"}:
                current_op = "subtract"
            elif token in {"multiply", "times", "product"}:
                current_op = "multiply"
            elif token in {"divide", "divided"}:
                current_op = "divide"

        if result is None:
            numbers = [float(part) for part in re.findall(r"-?\d+(?:\.\d+)?", problem)]
            if not numbers:
                return "I can calculate arithmetic expressions if you provide a numeric expression."
            return f"Result: {sum(numbers):g}"
        return f"Result: {result:g}"

    def _list_files(self, problem: str) -> str:
        root = Path.cwd()
        entries = sorted(
            [entry.name for entry in root.iterdir() if not entry.name.startswith(".")][:20],
            key=str.lower,
        )
        if not entries:
            return "No visible files or folders were found in the current directory."
        return "Files in the current workspace:\n- " + "\n- ".join(entries)

    def _read_file(self, problem: str) -> str:
        match = re.search(r"['\"]?([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+)['\"]?", problem)
        if match is None:
            return "Please specify a file path to read, such as 'samples/invoice.txt'."
        path = Path(match.group(1))
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists() or not path.is_file():
            return f"The file '{path}' does not exist."
        text = path.read_text(encoding="utf-8", errors="replace")
        preview = "\n".join(text.splitlines()[:8])
        return f"Contents of {path}:\n{preview}"

    def _summarize_text(self, problem: str) -> str:
        payload = problem.split(":", 1)[-1].strip() if ":" in problem else problem.strip()
        words = re.findall(r"\b\w+\b", payload)
        return f"Summary: {len(words)} words across {len(payload.split())} tokens."


def agent_main(argv: list[str] | None = None) -> int:
    parser = __import__("argparse").ArgumentParser(description="Run a lightweight autonomous local agent.")
    parser.add_argument("task", nargs="*", help="Text of the task to solve")
    args = parser.parse_args(argv)
    task = " ".join(args.task)
    if not task:
        parser.error("Provide a task description for the autonomous agent.")
    print(ToolAgent().run(task))
    return 0
