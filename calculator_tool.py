"""
This file contains the CalculatorTool class, which implements a basic calculator
functionality for mathematical operations.
"""

import math
from langchain.tools import Tool

class CalculatorTool:
    def calculate(self, expression: str) -> str:
        try:
            # Use Python's eval function with a safe subset of operations
            allowed_names = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow, 'sqrt': math.sqrt,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'pi': math.pi, 'e': math.e
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error in calculation: {str(e)}"

    def get_tool(self) -> Tool:
        return Tool(
            name="Calculator",
            func=self.calculate,
            description="Useful for performing mathematical calculations. Input should be a mathematical expression as a string."
        )