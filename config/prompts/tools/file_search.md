### file_search
Recursive glob search for files. Prefix path with `project/` to search source code.
<|tool_call|>call: file_search, {"path": ".", "pattern": "*.py"}<|tool_call|>
<|tool_call|>call: file_search, {"path": "project/", "pattern": "*.md"}<|tool_call|>
