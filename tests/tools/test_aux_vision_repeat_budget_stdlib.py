"""Regression for the real aux-vision handler, no network or model invocation.

Extract only the function under test from the live Python module to avoid
installing the whole Hermes runtime in a test environment. The complete
async handler body is compiled unchanged, with fake provider dependencies.
"""
from __future__ import annotations

import ast
import asyncio
import json
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[2] / "tools/vision_tools.py"


class AuxVisionRepeatTests(unittest.TestCase):
    def make_handler(self, *, answers=None, native=False):
        tree = ast.parse(SRC.read_text(encoding="utf-8"))
        func = next(node for node in tree.body
                    if isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "_handle_vision_analyze")
        calls = []
        slots = {}
        released = []
        outputs = list(answers or [json.dumps({"success": True, "analysis": "x+1/x=4"})])
        def reserve(path):
            count = slots.get(path, 0)
            if count >= 2:
                return json.dumps({"success": False, "error": "limit reached"})
            slots[path] = count + 1
            return None
        def release(path):
            released.append(path)
            slots[path] = max(slots[path]-1, 0)
        async def aux(path, prompt, model, task_id=None, region=None):
            calls.append((path, region))
            answer = outputs.pop(0) if outputs else json.dumps({"success": True, "analysis": "ok"})
            if isinstance(answer, BaseException):
                raise answer
            return answer
        async def native_handler(path, question, task_id=None, region=None):
            return "native"
        scope = {
            "_should_use_native_vision_fast_path": lambda: native,
            "_vision_analyze_native": native_handler,
            "_repeat_refusal": reserve,
            "_release_embed": release,
            "_configured_aux_model": lambda *a: "MiniMax-M3",
            "vision_analyze_tool": aux,
            "json": json,
        }
        exec(compile(ast.Module(body=[func], type_ignores=[]), str(SRC), "exec"), scope)
        return scope["_handle_vision_analyze"], calls, slots, released

    def test_same_image_crops_share_two_call_cap(self):
        handler, calls, slots, _ = self.make_handler()
        async def run():
            args = [
                {"image_url": "worksheet.png", "question": "whole"},
                {"image_url": "worksheet.png", "question": "crop", "region": [0, 0, 10, 10]},
                {"image_url": "worksheet.png", "question": "crop again", "region": [0, 0, 5, 5]},
                {"image_url": "worksheet.png", "question": "repeat"},
            ]
            return await asyncio.gather(*(handler(item) for item in args))
        results = asyncio.run(run())
        self.assertEqual(len(calls), 2)
        self.assertEqual(slots, {"worksheet.png": 2})
        self.assertEqual([json.loads(v)["success"] for v in results], [True, True, False, False])

    def test_failed_result_frees_budget(self):
        handler, calls, slots, released = self.make_handler(
            answers=['{"success": false}', '{"success": true, "analysis": "ok"}'])
        async def run():
            return await handler({"image_url": "worksheet.png", "question": "first"}), await handler(
                {"image_url": "worksheet.png", "question": "second"})
        r1, r2 = asyncio.run(run())
        self.assertEqual((json.loads(r1)["success"], json.loads(r2)["success"]), (False, True))
        self.assertEqual(len(calls), 2)
        self.assertEqual(released, ["worksheet.png"])
        self.assertEqual(slots["worksheet.png"], 1)

    def test_exception_frees_budget(self):
        handler, calls, slots, released = self.make_handler(answers=[RuntimeError("backend failure")])
        with self.assertRaisesRegex(RuntimeError, "backend failure"):
            asyncio.run(handler({"image_url": "worksheet.png", "question": "q"}))
        self.assertEqual(slots["worksheet.png"], 0)
        self.assertEqual(released, ["worksheet.png"])

    def test_native_path_unchanged(self):
        handler, calls, slots, _ = self.make_handler(native=True)
        output = asyncio.run(handler({"image_url": "worksheet.png", "question": "q"}))
        self.assertEqual(output, "native")
        self.assertEqual(calls, [])
        self.assertEqual(slots, {})


if __name__ == "__main__":
    unittest.main()
