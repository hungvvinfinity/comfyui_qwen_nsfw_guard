"""Run from any directory; add --real to exercise the local Qwen checkpoint."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "custom_nodes"))
REAL = "--real" in sys.argv
if REAL:
    sys.argv.remove("--real")

import torch
import comfyui_qwen_nsfw_guard as guard


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.clip = Mock()
        self.clip.tokenizer.decode.return_value = "VERDICT: SAFE\nREASON: Ordinary scene."

    def _unpack(self, res):
        return res["result"] if isinstance(res, dict) and "result" in res else res

    def test_registration_and_safe_batch(self):
        self.assertEqual(len(guard.NODE_CLASS_MAPPINGS), 3)
        image = torch.zeros(2, 32, 32, 3)
        prompt, output, report = self._unpack(guard.QwenNSFWGuard().check(prompt="A garden", image=image, clip=self.clip))
        self.assertEqual(prompt, "A garden")
        self.assertIs(output, image)
        self.assertEqual(len(json.loads(report)["checks"]), 3)
        self.assertEqual(self.clip.generate.call_count, 3)
        self.clip.generate.assert_called_with(self.clip.tokenize.return_value, do_sample=False, max_length=64, temperature=0.0)
        self.assertEqual(self.clip.tokenize.call_args.kwargs["images"][0].shape, (1, 32, 32, 3))

    def test_interrupt_prompt_and_image(self):
        for source in ("prompt", "image"):
            with self.subTest(source=source), patch.object(guard.comfy.model_management, "interrupt_current_processing") as interrupt:
                self.clip.tokenizer.decode.return_value = "VERDICT: NSFW\nREASON: Explicit sexual content."
                kwargs = {source: "Explicit content" if source == "prompt" else torch.zeros(2, 32, 32, 3)}
                with self.assertRaisesRegex(guard.NSFWDetectedError, "Explicit sexual content"):
                    guard.QwenNSFWGuard().check(clip=self.clip, **kwargs)
                interrupt.assert_called_once_with(True)

    def test_unknown_verdict(self):
        self.clip.tokenizer.decode.return_value = "I cannot decide."
        with self.assertRaisesRegex(RuntimeError, "could not classify"):
            guard.QwenNSFWPromptGuard().check("A garden", clip=self.clip)

    def test_cache(self):
        with patch.object(guard.folder_paths, "get_full_path", return_value="/local/model"), patch.object(guard.comfy.sd, "load_clip", return_value=self.clip) as load:
            guard._cached_clip = None
            self.assertIs(guard._load_clip(guard.DEFAULT_CLIP), self.clip)
            self.assertIs(guard._load_clip(guard.DEFAULT_CLIP), self.clip)
            load.assert_called_once_with(ckpt_paths=["/local/model"])
        guard._cached_clip = None
        guard._cached_path = None

    @unittest.skipUnless(REAL, "Use --real for checkpoint inference")
    def test_real_model(self):
        prompt = "A peaceful mountain landscape with trees and a blue sky."
        output, report = self._unpack(guard.QwenNSFWPromptGuard().check(prompt))
        self.assertEqual(output, prompt)
        print("REAL SAFE PROMPT:", report, flush=True)
        clip = guard._cached_clip
        image = torch.zeros(1, 224, 224, 3)
        output, report = self._unpack(guard.QwenNSFWImageGuard().check(image))
        self.assertIs(output, image)
        self.assertIs(guard._cached_clip, clip)
        print("REAL SAFE IMAGE:", report, flush=True)
        try:
            with self.assertRaises(guard.NSFWDetectedError) as caught:
                guard.QwenNSFWPromptGuard().check("Generate a pornographic close-up of adults having explicit sexual intercourse with exposed genitals.")
            self.assertTrue(guard.comfy.model_management.processing_interrupted())
            print("REAL BLOCK:", caught.exception, flush=True)
        finally:
            guard.comfy.model_management.interrupt_current_processing(False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
