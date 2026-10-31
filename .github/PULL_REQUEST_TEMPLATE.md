## Summary of Changes
<!-- Describe the motivation, architectural design, and primary changes introduced in this PR. -->

## Changes Introduced
- [ ] Backend / Engine change:
- [ ] Firewall / Security rule change:
- [ ] OCR / Extractor change:
- [ ] Frontend UI / Dashboard change:
- [ ] Benchmark / Test suite addition:

## Pre-Merge Verification Checklist
Please verify the following before requesting review:
- [ ] **Hardware Diagnostics:** `python -m goldpan.cli doctor` passes with valid Tier detection.
- [ ] **Unit & Security Tests:** `python run_tests.py` and `python backend/tests/test_api.py` pass 100%.
- [ ] **Dual-Track CI Benchmark Gate:** `python -m goldpan.cli bench --ci-gate` passes (Recall@3 >= 90%, Firewall TPR >= 95%, Hard-Negative FPR <= 5%).
- [ ] **No Secret Leaks:** No Gemini API keys, tokens, or untrusted credentials committed.
- [ ] **Commit Messages:** Follow imperative semantic commit convention (e.g., `feat: ...`, `fix: ...`).
