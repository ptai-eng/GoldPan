"""Core codebase scanner orchestrating RepoWalker and AgentContextFirewall."""

import time
from pathlib import Path

from ..filters.vn_encoding import normalize_vietnamese_text
from ..firewall.context_firewall import AgentContextFirewall
from .report import Finding, ScanCoverage, ScanReport
from .walker import RepoWalker


class CodeScanner:
    """Orchestrates recursive repository scanning, code-profile firewalling, and report generation."""

    def __init__(
        self,
        root_dir: str | Path,
        mode: str = "audit",
        profile: str = "code",
        out_dir: str | Path | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_file_kb: int = 1024,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.mode = mode
        self.profile = profile
        self.out_dir = Path(out_dir).resolve() if out_dir else None
        self.include_patterns = include_patterns
        self.exclude_patterns = exclude_patterns
        self.max_file_kb = max_file_kb

        self.firewall = AgentContextFirewall(mode=self.mode, profile=self.profile)

    def scan(self) -> ScanReport:
        t0 = time.perf_counter()
        walker = RepoWalker(
            root_dir=self.root_dir,
            include_patterns=self.include_patterns,
            exclude_patterns=self.exclude_patterns,
            max_file_kb=self.max_file_kb,
        )

        all_findings: list[Finding] = []
        clean_files: list[str] = []
        suspicious_files: list[str] = []
        files_scanned = 0
        highest_risk = 0.0

        gen = walker.walk()
        while True:
            try:
                full_path, rel_posix = next(gen)
            except StopIteration as stop:
                files_skipped, residual_unknowns = stop.value if stop.value else (0, [])
                break

            try:
                raw_text = full_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                files_skipped += 1
                residual_unknowns.append(f"Failed to read file ({e}): {rel_posix}")
                continue

            files_scanned += 1
            # Normalize Vietnamese encodings if legacy fonts are present
            norm_text, _ = normalize_vietnamese_text(raw_text)

            # 1. Collect exact line-aware findings
            file_findings = self.firewall.scan_text_findings(
                norm_text, filename=rel_posix
            )
            all_findings.extend(file_findings)

            # 2. Process via firewall for sanitization / report telemetry
            cleansed_text, fw_rep = self.firewall.process_text(norm_text)
            if self.profile == "code":
                file_is_suspicious = any(
                    f.severity == "suspicious" for f in file_findings
                )
                file_risk = max(
                    (f.risk_score for f in file_findings if f.severity == "suspicious"),
                    default=0.0,
                )
                highest_risk = max(highest_risk, file_risk)
            else:
                file_is_suspicious = fw_rep.is_suspicious or any(
                    f.severity == "suspicious" for f in file_findings
                )
                highest_risk = max(highest_risk, fw_rep.risk_score)

            if file_is_suspicious:
                suspicious_files.append(rel_posix)
            else:
                clean_files.append(rel_posix)

            # 3. If mode is sanitize and out_dir is provided, write sanitized copy to out_dir
            if self.mode == "sanitize" and self.out_dir:
                target_file = self.out_dir / rel_posix
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_text(cleansed_text, encoding="utf-8")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        completeness = (
            "complete" if not residual_unknowns and files_skipped == 0 else "partial"
        )
        coverage = ScanCoverage(
            completeness=completeness,
            residual_unknowns=residual_unknowns,
        )

        return ScanReport(
            root_path=str(self.root_dir).replace("\\", "/"),
            files_scanned=files_scanned,
            files_skipped=files_skipped,
            findings=all_findings,
            clean_files=clean_files,
            suspicious_files=suspicious_files,
            coverage=coverage,
            highest_risk=round(highest_risk, 4),
            is_suspicious=len(suspicious_files) > 0,
            duration_ms=round(elapsed_ms, 2),
        )
