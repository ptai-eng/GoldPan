"""Scan telemetry, finding schemas, and SARIF/JSON/Markdown serializers for GoldPan Repo Scan."""

import json
from typing import Any

from pydantic import BaseModel, Field

from ..core.schemas import Finding


class ScanCoverage(BaseModel):
    """Explicit coverage and residual unknowns tracking (rea-tool-design compliant)."""

    completeness: str = "complete"  # "complete" | "partial"
    residual_unknowns: list[str] = Field(default_factory=list)


class ScanReport(BaseModel):
    """Complete report produced by GoldPan Repo Context Firewall."""

    root_path: str
    files_scanned: int = 0
    files_skipped: int = 0
    findings: list[Finding] = Field(default_factory=list)
    clean_files: list[str] = Field(default_factory=list)
    suspicious_files: list[str] = Field(default_factory=list)
    coverage: ScanCoverage = Field(default_factory=ScanCoverage)
    highest_risk: float = 0.0
    is_suspicious: bool = False
    duration_ms: float = 0.0

    def to_json(self, indent: int = 2) -> str:
        data = self.model_dump()
        data["_meta"] = {
            "provenance": {
                "untrusted_content": True,
                "source": self.root_path,
                "tool": "goldpan_scan",
            }
        }
        return json.dumps(data, ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        lines = [
            "# GoldPan Repo Context Firewall Report",
            "",
            f"- **Target Root:** `{self.root_path}`",
            f"- **Files Scanned:** `{self.files_scanned}`",
            f"- **Files Skipped:** `{self.files_skipped}`",
            f"- **Status:** {'🚨 SUSPICIOUS' if self.is_suspicious else '✅ CLEAN'}",
            f"- **Highest Risk Score:** `{self.highest_risk:.2f}`",
            f"- **Coverage:** `{self.coverage.completeness}`",
        ]
        if self.coverage.residual_unknowns:
            lines.append(
                f"- **Residual Unknowns:** {len(self.coverage.residual_unknowns)} skipped items"
            )
            for u in self.coverage.residual_unknowns[:5]:
                lines.append(f"  - {u}")
            if len(self.coverage.residual_unknowns) > 5:
                lines.append(
                    f"  - *(and {len(self.coverage.residual_unknowns) - 5} more...)*"
                )

        lines.extend(["", "---", "", "## Security Findings", ""])
        if not self.findings:
            lines.append(
                "No prompt injection or steganographic stealth payloads detected."
            )
        else:
            lines.extend(
                [
                    "| File | Line | Signal | Severity | Risk | Snippet |",
                    "| :--- | :---: | :--- | :---: | :---: | :--- |",
                ]
            )
            for f in self.findings:
                clean_snippet = f.snippet.replace("|", "\\|").replace("\n", " ").strip()
                if len(clean_snippet) > 60:
                    clean_snippet = clean_snippet[:57] + "..."
                lines.append(
                    f"| `{f.path}` | `{f.line}` | `{f.signal}` | `{f.severity}` | `{f.risk_score:.2f}` | `{clean_snippet}` |"
                )

        return "\n".join(lines)

    def to_sarif(self) -> dict[str, Any]:
        """Convert findings into SARIF 2.1.0 format for GitHub Code Scanning."""
        rule_registry: dict[str, dict[str, Any]] = {}
        results = []

        for f in self.findings:
            if f.signal not in rule_registry:
                rule_registry[f.signal] = {
                    "id": f.signal,
                    "name": f.signal.replace("_", " ").title().replace(" ", ""),
                    "shortDescription": {
                        "text": f"Potential indirect prompt injection via {f.signal}"
                    },
                    "defaultConfiguration": {
                        "level": "error" if f.severity == "suspicious" else "note"
                    },
                }

            results.append(
                {
                    "ruleId": f.signal,
                    "level": "error" if f.severity == "suspicious" else "note",
                    "message": {
                        "text": f"Detected {f.signal} (risk: {f.risk_score:.2f}): {f.snippet}"
                    },
                    "locations": [
                        {
                            "physicalLocation": {
                                "artifactLocation": {
                                    "uri": f.path.replace("\\", "/"),
                                    "uriBaseId": "%SRCROOT%",
                                },
                                "region": {
                                    "startLine": max(1, f.line),
                                    "startColumn": max(1, f.column),
                                },
                            }
                        }
                    ],
                }
            )

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "GoldPan Repo Context Firewall",
                            "version": "3.3.0",
                            "informationUri": "https://github.com/ptai-eng/GoldPan",
                            "rules": list(rule_registry.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }
