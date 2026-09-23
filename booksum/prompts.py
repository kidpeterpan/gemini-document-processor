"""Versioned prompt registry.

Every prompt has a *name* and a *version*. The version is recorded on every
cached result and generated artifact, and is part of the idempotency key, so
editing a template's text REQUIRES bumping its version (feature 002 R4).

Templates use ``string.Template`` syntax (``$name``) so JSON braces inside the
prompt body stay literal.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Template

from .config import DEFAULT_PROMPT_VERSIONS


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    template: str


_UNIT_V1 = """คุณคือ expert ด้าน summarizer

ช่วยสรุปเนื้อหา**จาก$doc_type** ($doc_filename) **เป็นภาษาไทย** โดย:
1. ใช้หัวข้อเดิมตามไฟล์ได้เลย (ไม่ต้องแปลหัวข้อ)
2. อย่าให้ตกหล่นแม้แต่เรื่องเดียว (ขอแบบละเอียดจนไม่ต้องกลับไปอ่านต้นฉบับเลย)
3. บอกด้วยว่ากำลังสรุป$page_or_chapterไหนของไฟล์ เช่น <!-- 1 -->, <!-- 2 -->, <!-- end --> (use markdown comment)
4. ไม่จำเป็นต้องกระชับ และรักษาความถูกต้องของข้อมูลสำคัญ เนื้อหาสำคัญไม่ตกหล่น
5. ไม่ต้องแปล technical terminology จากภาษาอังกฤษให้เป็นภาษาไทย
6. ถ้าใน file มีตัวอย่าง code ก็ใส่มาให้ด้วย
7. output ใน format ที่ดีที่สุด

เนื้อหาต่อไปนี้มาจาก$page_or_chapter $start_ref ถึง$page_or_chapter $end_ref:

$text
"""

_SYNTHESIS_V1 = """You are an expert editor. You are given the per-section summaries of one
document titled "$doc_title". Produce a whole-document synthesis in Thai.

Return ONLY one JSON object (no markdown fences, no commentary) with exactly:
- "overview": Thai prose synthesizing the whole document. Consolidate recurring
  themes across sections; do NOT restate each section in order.
- "key_ideas": array of Thai strings. The most important ideas, deduplicated.
- "glossary": array of {"term": string, "definition": string}. Keep English
  technical terms in English.

Rules:
- Do not invent content absent from the summaries.
- Merge duplicated material into a single statement.

Section summaries:
$summaries
"""

_COVERAGE_V1 = """You are a strict fact auditor. For each section you are given the SOURCE text
and the SUMMARY produced from it.

For each section, list the salient facts present in the SOURCE but ABSENT from
the SUMMARY. Be conservative: paraphrase is not an omission, and only include
facts a reader would need.

Return ONLY one JSON object (no markdown fences, no commentary):
{"units": [{"label": string, "missing": [string], "score": number}]}

"score" is your judgement of coverage in [0,1]; 1 means nothing salient is missing.

Sections:
$units
"""

_TEMPLATES: dict[str, PromptTemplate] = {
    "unit": PromptTemplate("unit", DEFAULT_PROMPT_VERSIONS["unit"], _UNIT_V1),
    "synthesis": PromptTemplate("synthesis", DEFAULT_PROMPT_VERSIONS["synthesis"], _SYNTHESIS_V1),
    "coverage": PromptTemplate("coverage", DEFAULT_PROMPT_VERSIONS["coverage"], _COVERAGE_V1),
}


class PromptRegistry:
    """Look up and render versioned prompt templates."""

    def __init__(self, templates: dict[str, PromptTemplate] | None = None) -> None:
        self._templates = dict(templates or _TEMPLATES)

    def get(self, name: str) -> PromptTemplate:
        try:
            return self._templates[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"unknown prompt: {name!r}") from exc

    def render(self, name: str, **kwargs: object) -> str:
        template = self.get(name)
        try:
            return Template(template.template).substitute(**kwargs)
        except KeyError as exc:
            raise ValueError(f"missing placeholder {exc.args[0]!r} for prompt {name!r}") from exc

    def versions(self) -> dict[str, str]:
        return {name: template.version for name, template in self._templates.items()}
