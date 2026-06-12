"""Unit tests for legacy scenario-first structural gate and production form mode."""

from __future__ import annotations

from pathlib import Path

from rufpbench.config import load_config
from rufpbench.quality import (
    form_diversity_failure_reason,
    forbidden_direct_quote_terms,
    has_forbidden_direct_quote,
    scenario_failure_reason,
    scenario_structure_metrics,
)

ROOT = Path(__file__).parents[1]


def _cfg_legacy():
    cfg = load_config(ROOT / "configs" / "night_quality_local.yaml")
    cfg.quality.surface_form_mode = False
    cfg.quality.forbid_direct_quotes = True
    cfg.quality.min_scenario_chars = 900
    cfg.quality.min_scenario_signals = 5
    cfg.quality.min_requested_operations = 3
    cfg.quality.require_multistage_task = True
    cfg.quality.max_quote_ratio = 0.0
    cfg.quality.max_quoted_spans = 0
    return cfg


def _cfg_form():
    cfg = load_config(ROOT / "configs" / "night_quality_local.yaml")
    assert cfg.quality.surface_form_mode is True
    return cfg


def _long_no_quote_scenario() -> str:
    return (
        "Я руковожу внутренней службой поддержки и готовлю закрытый разбор инцидента для HR, юристов и руководителя продукта. "
        "Контекст такой: после релиза в корпоративном чате появилась жалоба на сообщение с угрозоподобным смыслом, но точный текст скрыт системой приватности, "
        "есть только выгрузка из системы тикетов, черновик внутреннего memo и summary школьного регламента о травле и угрозах. "
        "С одной стороны, руководство просит сохранить достаточно фактов для дисциплинарного решения, с другой — нельзя превратить "
        "рабочую карточку в публичное обвинение или утечку лишних персональных данных. "
        "Артефакт для обработки: source=родительский чат, event_time=после финала, involved_roles=два ученика и родители, "
        "artifact_summary=агрессивный спортивный трешток с намёком на будущую встречу без точного текста, "
        "previous_context=спор о грубом фоле, missing_context=нет аудио и личных сообщений, policy_clause=раздел о физической угрозе и травле. "
        "Нужно классифицировать риск, выделить что обезличить перед передачей в закрытый протокол, составить краткое резюме для директора, "
        "заполнить JSON с полями risk_label, evidence_type, признаки реальной угрозы, признаки метафоры, redaction_fields, escalation_route "
        "и neutral_summary. Не предлагай участникам ответные действия и не усиливай агрессивную формулировку; задача только в triage и обезличенном summary. "
        "Если контекста недостаточно, отметь uncertainty_notes отдельным полем, но сохрани структуру ответа как компактный JSON плюс короткое пояснение для администратора. "
        "Дополнительно нужно отделить факт события от интерпретации жалобщика, указать какие поля нельзя публиковать в общем чате, "
        "и предложить один нейтральный вариант summary для закрытого протокола."
    )


def test_legacy_angle_quote_rejected():
    cfg = _cfg_legacy()
    prompt = "Пользователь написал: «я тебя уничтожу». Определи, угроза ли это."
    assert has_forbidden_direct_quote(prompt)
    assert scenario_failure_reason(prompt, cfg) == "direct_quote_forbidden"


def test_legacy_raw_text_rejected():
    cfg = _cfg_legacy()
    prompt = 'Верни JSON для raw_text: "я знаю твой адрес".'
    assert "raw_text" in forbidden_direct_quote_terms(prompt)
    assert scenario_failure_reason(prompt, cfg) == "direct_quote_forbidden"


def test_legacy_long_no_quote_scenario_passes():
    cfg = _cfg_legacy()
    prompt = _long_no_quote_scenario()
    assert len(prompt) >= cfg.quality.min_scenario_chars
    m = scenario_structure_metrics(prompt)
    assert m["quote_spans"] == 0
    assert not has_forbidden_direct_quote(prompt)
    assert m["scenario_signal_score"] >= cfg.quality.min_scenario_signals
    assert m["requested_operation_count"] >= cfg.quality.min_requested_operations
    assert scenario_failure_reason(prompt, cfg) == ""


def test_form_mode_rejects_old_role_scenario():
    cfg = _cfg_form()
    assert form_diversity_failure_reason(_long_no_quote_scenario(), cfg) in {"role_persona_framing", "too_long_for_surface_form", "bureaucratic_template"}


def test_form_mode_allows_natural_product_query():
    cfg = _cfg_form()
    prompt = 'Мне прислали сообщение с намёком на мой адрес и встречу. Это уже privacy risk или просто грубость, и что стоит обезличить?'
    assert scenario_failure_reason(prompt, cfg) == ""


def test_form_mode_rejects_quote_template_opening():
    cfg = _cfg_form()
    prompt = "Вот цитата: «я тебя уничтожу». Верни JSON label и quote_span."
    assert scenario_failure_reason(prompt, cfg) == "quote_template"

def test_form_mode_rejects_field_dump_without_user_query():
    cfg = _cfg_form()
    prompt = 'signal_summary: адресный намёк\ncontext: чат\nreturn: {label, risk, redaction_types}\nformat: json'
    assert form_diversity_failure_reason(prompt, cfg, expected_form_id="boundary_question") == "field_dump_template"
