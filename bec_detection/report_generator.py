"""
report_generator.py  —  PDF & CSV report export for BEC Detection System

Generates:
  - Single-email PDF report (detailed)
  - Batch CSV export of email history

Dependencies: reportlab, csv (stdlib)
"""

import csv
import io
import os
from datetime import datetime
from typing import List, Optional

# ── PDF generation via ReportLab ─────────────────────────────────────────────

def generate_pdf_report(email_result: dict) -> bytes:
    """
    Generate a detailed PDF report for a single analyzed email.
    Returns raw PDF bytes.
    Falls back to a plain-text pseudo-PDF if reportlab is unavailable.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, KeepTogether
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm
        )

        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "BECTitle", parent=styles["Title"],
            fontSize=20, textColor=colors.HexColor("#1e293b"),
            spaceAfter=4
        )
        subtitle_style = ParagraphStyle(
            "BECSub", parent=styles["Normal"],
            fontSize=10, textColor=colors.HexColor("#64748b"),
            spaceAfter=12
        )
        section_style = ParagraphStyle(
            "BECSection", parent=styles["Heading2"],
            fontSize=13, textColor=colors.HexColor("#0f172a"),
            spaceAfter=6, spaceBefore=14
        )
        body_style = ParagraphStyle(
            "BECBody", parent=styles["Normal"],
            fontSize=10, textColor=colors.HexColor("#334155"),
            leading=15
        )

        # Risk level colours
        RISK_COLORS_MAP = {
            "CRITICAL": colors.HexColor("#ef4444"),
            "HIGH":     colors.HexColor("#f97316"),
            "MEDIUM":   colors.HexColor("#eab308"),
            "LOW":      colors.HexColor("#22c55e"),
        }
        risk_level = email_result.get("risk_level", "LOW")
        risk_color = RISK_COLORS_MAP.get(risk_level, colors.gray)

        story = []

        # ── Header ──────────────────────────────────────────────────────────
        story.append(Paragraph("🛡️ BEC Shield", title_style))
        story.append(Paragraph(
            "Business Email Compromise Detection Report", subtitle_style
        ))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  •  "
            f"Email ID: <b>{email_result.get('email_id', '—')}</b>",
            subtitle_style
        ))
        story.append(HRFlowable(width="100%", color=colors.HexColor("#e2e8f0")))
        story.append(Spacer(1, 0.3*cm))

        # ── Risk Summary table ───────────────────────────────────────────────
        story.append(Paragraph("Risk Summary", section_style))
        risk_score_pct = f"{email_result.get('risk_score', 0)*100:.1f}%"
        summary_data = [
            ["Field", "Value"],
            ["Risk Score",        risk_score_pct],
            ["Risk Level",        risk_level],
            ["Classification",    email_result.get("classification", "—")],
            ["Confidence",        f"{email_result.get('confidence', 0)*100:.1f}%"],
            ["BEC Type",          email_result.get("bec_type", "—")],
            ["Recommended Action",email_result.get("recommended_action", "—")],
            ["Analyzed At",       email_result.get("timestamp", "—")],
        ]
        tbl = Table(summary_data, colWidths=[5*cm, 11*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 10),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
                [colors.HexColor("#f8fafc"), colors.white]),
            ("TEXTCOLOR",   (1,1), (1,1), risk_color),
            ("FONTNAME",    (1,1), (1,1), "Helvetica-Bold"),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING",  (0,0), (-1,-1), 6),
            ("BOTTOMPADDING",(0,0),(-1,-1), 6),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.4*cm))

        # ── Individual model scores ──────────────────────────────────────────
        model_scores = email_result.get("model_scores", {})
        if model_scores:
            story.append(Paragraph("Individual Model Scores", section_style))
            model_labels = {
                "lr": "Logistic Regression",
                "xgb": "XGBoost",
                "gb": "Gradient Boosting",
                "svm": "SVM",
                "lightgbm": "LightGBM",
                "catboost": "CatBoost",
                "lstm": "Bi-LSTM",
                "iso_forest": "Isolation Forest"
            }
            ms_data = [["Model", "Score"]]
            for k, v in model_scores.items():
                ms_data.append([model_labels.get(k, k), f"{v*100:.1f}%"])
            ms_tbl = Table(ms_data, colWidths=[8*cm, 8*cm])
            ms_tbl.setStyle(TableStyle([
                ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
                ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",    (0,0), (-1,-1), 10),
                ("ROWBACKGROUNDS", (0,1), (-1,-1),
                    [colors.HexColor("#f8fafc"), colors.white]),
                ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING",  (0,0), (-1,-1), 5),
                ("BOTTOMPADDING",(0,0),(-1,-1), 5),
                ("LEFTPADDING", (0,0), (-1,-1), 8),
            ]))
            story.append(ms_tbl)
            story.append(Spacer(1, 0.4*cm))

        # ── Threat indicators / flags ────────────────────────────────────────
        flags = email_result.get("flags", [])
        story.append(Paragraph(
            f"Threat Indicators ({len(flags)} detected)", section_style
        ))
        if flags:
            flag_type_labels = {
                "header":      "📨 Header",
                "linguistic":  "🗣️ Linguistic",
                "social_eng":  "🎭 Social Eng.",
                "url":         "🔗 URL",
                "domain":      "🌐 Domain",
                "content":     "📝 Content",
                "thread":      "🧵 Thread",
                "threat_intel":"🛡️ Threat Intel",
            }
            flags_data = [["Type", "Finding"]]
            for flag in flags:
                flags_data.append([
                    flag_type_labels.get(flag["type"], flag["type"]),
                    flag["message"]
                ])
            fl_tbl = Table(flags_data, colWidths=[4.5*cm, 11.5*cm])
            fl_tbl.setStyle(TableStyle([
                ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#7f1d1d")),
                ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
                ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",    (0,0), (-1,-1), 9),
                ("ROWBACKGROUNDS", (0,1), (-1,-1),
                    [colors.HexColor("#fff1f2"), colors.HexColor("#fff8f8")]),
                ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#fecaca")),
                ("TOPPADDING",  (0,0), (-1,-1), 5),
                ("BOTTOMPADDING",(0,0),(-1,-1), 5),
                ("LEFTPADDING", (0,0), (-1,-1), 8),
                ("WORDWRAP",    (1,1), (1,-1), True),
            ]))
            story.append(fl_tbl)
        else:
            story.append(Paragraph(
                "✅ No threat indicators detected.", body_style
            ))
        story.append(Spacer(1, 0.4*cm))

        # ── SHAP top features (if available) ────────────────────────────────
        shap_features = email_result.get("top_shap_features", [])
        if shap_features:
            story.append(Paragraph("Top SHAP Feature Contributions", section_style))
            shap_data = [["Feature", "SHAP Impact"]]
            for sf in shap_features:
                shap_data.append([sf.get("feature","—"), f"{sf.get('impact',0):.4f}"])
            shap_tbl = Table(shap_data, colWidths=[10*cm, 6*cm])
            shap_tbl.setStyle(TableStyle([
                ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#1e3a5f")),
                ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
                ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",    (0,0), (-1,-1), 10),
                ("ROWBACKGROUNDS", (0,1), (-1,-1),
                    [colors.HexColor("#eff6ff"), colors.white]),
                ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#bfdbfe")),
                ("TOPPADDING",  (0,0), (-1,-1), 5),
                ("BOTTOMPADDING",(0,0),(-1,-1), 5),
                ("LEFTPADDING", (0,0), (-1,-1), 8),
            ]))
            story.append(shap_tbl)
            story.append(Spacer(1, 0.4*cm))

        # ── Footer ───────────────────────────────────────────────────────────
        story.append(HRFlowable(width="100%", color=colors.HexColor("#e2e8f0")))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            "Generated by BEC Shield v2.0 — AI-Based Business Email Compromise Detection System",
            ParagraphStyle("footer", parent=styles["Normal"],
                           fontSize=8, textColor=colors.HexColor("#94a3b8"),
                           alignment=TA_CENTER)
        ))

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes

    except ImportError:
        # Fallback: plain text if reportlab not installed
        lines = [
            "BEC Shield — Email Analysis Report",
            "=" * 50,
            f"Email ID:    {email_result.get('email_id','—')}",
            f"Risk Score:  {email_result.get('risk_score',0)*100:.1f}%",
            f"Risk Level:  {email_result.get('risk_level','—')}",
            f"BEC Type:    {email_result.get('bec_type','—')}",
            f"Action:      {email_result.get('recommended_action','—')}",
            f"Timestamp:   {email_result.get('timestamp','—')}",
            "",
            "Threat Indicators:",
        ]
        for flag in email_result.get("flags", []):
            lines.append(f"  [{flag['type']}] {flag['message']}")
        if not email_result.get("flags"):
            lines.append("  None detected.")
        lines.append("")
        lines.append("Generated by BEC Shield v2.0")
        return "\n".join(lines).encode("utf-8")


# ── CSV export ────────────────────────────────────────────────────────────────

def generate_csv_export(email_log: List[dict]) -> bytes:
    """
    Export the full email analysis history as a CSV file.
    Returns raw CSV bytes (UTF-8 with BOM for Excel compatibility).
    """
    buffer = io.StringIO()
    fieldnames = [
        "email_id", "timestamp", "subject_preview", "sender",
        "risk_score", "risk_level", "classification", "confidence",
        "bec_type", "recommended_action", "flag_count"
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for entry in email_log:
        row = {k: entry.get(k, "") for k in fieldnames}
        row["risk_score"]  = f"{entry.get('risk_score', 0)*100:.1f}%"
        row["confidence"]  = f"{entry.get('confidence', 0)*100:.1f}%"
        row["flag_count"]  = len(entry.get("flags", []))
        writer.writerow(row)

    # Return UTF-8 BOM + CSV content
    csv_content = buffer.getvalue()
    buffer.close()
    return ("\ufeff" + csv_content).encode("utf-8")
