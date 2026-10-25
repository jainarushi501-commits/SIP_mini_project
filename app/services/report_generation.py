import json
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from app.config import settings
from app.schemas import AnalysisReport

def generate_pdf_report(report: AnalysisReport) -> str:
    """
    Generates an official, publication-quality forensic PDF summary.
    Returns relative web path to the generated PDF.
    """
    pdf_filename = f"forensic_report_{report.analysis_id}.pdf"
    pdf_path = settings.REPORTS_DIR / pdf_filename

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a')
    )
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )
    verdict_style = ParagraphStyle(
        'Verdict',
        parent=styles['Normal'],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#dc2626') if report.verdict == "POTENTIAL_MANIPULATION" else colors.HexColor('#166534'),
        fontName='Helvetica-Bold'
    )

    elements = []

    # Title & Metadata
    elements.append(Paragraph("FORENSIGHT DIGITAL IMAGE FORENSICS REPORT", title_style))
    elements.append(Paragraph(f"Analysis ID: <b>{report.analysis_id}</b> &bull; Generated: {report.created_at}", body_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#cbd5e1')))
    elements.append(Spacer(1, 10))

    # Image Details Table
    elements.append(Paragraph("IMAGE SPECIFICATIONS & INTEGRITY", section_style))
    meta_data = [
        ["Filename:", report.image_metadata.filename, "SHA-256 Hash:", report.image_metadata.sha256[:28] + "..."],
        ["Dimensions:", f"{report.image_metadata.width} x {report.image_metadata.height} px", "File Format:", report.image_metadata.format],
        ["Color Mode:", report.image_metadata.color_mode, "Payload Size:", f"{round(report.image_metadata.file_size_bytes / 1024, 1)} KB"]
    ]
    meta_table = Table(meta_data, colWidths=[90, 175, 95, 170])
    meta_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#334155')),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 12))

    # Overall Evidence Verdict
    elements.append(Paragraph("OVERALL FORENSIC VERDICT", section_style))
    elements.append(Paragraph(f"Verdict: {report.verdict.value}", verdict_style))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(f"<b>Summary:</b> {report.verdict_summary}", body_style))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(f"<b>Provenance Check (C2PA):</b> {report.provenance_status.value}", body_style))
    elements.append(Spacer(1, 12))

    # Module Results Table
    elements.append(Paragraph("FORENSIC MODULE SUMMARY", section_style))
    module_rows = [["Forensic Analysis Module", "Execution Status", "Findings & Verdict Contribution"]]
    for name, mod in report.module_results.items():
        module_rows.append([
            mod.module_name,
            mod.status.value,
            mod.verdict_contribution or "N/A"
        ])

    mod_table = Table(module_rows, colWidths=[180, 110, 240])
    mod_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    elements.append(mod_table)
    elements.append(Spacer(1, 14))

    # Technical Limitations & Legal Disclaimer
    elements.append(Paragraph("LIMITATIONS & SCIENTIFIC INTEGRITY STATEMENT", section_style))
    elements.append(Paragraph(report.disclaimer, body_style))

    doc.build(elements)
    return f"/reports/{pdf_filename}"
