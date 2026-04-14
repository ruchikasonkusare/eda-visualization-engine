import os
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


REPORT_FOLDER = "app/static/reports"
os.makedirs(REPORT_FOLDER, exist_ok=True)


def generate_pdf_report(dataset_id, summary, visualizations):
    file_path = os.path.join(
        REPORT_FOLDER,
        f"{dataset_id}_eda_report.pdf"
    )

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    story = []

    # =====================================================
    # TITLE
    # =====================================================
    story.append(
        Paragraph(
            f"<b>EDA Report - Dataset {dataset_id}</b>",
            styles["Title"]
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    # =====================================================
    # DATASET SUMMARY
    # =====================================================
    story.append(Paragraph("<b>Dataset Summary</b>", styles["Heading2"]))
    story.append(
        Paragraph(f"Rows: {summary['rows']}", styles["BodyText"])
    )
    story.append(
        Paragraph(f"Columns: {summary['columns']}", styles["BodyText"])
    )
    story.append(
        Paragraph(
            f"Duplicate Rows: {summary['duplicated_rows']}",
            styles["BodyText"]
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    # =====================================================
    # CHARTS + INSIGHTS
    # =====================================================
    story.append(
        Paragraph("<b>Visualization Insights</b>", styles["Heading2"])
    )

    for item in visualizations:
        story.append(
            Paragraph(
                f"<b>Chart:</b> {item['chart']}",
                styles["BodyText"]
            )
        )

        story.append(
            Paragraph(
                f"<b>Columns:</b> {', '.join(item['columns'])}",
                styles["BodyText"]
            )
        )

        story.append(
            Paragraph(
                f"<b>Insight:</b> {item.get('insight', '')}",
                styles["BodyText"]
            )
        )

        # add image if exists
        image_path = item["image_url"].replace("/static/", "app/static/")
        if os.path.exists(image_path):
            story.append(Image(image_path, width=5 * inch, height=3 * inch))

        story.append(Spacer(1, 0.3 * inch))

    doc.build(story)

    return f"/static/reports/{dataset_id}_eda_report.pdf"