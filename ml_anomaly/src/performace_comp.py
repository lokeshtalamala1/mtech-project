from PIL import Image, ImageDraw, ImageFont
import textwrap, os

W, H = 1800, 1200
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

title_font = ImageFont.load_default()
text_font = ImageFont.load_default()

for path in [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
]:
    if os.path.exists(path):
        title_font = ImageFont.truetype(path, 28)
        text_font = ImageFont.truetype(path, 20)
        break

title = "Performance Comparison of Anomaly Detection Algorithms"
table_lines = [
"+-------------------+----------------------+------------------+-----------+--------+----------+---------+---------+----------------------------------------------+",
"| Model             | Type                 | Eval Level       | Precision | Recall | F1-Score | ROC-AUC | PR-AUC  | Remark                                       |",
"+-------------------+----------------------+------------------+-----------+--------+----------+---------+---------+----------------------------------------------+",
"| Isolation Forest  | Traditional ML       | Row-level        | 0.5085    | 1.0000 | 0.6742   | 0.9964  | 0.8525  | Strong baseline; best practical model.       |",
"+-------------------+----------------------+------------------+-----------+--------+----------+---------+---------+----------------------------------------------+",
"| One-Class SVM     | Traditional ML       | Row-level        | 0.4839    | 1.0000 | 0.6522   | 0.9990  | 0.9447  | Strong ranking; slightly more false alerts.  |",
"+-------------------+----------------------+------------------+-----------+--------+----------+---------+---------+----------------------------------------------+",
"| LOF               | Traditional ML       | Row-level        | 0.0000    | 0.0000 | 0.0000   | 0.9159  | 0.1084  | Poor fit for repeated-flow data.             |",
"+-------------------+----------------------+------------------+-----------+--------+----------+---------+---------+----------------------------------------------+",
"| Autoencoder       | Deep Learning        | Row-level        | 0.5085    | 1.0000 | 0.6742   | 0.9975  | 0.8925  | Best row-level deep model.                   |",
"+-------------------+----------------------+------------------+-----------+--------+----------+---------+---------+----------------------------------------------+",
"| LSTM Autoencoder  | Deep Learning / UEBA | Sequence-level   | 0.7611    | 0.7049 | 0.7319   | 0.7948  | 0.7393  | Best temporal model; highest F1-score.       |",
"+-------------------+----------------------+------------------+-----------+--------+----------+---------+---------+----------------------------------------------+",
"| GraphSAGE         | Graph-based ML       | Node-level       | 0.0000    | 0.0000 | 0.0000   | 0.6471  | 0.1964  | Limited by small graph and few anomalies.    |",
"+-------------------+----------------------+------------------+-----------+--------+----------+---------+---------+----------------------------------------------+",
]

remark = (
    "Final Remark: Isolation Forest is the best practical baseline, while "
    "LSTM Autoencoder is the best advanced behavioral model for UEBA-oriented anomaly detection."
)

y = 30
draw.text((40, y), title, fill="black", font=title_font)
y += 60

for line in table_lines:
    draw.text((40, y), line, fill="black", font=text_font)
    y += 34

y += 20
for line in textwrap.wrap(remark, width=110):
    draw.text((40, y), line, fill="black", font=text_font)
    y += 32

out = "outputs/performance_comparison_table.png"
img.save(out)
print(out)
