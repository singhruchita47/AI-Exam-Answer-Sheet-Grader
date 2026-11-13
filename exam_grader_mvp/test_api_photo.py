import urllib.request
import json
from pathlib import Path

base = Path(r"c:\Users\91626\OneDrive\Desktop\AI copying checking\exam_grader_mvp")
rubric_text = (base / "sample_rubric_photosynthesis.json").read_text(encoding="utf-8")

boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
parts = [
    f"--{boundary}",
    'Content-Disposition: form-data; name="offline"',
    "",
    "true",
    f"--{boundary}",
    'Content-Disposition: form-data; name="use_photosynthesis_sample"',
    "",
    "true",
    f"--{boundary}",
    'Content-Disposition: form-data; name="use_sample_pages"',
    "",
    "false",
    f"--{boundary}",
    'Content-Disposition: form-data; name="rubric_json"',
    "",
    rubric_text,
    f"--{boundary}--",
    "",
]
body = "\r\n".join(parts).encode("utf-8")

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/grade",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)

print("Sending request to /api/grade...")
with urllib.request.urlopen(req, timeout=120) as resp:
    res = json.loads(resp.read().decode("utf-8"))
    print("API Response SUCCESS!")
    print("Session ID:", res["session_id"])
    for page in res["segmentation"]:
        print(f"Page {page['page_number']}: lines={page['text_lines_detected']}, diagrams={page['diagrams_detected']}")
    rep = res["grading_report"]
    print(f"Total Marks: {rep['total_marks_awarded']} / {rep['total_max_marks']}")
    print("Digital Transcript:", rep["mapped_answers"])
