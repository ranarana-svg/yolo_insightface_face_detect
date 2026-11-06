import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
os.environ["USE_NNPACK"] = "0"
# ---------------------- 配置 ----------------------
MODEL_PATH = "yolov8n-face.pt"
FACE_DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_db.json")
SKIP_FRAME = 2
THRESHOLD = 0.55  # 相似度阈值，越大越严格
# -------------------------------------------------

print("人脸库文件路径：", FACE_DB_FILE)

yolo_model = YOLO(MODEL_PATH)

face_app = FaceAnalysis(providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(320, 320))

try:
    with open(FACE_DB_FILE, "r", encoding="utf-8") as f:
        face_db = json.load(f)
except FileNotFoundError:
    face_db = {}


def sanitize_name(name: str) -> str:
    """清理文件名非法字符，避免保存失败"""
    return "".join(c for c in name if c not in r'\/:*?"<>|').strip()


def cos_sim(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def recognize(feat):
    best_score = 0.0
    best_name = "unknown"
    for db_name, db_feat in face_db.items():
        score = cos_sim(feat, db_feat)
        if score > best_score:
            best_score = score
            best_name = db_name
    if best_score >= THRESHOLD:
        return best_name, best_score
    return "unknown", best_score


def extract_face_embedding(crop):
    """从裁剪图中提取人脸 embedding，失败返回 None"""
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return None

    pad = 20
    padded = np.zeros((h + pad * 2, w + pad * 2, 3), dtype=crop.dtype)
    padded[pad:pad + h, pad:pad + w] = crop

    if min(padded.shape[:2]) < 112:
        scale = 112.0 / min(padded.shape[:2])
        new_w = max(int(padded.shape[1] * scale), 112)
        new_h = max(int(padded.shape[0] * scale), 112)
        padded = cv2.resize(padded, (new_w, new_h))

    faces = face_app.get(padded)
    if len(faces) == 0:
        return None
    emb = getattr(faces[0], "embedding", None)
    if emb is None:
        return None
    return emb.tolist()


def save_face_db():
    with open(FACE_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(face_db, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())


def save_single_face(name, emb):
    """单独保存一份以名字为前缀的人脸文件"""
    dir_path = os.path.dirname(FACE_DB_FILE)
    safe = sanitize_name(name)
    out = os.path.join(dir_path, f"{safe}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({name: emb}, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    return out


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

frame_cnt = 0
last_boxes = []  # (x1, y1, x2, y2, name)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_cnt += 1
    show_frame = frame.copy()
    h, w = frame.shape[:2]

    # 隔帧推理
    if frame_cnt % SKIP_FRAME == 0:
        results = yolo_model(frame, conf=0.4)
        new_boxes = []
        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                face_crop = frame[y1:y2, x1:x2]
                name = "unknown"
                emb = extract_face_embedding(face_crop)
                if emb is not None:
                    name, _ = recognize(emb)
                new_boxes.append((x1, y1, x2, y2, name))
        last_boxes = new_boxes

    for (x1, y1, x2, y2, name) in last_boxes:
        color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
        cv2.rectangle(show_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            show_frame,
            name,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    cv2.imshow("Face Recognition", show_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        if not last_boxes:
            print("❌ 当前画面没有检测到人脸")
            continue

        # 选面积最大的人脸入库
        best = max(last_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x1, y1, x2, y2, _ = best

        crop = frame[y1:y2, x1:x2].copy()
        print("crop size:", crop.shape)

        emb = extract_face_embedding(crop)
        if emb is None:
            print("❌ 没有提取到人脸特征")
            print("调试建议：")
            print("1. 人脸正对摄像头，不要侧脸、低头、遮挡")
            print("2. 光线不要太暗或背光")
            print("3. 人脸尽量大一些，靠近摄像头")
            print("4. 可尝试把 det_size 改为 (128,128) 或 (640,640)")
            continue

        print("请输入这个人名字：")
        new_name = input().strip()
        new_name = sanitize_name(new_name)
        if not new_name:
            print("❌ 名字不能为空")
            continue

        if new_name in face_db:
            print(f"⚠️ 名字【{new_name}】已存在，将覆盖旧特征")

        face_db[new_name] = emb
        save_face_db()
        single_path = save_single_face(new_name, emb)

        print(f"✅ {new_name} 录入成功！")
        print(f"总人脸库文件：{FACE_DB_FILE}")
        print(f"单人人脸文件：{single_path}")
        print(f"当前人脸库人数：{len(face_db)}")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()