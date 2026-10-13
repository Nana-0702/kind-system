from pathlib import Path
import numpy as np
from django.conf import settings
 
 
def predict_feature(feature, active_ids):
    path = Path(settings.FACE_GALLERY_PATH)
    if not path.is_file():
        raise FileNotFoundError("先に実行してください build_face_gallery")
    with np.load(path, allow_pickle=False) as data:
        features = data["features"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
    if (features.ndim != 2 or labels.ndim != 1
            or len(features) != len(labels)
            or features.shape[1] != feature.size):
        raise ValueError("特徴量データベースの形式が一致しません。再構築してください")
    mask = np.isin(labels, list(active_ids))
    features, labels = features[mask], labels[mask]
    if not len(labels):
        raise ValueError("特徴量データベースに在園児が含まれていません。再構築してください")
    norms = np.linalg.norm(features, axis=1)
    if (not np.isfinite(features).all() or np.any(norms <= 0)):
        raise ValueError("特徴量データ破損。再構築してください")
    query = np.asarray(feature, dtype=np.float32).reshape(-1)
    query_norm = np.linalg.norm(query)
    if not np.isfinite(query).all() or query_norm <= 0:
        raise ValueError("照合特徴量無効")
    # 单位向量的点积就是余弦相似度。
    scores = (features / norms[:, None]) @ (query / query_norm)
    # 每名学生取其照片的最高分，再比较不同学生。
    ranked = sorted(
        ((int(label), float(scores[labels == label].max()))
         for label in np.unique(labels)),
        key=lambda item: item[1], reverse=True)
    label, score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else None
    threshold = float(settings.FACE_COSINE_THRESHOLD)
    margin = float(settings.FACE_COSINE_MARGIN)
    accepted = score >= threshold
    if second is not None:
        accepted = accepted and score - second >= margin
    gap = None if second is None else score - second
    if score < threshold:
        reason = "LOW_SIMILARITY"
    elif gap is not None and gap < margin:
        reason = "AMBIGUOUS"
    else:
        reason = "MATCHED"

    diagnostics = {
        "best_score": score,
        "second_score": second,
        "gap": gap,
        "threshold": threshold,
        "margin": margin,
        "reason": reason,
    }
    return label, score, accepted, diagnostics

