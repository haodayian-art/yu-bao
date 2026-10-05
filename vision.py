import json
import time
import cv2
import cv2.aruco as aruco

TARGET_IDS = {1: "紅茶", 2: "珍珠", 3: "空杯區"}


def scan_and_save_targets(output_path="targets.json"):
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    targets = {}
    print("[視覺系統] 開始掃描環境，請將標記 1、2、3 放入視野...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            corners, ids, _ = detector.detectMarkers(frame)

            if ids is not None and len(ids) > 0:
                ids_flat = ids.flatten()
                for i, raw_id in enumerate(ids_flat):
                    mid = int(raw_id)
                    if mid in TARGET_IDS:
                        marker_corners = corners[i][0]
                        cx = int(sum(pt[0] for pt in marker_corners) / 4)
                        cy = int(sum(pt[1] for pt in marker_corners) / 4)
                        targets[mid] = {"name": TARGET_IDS[mid], "x": cx, "y": cy}

                        # 畫中心與框線
                        cv2.polylines(
                            frame,
                            [marker_corners.astype(int)],
                            True,
                            (0, 255, 0),
                            2,
                        )
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.putText(
                            frame,
                            f"ID:{mid} ({cx},{cy})",
                            (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

            cv2.imshow("ArUco Scanner", frame)
            cv2.waitKey(1)

            # 當 1, 2, 3 全部辨識完畢
            if len(targets) == 3:
                print("[視覺系統] 成功抓取 1、2、3 座標！")
                time.sleep(1.0)
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    # 儲存座標至檔案
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(targets, f, ensure_ascii=False, indent=4)
    print(f"[視覺系統] 已將目標座標寫入 {output_path}")


if __name__ == "__main__":
    scan_and_save_targets()