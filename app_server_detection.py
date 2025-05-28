from aiohttp import web
import cv2
import asyncio
from ultralytics import YOLO
import logging
import timeit
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css"
          integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">

    <title>Webcam Live Streaming</title>
</head>
<body>
<div class="container">
    <div class="row">
        <div class="col-lg-8  offset-lg-2">
            <h3 class="mt-5">Webcam Live Streaming</h3>
            <img src="/video_feed" width="100%" height="auto" class="img-fluid" alt="Live Video Feed">
        </div>
    </div>
</div>
</body>
</html>
"""

model = YOLO("yolov8n.pt")
executor = ThreadPoolExecutor(max_workers=3)

async def index(request):
    return web.Response(text=HTML_PAGE, content_type='text/html')

async def video_feed(request):
    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'multipart/x-mixed-replace; boundary=frame'
        }
    )
    await response.prepare(request)

    loop = asyncio.get_event_loop()
    cap = await loop.run_in_executor(executor, lambda: cv2.VideoCapture(1))

    async def get_frame():
        return await loop.run_in_executor(executor, cap.read)

    async def detect(frame):
        return await loop.run_in_executor(executor, detect_objects_sync, frame)

    async def encode(frame):
        return await loop.run_in_executor(executor, lambda: cv2.imencode('.jpg', frame)[1].tobytes())

    try:
        prev_time = timeit.default_timer()
        fps = 0.0
        while True:
            t0 = timeit.default_timer()
            ret, frame = await get_frame()
            if not ret:
                await asyncio.sleep(0.01)
                continue
            
            annotated_frame = await detect(frame)
            t1 = timeit.default_timer()
            # FPS
            fps = 1.0 / (t1 - prev_time) if (t1 - prev_time) > 0 else 0.0
            prev_time = t1
            # 
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            jpg_bytes = await encode(annotated_frame)
            try:
                await response.write(
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n'
                )
            except (ConnectionResetError, asyncio.CancelledError):
                break
    finally:
        await loop.run_in_executor(executor, cap.release)
        try:
            await response.write_eof()
        except Exception:
            pass
    return response

def detect_objects_sync(frame):
    results = model(frame, verbose=False)
    annotated_frame = frame.copy()
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return annotated_frame

app = web.Application()
app.router.add_get('/', index)
app.router.add_get('/video_feed', video_feed)

if __name__ == '__main__':
    web.run_app(app, port=8080)
