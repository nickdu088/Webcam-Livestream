from aiohttp import web
import cv2
import asyncio
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
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.js" crossorigin="anonymous"></script>
    <title>Webcam Live Streaming</title>
</head>
<body>

<img id="video" src="/video_feed" style="display:none;">
<div class="container">
    <div class="row">
        <div class="col-lg-8  offset-lg-2">
            <h3 class="mt-5">Webcam Live Streaming</h3>
            <canvas id="canvas" width="720" height="540" alt="Live Video Feed"></canvas>
            <!-- img src="/video_feed" width="100%" height="auto" class="img-fluid" alt="Live Video Feed" -->
        </div>
    </div>
</div>

<script type="module">
import { ObjectDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const vision = await FilesetResolver.forVisionTasks(
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
);
const detector = await ObjectDetector.createFromOptions(vision, {
  baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0_uint8.tflite" }
});

video.onload = async function draw() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const result = await detector.detect(canvas);
  // draw on canvas
  for (const det of result.detections) {
    const box = det.boundingBox;
    ctx.strokeStyle = "red";
    ctx.strokeRect(box.originX, box.originY, box.width, box.height);
    ctx.strokeText(
      `${det.categories[0].categoryName} (${(det.categories[0].score * 100).toFixed(1)}%)`,
      box.originX,
      box.originY > 10 ? box.originY - 5 : 10
    );
  }
  requestAnimationFrame(draw);
};
</script>

</body>
</html>
"""
executor = ThreadPoolExecutor(max_workers=2)

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
    cap = await loop.run_in_executor(executor, lambda: cv2.VideoCapture(0))

    async def get_frame():
        return await loop.run_in_executor(executor, cap.read)

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
            
            annotated_frame = frame
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

app = web.Application()
app.router.add_get('/', index)
app.router.add_get('/video_feed', video_feed)

if __name__ == '__main__':
    web.run_app(app, port=8080)
