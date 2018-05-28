import * as posenet from '@tensorflow-models/posenet';

export async function captureWebcam(size) {
  const video = document.createElement('video');
  video.width = size.width;
  video.height = size.height;
  video.srcObject = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
  await new Promise((resolve) => video.onloadedmetadata = resolve);
  video.play();
  return video;
}

export function drawFrame(ctx, frame, size) {
  ctx.clearRect(0, 0, size.width, size.height);
  ctx.save();
  ctx.scale(-1, 1);
  ctx.translate(-size.width, 0);
  ctx.drawImage(frame, 0, 0, size.width, size.height);
  ctx.restore();
}

export function drawKeypoints(ctx, keypoints, minScore, scale) {
  for (const { score, position } of keypoints) {
    if (score < minScore) continue;

    ctx.beginPath();
    ctx.arc(position.x * scale.x, position.y * scale.y, 3, 0, 2 * Math.PI);
    ctx.fillStyle = 'red';
    ctx.fill();
  }
}

export function drawSkeleton(ctx, keypoints, minScore, scale) {
  const points = posenet.getAdjacentKeyPoints(keypoints, minScore);

  for (const [{ position: from }, { position: to }] of points) {
    ctx.beginPath();
    ctx.moveTo(from.x * scale.x, from.y * scale.y);
    ctx.lineTo(to.x * scale.x, to.y * scale.y);
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'red';
    ctx.stroke();
  }
}
