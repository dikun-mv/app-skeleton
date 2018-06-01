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

async function loadImage(url) {
  const image = new Image();
  image.src = url;
  await new Promise((resolve) => image.onload = resolve);
  return image;
}

export async function loadAssets(names) {
  return Promise.all(names.map((name) => loadImage(`./assets/${name}`)));
}

function partToColor(part) {
  const map = {
    'rightWrist': '#ff0000',
    'leftWrist': '#0000ff',
  };

  return map[part] || '#00ff00';
}

export function drawFrame(ctx, frame, size) {
  ctx.clearRect(0, 0, size.width, size.height);
  ctx.drawImage(frame, 0, 0, size.width, size.height);
}

export function drawKeypoints(ctx, keypoints, minScore, scale) {
  for (const { score, position, part } of keypoints) {
    if (score < minScore) continue;

    ctx.beginPath();
    ctx.arc(position.x * scale.x, position.y * scale.y, 10, 0, 2 * Math.PI);
    ctx.fillStyle = partToColor(part)
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
