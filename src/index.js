import * as tf from '@tensorflow/tfjs';
import * as posenet from '@tensorflow-models/posenet';
import * as axios from 'axios';
import Stats from 'stats.js';

import { captureWebcam, drawFrame, drawKeypoints, drawSkeleton } from './utils';

async function main() {
  const {
    data: {
      posenetVersion,
      imageScaleFactor,
      outputStride,
      minPoseScore,
      minPartScore,
      maxPoseDetections,
      nmsRadius,
      frameSize
    }
  } = await axios.get('./config.json');

  const viewportSize = {
    width: window.innerWidth,
    height: window.innerHeight
  };

  const scaleRate = {
    x: viewportSize.width / frameSize.width,
    y: viewportSize.height / frameSize.height * frameSize.height / frameSize.width
  };

  const [network, video] = await Promise.all([
    posenet.load(posenetVersion),
    captureWebcam(frameSize)
  ]);

  const icanvas = document.createElement('canvas');
  icanvas.width = frameSize.width;
  icanvas.height = frameSize.height;
  const ictx = icanvas.getContext('2d');

  const ocanvas = document.getElementById('output');
  ocanvas.width = viewportSize.width;
  ocanvas.height = viewportSize.height;
  const octx = ocanvas.getContext('2d');

  octx.scale(-1, 1);
  octx.translate(-viewportSize.width, 0);

  const stats = new Stats();
  stats.showPanel(0);
  document.body.appendChild(stats.dom);

  const box = document.getElementById('box');
  box.addEventListener('show', ({ detail: { type } }) => {
    if (!box.style.display) {
      box.src = `./${type}.gif`;
      box.style.display = 'block';

      setTimeout(() => box.style.display = '', 2000);
    }
  });

  let counter = 0;

  while (true) {
    await new Promise((resolve) => requestAnimationFrame(resolve));

    stats.begin();

    drawFrame(ictx, video, frameSize);
    const poses = await network.estimateMultiplePoses(
      icanvas, imageScaleFactor, false, outputStride, maxPoseDetections, minPartScore, nmsRadius
    );
    drawFrame(octx, video, viewportSize);

    for (const { score, keypoints } of poses) {
      if (score < minPoseScore) continue;

      const points = keypoints.slice(5, 11);

      if (counter % 2 === 0) {
        const L = [
          (points[4].position.x - points[0].position.x) / -frameSize.width,
          (points[4].position.y - points[0].position.y) / -frameSize.height
        ];

        const R = [
          (points[5].position.x - points[1].position.x) / -frameSize.width,
          (points[5].position.y - points[1].position.y) / -frameSize.height
        ];

        if (L[1] > 0 && R[1] > 0) continue;

        if (L[1] > 0) {
          box.dispatchEvent(new CustomEvent('show', { detail: { type: 'left' } }));
          continue;
        }

        if (R[1] > 0) {
          box.dispatchEvent(new CustomEvent('show', { detail: { type: 'right' } }));
          continue;
        }
      }

      drawKeypoints(octx, points, minPartScore, scaleRate);
    }

    counter += 1;

    stats.end();
  }
}

window.onload = main;
