import * as tf from '@tensorflow/tfjs';
import * as posenet from '@tensorflow-models/posenet';
import * as axios from 'axios';
import Stats from 'stats.js';

import { captureWebcam, loadAssets, drawFrame } from './utils';

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

  const [network, video, assets] = await Promise.all([
    posenet.load(posenetVersion),
    captureWebcam(frameSize),
    loadAssets(['face.png', 'left.png', 'right.png'])
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

      const box = posenet.getBoundingBoxPoints(keypoints);
      const heightRate = Math.trunc((box[2].y - box[1].y) / frameSize.height * 10) / 10;

      if (keypoints[0].score > minPartScore) {
        const assetWidth = Math.trunc(150 * heightRate / 10) * 10;
        const assetHeight = Math.trunc(150 * heightRate / 10) * 10;

        octx.drawImage(
          assets[0],
          keypoints[0].position.x * scaleRate.x - assetWidth / 2,
          keypoints[0].position.y * scaleRate.y - assetHeight / 2,
          assetWidth,
          assetHeight
        );
      }

      if (keypoints[9].score > minPartScore) {
        const vector = [
          (keypoints[9].position.x - keypoints[5].position.x) / -frameSize.width,
          (keypoints[9].position.y - keypoints[5].position.y) / -frameSize.height
        ];

        if (vector[1] > 0) {
          const assetWidth = Math.trunc(150 * heightRate / 10) * 10;
          const assetHeight = Math.trunc(150 * heightRate / 10) * 10;
          const handOffset = Math.trunc(50 * heightRate / 10) * 10;

          octx.drawImage(
            assets[1],
            keypoints[9].position.x * scaleRate.x - assetWidth / 2,
            keypoints[9].position.y * scaleRate.y - assetHeight / 2 - handOffset,
            assetWidth,
            assetHeight
          );
        }
      }

      if (keypoints[10].score > minPartScore) {
        const vector = [
          (keypoints[10].position.x - keypoints[6].position.x) / -frameSize.width,
          (keypoints[10].position.y - keypoints[6].position.y) / -frameSize.height
        ];

        if (vector[1] > 0) {
          const assetWidth = Math.trunc(150 * heightRate / 10) * 10;
          const assetHeight = Math.trunc(150 * heightRate / 10) * 10;
          const handOffset = Math.trunc(50 * heightRate / 10) * 10;

          octx.drawImage(
            assets[2],
            keypoints[10].position.x * scaleRate.x - assetWidth / 2,
            keypoints[10].position.y * scaleRate.y - assetHeight / 2 - handOffset,
            assetWidth,
            assetHeight
          );
        }
      }
    }

    stats.end();
  }
}

window.onload = main;
