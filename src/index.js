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

  const stats = new Stats();
  stats.showPanel(0);
  document.body.appendChild(stats.dom);

  while (true) {
    await new Promise((resolve) => requestAnimationFrame(resolve));

    stats.begin();

    drawFrame(ictx, video, frameSize);
    const { score, keypoints } = await network.estimateSinglePose(icanvas, imageScaleFactor, false, outputStride);
    drawFrame(octx, video, viewportSize);

    if (score < minPoseScore) continue;

    drawKeypoints(octx, keypoints, minPartScore, scaleRate);
    drawSkeleton(octx, keypoints, minPartScore, scaleRate);

    stats.end();
  }
}

window.onload = main;
