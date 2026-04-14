import { FilesetResolver, HandLandmarker, type HandLandmarkerResult } from '@mediapipe/tasks-vision'
import type { HandResult } from './gesture'

export type HandTracker = {
  detect(video: HTMLVideoElement): HandResult | null
}

export async function loadHandTracker(): Promise<HandTracker> {
  const fileset = await FilesetResolver.forVisionTasks(
    // CDN-hosted WASM assets
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
  )

  const landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      // CDN-hosted model
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
    },
    runningMode: 'VIDEO',
    numHands: 1,
  })

  return {
    detect(video: HTMLVideoElement): HandResult | null {
      const now = performance.now()
      let res: HandLandmarkerResult
      try {
        res = landmarker.detectForVideo(video, now)
      } catch {
        return null
      }
      const lm = res.landmarks?.[0]
      if (!lm || lm.length < 9) return { landmarks: undefined }
      return { landmarks: lm.map((p) => ({ x: p.x, y: p.y })) }
    },
  }
}

