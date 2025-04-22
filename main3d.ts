// import cv from "@techstark/opencv-js";
// import * as tf from "@tensorflow/tfjs"; 
// import {
//   PoseLandmarker,
//   FilesetResolver,
//   NormalizedLandmark,
// } from "@mediapipe/tasks-vision";

// let runningMode: "VIDEO" | "IMAGE" = "IMAGE";
// let webcamRunning: Boolean = false;

// const videoHeight = `${360 * 2}px`;
// const videoWidth = `${480 * 2}px`;

// // Before we can use PoseLandmarker class we must wait for it to finish
// // loading. Machine Learning models can be large and take a moment to
// // get everything needed to run.

// const vision = await FilesetResolver.forVisionTasks(
//   "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
// );
// const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
//   baseOptions: {
//     modelAssetPath: `pose_landmarker_heavy.task`,
//     delegate: "GPU",
//   },
//   runningMode: runningMode,
//   numPoses: 1,
//   minPoseDetectionConfidence: 0.5,
//   minPosePresenceConfidence: 0.5,
//   minTrackingConfidence: 0.5,
// });

// let midas: any;
// async function loadMiDaSModel() {
//   // Load the MiDaS model from the TensorFlow Hub link
//   const model = await tf.loadGraphModel("midas_u8/model.json"); // Change to the model path if locally available
//   console.log('MiDaS Model loaded');
//   return model;
// }



//   midas = await loadMiDaSModel();
//   async function estimateDepth(video: HTMLVideoElement) {
//     const inputTensor = tf.browser.fromPixels(video); // Convert video frame to tensor
//     const processedImage = inputTensor.resizeBilinear([384, 384]).expandDims(0).toFloat(); // Preprocess image
  
//     // Get depth map prediction from MiDaS
//     const depthPrediction = midas.predict(processedImage);
//     const depthMap = depthPrediction.squeeze(); // Remove batch dimension
//     return depthMap;
//   }
//     // Load MiDaS model for depth estimation
// //midas = await midas.load();
// /********************************************************************
// // Demo 2: Continuously grab image from webcam stream and detect it.
// ********************************************************************/

// const video = document.getElementById("webcam") as HTMLVideoElement;
// const canvasElement = document.getElementById(
//   "output_canvas"
// ) as HTMLCanvasElement;

// const canvasCtx = canvasElement.getContext("2d")!;

// // Check if webcam access is supported.
// const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

// // If webcam supported, add event listener to button for when user
// // wants to activate it.
// if (hasGetUserMedia()) {
//   enableCam();
// } else {
//   console.warn("getUserMedia() is not supported by your browser");
// }

// // Enable the live webcam view and start detection.
// function enableCam() {
//   if (!poseLandmarker) {
//     console.log("Wait! poseLandmaker not loaded yet.");
//     return;
//   }

//   if (webcamRunning === true) {
//     webcamRunning = false;
//   } else {
//     webcamRunning = true;
//   }

//   // getUsermedia parameters.
//   const constraints = {
//     video: true,
//   };

//   // Activate the webcam stream.
//   navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
//     video.srcObject = stream;
//     video.addEventListener("loadeddata", predictWebcam);
//   });
// }



// // Create the camera
// const getLandMark = (index: number, landmark: NormalizedLandmark[][]) => {
//   if (!landmark[0]) return undefined;
//   if (!landmark[0][index]) return undefined;
//   return landmark[0][index];
// };
//    // Helper function to get depth from depth map using 2D pose coordinates
// function getDepthFromMap(landmark: NormalizedLandmark, depthMap: any): number {
//     const x = Math.floor(landmark.x * depthMap.width);
//     const y = Math.floor(landmark.y * depthMap.height);
//     return depthMap.data[y * depthMap.width + x];
//   }



// const dsize = new cv.Size(canvasElement.width, canvasElement.height);
// // const drawingUtils = new DrawingUtils(canvasCtx);

// // Pass in fov, near, far and camera position respectively
// let lastVideoTime = -1;
// async function predictWebcam() {
//   canvasElement.style.height = videoHeight;
//   video.style.height = videoHeight;
//   canvasElement.style.width = videoWidth;
//   video.style.width = videoWidth;
//   // Now let's start detecting the stream.
//   if (runningMode === "IMAGE") {
//     runningMode = "VIDEO";
//     await poseLandmarker.setOptions({ runningMode: "VIDEO" });
//   }
//   let startTimeMs = performance.now();
//   if (lastVideoTime !== video.currentTime) {
//     lastVideoTime = video.currentTime;
//     poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
//       canvasCtx.save();
//       canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
//       const ls = getLandMark(11, result.landmarks);
//       const rs = getLandMark(12, result.landmarks);
//       const lh = getLandMark(23, result.landmarks);
//       const rh = getLandMark(24, result.landmarks);
//       // Get depth map from MiDaS
//       const depthMap = await estimateDepth(video)
//       const depth = getDepthFromMap(ls, depthMap);
  
//             // Convert to 3D (x, y, z) using depth
//       const x3D = ls.x * depth;  // Multiply x with depth value to get 3D x
//       const y3D = ls.y * depth;  // Multiply y with depth value to get 3D y
//       const z3D = depth;  // Depth as z value
  
//             // Overlay the 3D keypoints on the image by drawing them on the canvas
//             canvasCtx.beginPath();
//             canvasCtx.arc(ls.x * canvasElement.width, ls.y * canvasElement.height, 5, 0, 2 * Math.PI);
//             canvasCtx.fillStyle = 'red';
//             canvasCtx.fill();
//             console.log(`3D Point at ls: X: ${x3D}, Y: ${y3D}, Z: ${z3D}`);
//             // Optional: log the 3D coordinates
//             const landmarks = result.landmarks;
//             for (let i = 0; i < landmarks[0].length; i++) {
//                 const landmark = landmarks[0][i];
//                 const depth = getDepthFromMap(landmark, depthMap);
        
//                 const x3D = landmark.x * depth;
//                 const y3D = landmark.y * depth;
//                 const z3D = depth;
        
//                 // Draw the 3D points on the canvas (in 2D)
//                 canvasCtx.beginPath();
//                 canvasCtx.arc(landmark.x * canvasElement.width, landmark.y * canvasElement.height, 5, 0, 2 * Math.PI);
//                 canvasCtx.fillStyle = "red";
//                 canvasCtx.fill();
        
//                 // Show 3D values in the console
//                 console.log(`3D Point at ${i}: X: ${x3D}, Y: ${y3D}, Z: ${z3D}`);
//               }


           
 
//       canvasCtx.restore();
//     });
//   }

//   // Call this function again to keep predicting when the browser is ready.
//   if (webcamRunning === true) {
//     window.requestAnimationFrame(predictWebcam);
//   }
// }
import * as tf from "@tensorflow/tfjs"; // Load TensorFlow.js
import cv from "@techstark/opencv-js"; // OpenCV for JS
import {
  PoseLandmarker,
  FilesetResolver,
  NormalizedLandmark,
} from "@mediapipe/tasks-vision";

let runningMode: "VIDEO" | "IMAGE" = "IMAGE";
let webcamRunning: Boolean = false;

const videoHeight = `720px`;
const videoWidth = `960px`;
const videoHeight1 = 720;
const videoWidth1 = 960;
// Declare midas variable for depth estimation
let midas: any;

async function loadMiDaSModel() {
  // Load the MiDaS model from a local or remote TensorFlow.js model file
  midas = await tf.loadGraphModel("midas_u8/model.json"); // Replace with correct path if needed
  console.log("MiDaS Model loaded");
}

// Load MiDaS model for depth estimation
await loadMiDaSModel();

// Setup MediaPipe Pose
const vision = await FilesetResolver.forVisionTasks(
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);

const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
  baseOptions: {
    modelAssetPath: `pose_landmarker_heavy.task`,
    delegate: "GPU",
  },
  runningMode: runningMode,
  numPoses: 1,
  minPoseDetectionConfidence: 0.5,
  minPosePresenceConfidence: 0.5,
  minTrackingConfidence: 0.5,
});

// Set up the video and canvas for pose detection
const video = document.getElementById("webcam") as HTMLVideoElement;
const canvasElement = document.getElementById("output_canvas") as HTMLCanvasElement;
const canvasCtx = canvasElement.getContext("2d")!;

// Check if webcam access is supported
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

// Start the webcam if supported
if (hasGetUserMedia()) {
  enableCam();
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

// Initialize webcam
function enableCam() {
  if (!poseLandmarker) {
    console.log("Wait! PoseLandmaker not loaded yet.");
    return;
  }

  if (webcamRunning === true) {
    webcamRunning = false;
  } else {
    webcamRunning = true;
  }

  const constraints = {
    video: true,
  };

  // Activate the webcam stream
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

// Create a helper function to get a specific pose landmark by index
const getLandMark = (index: number, landmark: NormalizedLandmark[][]) => {
  if (!landmark[0]) return undefined;
  if (!landmark[0][index]) return undefined;
  return landmark[0][index];
};

// Helper function to get depth from depth map using 2D pose coordinates
// function getDepthFromMap(landmark: NormalizedLandmark, depthMap: any): number {
//   const x = Math.floor(landmark.x * depthMap.shape[1]);
//   const y = Math.floor(landmark.y * depthMap.shape[0]);
//   const depthArray = depthMap.arraySync();  // Convert tensor to 2D array

//   // Return the depth value at the (x, y) position
//   return depthArray[y][x];

//   //return depthMap.get(y, x); // Get the depth value from the depth map
// }



function getDepthFromMap(landmark: NormalizedLandmark, depthMap: any): number {
  const x = Math.floor(landmark.x * depthMap.shape[0]);
  const y = Math.floor(landmark.y * depthMap.shape[1]);
  console.log(`  Point at   X: ${x }, Y: ${y}, dMW:  ${depthMap.shape[0]}, DMH: ${depthMap.shape[1]}`);
    
  // Check that the x, y coordinates are within valid bounds
  if (x < 0 || x >= depthMap.shape[1] || y < 0 || y >= depthMap.shape[0]) {
    console.log("Invalid depth map coordinates: x = ${x}, y = ${y}");
    return 0;  // Return a default value if coordinates are out of bounds
  }

  // Convert depth map tensor to a 2D array (if it's not already)
  const depthArray = depthMap.arraySync();  // Convert tensor to 2D array

  // Ensure that the depthArray is a valid 2D array
  if (!depthArray || !Array.isArray(depthArray) || !Array.isArray(depthArray[y])) {
    console.log("Failed to convert depth map to a valid array.");
    return 0;  // Return a default value if the conversion failed
  }

  // Return the depth value at the (x, y) position
  return depthArray[y][x];
}


// Process the webcam video to predict pose and depth
let lastVideoTime = -1;
async function predictWebcam() {
 



  canvasElement.style.height = videoHeight;
  video.style.height = videoHeight;
  canvasElement.style.width = videoWidth;
  video.style.width = videoWidth;

  // Now let's start detecting the stream
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await poseLandmarker.setOptions({ runningMode: "VIDEO" });
  }

  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    canvasCtx.save(); // Save current canvas state
  
    // // Flip the canvas horizontally
     canvasCtx.scale(-1, 1);
    canvasCtx.translate(-canvasElement.width, 0);
    
    // Draw the video frame
    canvasCtx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);
    
     canvasCtx.restore(); // Restore the original state
     const canvasImage = canvasElement;
    // Detect pose and process depth map
    poseLandmarker.detectForVideo(canvasImage, startTimeMs, async (result) => {
      // canvasCtx.save();
      // canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      // canvasCtx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);

      const landmarks = result.landmarks;
      const depthMap = await estimateDepth(canvasImage); // Get depth map from MiDaS

      console.log(canvasElement.width);
      console.log(canvasElement.height);
      console.log(depthMap.shape);

      //const resizedDepthMap = tf.image.resizeBilinear(depthMap, [ canvasElement.height,  canvasElement.width]);
      // Process each landmark and get the depth
      for (let i = 0; i < landmarks[0].length; i++) {
        const landmark = landmarks[0][i];
        const depth = getDepthFromMap(landmark, depthMap);

        // Convert to 3D (x, y, z) using depth
        const x3D = landmark.x * depth;
        const y3D = landmark.y * depth;
        const z3D = depth;

        // Draw the 3D points on the canvas (in 2D)
        canvasCtx.beginPath();
        canvasCtx.arc(landmark.x * canvasElement.width, landmark.y * canvasElement.height, 5, 0, 2 * Math.PI);
        canvasCtx.fillStyle = "red";
        canvasCtx.fill();

        // Show 3D values in the console
        console.log(`3D Point at ${i}: X: ${x3D}, Y: ${y3D}, Z: ${z3D}`);
      }

      canvasCtx.restore();
    });
  }

  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}
async function estimateDepth(video: HTMLCanvasElement) {
  let inputTensor = tf.browser.fromPixels(video); // Convert video frame to tensor

  // Normalize the image (divide by 255 to scale pixel values to [0, 1])
  inputTensor = tf.div(inputTensor, 255);

  // Resize the image to [256, 256]
  inputTensor = tf.image.resizeBilinear(inputTensor, [256, 256]);

  // Ensure the tensor has 3 channels (RGB). If the input tensor has more than 3 channels, we can use `slice()` to get the first 3 channels.
  if (inputTensor.shape[2] !== 3) {
    inputTensor = inputTensor.slice([0, 0, 0], [-1, -1, 3]);  // Keep only the first 3 channels
  }

  // Transpose the tensor to the required format [batch_size, channels, height, width] for the MiDaS model
  inputTensor = tf.transpose(inputTensor, [2, 0, 1]);  // [3, 256, 256]

  // Expand dims to add batch size: [1, 3, 256, 256]
  inputTensor = tf.expandDims(inputTensor);

  // Use executeAsync() to handle dynamic ops in the MiDaS model
  const depthPrediction = await midas.executeAsync(inputTensor); // Use executeAsync()

  // Remove batch dimension and return the depth map
  const depthMap = depthPrediction.squeeze();  // Shape becomes [256, 256]

  // Expand the depth map to have 3 dimensions [height, width, 1]
  const expandedDepthMap = depthMap.expandDims(-1);  // Shape becomes [256, 256, 1]

  // Resize the depth map back to the original video size (e.g., 640x480)
  const originalWidth = videoHeight1; //video.videoWidth;  // Get original width of the video
  const originalHeight = videoWidth1; //video.videoHeight;  // Get original height of the video
  const resizedDepthMap = tf.image.resizeBilinear(expandedDepthMap, [originalHeight, originalWidth]);

  return resizedDepthMap;
}


// async function estimateDepth(video: HTMLVideoElement) {
//   let inputTensor = tf.browser.fromPixels(video); // Convert video frame to tensor

//   // Normalize the image (divide by 255 to scale pixel values to [0, 1])
//   inputTensor = tf.div(inputTensor, 255);

//   // Resize the image to [256, 256]
//   inputTensor = tf.image.resizeBilinear(inputTensor, [256, 256]);

//   // Ensure the tensor has 3 channels (RGB). If the input tensor has more than 3 channels, we can use `slice()` to get the first 3 channels.
//   if (inputTensor.shape[2] !== 3) {
//     inputTensor = inputTensor.slice([0, 0, 0], [-1, -1, 3]);  // Keep only the first 3 channels
//   }

//   // Transpose the tensor to the required format [batch_size, channels, height, width] for the MiDaS model
//   inputTensor = tf.transpose(inputTensor, [2, 0, 1]);  // [3, 256, 256]

//   // Expand dims to add batch size: [1, 3, 256, 256]
//   inputTensor = tf.expandDims(inputTensor);

//   // Use executeAsync() to handle dynamic ops in the MiDaS model
//   const depthPrediction = await midas.executeAsync(inputTensor); // Use executeAsync()

//   // Remove batch dimension and return the depth map
//   const depthMap = depthPrediction.squeeze();  // Shape becomes [256, 256]

//   // Resize the depth map back to the original video size (e.g., 640x480)
//   const originalWidth = video.videoWidth;  // Get original width of the video
//   const originalHeight = video.videoHeight;  // Get original height of the video
//   const resizedDepthMap = tf.image.resizeBilinear(depthMap, [originalHeight, originalWidth]);

//   return resizedDepthMap;
// }
 
// async function estimateDepth(video: HTMLVideoElement) {
//   const inputTensor = tf.browser.fromPixels(video); // Convert video frame to tensor

//   // Preprocess image: Resize to [256, 256] and ensure correct order [1, 3, 256, 256]
//   const resizedImage = inputTensor.resizeBilinear([256, 256]).toFloat();  // Resize to 256x256
//   const normalizedImage = resizedImage.div(tf.scalar(255)); // Normalize to [0, 1]

//   // Rearrange dimensions from [height, width, channels] to [batch size, channels, height, width]
//   const batchedImage = normalizedImage.expandDims(0); // Add batch dimension: [1, 256, 256, 3]
//   const transposedImage = batchedImage.transpose([0, 3, 1, 2]); // Transpose to [1, 3, 256, 256]

//   // Use executeAsync() to handle dynamic ops in the MiDaS model
//   const depthPrediction = await midas.executeAsync(transposedImage); // Use executeAsync()
  
//   const depthMap = depthPrediction.squeeze(); // Remove batch dimension

//   // Expand the depth map to have 3 dimensions [height, width, 1]
//   const expandedDepthMap = depthMap.expandDims(-1);  // Shape becomes [height, width, 1]

//   // Resize the depth map back to the original video size (e.g., 640x480)
//   const originalWidth = video.videoWidth;  // Get original width of the video
//   const originalHeight = video.videoHeight;  // Get original height of the video
//   const resizedDepthMap = tf.image.resizeBilinear(expandedDepthMap, [originalHeight, originalWidth]);

//   return resizedDepthMap;
// }


// export async function infer(input) {
//   input = deserializeTensor(input);
//   await modelLoaded;
//   input = tf.div(input, 255);
//   input = tf.transpose(input, [2, 0, 1]);
//   input = tf.expandDims(input);

//   let output = await model.executeAsync(input);

//   output = tf.transpose(output, [1, 2, 0]);
//   output = tf.div(
//     tf.sub(output, tf.min(output)),
//     tf.sub(tf.max(output), tf.min(output))
//   );
//   return serializeTensor(output);
// }

// Function to estimate depth using MiDaS (local version)
 
  // async function estimateDepth(video: HTMLVideoElement) {
  //   const inputTensor = tf.browser.fromPixels(video); // Convert video frame to tensor
  
  //   // Preprocess image: Resize to [256, 256] and ensure correct order [1, 3, 256, 256]
  //   const resizedImage = inputTensor.resizeBilinear([256, 256]).toFloat();  // Resize to 256x256
  //   const normalizedImage = resizedImage.div(tf.scalar(255)); // Normalize to [0, 1]
  
  //   // Rearrange dimensions from [height, width, channels] to [batch size, channels, height, width]
  //   const batchedImage = normalizedImage.expandDims(0); // Add batch dimension: [1, 256, 256, 3]
  //   const transposedImage = batchedImage.transpose([0, 3, 1, 2]); // Transpose to [1, 3, 256, 256]
  
  //   // Get depth map prediction from MiDaS
  //   const depthPrediction =await midas.executeAsync(transposedImage); //midas.predict(transposedImage);
  //   const depthMap = depthPrediction.squeeze(); // Remove batch dimension
  
  //   // Resize the depth map back to the original video size (e.g., 640x480)
  //   const originalWidth = video.videoWidth;  // Get original width of the video
  //   const originalHeight = video.videoHeight;  // Get original height of the video
  //   const resizedDepthMap = tf.image.resizeBilinear(depthMap, [originalHeight, originalWidth]);
  
  //   return resizedDepthMap;
  // }
  
 
