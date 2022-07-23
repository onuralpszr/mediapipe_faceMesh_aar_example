package com.example.myfacemeshdetectionapp


import android.graphics.SurfaceTexture
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.app.AppCompatActivity
import com.example.myfacemeshdetectionapp.databinding.ActivityMainBinding
import com.google.mediapipe.components.CameraHelper.CameraFacing
import com.google.mediapipe.components.CameraXPreviewHelper
import com.google.mediapipe.components.ExternalTextureConverter
import com.google.mediapipe.components.FrameProcessor
import com.google.mediapipe.components.PermissionHelper
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList
import com.google.mediapipe.framework.AndroidAssetUtil
import com.google.mediapipe.framework.Packet
import com.google.mediapipe.framework.PacketGetter
import com.google.mediapipe.glutil.EglManager



class MainActivity : AppCompatActivity() {

    private val TAG = "MainActivity"

    private val INPUT_NUM_FACES_SIDE_PACKET_NAME = "num_faces"
    private val OUTPUT_LANDMARKS_STREAM_NAME = "multi_face_landmarks"
    private val BINARY_GRAPH_NAME = "face_detection_mobile_gpu.binarypb"
    private val INPUT_VIDEO_STREAM_NAME = "input_video"
    private val OUTPUT_VIDEO_STREAM_NAME = "output_video"
    private val NUM_BUFFERS = 2


    // Max number of faces to detect/process.
    private val NUM_FACES = 1
    private val FRONT_CAMERA = true
    private val FLIP_FRAMES_VERTICALLY = true

    companion object {
        init {
            System.loadLibrary("mediapipe_jni")
            System.loadLibrary("opencv_java3")
        }
    }

    // Number of output frames allocated in ExternalTextureConverter.
    // NOTE: use "converterNumBuffers" in manifest metadata to override number of buffers. For
    // example, when there is a FlowLimiterCalculator in the graph, number of buffers should be at
    // least `max_in_flight + max_in_queue + 1` (where max_in_flight and max_in_queue are used in
    // FlowLimiterCalculator options). That's because we need buffers for all the frames that are in
    // flight/queue plus one for the next frame from the camera.


    // Sends camera-preview frames into a MediaPipe graph for processing, and displays the processed
    // frames onto a {@link Surface}.
    protected var processor: FrameProcessor? = null

    // Handles camera access via the {@link CameraX} Jetpack support library.
    protected var cameraHelper: CameraXPreviewHelper? = null

    // {@link SurfaceTexture} where the camera-preview frames can be accessed.
    private var previewFrameTexture: SurfaceTexture? = null

    // {@link SurfaceView} that displays the camera-preview frames processed by a MediaPipe graph.
    private var previewDisplayView: SurfaceView? = null

    // Creates and manages an {@link EGLContext}.
    private var eglManager: EglManager? = null

    // Converts the GL_TEXTURE_EXTERNAL_OES texture from Android camera into a regular texture to be
    // consumed by {@link FrameProcessor} and the underlying MediaPipe graph.
    private var converter: ExternalTextureConverter? = null


    private lateinit var binding: ActivityMainBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(R.layout.activity_main)
        previewDisplayView = SurfaceView(this);
        setupPreviewDisplayView();

        // Initialize asset manager so that MediaPipe native libraries can access the app assets, e.g.,
        // binary graphs.
        // Initialize asset manager so that MediaPipe native libraries can access the app assets, e.g.,
        // binary graphs.
        AndroidAssetUtil.initializeNativeAssetManager(this)

        eglManager = EglManager(null)
        processor = FrameProcessor(
            this,
            eglManager!!.nativeContext,
            BINARY_GRAPH_NAME,
            INPUT_VIDEO_STREAM_NAME,
            OUTPUT_VIDEO_STREAM_NAME
        )
        processor!!
            .videoSurfaceOutput
            .setFlipY(
                FLIP_FRAMES_VERTICALLY
            )

        PermissionHelper.checkAndRequestCameraPermissions(this)



        // This is all for logging
        val packetCreator = processor!!.packetCreator // null error
        val inputSidePackets: MutableMap<String, Packet> = HashMap()
        inputSidePackets[INPUT_NUM_FACES_SIDE_PACKET_NAME] = packetCreator.createInt32(NUM_FACES)
        processor!!.setInputSidePackets(inputSidePackets)



        // To show verbose logging, run:
        // adb shell setprop log.tag.MainActivity VERBOSE
        if (Log.isLoggable(TAG, Log.VERBOSE)) {
            processor!!.addPacketCallback(
                OUTPUT_LANDMARKS_STREAM_NAME
            ) { packet: Packet ->
                Log.v(TAG, "Received multi face landmarks packet.")
                val multiFaceLandmarks =
                    PacketGetter.getProtoVector(
                        packet,
                        NormalizedLandmarkList.parser()
                    )

                Log.v(
                    TAG,
                    "[TS:"
                            + packet.timestamp
                            + "] "
                            + getMultiFaceLandmarksDebugString(multiFaceLandmarks)
                )
            }
        }

    }



    override fun onResume() {
        super.onResume()
        converter = ExternalTextureConverter(
            eglManager!!.context, NUM_BUFFERS
        )
        converter!!.setFlipY(FLIP_FRAMES_VERTICALLY)
        converter!!.setConsumer(processor)


        if (PermissionHelper.cameraPermissionsGranted(this)) {
            startCamera()
            //previewFrameTexture = converter!!.surfaceTexture
        }


    }

    override fun onPause() {
        super.onPause()
        converter!!.close()
        // Hide preview display until we re-open the camera again.
        previewDisplayView!!.visibility = View.GONE
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        PermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }



    private fun onCameraStarted(surfaceTexture: SurfaceTexture) {
        previewFrameTexture = surfaceTexture
        // Make the display view visible to start showing the preview. This triggers the
        // SurfaceHolder.Callback added to (the holder of) previewDisplayView.
        previewDisplayView!!.visibility = View.VISIBLE
    }

    private fun cameraTargetResolution(): Size? {
        return null // No preference and let the camera (helper) decide.
    }

    private fun startCamera() {
        cameraHelper = CameraXPreviewHelper()
        cameraHelper!!.setOnCameraStartedListener { surfaceTexture: SurfaceTexture? ->
            onCameraStarted(surfaceTexture!!)
        }
        val cameraFacing = if (FRONT_CAMERA) CameraFacing.FRONT else CameraFacing.BACK
        cameraHelper!!.startCamera(
            this, cameraFacing, /*surfaceTexture=*/null, cameraTargetResolution()
        )
    }

    private fun computeViewSize(width: Int, height: Int): Size {
        return Size(width, height)
    }

    private fun onPreviewDisplaySurfaceChanged(
        holder: SurfaceHolder?, format: Int, width: Int, height: Int
    ) {
        // (Re-)Compute the ideal size of the camera-preview display (the area that the
        // camera-preview frames get rendered onto, potentially with scaling and rotation)
        // based on the size of the SurfaceView that contains the display.
        val viewSize: Size = computeViewSize(width, height)
        val displaySize: Size = cameraHelper!!.computeDisplaySizeFromViewSize(viewSize)
        val isCameraRotated = cameraHelper!!.isCameraRotated

        converter!!.setSurfaceTextureAndAttachToGLContext(
            previewFrameTexture,
            if (isCameraRotated) displaySize.height else displaySize.width,
            if (isCameraRotated) displaySize.width else displaySize.height
        )

//        // Configure the output width and height as the computed display size.
//        converter!!.setDestinationSize(
//            if (isCameraRotated) displaySize.height else displaySize.width,
//            if (isCameraRotated) displaySize.width else displaySize.height
//        )
    }

    private fun setupPreviewDisplayView() {
        previewDisplayView!!.visibility = View.GONE
        val viewGroup = findViewById<ViewGroup>(R.id.preview_display_layout)
        viewGroup.addView(previewDisplayView)
        previewDisplayView!!
            .holder
            .addCallback(
                object : SurfaceHolder.Callback {
                    override fun surfaceCreated(holder: SurfaceHolder) {
                        processor!!.videoSurfaceOutput.setSurface(holder.surface)
                    }

                    override fun surfaceChanged(
                        holder: SurfaceHolder,
                        format: Int,
                        width: Int,
                        height: Int
                    ) {
                        onPreviewDisplaySurfaceChanged(holder, format, width, height)
                    }

                    override fun surfaceDestroyed(holder: SurfaceHolder) {
                        processor!!.videoSurfaceOutput.setSurface(null)
                    }
                })
    }


    private fun getMultiFaceLandmarksDebugString(
        multiFaceLandmarks: List<NormalizedLandmarkList>
    ): String {
        if (multiFaceLandmarks.isEmpty()) {
            return "No face landmarks"
        }
        var multiFaceLandmarksStr = "Number of faces detected: ${multiFaceLandmarks.size} ".trimIndent()
        for ((faceIndex, landmarks) in multiFaceLandmarks.withIndex()) {
            multiFaceLandmarksStr += "#Face landmarks for face[$faceIndex]: ${landmarks.landmarkCount}"
            for ((landmarkIndex, landmark) in landmarks.landmarkList.withIndex()) {
                multiFaceLandmarksStr += "Landmark [$landmarkIndex]: (${landmark.x}, ${landmark.y}, ${landmark.z})"
            }
        }
        return multiFaceLandmarksStr
    }

}