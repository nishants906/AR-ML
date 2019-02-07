package com.amitshekhar.tflite;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.support.annotation.RequiresApi;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.google.ar.core.Anchor;
import com.google.ar.core.AugmentedImage;
import com.google.ar.core.Frame;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.core.TrackingState;
import com.google.ar.core.exceptions.NotYetAvailableException;
import com.google.ar.sceneform.AnchorNode;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.Renderable;
import com.google.ar.sceneform.rendering.ViewRenderable;
import com.google.ar.sceneform.ux.ArFragment;
import com.google.ar.sceneform.ux.TransformableNode;
import com.wonderkiln.camerakit.CameraKitError;
import com.wonderkiln.camerakit.CameraKitEvent;
import com.wonderkiln.camerakit.CameraKitEventListener;
import com.wonderkiln.camerakit.CameraKitImage;
import com.wonderkiln.camerakit.CameraKitVideo;
import com.wonderkiln.camerakit.CameraView;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String MODEL_PATH = "mobilenet_quant_v1_224.tflite";
    private static final boolean QUANT = true;
    private static final String LABEL_PATH = "labels.txt";
    private static final int INPUT_SIZE = 224;

    private Classifier classifier;

    private Executor executor = Executors.newSingleThreadExecutor();
    private TextView textViewResult;
    private Button btnDetectObject, btnToggleCamera;
    private ImageView imageViewResult;
    private ArFragment fragment;

    Boolean hasplace=false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageViewResult = findViewById(R.id.imageViewResult);
        textViewResult = findViewById(R.id.textViewResult);
        textViewResult.setMovementMethod(new ScrollingMovementMethod());

        fragment = (ArFragment) getSupportFragmentManager().findFragmentById(R.id.cameraView);
        fragment.getPlaneDiscoveryController().hide();
        fragment.getPlaneDiscoveryController().setInstructionView(null);

        fragment.getArSceneView().getScene().addOnUpdateListener(this::onUpdateFrame);


        btnToggleCamera = findViewById(R.id.btnToggleCamera);
        btnDetectObject = findViewById(R.id.btnDetectObject);


        initTensorFlowAndLoadModel();
    }

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    private void onUpdateFrame(FrameTime frameTime) {

        Log.d("callingit","calling");


        // If there is no frame then don't process anything.
        if (fragment.getArSceneView().getArFrame() == null) {
            return;
        }

        // If ARCore is not tracking yet, then don't process anything.
        if (fragment.getArSceneView().getArFrame().getCamera().getTrackingState() != TrackingState.TRACKING) {
            return;
        }




        Image cameraImage = null;
        try {

            Frame frame = fragment.getArSceneView().getArFrame();


            cameraImage = frame.acquireCameraImage();

//The camera image received is in YUV YCbCr Format. Get buffers for each of the planes and use them to create a new bytearray defined by the size of all three buffers combined
            ByteBuffer cameraPlaneY = cameraImage.getPlanes()[0].getBuffer();
            ByteBuffer cameraPlaneU = cameraImage.getPlanes()[1].getBuffer();
            ByteBuffer cameraPlaneV = cameraImage.getPlanes()[2].getBuffer();

//Use the buffers to create a new byteArray that
            byte[] compositeByteArray = new byte[(cameraPlaneY.capacity() + cameraPlaneU.capacity() + cameraPlaneV.capacity())];

            cameraPlaneY.get(compositeByteArray, 0, cameraPlaneY.capacity());
            cameraPlaneU.get(compositeByteArray, cameraPlaneY.capacity(), cameraPlaneU.capacity());
            cameraPlaneV.get(compositeByteArray, cameraPlaneY.capacity() + cameraPlaneU.capacity(), cameraPlaneV.capacity());

            ByteArrayOutputStream baOutputStream = new ByteArrayOutputStream();
            YuvImage yuvImage = new YuvImage(compositeByteArray, ImageFormat.NV21, cameraImage.getWidth(), cameraImage.getHeight(), null);
            yuvImage.compressToJpeg(new Rect(0, 0, cameraImage.getWidth(), cameraImage.getHeight()), 75, baOutputStream);
            byte[] byteForBitmap = baOutputStream.toByteArray();
            Bitmap bitmap = BitmapFactory.decodeByteArray(byteForBitmap, 0, byteForBitmap.length);
            imageViewResult.setImageBitmap(bitmap);
            cameraImage.close();


                bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);

                imageViewResult.setImageBitmap(bitmap);

                final List<Classifier.Recognition> results = classifier.recognizeImage(bitmap);
                Log.d("resultsize", String.valueOf(results.size()));

            textViewResult.setText(String.valueOf(results));
                if (results.size()>0) {
                    if (!String.valueOf(results.get(0)).equals("null")&&!hasplace) {
                        if (String.valueOf(results.get(0).getTitle()).equals("laptop")){
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                            hasplace = true;
                            placeObject(fragment, Uri.parse("laptop.sfb"), " ");
                        }

                        }
                    }
                }


        }catch (NotYetAvailableException e) {
            Log.e("erroroccur",e.toString());
        };

    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.execute(new Runnable() {
            @Override
            public void run() {
                classifier.close();
            }
        });
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TensorFlowImageClassifier.create(
                            getAssets(),
                            MODEL_PATH,
                            LABEL_PATH,
                            INPUT_SIZE,
                            QUANT);
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    private void makeButtonVisible() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
            }
        });
    }


    @RequiresApi(api = Build.VERSION_CODES.N)
    public void placeObject(ArFragment fragment, Uri model, String name){

        Session session = fragment.getArSceneView().getSession();
        Pose p =fragment.getArSceneView().getArFrame().getCamera().getPose();
        float[] pos = p.getTranslation();
        float[] rotation = {0,0,0,1};
        Anchor anchor =  session.createAnchor(new Pose(pos,rotation));



        ModelRenderable.builder()
                .setSource(fragment.getContext(),model)
                .build()
                .thenAccept(renderable -> addnodetoscene(fragment, anchor, renderable))
                .exceptionally(throwable -> {
                    Log.d("enter1",throwable.getMessage());
                    AlertDialog.Builder builder = new AlertDialog.Builder(this);
                    builder.setMessage(throwable.getMessage()).setTitle("Error");
                    AlertDialog dialog = builder.create();
                    dialog.show();
                    return null;
                });


    }

    public void addnodetoscene(ArFragment fragment, Anchor anchor,Renderable renderable){

        AnchorNode anchorNode = new AnchorNode(anchor);
        anchorNode.setParent(fragment.getArSceneView().getScene());

        TransformableNode node = new TransformableNode(fragment.getTransformationSystem());
        node.setParent(anchorNode);
        node.setRenderable(renderable);
    }


}
