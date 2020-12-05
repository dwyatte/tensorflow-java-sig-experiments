import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TString;
import org.tensorflow.ndarray.StdArrays;

public class SavedModelPredictor {

    public static final String savedModelPath = "/Users/dwyatte/GitHub/tensorflow-java-sig-experiments/python/export/multimodal/0";
    public static final String tagSet = "serve";
    public static final String floatInputName = "serving_default_float_input";
    public static final String stringInputName = "serving_default_string_input";
    public static final String outputName = "StatefulPartitionedCall_1";

    public static void main(String[] args) throws Exception {
        SavedModelBundle savedModel = SavedModelBundle.load(savedModelPath, tagSet);
        Tensor<TFloat32> floatInputTensor = TFloat32.tensorOf(
          StdArrays.ndCopyOf(new float[][]
            {{0.0f}, {1.0f}}
        ));
        Tensor<TString> stringInputTensor = TString.tensorOf(
          StdArrays.ndCopyOf(new String[]
            {"a sentence", "b sentence"}
        ));
        Tensor<TFloat32> outputTensor = savedModel.session().runner()
                                .feed(floatInputName, floatInputTensor)
                                .feed(stringInputName, stringInputTensor)
                                .fetch(outputName).run().get(0)
                                .expect(TFloat32.DTYPE);

        float[][] output = new float[2][1];
        StdArrays.copyTo(output, outputTensor.data());
        System.out.println(output[0][0]);
    }
}