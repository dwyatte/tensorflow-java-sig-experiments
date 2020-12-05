import java.util.Map;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TString;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.proto.framework.MetaGraphDef;
import org.tensorflow.proto.framework.SignatureDef;
import org.tensorflow.proto.framework.TensorInfo;

public class SavedModelPredictorInspectMetaGraph {

    public static final String savedModelPath = "/Users/dwyatte/GitHub/tensorflow-java-sig-experiments/python/export/multimodal/0";
    public static final String tagSet = "serve";
    public static final String signatureDefKey = "serving_default";
    public static final Map<String, Object> inputMap = Map.ofEntries(
      Map.entry("float_input", new float[][]{{0.0f}, {1.0f}}),
      Map.entry("string_input", new String[]{"a sentence", "b sentence"})
    );
    public static final String outputKey = "output_1";

    public static void main(String[] args) throws Exception {
        SavedModelBundle savedModel = SavedModelBundle.load(savedModelPath, tagSet);
        MetaGraphDef metaGraphDef = savedModel.metaGraphDef();
        SignatureDef signatureDef = metaGraphDef.getSignatureDefMap().get(signatureDefKey);
        Map<String, TensorInfo> inputsMap = signatureDef.getInputsMap();
        Map<String, TensorInfo> outputsMap = signatureDef.getOutputsMap();

        // loop over inputMap, get hardcoded value and inject into graph
        Runner runner = savedModel.session().runner();
        for (Map.Entry<String, Object> inputEntry : inputMap.entrySet()) {
          String inputKey = inputEntry.getKey();
          String inputName = inputsMap.get(inputKey).getName();
          Object inputValue = inputEntry.getValue();

          if (inputValue instanceof float[][]) {
            runner = runner.feed(inputName, TFloat32.tensorOf(
              StdArrays.ndCopyOf( (float[][]) inputValue )
            ));
          }
          else if (inputValue instanceof String[]) {
            runner = runner.feed(inputName, TString.tensorOf(
              StdArrays.ndCopyOf( (String[]) inputValue )
            ));
          }
        }

        // get extract the output from the graph
        String outputName = outputsMap.get(outputKey).getName();
        Tensor<TFloat32> outputTensor = runner.fetch(outputName)
                                              .run()
                                              .get(0)
                                              .expect(TFloat32.DTYPE);
        System.out.println(outputTensor.data().getFloat(0, 0));
    }
}