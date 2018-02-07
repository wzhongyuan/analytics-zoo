## Image Classification example With BigDL and Storm

In some scenarios, we may want to select images of specific kind out of tons of images, this application shows how to 
apply BigDL in Storm streaming to solve this problem using pre-trained InceptionV1 model

We built a storm topology from top to bottom as listed below

                               .....................
                               .    ImageSpout     .
                               ..................... 
                                         .
                                         .
                               .....................
                               .   ImagePredictor  .
                               .....................   
                                         .
                                         .
                               .....................
                               .    LabelResult    .
                               ..................... 
                                         .
                                         .
                               .....................
                               .     ImageFilter   .
                               .....................

                                         
* **ImageSpout**  A Spout implementation to simulate continously feeding image data
* **ImagePredictor**  A bolt implementation to predict on top of InceptionV1 with incoming image
* **LabelResult** A bolt implementation to map prediction result with real imagenet labels
* **ImageFilter** A bolt implementation to filter out expected images with given keyword

### Steps for run the example

imageFoler=... #image folder you want to predict on, in the example the Spout will continously feed a random image to downstream bolt from this folder

modelPath=... #Bigdl model, you could find pre-trained models in [Model Zoo](https://github.com/intel-analytics/analytics-zoo/tree/master/models)

threshold=... #possibilty threshold, if a predicted category is of the specific kind and the possibility is not less than threshold, the image will be thought to be an expected one

target=... # keyword to specify which kind to filter, i.e. cat, fish, goldfish, etc.

resultFolder=... #where you want to put the expected images

labelPath=... #ImageLabel path, you can download it from [Imagenet labels](../../../../../../../../../../../models/src/main/resources/imagenet_classname.txt)

localMode=... #if run in local model or not

storm jar   models-0.1-SNAPSHOT-jar-with-dependencies.jar \
            com.intel.analytics.zoo.models.imageclassification.example.PredictStreaming \
            -f  $imageFoler \
            --model $modelPath\
            --threshold $threshold\
            --target $target
            --resultFolder $resultFolder
            --labelPath $labelPath
            --localMode $localMode 
