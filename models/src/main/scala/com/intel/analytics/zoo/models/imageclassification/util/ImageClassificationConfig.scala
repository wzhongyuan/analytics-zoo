/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.models.imageclassification.util

import java.net.URL

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFrameToSample, MatToTensor}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, PixelNormalizer, Resize}
import com.intel.analytics.zoo.models.Configure
import com.intel.analytics.zoo.models.imageclassification.util.Dataset.{Imagenet, Places365}

import scala.io.Source

object ImageClassificationConfig {
  val models = Set("alexnet",
    "inception-v1",
    "resnet-50",
    "vgg-16",
    "vgg-19",
    "densenet-161",
    "squeezenet",
    "mobilenet")

  def apply(model: String, dataset: String, version: String): Configure = {
    dataset match {
      case "imagenet" => ImagenetConfig(model, dataset, version)
      case "places365" => Places365Config(model, dataset, version)
      case _ => throw new RuntimeException(s"dataset $dataset not supported for now")
    }
  }

  private[models] def createMean(meanFile : URL) : Array[Float] = {
    val lines = Source.fromURL(meanFile).getLines.toArray
    val array = new Array[Float](lines.size)
    lines.zipWithIndex.foreach(data => {
      array(data._2) = data._1.toFloat
    })
    array
  }
}

object ImagenetConfig {

  val meanFile = getClass().getResource("/mean.txt")

  val mean : Array[Float] = ImageClassificationConfig.createMean(meanFile)

  val imagenetLabelMap = LabelReader(Imagenet.value)

  def apply(model: String, dataset: String, version: String): Configure = {
    model match {
      case "alexnet" => Configure(preProcessor = alexnetImagenetPreprocessor,
        labelMap = imagenetLabelMap)
      case "inception-v1" => Configure(preProcessor = inceptionV1ImagenetPreprocessor,
        labelMap = imagenetLabelMap)
      case "resnet-50" => Configure(preProcessor = resnetImagenetPreprocessor,
        labelMap = imagenetLabelMap)
      case "vgg-16" => Configure(preProcessor = vggImagenetPreprocessor,
        labelMap = imagenetLabelMap)
      case "vgg-19" => Configure(preProcessor = vggImagenetPreprocessor,
        labelMap = imagenetLabelMap)
      case "densenet-161" => Configure(preProcessor = densenetImagenetPreprocessor,
        labelMap = imagenetLabelMap)
      case "squeezenet" => Configure(preProcessor = squeezenetImagenetPreprocessor,
        labelMap = imagenetLabelMap)
      case "mobilenet" => Configure(preProcessor = mobilenetImagenetPreprocessor,
        labelMap = imagenetLabelMap)
    }
  }

  def alexnetImagenetPreprocessor() : FeatureTransformer = {
    Resize(Consts.IMAGENET_RESIZE, Consts.IMAGENET_RESIZE) ->
      PixelNormalizer(mean) -> CenterCrop(227, 227) ->
      MatToTensor() -> ImageFrameToSample()
  }

  def commonPreprocessor(imageSize : Int, meanR: Float, meanG: Float, meanB: Float,
                         stdR: Float = 1, stdG: Float = 1, stdB: Float = 1) : FeatureTransformer = {
    Resize(Consts.IMAGENET_RESIZE, Consts.IMAGENET_RESIZE) ->
      CenterCrop(imageSize, imageSize) -> ChannelNormalize(meanR, meanG, meanB,
      stdR, stdG, stdB) ->
      MatToTensor() -> ImageFrameToSample()
  }

  def inceptionV1ImagenetPreprocessor(): FeatureTransformer = {
    commonPreprocessor(224, 123, 117, 104)
  }

  def resnetImagenetPreprocessor() : FeatureTransformer = {
    commonPreprocessor(224, 0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f)
  }

  def vggImagenetPreprocessor(): FeatureTransformer = {
    commonPreprocessor(224, 123, 117, 104)
  }

  def densenetImagenetPreprocessor() : FeatureTransformer = {
    commonPreprocessor(224, 123, 117, 104, 1/0.017f, 1/0.017f, 1/0.017f)
  }

  def mobilenetImagenetPreprocessor() : FeatureTransformer = {
    commonPreprocessor(224, 123.68f, 116.78f, 103.94f, 1/0.017f, 1/0.017f, 1/0.017f )
  }

  def squeezenetImagenetPreprocessor(): FeatureTransformer = {
    commonPreprocessor(227, 123, 117, 104)
  }
}

object Places365Config {

  val meanFile = getClass().getResource("/places365_mean.txt")

  val mean : Array[Float] = ImageClassificationConfig.createMean(meanFile)

  val places365LabelMap = LabelReader(Places365.value)

  def apply(model: String, dataset: String, version: String): Configure = {
    model match {
      case "alexnet" => Configure(preProcessor = alexnetPlaces365Preprocessor,
        labelMap = places365LabelMap)
      case "vgg-16" => Configure(preProcessor = vggPlaces365Preprocessor,
        labelMap = places365LabelMap)
    }
  }

  def alexnetPlaces365Preprocessor() : FeatureTransformer = {
    Resize(Consts.IMAGENET_RESIZE, Consts.IMAGENET_RESIZE) ->
      PixelNormalizer(mean) -> CenterCrop(227, 227) ->
      MatToTensor() -> ImageFrameToSample()
  }

  def vggPlaces365Preprocessor() : FeatureTransformer = {
    Resize(Consts.IMAGENET_RESIZE, Consts.IMAGENET_RESIZE) ->
      PixelNormalizer(mean) -> CenterCrop(224, 224) ->
      MatToTensor() -> ImageFrameToSample()
  }
}

sealed trait Dataset {
  val value: String
}

object Dataset {
  def apply(datasetString: String): Dataset = {
    datasetString.toUpperCase match {
      case Imagenet.value => Imagenet
      case Places365.value => Places365
    }
  }

  case object Imagenet extends Dataset {
    val value = "IMAGENET"
  }

  case object Places365 extends Dataset {
    val value = "PLACES365"
  }

}

object Consts {
  val IMAGENET_RESIZE : Int = 256
}
