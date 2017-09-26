
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

package com.intel.analytics.zoo.models.resnet

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.example.loadmodel.AlexNetPreprocessor.imageSize
import com.intel.analytics.bigdl.utils.serializer.ModuleLoader
import com.intel.analytics.zoo.models.Predictor
import com.intel.analytics.zoo.models.dataset.{ImageSample, ImageToMate, MateToSample, PredictResult}
import com.intel.analytics.zoo.transform.vision.image.BytesToMat
import com.intel.analytics.zoo.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, Resize}
import org.apache.spark.rdd.RDD

class ResnetPreprocessor(modelPath: String)  extends Predictor with Serializable {
  model = ModuleLoader.
    loadFromFile[Float](modelPath).evaluate()

  val transformer = ImageToMate() -> BytesToMat() -> Resize(256 , 256) ->
    CenterCrop(imageSize, imageSize) -> ChannelNormalize(0.485f, 0.456f, 0.406f) ->
    MateToSample(false)

  override def predictLocal(path : String, topNum : Int,
                            preprocessor: Transformer[String, ImageSample] = transformer)
  : Array[PredictResult] = {
    doPredictLocal(path, topNum, preprocessor)
  }

  override def predictDistributed(paths : RDD[String], topNum : Int ,
                                  preprocessor: Transformer[String, ImageSample] = transformer):
  RDD[Array[PredictResult]] = {
    doPredictDistributed(paths, topNum, preprocessor)
  }
}